// scene_renderer.cpp - 场景渲染器实现
#include "scene_renderer.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <cmath>

// 图像保存库（需要添加到项目）
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// 可选：OpenEXR 支持
#ifdef USE_OPENEXR
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#endif

namespace diff_optics {

// 辅助函数
static void check_cuda_render(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

static void check_optix_render(OptixResult res, const char* msg) {
    if (res != OPTIX_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": OptiX error " + std::to_string(res));
    }
}

SceneRenderer::SceneRenderer() {}

SceneRenderer::~SceneRenderer() {
    unload_environment_map();
    
    if (m_color_buffer) cudaFree(reinterpret_cast<void*>(m_color_buffer));
    if (m_sample_count) cudaFree(reinterpret_cast<void*>(m_sample_count));
    if (m_launch_params_buffer) cudaFree(reinterpret_cast<void*>(m_launch_params_buffer));
    
    if (m_raygen_record) cudaFree(reinterpret_cast<void*>(m_raygen_record));
    if (m_miss_record) cudaFree(reinterpret_cast<void*>(m_miss_record));
    if (m_hitgroup_records) cudaFree(reinterpret_cast<void*>(m_hitgroup_records));
    
    if (m_render_pipeline) optixPipelineDestroy(m_render_pipeline);
    if (m_render_hitgroup_pg) optixProgramGroupDestroy(m_render_hitgroup_pg);
    if (m_render_miss_pg) optixProgramGroupDestroy(m_render_miss_pg);
    if (m_render_raygen_pg) optixProgramGroupDestroy(m_render_raygen_pg);
    if (m_render_module) optixModuleDestroy(m_render_module);
    
    // 注意：不销毁 m_context，因为它可能被 OptiXSceneManager 共享
}

bool SceneRenderer::initialize(const std::string& ptx_dir) {
    if (m_initialized) return true;
    
    m_ptx_dir = ptx_dir;
    
    try {
        // 获取 OptiX context（从全局管理器）
        OptiXSceneManager* mgr = get_optix_manager();
        if (!mgr || !mgr->is_initialized()) {
            std::cerr << "[SceneRenderer] OptiXSceneManager not initialized" << std::endl;
            return false;
        }
        
        // 共享 context（需要在 OptiXSceneManager 中暴露）
        // 这里我们重新创建一个，但使用相同的 CUDA context
        CUcontext cu_ctx = nullptr;
        cuCtxGetCurrent(&cu_ctx);
        
        OptixDeviceContextOptions options = {};
        options.logCallbackLevel = 3;
        
        check_optix_render(optixDeviceContextCreate(cu_ctx, &options, &m_context), "Create render context");
        
        // 加载渲染用 PTX
        std::string render_ptx_path = m_ptx_dir + "/optix_render_kernels.ptx";
        std::ifstream ptx_file(render_ptx_path);
        if (!ptx_file.is_open()) {
            std::cerr << "[SceneRenderer] Cannot open PTX: " << render_ptx_path << std::endl;
            return false;
        }
        
        std::stringstream buffer;
        buffer << ptx_file.rdbuf();
        m_render_ptx_code = buffer.str();
        ptx_file.close();
        
        if (!create_render_module()) {
            return false;
        }
        
        create_render_program_groups();
        create_render_pipeline();
        
        check_cuda_render(cudaMalloc(reinterpret_cast<void**>(&m_launch_params_buffer), 
                                     sizeof(RenderLaunchParams)), "Alloc launch params");
        
        m_initialized = true;
        std::cout << "[SceneRenderer] Initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[SceneRenderer] Init failed: " << e.what() << std::endl;
        return false;
    }
}

bool SceneRenderer::create_render_module() {
    OptixModuleCompileOptions module_opts = {};
    module_opts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    
    m_pipeline_compile_options = {};
    m_pipeline_compile_options.usesMotionBlur = false;
    m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipeline_compile_options.numPayloadValues = 15;  // 更多 payload 用于路径追踪
    m_pipeline_compile_options.numAttributeValues = 2;
    m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "render_params";
    m_pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    
    char log[4096];
    size_t log_size = sizeof(log);
    
    OptixResult result = optixModuleCreate(
        m_context,
        &module_opts,
        &m_pipeline_compile_options,
        m_render_ptx_code.c_str(),
        m_render_ptx_code.size(),
        log,
        &log_size,
        &m_render_module
    );
    
    if (result != OPTIX_SUCCESS) {
        std::cerr << "[SceneRenderer] Module creation failed: " << result << std::endl;
        if (log_size > 1) {
            std::cerr << "[SceneRenderer] Module log: " << log << std::endl;
        }
        return false;
    }
    
    return true;
}

void SceneRenderer::create_render_program_groups() {
    char log[2048];
    size_t log_size;
    
    // Raygen
    {
        OptixProgramGroupOptions pg_opts = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pg_desc.raygen.module = m_render_module;
        pg_desc.raygen.entryFunctionName = "__raygen__render";
        
        log_size = sizeof(log);
        check_optix_render(optixProgramGroupCreate(m_context, &pg_desc, 1, &pg_opts, log, &log_size, &m_render_raygen_pg), 
                          "Create render raygen PG");
    }
    
    // Miss
    {
        OptixProgramGroupOptions pg_opts = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pg_desc.miss.module = m_render_module;
        pg_desc.miss.entryFunctionName = "__miss__render";
        
        log_size = sizeof(log);
        check_optix_render(optixProgramGroupCreate(m_context, &pg_desc, 1, &pg_opts, log, &log_size, &m_render_miss_pg), 
                          "Create render miss PG");
    }
    
    // Hitgroup
    {
        OptixProgramGroupOptions pg_opts = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pg_desc.hitgroup.moduleCH = m_render_module;
        pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__render";
        
        log_size = sizeof(log);
        check_optix_render(optixProgramGroupCreate(m_context, &pg_desc, 1, &pg_opts, log, &log_size, &m_render_hitgroup_pg), 
                          "Create render hitgroup PG");
    }
}

void SceneRenderer::create_render_pipeline() {
    OptixProgramGroup pgs[] = { m_render_raygen_pg, m_render_miss_pg, m_render_hitgroup_pg };
    
    OptixPipelineLinkOptions link_opts = {};
    link_opts.maxTraceDepth = 16;  // 路径追踪需要更深的追踪
    
    char log[2048];
    size_t log_size = sizeof(log);
    
    check_optix_render(optixPipelineCreate(
        m_context,
        &m_pipeline_compile_options,
        &link_opts,
        pgs,
        sizeof(pgs) / sizeof(pgs[0]),
        log,
        &log_size,
        &m_render_pipeline
    ), "Create render pipeline");
    
    // 设置栈大小
    OptixStackSizes stack_sizes = {};
    for (auto pg : pgs) {
        OptixStackSizes pg_ss;
        check_optix_render(optixProgramGroupGetStackSize(pg, &pg_ss, m_render_pipeline), "Get stack size");
        stack_sizes.cssRG = std::max(stack_sizes.cssRG, pg_ss.cssRG);
        stack_sizes.cssMS = std::max(stack_sizes.cssMS, pg_ss.cssMS);
        stack_sizes.cssCH = std::max(stack_sizes.cssCH, pg_ss.cssCH);
    }
    
    unsigned int cont_stack = stack_sizes.cssRG + 
        std::max(stack_sizes.cssCH, stack_sizes.cssMS) * (link_opts.maxTraceDepth + 1);
    
    check_optix_render(optixPipelineSetStackSize(m_render_pipeline, 0, 0, cont_stack, 2), "Set stack size");
}

ShadingType SceneRenderer::map_bsdf_to_shading_type(const std::shared_ptr<BSDF>& bsdf) const {
    if (!bsdf) {
        return ShadingType::Diffuse;
    }
    
    std::string type = bsdf->type_name();
    
    if (type == "SpecularReflector") {
        return ShadingType::Mirror;
    } else if (type == "SpecularRefractor" || type == "FresnelRefractor") {
        return ShadingType::Glass;
    } else if (type == "Absorber") {
        return ShadingType::Absorber;
    }
    
    return ShadingType::Diffuse;
}

bool SceneRenderer::build_from_scene(const Scene& scene, OptiXSceneManager* optix_manager) {
    if (!m_initialized) {
        std::cerr << "[SceneRenderer] Not initialized" << std::endl;
        return false;
    }
    
    if (!optix_manager || !optix_manager->is_initialized()) {
        std::cerr << "[SceneRenderer] OptiXSceneManager not ready" << std::endl;
        return false;
    }
    
    // 获取加速结构句柄
    m_traversable = optix_manager->get_traversable_handle();
    if (m_traversable == 0) {
        std::cerr << "[SceneRenderer] No traversable handle available" << std::endl;
        return false;
    }
    
    // 保存表面信息
    m_surfaces = scene.get_all_surfaces_for_optix();
    
    // 构建 SBT
    build_render_sbt(scene);
    
    std::cout << "[SceneRenderer] Built from scene with " << m_surfaces.size() << " surfaces" << std::endl;
    return true;
}

void SceneRenderer::build_render_sbt(const Scene& scene) {
    // Raygen record
    RenderRaygenRecord rg_rec = {};
    check_optix_render(optixSbtRecordPackHeader(m_render_raygen_pg, &rg_rec), "Pack raygen");
    
    if (m_raygen_record) cudaFree(reinterpret_cast<void*>(m_raygen_record));
    check_cuda_render(cudaMalloc(reinterpret_cast<void**>(&m_raygen_record), sizeof(RenderRaygenRecord)), "Alloc raygen");
    check_cuda_render(cudaMemcpy(reinterpret_cast<void*>(m_raygen_record), &rg_rec, sizeof(RenderRaygenRecord), cudaMemcpyHostToDevice), "Upload raygen");
    
    // Miss record
    RenderMissRecord ms_rec = {};
    check_optix_render(optixSbtRecordPackHeader(m_render_miss_pg, &ms_rec), "Pack miss");
    
    if (m_miss_record) cudaFree(reinterpret_cast<void*>(m_miss_record));
    check_cuda_render(cudaMalloc(reinterpret_cast<void**>(&m_miss_record), sizeof(RenderMissRecord)), "Alloc miss");
    check_cuda_render(cudaMemcpy(reinterpret_cast<void*>(m_miss_record), &ms_rec, sizeof(RenderMissRecord), cudaMemcpyHostToDevice), "Upload miss");
    
    // Hitgroup records
    size_t num_surfaces = m_surfaces.size();
    if (num_surfaces == 0) {
        m_render_sbt.raygenRecord = m_raygen_record;
        m_render_sbt.missRecordBase = m_miss_record;
        m_render_sbt.missRecordStrideInBytes = sizeof(RenderMissRecord);
        m_render_sbt.missRecordCount = 1;
        return;
    }
    
    std::vector<RenderHitGroupRecord> hg_records(num_surfaces);
    
    for (size_t i = 0; i < num_surfaces; ++i) {
        const auto& [name, surface] = m_surfaces[i];
        
        check_optix_render(optixSbtRecordPackHeader(m_render_hitgroup_pg, &hg_records[i]), "Pack hitgroup");
        
        hg_records[i].data.surface_id = static_cast<int>(i);
        hg_records[i].data.shading_type = map_bsdf_to_shading_type(surface->get_bsdf());
        
        // 默认白色
        hg_records[i].data.albedo[0] = 0.8f;
        hg_records[i].data.albedo[1] = 0.8f;
        hg_records[i].data.albedo[2] = 0.8f;
        hg_records[i].data.ior = 1.5f;
        
        // 吸收器使用黑色
        if (hg_records[i].data.shading_type == ShadingType::Absorber) {
            hg_records[i].data.albedo[0] = 0.0f;
            hg_records[i].data.albedo[1] = 0.0f;
            hg_records[i].data.albedo[2] = 0.0f;
        }
    }
    
    size_t hg_size = num_surfaces * sizeof(RenderHitGroupRecord);
    
    if (m_hitgroup_records) cudaFree(reinterpret_cast<void*>(m_hitgroup_records));
    check_cuda_render(cudaMalloc(reinterpret_cast<void**>(&m_hitgroup_records), hg_size), "Alloc hitgroups");
    check_cuda_render(cudaMemcpy(reinterpret_cast<void*>(m_hitgroup_records), hg_records.data(), hg_size, cudaMemcpyHostToDevice), "Upload hitgroups");
    
    m_render_sbt.raygenRecord = m_raygen_record;
    m_render_sbt.missRecordBase = m_miss_record;
    m_render_sbt.missRecordStrideInBytes = sizeof(RenderMissRecord);
    m_render_sbt.missRecordCount = 1;
    m_render_sbt.hitgroupRecordBase = m_hitgroup_records;
    m_render_sbt.hitgroupRecordStrideInBytes = sizeof(RenderHitGroupRecord);
    m_render_sbt.hitgroupRecordCount = static_cast<unsigned int>(num_surfaces);
}

void SceneRenderer::resize_buffers(int width, int height) {
    if (width == m_width && height == m_height) return;
    
    m_width = width;
    m_height = height;
    size_t buffer_size = static_cast<size_t>(width) * static_cast<size_t>(height);
    
    if (m_color_buffer) cudaFree(reinterpret_cast<void*>(m_color_buffer));
    if (m_sample_count) cudaFree(reinterpret_cast<void*>(m_sample_count));
    
    check_cuda_render(cudaMalloc(reinterpret_cast<void**>(&m_color_buffer), buffer_size * sizeof(float4)), "Alloc color buffer");
    check_cuda_render(cudaMalloc(reinterpret_cast<void**>(&m_sample_count), buffer_size * sizeof(unsigned int)), "Alloc sample count");
    
    reset_accumulation();
}

void SceneRenderer::reset_accumulation() {
    if (m_width > 0 && m_height > 0) {
        size_t buffer_size = static_cast<size_t>(m_width) * static_cast<size_t>(m_height);
        check_cuda_render(cudaMemset(reinterpret_cast<void*>(m_color_buffer), 0, buffer_size * sizeof(float4)), "Clear color");
        check_cuda_render(cudaMemset(reinterpret_cast<void*>(m_sample_count), 0, buffer_size * sizeof(unsigned int)), "Clear count");
    }
    m_frame_number = 0;
}

void SceneRenderer::render(const Camera& camera, const RenderConfig& config) {
    resize_buffers(config.width, config.height);
    reset_accumulation();
    render_progressive(camera, config, 0);
}

void SceneRenderer::render_progressive(const Camera& camera, const RenderConfig& config, int frame_number) {
    if (!m_initialized || m_traversable == 0) {
        std::cerr << "[SceneRenderer] Not ready for rendering" << std::endl;
        return;
    }
    
    resize_buffers(config.width, config.height);
    
    // 准备启动参数
    RenderLaunchParams params = {};
    params.color_buffer = reinterpret_cast<float4*>(m_color_buffer);
    params.sample_count = reinterpret_cast<unsigned int*>(m_sample_count);
    params.width = config.width;
    params.height = config.height;
    
    // 相机参数
    float eye[3], U[3], V[3], W[3];
    camera.get_render_params(eye, U, V, W, config.width, config.height);
    
    params.eye[0] = eye[0]; params.eye[1] = eye[1]; params.eye[2] = eye[2];
    params.U[0] = U[0]; params.U[1] = U[1]; params.U[2] = U[2];
    params.V[0] = V[0]; params.V[1] = V[1]; params.V[2] = V[2];
    params.W[0] = W[0]; params.W[1] = W[1]; params.W[2] = W[2];
    params.is_orthographic = camera.is_orthographic() ? 1 : 0;
    
    // 渲染参数
    params.samples_per_pixel = config.progressive ? 1 : config.samples_per_pixel;
    params.max_depth = config.max_depth;
    params.frame_number = static_cast<unsigned int>(frame_number);
    
    // 环境光
    params.environment_map = m_environment_texture;
    params.has_environment_map = m_has_environment_map ? 1 : 0;
    params.environment_intensity = config.environment_intensity;
    params.environment_rotation = config.environment_rotation;
    params.sky_color_top[0] = config.sky_color_top[0];
    params.sky_color_top[1] = config.sky_color_top[1];
    params.sky_color_top[2] = config.sky_color_top[2];
    params.sky_color_bottom[0] = config.sky_color_bottom[0];
    params.sky_color_bottom[1] = config.sky_color_bottom[1];
    params.sky_color_bottom[2] = config.sky_color_bottom[2];
    
    // 材质默认值
    params.glass_ior = config.glass_ior;
    params.diffuse_color[0] = config.diffuse_color[0];
    params.diffuse_color[1] = config.diffuse_color[1];
    params.diffuse_color[2] = config.diffuse_color[2];
    
    params.traversable = m_traversable;
    
    // 上传参数
    check_cuda_render(cudaMemcpy(reinterpret_cast<void*>(m_launch_params_buffer), &params, 
                                 sizeof(RenderLaunchParams), cudaMemcpyHostToDevice), "Upload render params");
    
    // 启动渲染
    check_optix_render(optixLaunch(
        m_render_pipeline,
        0,
        m_launch_params_buffer,
        sizeof(RenderLaunchParams),
        &m_render_sbt,
        static_cast<unsigned int>(config.width),
        static_cast<unsigned int>(config.height),
        1
    ), "Launch render");
    
    cudaDeviceSynchronize();
    
    m_frame_number = frame_number + 1;
}

std::vector<float> SceneRenderer::get_image_data() const {
    if (m_width <= 0 || m_height <= 0) return {};
    
    size_t buffer_size = static_cast<size_t>(m_width) * static_cast<size_t>(m_height);
    std::vector<float4> device_data(buffer_size);
    
    check_cuda_render(cudaMemcpy(device_data.data(), reinterpret_cast<void*>(m_color_buffer),
                                 buffer_size * sizeof(float4), cudaMemcpyDeviceToHost), "Download image");
    
    std::vector<float> rgb_data(buffer_size * 3);
    for (size_t i = 0; i < buffer_size; ++i) {
        rgb_data[i * 3 + 0] = device_data[i].x;
        rgb_data[i * 3 + 1] = device_data[i].y;
        rgb_data[i * 3 + 2] = device_data[i].z;
    }
    
    return rgb_data;
}

std::vector<unsigned char> SceneRenderer::get_image_data_8bit() const {
    auto float_data = get_image_data();
    std::vector<unsigned char> byte_data(float_data.size());
    
    for (size_t i = 0; i < float_data.size(); ++i) {
        // 简单的 gamma 校正和截断
        float val = std::pow(float_data[i], 1.0f / 2.2f);
        val = std::max(0.0f, std::min(1.0f, val));
        byte_data[i] = static_cast<unsigned char>(val * 255.0f);
    }
    
    return byte_data;
}

bool SceneRenderer::save_png(const std::string& filename) const {
    if (m_width <= 0 || m_height <= 0) return false;
    
    auto data = get_image_data_8bit();
    
    // 垂直翻转（OpenGL 坐标系）
    std::vector<unsigned char> flipped(data.size());
    for (int y = 0; y < m_height; ++y) {
        int src_row = m_height - 1 - y;
        memcpy(&flipped[y * m_width * 3], &data[src_row * m_width * 3], m_width * 3);
    }
    
    int result = stbi_write_png(filename.c_str(), m_width, m_height, 3, flipped.data(), m_width * 3);
    
    if (result) {
        std::cout << "[SceneRenderer] Saved PNG: " << filename << std::endl;
    } else {
        std::cerr << "[SceneRenderer] Failed to save PNG: " << filename << std::endl;
    }
    
    return result != 0;
}

bool SceneRenderer::save_exr(const std::string& filename) const {
#ifdef USE_OPENEXR
    if (m_width <= 0 || m_height <= 0) return false;
    
    auto data = get_image_data();
    
    // 分离通道
    std::vector<float> r_channel(m_width * m_height);
    std::vector<float> g_channel(m_width * m_height);
    std::vector<float> b_channel(m_width * m_height);
    
    for (int i = 0; i < m_width * m_height; ++i) {
        r_channel[i] = data[i * 3 + 0];
        g_channel[i] = data[i * 3 + 1];
        b_channel[i] = data[i * 3 + 2];
    }
    
    try {
        Imf::Header header(m_width, m_height);
        header.channels().insert("R", Imf::Channel(Imf::FLOAT));
        header.channels().insert("G", Imf::Channel(Imf::FLOAT));
        header.channels().insert("B", Imf::Channel(Imf::FLOAT));
        
        Imf::OutputFile file(filename.c_str(), header);
        
        Imf::FrameBuffer frameBuffer;
        frameBuffer.insert("R", Imf::Slice(Imf::FLOAT, (char*)r_channel.data(), sizeof(float), m_width * sizeof(float)));
        frameBuffer.insert("G", Imf::Slice(Imf::FLOAT, (char*)g_channel.data(), sizeof(float), m_width * sizeof(float)));
        frameBuffer.insert("B", Imf::Slice(Imf::FLOAT, (char*)b_channel.data(), sizeof(float), m_width * sizeof(float)));
        
        file.setFrameBuffer(frameBuffer);
        file.writePixels(m_height);
        
        std::cout << "[SceneRenderer] Saved EXR: " << filename << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[SceneRenderer] Failed to save EXR: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "[SceneRenderer] EXR support not compiled in" << std::endl;
    
    // 回退：保存为 HDR 格式（stb_image_write 支持）
    auto data = get_image_data();
    std::string hdr_filename = filename.substr(0, filename.find_last_of('.')) + ".hdr";
    
    int result = stbi_write_hdr(hdr_filename.c_str(), m_width, m_height, 3, data.data());
    
    if (result) {
        std::cout << "[SceneRenderer] Saved HDR instead: " << hdr_filename << std::endl;
    }
    
    return result != 0;
#endif
}

bool SceneRenderer::load_environment_map(const std::string& filepath) {
    unload_environment_map();
    
    int width, height, channels;
    float* data = stbi_loadf(filepath.c_str(), &width, &height, &channels, 4);
    
    if (!data) {
        std::cerr << "[SceneRenderer] Failed to load environment map: " << filepath << std::endl;
        return false;
    }
    
    // 创建 CUDA 数组
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    check_cuda_render(cudaMallocArray(&m_environment_array, &channel_desc, width, height), "Alloc env array");
    
    check_cuda_render(cudaMemcpy2DToArray(m_environment_array, 0, 0, data, width * sizeof(float4),
                                          width * sizeof(float4), height, cudaMemcpyHostToDevice), "Upload env map");
    
    stbi_image_free(data);
    
    // 创建纹理对象
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = m_environment_array;
    
    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    
    check_cuda_render(cudaCreateTextureObject(&m_environment_texture, &res_desc, &tex_desc, nullptr), "Create env texture");
    
    m_has_environment_map = true;
    std::cout << "[SceneRenderer] Loaded environment map: " << filepath << " (" << width << "x" << height << ")" << std::endl;
    
    return true;
}

void SceneRenderer::unload_environment_map() {
    if (m_environment_texture) {
        cudaDestroyTextureObject(m_environment_texture);
        m_environment_texture = 0;
    }
    if (m_environment_array) {
        cudaFreeArray(m_environment_array);
        m_environment_array = nullptr;
    }
    m_has_environment_map = false;
}

} // namespace diff_optics