// ==================== optix_scene.cpp ====================
#include "optix_scene.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <optix_function_table_definition.h>

extern "C" {
    void launch_pack_rays(
        const float* ox, const float* oy, const float* oz,
        const float* dx, const float* dy, const float* dz,
        float3* origins, float3* directions,
        unsigned int n, cudaStream_t stream
    );
    
    void launch_transform_vertices(
        const float* local_x, const float* local_y, const float* local_z,
        const float* matrix,
        float3* world_vertices,
        unsigned int n, cudaStream_t stream
    );
}

namespace diff_optics {

// ==================== 全局实例 ====================

std::shared_ptr<OptiXSceneManager> g_optix_manager = nullptr;

bool init_optix_manager(const std::string& ptx_dir) {
    if (!g_optix_manager) {
        g_optix_manager = std::make_shared<OptiXSceneManager>();
    }
    return g_optix_manager->initialize(ptx_dir);
}

OptiXSceneManager* get_optix_manager() {
    return g_optix_manager.get();
}

bool build_optix_scene(const Scene& scene) {
    OptiXSceneManager* manager = get_optix_manager();
    if (!manager) return false;
    return manager->build_from_scene(scene);
}

IntersectionResult optix_trace_rays(const Ray& rays) {
    OptiXSceneManager* manager = get_optix_manager();
    if (!manager) {
        return IntersectionResult(rays.size());
    }
    return manager->trace_rays(rays);
}

IntersectionResult optix_trace_single_surface(const Ray& rays, const std::string& surface_name) {
    OptiXSceneManager* manager = get_optix_manager();
    if (!manager) {
        return IntersectionResult(rays.size());
    }
    return manager->trace_rays_single_surface(rays, surface_name);
}

// ==================== OptiX 日志回调 ====================

static void optix_log_callback(unsigned int level, const char* tag, const char* message, void*) {
    if (level <= 2) {
        std::cerr << "[OptiX][" << level << "][" << tag << "]: " << message << std::endl;
    }
}

// ==================== OptiXSceneManager 实现 ====================

OptiXSceneManager::OptiXSceneManager() {}

OptiXSceneManager::~OptiXSceneManager() {
    clear();
    
    if (m_pipeline) optixPipelineDestroy(m_pipeline);
    if (m_hitgroup_pg) optixProgramGroupDestroy(m_hitgroup_pg);
    if (m_miss_pg) optixProgramGroupDestroy(m_miss_pg);
    if (m_raygen_pg) optixProgramGroupDestroy(m_raygen_pg);
    if (m_module) optixModuleDestroy(m_module);
    if (m_context) optixDeviceContextDestroy(m_context);
    
    if (m_launch_params_buffer) cudaFree(reinterpret_cast<void*>(m_launch_params_buffer));
}

void OptiXSceneManager::check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

void OptiXSceneManager::check_optix(OptixResult res, const char* msg) {
    if (res != OPTIX_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": OptiX error " + std::to_string(res));
    }
}

bool OptiXSceneManager::initialize(const std::string& ptx_dir) {
    if (m_initialized) {
        return true;
    }
    
    m_ptx_dir = ptx_dir;
    
    try {
        jit_sync_thread();
        cudaDeviceSynchronize();
        
        check_optix(optixInit(), "OptiX init");
        
        CUcontext cu_ctx = nullptr;
        cuCtxGetCurrent(&cu_ctx);
        
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = optix_log_callback;
        options.logCallbackLevel = 3;
        
        check_optix(optixDeviceContextCreate(cu_ctx, &options, &m_context), "Create context");
        
        std::string ptx_path = m_ptx_dir + "/optix_kernels.ptx";
        std::ifstream ptx_file(ptx_path);
        if (!ptx_file.is_open()) {
            throw std::runtime_error("Cannot open PTX: " + ptx_path);
        }
        
        std::stringstream buffer;
        buffer << ptx_file.rdbuf();
        m_ptx_code = buffer.str();
        ptx_file.close();
        
        if (!create_module()) {
            throw std::runtime_error("Failed to create module");
        }
        
        create_program_groups();
        create_pipeline();
        
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&m_launch_params_buffer), sizeof(LaunchParams)), "Alloc launch params");
        
        m_initialized = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[OptiXSceneManager] Init failed: " << e.what() << std::endl;
        return false;
    }
}

bool OptiXSceneManager::create_module() {
    jit_sync_thread();
    cudaDeviceSynchronize();
    
    OptixModuleCompileOptions module_opts = {};
    module_opts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    
    m_pipeline_compile_options = {};
    m_pipeline_compile_options.usesMotionBlur = false;
    m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipeline_compile_options.numPayloadValues = 5;
    m_pipeline_compile_options.numAttributeValues = 2;
    m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    m_pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    
    char log[4096];
    size_t log_size = sizeof(log);
    
    OptixResult result = optixModuleCreate(
        m_context,
        &module_opts,
        &m_pipeline_compile_options,
        m_ptx_code.c_str(),
        m_ptx_code.size(),
        log,
        &log_size,
        &m_module
    );
    
    if (result != OPTIX_SUCCESS) {
        std::cerr << "[OptiXSceneManager] Module creation failed: " << result << std::endl;
        if (log_size > 1) {
            std::cerr << "[OptiXSceneManager] Module log: " << log << std::endl;
        }
        return false;
    }
    
    return true;
}

void OptiXSceneManager::create_program_groups() {
    char log[2048];
    size_t log_size;
    
    {
        OptixProgramGroupOptions pg_opts = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pg_desc.raygen.module = m_module;
        pg_desc.raygen.entryFunctionName = "__raygen__main";
        
        log_size = sizeof(log);
        check_optix(optixProgramGroupCreate(m_context, &pg_desc, 1, &pg_opts, log, &log_size, &m_raygen_pg), "Create raygen PG");
    }
    
    {
        OptixProgramGroupOptions pg_opts = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pg_desc.miss.module = m_module;
        pg_desc.miss.entryFunctionName = "__miss__main";
        
        log_size = sizeof(log);
        check_optix(optixProgramGroupCreate(m_context, &pg_desc, 1, &pg_opts, log, &log_size, &m_miss_pg), "Create miss PG");
    }
    
    {
        OptixProgramGroupOptions pg_opts = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pg_desc.hitgroup.moduleCH = m_module;
        pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";
        pg_desc.hitgroup.moduleAH = m_module;
        pg_desc.hitgroup.entryFunctionNameAH = "__anyhit__triangle";
        
        log_size = sizeof(log);
        check_optix(optixProgramGroupCreate(m_context, &pg_desc, 1, &pg_opts, log, &log_size, &m_hitgroup_pg), "Create hitgroup PG");
    }
}

void OptiXSceneManager::create_pipeline() {
    OptixProgramGroup pgs[] = { m_raygen_pg, m_miss_pg, m_hitgroup_pg };
    
    OptixPipelineLinkOptions link_opts = {};
    link_opts.maxTraceDepth = 1;
    
    char log[2048];
    size_t log_size = sizeof(log);
    
    check_optix(optixPipelineCreate(
        m_context,
        &m_pipeline_compile_options,
        &link_opts,
        pgs,
        sizeof(pgs) / sizeof(pgs[0]),
        log,
        &log_size,
        &m_pipeline
    ), "Create pipeline");
    
    OptixStackSizes stack_sizes = {};
    for (auto pg : pgs) {
        OptixStackSizes pg_ss;
        check_optix(optixProgramGroupGetStackSize(pg, &pg_ss, m_pipeline), "Get stack size");
        stack_sizes.cssRG = std::max(stack_sizes.cssRG, pg_ss.cssRG);
        stack_sizes.cssMS = std::max(stack_sizes.cssMS, pg_ss.cssMS);
        stack_sizes.cssCH = std::max(stack_sizes.cssCH, pg_ss.cssCH);
    }
    
    unsigned int cont_stack = stack_sizes.cssRG + std::max(stack_sizes.cssCH, stack_sizes.cssMS);
    
    check_optix(optixPipelineSetStackSize(m_pipeline, 0, 0, cont_stack, 2), "Set stack size");
}

void OptiXSceneManager::clear() {
    if (m_raygen_record) {
        cudaFree(reinterpret_cast<void*>(m_raygen_record));
        m_raygen_record = 0;
    }
    if (m_miss_record) {
        cudaFree(reinterpret_cast<void*>(m_miss_record));
        m_miss_record = 0;
    }
    if (m_hitgroup_records) {
        cudaFree(reinterpret_cast<void*>(m_hitgroup_records));
        m_hitgroup_records = 0;
    }
    
    if (m_ias_buffer) {
        cudaFree(reinterpret_cast<void*>(m_ias_buffer));
        m_ias_buffer = 0;
    }
    if (m_instances_buffer) {
        cudaFree(reinterpret_cast<void*>(m_instances_buffer));
        m_instances_buffer = 0;
    }
    
    for (auto& mesh : m_surface_meshes) {
        if (mesh.d_vertices) cudaFree(reinterpret_cast<void*>(mesh.d_vertices));
        if (mesh.d_indices) cudaFree(reinterpret_cast<void*>(mesh.d_indices));
        if (mesh.d_gas_buffer) cudaFree(reinterpret_cast<void*>(mesh.d_gas_buffer));
        if (mesh.d_hitgroup_record) cudaFree(reinterpret_cast<void*>(mesh.d_hitgroup_record));
    }
    
    m_surface_meshes.clear();
    m_surface_name_to_id.clear();
    m_top_level_handle = 0;
    m_current_scene = nullptr;
    
    memset(&m_sbt, 0, sizeof(m_sbt));
}

bool OptiXSceneManager::upload_surface_mesh(
    const std::string& name,
    std::shared_ptr<Surface> surface,
    int surface_id
) {
    SurfaceMeshGPU mesh_gpu;
    mesh_gpu.name = name;
    mesh_gpu.surface_id = surface_id;
    
    const TriangleMesh& mesh = surface->get_mesh();
    
    if (!mesh.is_valid()) {
        return false;
    }
    
    mesh_gpu.num_vertices = mesh.num_vertices;
    mesh_gpu.num_triangles = mesh.num_triangles;
    
    drjit::eval(mesh.vertices_x, mesh.vertices_y, mesh.vertices_z, mesh.indices);
    drjit::sync_thread();
    
    const Transform& transform = surface->get_transform();
    Matrix4 mat = transform.compute_matrix();
    
    std::vector<float> matrix_host(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix_host[i * 4 + j] = to_scalar(mat(i, j));
        }
    }
    
    CUdeviceptr d_matrix;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_matrix), 16 * sizeof(float)), "Alloc matrix");
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(d_matrix), matrix_host.data(), 16 * sizeof(float), cudaMemcpyHostToDevice), "Upload matrix");
    
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&mesh_gpu.d_vertices), mesh_gpu.num_vertices * sizeof(float3)), "Alloc vertices");
    
    launch_transform_vertices(
        mesh.vertices_x.data(),
        mesh.vertices_y.data(),
        mesh.vertices_z.data(),
        reinterpret_cast<float*>(d_matrix),
        reinterpret_cast<float3*>(mesh_gpu.d_vertices),
        static_cast<unsigned int>(mesh_gpu.num_vertices),
        0
    );
    cudaDeviceSynchronize();
    
    cudaFree(reinterpret_cast<void*>(d_matrix));
    
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&mesh_gpu.d_indices), mesh_gpu.num_triangles * 3 * sizeof(uint32_t)), "Alloc indices");
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(mesh_gpu.d_indices), mesh.indices.data(), mesh_gpu.num_triangles * 3 * sizeof(uint32_t), cudaMemcpyDeviceToDevice), "Copy indices");
    
    if (!build_surface_gas(mesh_gpu)) {
        return false;
    }
    
    m_surface_meshes.push_back(std::move(mesh_gpu));
    m_surface_name_to_id[name] = surface_id;
    
    return true;
}

bool OptiXSceneManager::build_surface_gas(SurfaceMeshGPU& mesh_gpu) {
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    CUdeviceptr vertex_buffers[] = { mesh_gpu.d_vertices };
    build_input.triangleArray.vertexBuffers = vertex_buffers;
    build_input.triangleArray.numVertices = static_cast<unsigned int>(mesh_gpu.num_vertices);
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    
    build_input.triangleArray.indexBuffer = mesh_gpu.d_indices;
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh_gpu.num_triangles);
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
    
    uint32_t flags = OPTIX_GEOMETRY_FLAG_NONE;
    build_input.triangleArray.flags = &flags;
    build_input.triangleArray.numSbtRecords = 1;
    
    OptixAccelBuildOptions accel_opts = {};
    accel_opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_opts.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes buffer_sizes;
    check_optix(optixAccelComputeMemoryUsage(m_context, &accel_opts, &build_input, 1, &buffer_sizes), "Compute GAS size");
    
    CUdeviceptr d_temp;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_temp), buffer_sizes.tempSizeInBytes), "Alloc temp");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&mesh_gpu.d_gas_buffer), buffer_sizes.outputSizeInBytes), "Alloc GAS");
    
    check_optix(optixAccelBuild(
        m_context,
        0,
        &accel_opts,
        &build_input,
        1,
        d_temp,
        buffer_sizes.tempSizeInBytes,
        mesh_gpu.d_gas_buffer,
        buffer_sizes.outputSizeInBytes,
        &mesh_gpu.gas_handle,
        nullptr,
        0
    ), "Build GAS");
    
    cudaFree(reinterpret_cast<void*>(d_temp));
    
    HitGroupRecord hg_rec = {};
    check_optix(optixSbtRecordPackHeader(m_hitgroup_pg, &hg_rec), "Pack hitgroup");
    hg_rec.data.surface_id = mesh_gpu.surface_id;
    
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&mesh_gpu.d_hitgroup_record), sizeof(HitGroupRecord)), "Alloc HG record");
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(mesh_gpu.d_hitgroup_record), &hg_rec, sizeof(HitGroupRecord), cudaMemcpyHostToDevice), "Upload HG record");
    
    return true;
}

bool OptiXSceneManager::build_top_level_as() {
    if (m_surface_meshes.empty()) {
        return true;
    }
    
    size_t num_instances = m_surface_meshes.size();
    
    std::vector<OptixInstance> instances(num_instances);
    
    for (size_t i = 0; i < num_instances; ++i) {
        OptixInstance& inst = instances[i];
        memset(&inst, 0, sizeof(OptixInstance));
        
        inst.transform[0] = 1.0f;
        inst.transform[5] = 1.0f;
        inst.transform[10] = 1.0f;
        
        inst.instanceId = static_cast<unsigned int>(i);
        inst.sbtOffset = static_cast<unsigned int>(i);
        inst.visibilityMask = 255;
        inst.flags = OPTIX_INSTANCE_FLAG_NONE;
        inst.traversableHandle = m_surface_meshes[i].gas_handle;
    }
    
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&m_instances_buffer), num_instances * sizeof(OptixInstance)), "Alloc instances");
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(m_instances_buffer), instances.data(), num_instances * sizeof(OptixInstance), cudaMemcpyHostToDevice), "Upload instances");
    
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = m_instances_buffer;
    build_input.instanceArray.numInstances = static_cast<unsigned int>(num_instances);
    
    OptixAccelBuildOptions accel_opts = {};
    accel_opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_opts.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes buffer_sizes;
    check_optix(optixAccelComputeMemoryUsage(m_context, &accel_opts, &build_input, 1, &buffer_sizes), "Compute IAS size");
    
    CUdeviceptr d_temp;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_temp), buffer_sizes.tempSizeInBytes), "Alloc temp");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&m_ias_buffer), buffer_sizes.outputSizeInBytes), "Alloc IAS");
    
    check_optix(optixAccelBuild(
        m_context,
        0,
        &accel_opts,
        &build_input,
        1,
        d_temp,
        buffer_sizes.tempSizeInBytes,
        m_ias_buffer,
        buffer_sizes.outputSizeInBytes,
        &m_top_level_handle,
        nullptr,
        0
    ), "Build IAS");
    
    cudaFree(reinterpret_cast<void*>(d_temp));
    
    return true;
}

void OptiXSceneManager::build_sbt() {
    RaygenRecord rg_rec = {};
    check_optix(optixSbtRecordPackHeader(m_raygen_pg, &rg_rec), "Pack raygen");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&m_raygen_record), sizeof(RaygenRecord)), "Alloc raygen");
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(m_raygen_record), &rg_rec, sizeof(RaygenRecord), cudaMemcpyHostToDevice), "Upload raygen");
    
    MissRecord ms_rec = {};
    check_optix(optixSbtRecordPackHeader(m_miss_pg, &ms_rec), "Pack miss");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&m_miss_record), sizeof(MissRecord)), "Alloc miss");
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(m_miss_record), &ms_rec, sizeof(MissRecord), cudaMemcpyHostToDevice), "Upload miss");
    
    size_t num_surfaces = m_surface_meshes.size();
    if (num_surfaces == 0) {
        memset(&m_sbt, 0, sizeof(m_sbt));
        m_sbt.raygenRecord = m_raygen_record;
        m_sbt.missRecordBase = m_miss_record;
        m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
        m_sbt.missRecordCount = 1;
        return;
    }
    
    std::vector<HitGroupRecord> hg_records(num_surfaces);
    for (size_t i = 0; i < num_surfaces; ++i) {
        check_optix(optixSbtRecordPackHeader(m_hitgroup_pg, &hg_records[i]), "Pack hitgroup");
        hg_records[i].data.surface_id = m_surface_meshes[i].surface_id;
    }
    
    size_t hg_size = num_surfaces * sizeof(HitGroupRecord);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&m_hitgroup_records), hg_size), "Alloc hitgroups");
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(m_hitgroup_records), hg_records.data(), hg_size, cudaMemcpyHostToDevice), "Upload hitgroups");
    
    m_sbt.raygenRecord = m_raygen_record;
    m_sbt.missRecordBase = m_miss_record;
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = 1;
    m_sbt.hitgroupRecordBase = m_hitgroup_records;
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    m_sbt.hitgroupRecordCount = static_cast<unsigned int>(num_surfaces);
}

// optix_scene.cpp 修改 - build_from_scene 方法

bool OptiXSceneManager::build_from_scene(const Scene& scene) {
    if (!m_initialized) {
        return false;
    }
    
    clear();
    m_current_scene = &scene;
    
    // 使用新的统一接口获取所有 Surface
    auto all_surfaces = scene.get_all_surfaces_for_optix();
    
    if (all_surfaces.empty()) {
        return true;
    }
    
    for (size_t i = 0; i < all_surfaces.size(); ++i) {
        const auto& [surface_name, surface] = all_surfaces[i];
        
        if (!upload_surface_mesh(surface_name, surface, static_cast<int>(i))) {
            return false;
        }
    }
    
    if (!build_top_level_as()) {
        return false;
    }
    
    build_sbt();
    
    return true;
}

bool OptiXSceneManager::update_surface_mesh(const std::string& surface_name, const Scene& scene) {
    auto it = m_surface_name_to_id.find(surface_name);
    if (it == m_surface_name_to_id.end()) {
        return false;
    }
    
    int surface_id = it->second;
    auto surface = scene.get_surface(surface_name);
    
    surface->update_mesh_vertices();
    const TriangleMesh& mesh = surface->get_mesh();
    
    drjit::eval(mesh.vertices_x, mesh.vertices_y, mesh.vertices_z);
    drjit::sync_thread();
    
    SurfaceMeshGPU& mesh_gpu = m_surface_meshes[surface_id];
    
    const Transform& transform = surface->get_transform();
    Matrix4 mat = transform.compute_matrix();
    
    std::vector<float> matrix_host(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix_host[i * 4 + j] = to_scalar(mat(i, j));
        }
    }
    
    CUdeviceptr d_matrix;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_matrix), 16 * sizeof(float)), "Alloc matrix");
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(d_matrix), matrix_host.data(), 16 * sizeof(float), cudaMemcpyHostToDevice), "Upload matrix");
    
    launch_transform_vertices(
        mesh.vertices_x.data(),
        mesh.vertices_y.data(),
        mesh.vertices_z.data(),
        reinterpret_cast<float*>(d_matrix),
        reinterpret_cast<float3*>(mesh_gpu.d_vertices),
        static_cast<unsigned int>(mesh_gpu.num_vertices),
        0
    );
    cudaDeviceSynchronize();
    
    cudaFree(reinterpret_cast<void*>(d_matrix));
    
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    CUdeviceptr vertex_buffers[] = { mesh_gpu.d_vertices };
    build_input.triangleArray.vertexBuffers = vertex_buffers;
    build_input.triangleArray.numVertices = static_cast<unsigned int>(mesh_gpu.num_vertices);
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    
    build_input.triangleArray.indexBuffer = mesh_gpu.d_indices;
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh_gpu.num_triangles);
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
    
    uint32_t flags = OPTIX_GEOMETRY_FLAG_NONE;
    build_input.triangleArray.flags = &flags;
    build_input.triangleArray.numSbtRecords = 1;
    
    OptixAccelBuildOptions accel_opts = {};
    accel_opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_opts.operation = OPTIX_BUILD_OPERATION_UPDATE;
    
    OptixAccelBufferSizes buffer_sizes;
    check_optix(optixAccelComputeMemoryUsage(m_context, &accel_opts, &build_input, 1, &buffer_sizes), "Compute GAS update size");
    
    CUdeviceptr d_temp;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_temp), buffer_sizes.tempUpdateSizeInBytes), "Alloc temp update");
    
    check_optix(optixAccelBuild(
        m_context,
        0,
        &accel_opts,
        &build_input,
        1,
        d_temp,
        buffer_sizes.tempUpdateSizeInBytes,
        mesh_gpu.d_gas_buffer,
        buffer_sizes.outputSizeInBytes,
        &mesh_gpu.gas_handle,
        nullptr,
        0
    ), "Update GAS");
    
    cudaFree(reinterpret_cast<void*>(d_temp));
    
    return true;
}

IntersectionResult OptiXSceneManager::execute_trace(
    const Ray& rays,
    OptixTraversableHandle traversable,
    int target_surface_id
) const {
    size_t n = rays.size();

    if (n == 0 || traversable == 0) {
        return IntersectionResult(n);
    }

    IntersectionResult result;

    try {
        // 确保输入光线数据就绪
        drjit::eval(rays.origin, rays.direction);
        drjit::sync_thread();

        // 分配并初始化输出数组（非微分类型）
        result.t = drjit::zeros<Float>(n);
        result.surface_id = drjit::full<Int32>(-1, n);
        result.prim_id = drjit::full<UInt32>(0xFFFFFFFF, n);
        result.bary_u = drjit::zeros<Float>(n);
        result.bary_v = drjit::zeros<Float>(n);

        drjit::eval(result.t, result.surface_id, result.prim_id, result.bary_u, result.bary_v);
        drjit::sync_thread();

        // 从微分类型获取底层非微分数组
        // 使用 .value() 或直接访问底层 index
        const Float& ray_ox = rays.origin[0];
        const Float& ray_oy = rays.origin[1];
        const Float& ray_oz = rays.origin[2];
        const Float& ray_dx = rays.direction[0];
        const Float& ray_dy = rays.direction[1];
        const Float& ray_dz = rays.direction[2];

        drjit::eval(ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz);
        drjit::sync_thread();

        LaunchParams params = {};
        params.ray_ox = ray_ox.data();
        params.ray_oy = ray_oy.data();
        params.ray_oz = ray_oz.data();
        params.ray_dx = ray_dx.data();
        params.ray_dy = ray_dy.data();
        params.ray_dz = ray_dz.data();
        params.num_rays = static_cast<unsigned int>(n);
        params.hit_t = result.t.data();
        params.hit_surface_id = result.surface_id.data();
        params.hit_prim_id = result.prim_id.data();
        params.hit_u = result.bary_u.data();
        params.hit_v = result.bary_v.data();
        params.traversable = traversable;
        params.target_surface_id = target_surface_id;

        check_cuda(cudaMemcpy(
            reinterpret_cast<void*>(m_launch_params_buffer),
            &params,
            sizeof(LaunchParams),
            cudaMemcpyHostToDevice
        ), "Upload params");

        check_optix(optixLaunch(
            m_pipeline,
            0,
            m_launch_params_buffer,
            sizeof(LaunchParams),
            &m_sbt,
            static_cast<unsigned int>(n),
            1,
            1
        ), "Launch");

        cudaDeviceSynchronize();

        // 计算有效掩码
        result.valid = result.t > Float(0.0f);

    }
    catch (const std::exception& e) {
        std::cerr << "[OptiXSceneManager] execute_trace failed: " << e.what() << std::endl;
        return IntersectionResult(n);
    }

    return result;
}

IntersectionResult OptiXSceneManager::trace_rays(const Ray& rays) const {
    if (!m_initialized || m_top_level_handle == 0) {
        return IntersectionResult(rays.size());
    }
    
    return execute_trace(rays, m_top_level_handle, -1);
}

IntersectionResult OptiXSceneManager::trace_rays_single_surface(
    const Ray& rays,
    const std::string& surface_name
) const {
    auto it = m_surface_name_to_id.find(surface_name);
    if (it == m_surface_name_to_id.end()) {
        return IntersectionResult(rays.size());
    }
    
    int surface_id = it->second;
    return execute_trace(rays, m_top_level_handle, surface_id);
}

} // namespace diff_optics