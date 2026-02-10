// scene_renderer.hpp - 场景渲染器
#pragma once

#include "camera.hpp"
#include "render_config.hpp"
#include "../scene.hpp"
#include "../optix_scene.hpp"
#include <optix.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>

namespace diff_optics {

// 渲染用着色类型
enum class ShadingType : int {
    Diffuse = 0,
    Mirror = 1,
    Glass = 2,
    Absorber = 3
};

// 渲染用 HitGroup 数据
struct RenderHitGroupData {
    int surface_id;
    ShadingType shading_type;
    float albedo[3];
    float ior;
};

// 渲染用启动参数
struct RenderLaunchParams {
    float4* color_buffer;
    unsigned int* sample_count;
    int width;
    int height;
    
    float eye[3];
    float U[3], V[3], W[3];
    int is_orthographic;
    
    int samples_per_pixel;
    int max_depth;
    unsigned int frame_number;
    
    cudaTextureObject_t environment_map;
    int has_environment_map;
    float environment_intensity;
    float environment_rotation;
    float sky_color_top[3];
    float sky_color_bottom[3];
    
    float glass_ior;
    float diffuse_color[3];
    
    OptixTraversableHandle traversable;
};

// SBT 记录模板
template<typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RenderSbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RenderRaygenRecord = RenderSbtRecord<int>;
using RenderMissRecord = RenderSbtRecord<int>;
using RenderHitGroupRecord = RenderSbtRecord<RenderHitGroupData>;

class SceneRenderer {
public:
    SceneRenderer();
    ~SceneRenderer();
    
    // 初始化
    bool initialize(const std::string& ptx_dir);
    
    // 从场景构建渲染数据（复用 OptiXSceneManager 的加速结构）
    bool build_from_scene(const Scene& scene, OptiXSceneManager* optix_manager);
    
    // 渲染单帧
    void render(const Camera& camera, const RenderConfig& config);
    
    // 渐进式渲染（累积多帧）
    void render_progressive(const Camera& camera, const RenderConfig& config, int frame_number);
    
    // 获取渲染结果
    std::vector<float> get_image_data() const;  // RGB float
    std::vector<unsigned char> get_image_data_8bit() const;  // RGB 8-bit
    
    // 保存图像
    bool save_png(const std::string& filename) const;
    bool save_exr(const std::string& filename) const;
    
    // 重置累积缓冲
    void reset_accumulation();
    
    // 加载环境贴图
    bool load_environment_map(const std::string& filepath);
    void unload_environment_map();
    
    // 获取当前帧数
    unsigned int get_frame_count() const { return m_frame_number; }
    
    // 获取分辨率
    int get_width() const { return m_width; }
    int get_height() const { return m_height; }
    
    bool is_initialized() const { return m_initialized; }
    
private:
    bool create_render_module();
    void create_render_program_groups();
    void create_render_pipeline();
    void build_render_sbt(const Scene& scene);
    void resize_buffers(int width, int height);
    
    ShadingType map_bsdf_to_shading_type(const std::shared_ptr<BSDF>& bsdf) const;
    
private:
    bool m_initialized = false;
    std::string m_ptx_dir;
    std::string m_render_ptx_code;
    
    // OptiX 对象
    OptixDeviceContext m_context = nullptr;
    OptixModule m_render_module = nullptr;
    OptixProgramGroup m_render_raygen_pg = nullptr;
    OptixProgramGroup m_render_miss_pg = nullptr;
    OptixProgramGroup m_render_hitgroup_pg = nullptr;
    OptixPipeline m_render_pipeline = nullptr;
    OptixPipelineCompileOptions m_pipeline_compile_options = {};
    
    // SBT
    OptixShaderBindingTable m_render_sbt = {};
    CUdeviceptr m_raygen_record = 0;
    CUdeviceptr m_miss_record = 0;
    CUdeviceptr m_hitgroup_records = 0;
    
    // 复用的加速结构句柄
    OptixTraversableHandle m_traversable = 0;
    
    // 表面信息（用于构建 SBT）
    std::vector<std::pair<std::string, std::shared_ptr<Surface>>> m_surfaces;
    
    // 输出缓冲
    CUdeviceptr m_color_buffer = 0;
    CUdeviceptr m_sample_count = 0;
    int m_width = 0;
    int m_height = 0;
    
    // 启动参数
    CUdeviceptr m_launch_params_buffer = 0;
    
    // 帧计数
    unsigned int m_frame_number = 0;
    
    // 环境贴图
    cudaTextureObject_t m_environment_texture = 0;
    cudaArray_t m_environment_array = nullptr;
    bool m_has_environment_map = false;
};

} // namespace diff_optics