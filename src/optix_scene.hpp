// ==================== optix_scene.hpp ====================
#pragma once

#include "utils.hpp"
#include "surface.hpp"
#include "scene.hpp"
#include <optix.h>
#include <optix_stubs.h>
#include <vector>
#include <unordered_map>

namespace diff_optics {

    // ==================== 求交结果 ====================

    struct IntersectionResult {
        Float t;
        Int32 surface_id;
        UInt32 prim_id;
        Float bary_u;
        Float bary_v;
        Mask valid;

        IntersectionResult() = default;

        IntersectionResult(size_t n) {
            // 使用非微分类型初始化
            t = drjit::full<Float>(-1.0f, n);
            surface_id = drjit::full<Int32>(-1, n);
            prim_id = drjit::full<UInt32>(0xFFFFFFFF, n);
            bary_u = drjit::zeros<Float>(n);
            bary_v = drjit::zeros<Float>(n);
            valid = drjit::full<Mask>(false, n);
        }
    };

    // ==================== 表面网格 GPU 数据 ====================

    struct SurfaceMeshGPU {
        std::string name;
        int surface_id;

        CUdeviceptr d_vertices;
        size_t num_vertices;

        CUdeviceptr d_indices;
        size_t num_triangles;

        OptixTraversableHandle gas_handle;
        CUdeviceptr d_gas_buffer;

        CUdeviceptr d_hitgroup_record;

        SurfaceMeshGPU() : surface_id(-1), d_vertices(0), num_vertices(0),
            d_indices(0), num_triangles(0), gas_handle(0),
            d_gas_buffer(0), d_hitgroup_record(0) {}
    };

    // ==================== SBT 记录 ====================

    template<typename T>
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    struct HitGroupData {
        int surface_id;
    };

    using RaygenRecord = SbtRecord<int>;
    using MissRecord = SbtRecord<int>;
    using HitGroupRecord = SbtRecord<HitGroupData>;

    // ==================== 启动参数 ====================

    struct LaunchParams {
        const float* ray_ox;
        const float* ray_oy;
        const float* ray_oz;
        const float* ray_dx;
        const float* ray_dy;
        const float* ray_dz;
        unsigned int num_rays;

        float* hit_t;
        int* hit_surface_id;
        unsigned int* hit_prim_id;
        float* hit_u;
        float* hit_v;

        OptixTraversableHandle traversable;
        int target_surface_id;
    };

    // ==================== OptiX 场景管理器 ====================

    class OptiXSceneManager {
    public:
        OptiXSceneManager();
        ~OptiXSceneManager();

        bool initialize(const std::string& ptx_dir);
        bool build_from_scene(const Scene& scene);
        bool update_surface_mesh(const std::string& surface_name, const Scene& scene);
        IntersectionResult trace_rays(const Ray& rays) const;
        IntersectionResult trace_rays_single_surface(const Ray& rays, const std::string& surface_name) const;
        void clear();

        bool is_initialized() const { return m_initialized; }
        size_t surface_count() const { return m_surface_meshes.size(); }

        int get_surface_id(const std::string& name) const {
            auto it = m_surface_name_to_id.find(name);
            return (it != m_surface_name_to_id.end()) ? it->second : -1;
        }

    private:
        static void check_cuda(cudaError_t err, const char* msg);
        static void check_optix(OptixResult res, const char* msg);

        bool create_module();
        void create_program_groups();
        void create_pipeline();

        bool upload_surface_mesh(const std::string& name, std::shared_ptr<Surface> surface, int surface_id);
        bool build_surface_gas(SurfaceMeshGPU& mesh_gpu);
        bool build_top_level_as();
        void build_sbt();

        IntersectionResult execute_trace(const Ray& rays, OptixTraversableHandle traversable, int target_surface_id) const;

    private:
        bool m_initialized = false;
        std::string m_ptx_dir;
        std::string m_ptx_code;

        OptixDeviceContext m_context = nullptr;
        OptixModule m_module = nullptr;
        OptixProgramGroup m_raygen_pg = nullptr;
        OptixProgramGroup m_miss_pg = nullptr;
        OptixProgramGroup m_hitgroup_pg = nullptr;
        OptixPipeline m_pipeline = nullptr;
        OptixPipelineCompileOptions m_pipeline_compile_options = {};

        OptixShaderBindingTable m_sbt = {};
        CUdeviceptr m_raygen_record = 0;
        CUdeviceptr m_miss_record = 0;
        CUdeviceptr m_hitgroup_records = 0;

        OptixTraversableHandle m_top_level_handle = 0;
        CUdeviceptr m_ias_buffer = 0;
        CUdeviceptr m_instances_buffer = 0;

        std::vector<SurfaceMeshGPU> m_surface_meshes;
        std::unordered_map<std::string, int> m_surface_name_to_id;

        mutable CUdeviceptr m_launch_params_buffer = 0;

        const Scene* m_current_scene = nullptr;
    };

    // ==================== 全局实例 ====================

    extern std::shared_ptr<OptiXSceneManager> g_optix_manager;

    bool init_optix_manager(const std::string& ptx_dir);
    OptiXSceneManager* get_optix_manager();

    // ==================== 便捷接口 ====================

    bool build_optix_scene(const Scene& scene);
    IntersectionResult optix_trace_rays(const Ray& rays);
    IntersectionResult optix_trace_single_surface(const Ray& rays, const std::string& surface_name);

} // namespace diff_optics