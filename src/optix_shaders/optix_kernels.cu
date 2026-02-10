// optix_kernels.cu
#include <optix.h>
#include <cuda_runtime.h>

// ==================== 启动参数 ====================
struct LaunchParams {
    // 输入：光线数据
    const float* ray_ox;
    const float* ray_oy;
    const float* ray_oz;
    const float* ray_dx;
    const float* ray_dy;
    const float* ray_dz;
    unsigned int num_rays;
    
    // 输出：求交结果
    float* hit_t;
    int* hit_surface_id;
    unsigned int* hit_prim_id;  // 三角形索引
    float* hit_u;               // 重心坐标 u
    float* hit_v;               // 重心坐标 v
    
    // 场景
    OptixTraversableHandle traversable;
    
    // 表面 ID 映射（用于单表面追踪）
    int target_surface_id;      // -1 表示追踪所有表面
};

extern "C" {
    __constant__ LaunchParams params;
}

// ==================== Payload 辅助函数 ====================

__device__ __forceinline__ void setPayload(float t, int surface_id, unsigned int prim_id, float u, float v) {
    optixSetPayload_0(__float_as_uint(t));
    optixSetPayload_1(static_cast<unsigned int>(surface_id));
    optixSetPayload_2(prim_id);
    optixSetPayload_3(__float_as_uint(u));
    optixSetPayload_4(__float_as_uint(v));
}

__device__ __forceinline__ void setPayloadMiss() {
    optixSetPayload_0(__float_as_uint(-1.0f));
    optixSetPayload_1(static_cast<unsigned int>(-1));
    optixSetPayload_2(0xFFFFFFFF);
    optixSetPayload_3(__float_as_uint(0.0f));
    optixSetPayload_4(__float_as_uint(0.0f));
}

// ==================== 光线生成程序 ====================

extern "C" __global__ void __raygen__main() {
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int ray_idx = idx.x;
    
    if (ray_idx >= params.num_rays) return;
    
    float3 ray_origin = make_float3(
        params.ray_ox[ray_idx],
        params.ray_oy[ray_idx],
        params.ray_oz[ray_idx]
    );
    float3 ray_direction = make_float3(
        params.ray_dx[ray_idx],
        params.ray_dy[ray_idx],
        params.ray_dz[ray_idx]
    );

    // 初始化 payload
    unsigned int p0 = __float_as_uint(-1.0f);   // t
    unsigned int p1 = static_cast<unsigned int>(-1);  // surface_id
    unsigned int p2 = 0xFFFFFFFF;               // prim_id
    unsigned int p3 = __float_as_uint(0.0f);    // u
    unsigned int p4 = __float_as_uint(0.0f);    // v
    
    optixTrace(
        params.traversable,
        ray_origin,
        ray_direction,
        1e-3f,              // tmin
        1e16f,              // tmax
        0.0f,               // ray time
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,                  // SBT offset
        1,                  // SBT stride
        0,                  // miss SBT index
        p0, p1, p2, p3, p4
    );
    
    // 写入结果
    params.hit_t[ray_idx] = __uint_as_float(p0);
    params.hit_surface_id[ray_idx] = static_cast<int>(p1);
    params.hit_prim_id[ray_idx] = p2;
    params.hit_u[ray_idx] = __uint_as_float(p3);
    params.hit_v[ray_idx] = __uint_as_float(p4);
}

// ==================== 未命中程序 ====================

extern "C" __global__ void __miss__main() {
    setPayloadMiss();
}

// ==================== Closest Hit 程序（三角形）====================

extern "C" __global__ void __closesthit__triangle() {
    // 获取 SBT 数据中存储的 surface_id
    const int surface_id = *reinterpret_cast<const int*>(optixGetSbtDataPointer());
    
    // 获取求交信息
    const float t = optixGetRayTmax();
    const unsigned int prim_id = optixGetPrimitiveIndex();
    
    // 获取重心坐标
    const float2 bary = optixGetTriangleBarycentrics();
    const float u = bary.x;
    const float v = bary.y;
    
    setPayload(t, surface_id, prim_id, u, v);
}

// ==================== Any Hit 程序（用于单表面过滤）====================

extern "C" __global__ void __anyhit__triangle() {
    // 如果指定了目标表面，过滤非目标表面的命中
    if (params.target_surface_id >= 0) {
        const int surface_id = *reinterpret_cast<const int*>(optixGetSbtDataPointer());
        if (surface_id != params.target_surface_id) {
            optixIgnoreIntersection();
        }
    }
}