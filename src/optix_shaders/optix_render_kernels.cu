// optix_render_kernels.cu - 渲染专用OptiX着色器
#include <optix.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ==================== 渲染用数据结构 ====================

enum class ShadingType : int {
    Diffuse = 0,
    Mirror = 1,
    Glass = 2,
    Absorber = 3
};

struct RenderLaunchParams {
    // 输出缓冲
    float4* color_buffer;
    unsigned int* sample_count;
    int width;
    int height;
    
    // 相机参数
    float3 eye;
    float3 U, V, W;
    int is_orthographic;
    
    // 渲染参数
    int samples_per_pixel;
    int max_depth;
    unsigned int frame_number;
    
    // 环境光
    cudaTextureObject_t environment_map;
    int has_environment_map;
    float environment_intensity;
    float environment_rotation;
    float3 sky_color_top;
    float3 sky_color_bottom;
    
    // 材质默认值
    float glass_ior;
    float3 diffuse_color;
    
    // 场景
    OptixTraversableHandle traversable;
};

struct RenderHitGroupData {
    int surface_id;
    ShadingType shading_type;
    float3 albedo;
    float ior;
};

extern "C" {
    __constant__ RenderLaunchParams render_params;
}

// ==================== 辅助函数 ====================

__device__ __forceinline__ float3 make_float3_v(float v) {
    return make_float3(v, v, v);
}

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __forceinline__ float3 operator*(float a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ __forceinline__ float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__device__ __forceinline__ float3 normalize(const float3& v) {
    float len = length(v);
    return len > 1e-8f ? v / len : make_float3(0, 0, 1);
}

__device__ __forceinline__ float3 reflect(const float3& I, const float3& N) {
    return I - 2.0f * dot(N, I) * N;
}

__device__ __forceinline__ float3 refract(const float3& I, const float3& N, float eta) {
    float cos_i = -dot(N, I);
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    
    if (sin2_t > 1.0f) {
        return make_float3(0, 0, 0); // TIR
    }
    
    float cos_t = sqrtf(1.0f - sin2_t);
    return eta * I + (eta * cos_i - cos_t) * N;
}

__device__ __forceinline__ float fresnel_schlick(float cos_i, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    float x = 1.0f - cos_i;
    return r0 + (1.0f - r0) * x * x * x * x * x;
}

// ==================== 随机数生成 ====================

__device__ __forceinline__ unsigned int tea(unsigned int val0, unsigned int val1) {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;
    
    for (int n = 0; n < 16; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    
    return v0;
}

__device__ __forceinline__ float rnd(unsigned int& seed) {
    seed = seed * 1664525u + 1013904223u;
    return float(seed & 0x00FFFFFF) / float(0x01000000);
}

// ==================== 环境采样 ====================

__device__ float3 sample_environment(const float3& direction) {
    if (render_params.has_environment_map) {
        // 将方向转换为球面坐标
        float theta = acosf(fmaxf(-1.0f, fminf(1.0f, direction.y)));
        float phi = atan2f(direction.z, direction.x) + render_params.environment_rotation * M_PI / 180.0f;
        
        float u = phi / (2.0f * M_PI) + 0.5f;
        float v = theta / M_PI;
        
        // 采样环境贴图
        float4 env = tex2D<float4>(render_params.environment_map, u, v);
        return make_float3(env.x, env.y, env.z) * render_params.environment_intensity;
    } else {
        // 天空渐变
        float t = direction.y * 0.5f + 0.5f;
        return render_params.sky_color_bottom * (1.0f - t) + render_params.sky_color_top * t;
    }
}

// ==================== 漫反射采样 ====================

__device__ float3 sample_cosine_hemisphere(const float3& N, float u1, float u2) {
    float r = sqrtf(u1);
    float theta = 2.0f * M_PI * u2;
    
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    
    // 构建局部坐标系
    float3 T, B;
    if (fabsf(N.x) > 0.9f) {
        T = normalize(cross(make_float3(0, 1, 0), N));
    } else {
        T = normalize(cross(make_float3(1, 0, 0), N));
    }
    B = cross(N, T);
    
    return normalize(T * x + B * y + N * z);
}

// ==================== Payload ====================

struct RenderPayload {
    float3 radiance;
    float3 throughput;
    float3 origin;
    float3 direction;
    unsigned int seed;
    int depth;
    bool done;
};

__device__ __forceinline__ void set_payload(const RenderPayload& p) {
    optixSetPayload_0(__float_as_uint(p.radiance.x));
    optixSetPayload_1(__float_as_uint(p.radiance.y));
    optixSetPayload_2(__float_as_uint(p.radiance.z));
    optixSetPayload_3(__float_as_uint(p.throughput.x));
    optixSetPayload_4(__float_as_uint(p.throughput.y));
    optixSetPayload_5(__float_as_uint(p.throughput.z));
    optixSetPayload_6(__float_as_uint(p.origin.x));
    optixSetPayload_7(__float_as_uint(p.origin.y));
    optixSetPayload_8(__float_as_uint(p.origin.z));
    optixSetPayload_9(__float_as_uint(p.direction.x));
    optixSetPayload_10(__float_as_uint(p.direction.y));
    optixSetPayload_11(__float_as_uint(p.direction.z));
    optixSetPayload_12(p.seed);
    optixSetPayload_13(static_cast<unsigned int>(p.depth));
    optixSetPayload_14(p.done ? 1u : 0u);
}

__device__ __forceinline__ RenderPayload get_payload() {
    RenderPayload p;
    p.radiance.x = __uint_as_float(optixGetPayload_0());
    p.radiance.y = __uint_as_float(optixGetPayload_1());
    p.radiance.z = __uint_as_float(optixGetPayload_2());
    p.throughput.x = __uint_as_float(optixGetPayload_3());
    p.throughput.y = __uint_as_float(optixGetPayload_4());
    p.throughput.z = __uint_as_float(optixGetPayload_5());
    p.origin.x = __uint_as_float(optixGetPayload_6());
    p.origin.y = __uint_as_float(optixGetPayload_7());
    p.origin.z = __uint_as_float(optixGetPayload_8());
    p.direction.x = __uint_as_float(optixGetPayload_9());
    p.direction.y = __uint_as_float(optixGetPayload_10());
    p.direction.z = __uint_as_float(optixGetPayload_11());
    p.seed = optixGetPayload_12();
    p.depth = static_cast<int>(optixGetPayload_13());
    p.done = optixGetPayload_14() != 0u;
    return p;
}

// ==================== 光线生成程序 ====================

extern "C" __global__ void __raygen__render() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    const int pixel_idx = idx.y * render_params.width + idx.x;
    
    // 初始化随机数种子
    unsigned int seed = tea(pixel_idx, render_params.frame_number);
    
    float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int s = 0; s < render_params.samples_per_pixel; ++s) {
        // 抗锯齿抖动
        float u = (float(idx.x) + rnd(seed)) / float(dim.x);
        float v = (float(idx.y) + rnd(seed)) / float(dim.y);
        
        // 转换到 [-1, 1]
        float2 d = make_float2(u * 2.0f - 1.0f, v * 2.0f - 1.0f);
        
        float3 ray_origin, ray_direction;
        
        if (render_params.is_orthographic) {
            // 正交投影
            ray_origin = render_params.eye + d.x * render_params.U + d.y * render_params.V;
            ray_direction = normalize(render_params.W);
        } else {
            // 透视投影
            ray_origin = render_params.eye;
            ray_direction = normalize(d.x * render_params.U + d.y * render_params.V + render_params.W);
        }
        
        // 路径追踪
        float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
        float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
        
        for (int depth = 0; depth < render_params.max_depth; ++depth) {
            // 初始化 payload
            unsigned int p0 = __float_as_uint(radiance.x);
            unsigned int p1 = __float_as_uint(radiance.y);
            unsigned int p2 = __float_as_uint(radiance.z);
            unsigned int p3 = __float_as_uint(throughput.x);
            unsigned int p4 = __float_as_uint(throughput.y);
            unsigned int p5 = __float_as_uint(throughput.z);
            unsigned int p6 = __float_as_uint(ray_origin.x);
            unsigned int p7 = __float_as_uint(ray_origin.y);
            unsigned int p8 = __float_as_uint(ray_origin.z);
            unsigned int p9 = __float_as_uint(ray_direction.x);
            unsigned int p10 = __float_as_uint(ray_direction.y);
            unsigned int p11 = __float_as_uint(ray_direction.z);
            unsigned int p12 = seed;
            unsigned int p13 = static_cast<unsigned int>(depth);
            unsigned int p14 = 0u; // not done
            
            optixTrace(
                render_params.traversable,
                ray_origin,
                ray_direction,
                1e-4f,              // tmin
                1e16f,              // tmax
                0.0f,               // ray time
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0,                  // SBT offset
                1,                  // SBT stride
                0,                  // miss SBT index
                p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14
            );
            
            // 读取更新后的 payload
            radiance.x = __uint_as_float(p0);
            radiance.y = __uint_as_float(p1);
            radiance.z = __uint_as_float(p2);
            throughput.x = __uint_as_float(p3);
            throughput.y = __uint_as_float(p4);
            throughput.z = __uint_as_float(p5);
            ray_origin.x = __uint_as_float(p6);
            ray_origin.y = __uint_as_float(p7);
            ray_origin.z = __uint_as_float(p8);
            ray_direction.x = __uint_as_float(p9);
            ray_direction.y = __uint_as_float(p10);
            ray_direction.z = __uint_as_float(p11);
            seed = p12;
            bool done = p14 != 0u;
            
            if (done) break;
            
            // 俄罗斯轮盘赌
            if (depth > 3) {
                float max_throughput = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
                if (rnd(seed) > max_throughput) {
                    break;
                }
                throughput = throughput / max_throughput;
            }
        }
        
        accumulated_color = accumulated_color + radiance;
    }
    
    // 平均采样结果
    accumulated_color = accumulated_color / float(render_params.samples_per_pixel);
    
    // 渐进式累积
    if (render_params.frame_number > 0) {
        float4 prev = render_params.color_buffer[pixel_idx];
        float weight = 1.0f / float(render_params.frame_number + 1);
        accumulated_color = make_float3(prev.x, prev.y, prev.z) * (1.0f - weight) + accumulated_color * weight;
    }
    
    render_params.color_buffer[pixel_idx] = make_float4(
        accumulated_color.x, accumulated_color.y, accumulated_color.z, 1.0f
    );
}

// ==================== 未命中程序 ====================

extern "C" __global__ void __miss__render() {
    float3 direction = optixGetWorldRayDirection();
    float3 env_color = sample_environment(normalize(direction));
    
    // 更新 payload
    float3 throughput = make_float3(
        __uint_as_float(optixGetPayload_3()),
        __uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5())
    );
    
    float3 radiance = make_float3(
        __uint_as_float(optixGetPayload_0()),
        __uint_as_float(optixGetPayload_1()),
        __uint_as_float(optixGetPayload_2())
    );
    
    radiance = radiance + throughput * env_color;
    
    optixSetPayload_0(__float_as_uint(radiance.x));
    optixSetPayload_1(__float_as_uint(radiance.y));
    optixSetPayload_2(__float_as_uint(radiance.z));
    optixSetPayload_14(1u); // done
}

// ==================== Closest Hit 程序 ====================

extern "C" __global__ void __closesthit__render() {
    const RenderHitGroupData* data = reinterpret_cast<const RenderHitGroupData*>(optixGetSbtDataPointer());
    
    // 获取交点信息
    const float t = optixGetRayTmax();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    float3 hit_point = ray_origin + t * ray_direction;
    
    // 获取法线（从三角形计算）
    const float2 bary = optixGetTriangleBarycentrics();
    
    // 获取三角形顶点
    OptixTraversableHandle gas = optixGetGASTraversableHandle();
    unsigned int prim_idx = optixGetPrimitiveIndex();
    unsigned int sbt_idx = optixGetSbtGASIndex();
    float3 vertices[3];
    optixGetTriangleVertexData(gas, prim_idx, sbt_idx, 0.0f, vertices);
    
    float3 e1 = vertices[1] - vertices[0];
    float3 e2 = vertices[2] - vertices[0];
    float3 geometric_normal = normalize(cross(e1, e2));
    
    // 确保法线朝向光线
    float3 N = geometric_normal;
    bool front_face = dot(ray_direction, N) < 0.0f;
    if (!front_face) {
        N = N * (-1.0f);
    }
    
    // 读取当前 payload
    unsigned int seed = optixGetPayload_12();
    float3 throughput = make_float3(
        __uint_as_float(optixGetPayload_3()),
        __uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5())
    );
    float3 radiance = make_float3(
        __uint_as_float(optixGetPayload_0()),
        __uint_as_float(optixGetPayload_1()),
        __uint_as_float(optixGetPayload_2())
    );
    
    float3 new_direction;
    float3 new_throughput = throughput;
    bool done = false;
    
    switch (data->shading_type) {
        case ShadingType::Diffuse: {
            // 漫反射
            float u1 = rnd(seed);
            float u2 = rnd(seed);
            new_direction = sample_cosine_hemisphere(N, u1, u2);
            new_throughput = throughput * data->albedo;
            break;
        }
        
        case ShadingType::Mirror: {
            // 镜面反射
            new_direction = reflect(ray_direction, N);
            new_throughput = throughput * data->albedo;
            break;
        }
        
        case ShadingType::Glass: {
            // 玻璃折射
            float eta = front_face ? (1.0f / data->ior) : data->ior;
            float cos_i = fminf(fabsf(dot(ray_direction, N)), 1.0f);
            float Fr = fresnel_schlick(cos_i, data->ior);
            
            if (rnd(seed) < Fr) {
                // 反射
                new_direction = reflect(ray_direction, N);
            } else {
                // 折射
                float3 refracted = refract(ray_direction, N, eta);
                if (length(refracted) < 0.001f) {
                    // 全反射
                    new_direction = reflect(ray_direction, N);
                } else {
                    new_direction = normalize(refracted);
                }
            }
            new_throughput = throughput * data->albedo;
            break;
        }
        
        case ShadingType::Absorber:
        default: {
            // 吸收 - 终止路径
            done = true;
            break;
        }
    }
    
    // 写入更新后的 payload
    optixSetPayload_0(__float_as_uint(radiance.x));
    optixSetPayload_1(__float_as_uint(radiance.y));
    optixSetPayload_2(__float_as_uint(radiance.z));
    optixSetPayload_3(__float_as_uint(new_throughput.x));
    optixSetPayload_4(__float_as_uint(new_throughput.y));
    optixSetPayload_5(__float_as_uint(new_throughput.z));
    
    // 偏移避免自相交
    float3 new_origin = hit_point + N * 1e-3f;
    if (data->shading_type == ShadingType::Glass && !done) {
        // 玻璃需要特殊处理偏移方向
        if (dot(new_direction, geometric_normal) < 0) {
            new_origin = hit_point - geometric_normal * 1e-3f;
        }
    }
    
    optixSetPayload_6(__float_as_uint(new_origin.x));
    optixSetPayload_7(__float_as_uint(new_origin.y));
    optixSetPayload_8(__float_as_uint(new_origin.z));
    optixSetPayload_9(__float_as_uint(new_direction.x));
    optixSetPayload_10(__float_as_uint(new_direction.y));
    optixSetPayload_11(__float_as_uint(new_direction.z));
    optixSetPayload_12(seed);
    optixSetPayload_14(done ? 1u : 0u);
}