// ==================== cuda_kernels.cu ====================
#include <cuda_runtime.h>

extern "C" {

// 打包光线数据
__global__ void pack_rays_kernel(
    const float* __restrict__ ox, const float* __restrict__ oy, const float* __restrict__ oz,
    const float* __restrict__ dx, const float* __restrict__ dy, const float* __restrict__ dz,
    float3* __restrict__ origins, float3* __restrict__ directions,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        origins[i] = make_float3(ox[i], oy[i], oz[i]);
        directions[i] = make_float3(dx[i], dy[i], dz[i]);
    }
}

// 变换顶点到世界空间
__global__ void transform_vertices_kernel(
    const float* __restrict__ local_x,
    const float* __restrict__ local_y,
    const float* __restrict__ local_z,
    const float* __restrict__ matrix,  // 4x4 行优先
    float3* __restrict__ world_vertices,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float lx = local_x[i];
        float ly = local_y[i];
        float lz = local_z[i];
        
        // 矩阵乘法 (4x4 * [x, y, z, 1])
        float wx = matrix[0] * lx + matrix[1] * ly + matrix[2] * lz + matrix[3];
        float wy = matrix[4] * lx + matrix[5] * ly + matrix[6] * lz + matrix[7];
        float wz = matrix[8] * lx + matrix[9] * ly + matrix[10] * lz + matrix[11];
        
        world_vertices[i] = make_float3(wx, wy, wz);
    }
}

// 复制结果
__global__ void copy_results_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

// 启动打包光线核函数
void launch_pack_rays(
    const float* ox, const float* oy, const float* oz,
    const float* dx, const float* dy, const float* dz,
    float3* origins, float3* directions,
    unsigned int n, cudaStream_t stream
) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    pack_rays_kernel<<<num_blocks, block_size, 0, stream>>>(ox, oy, oz, dx, dy, dz, origins, directions, n);
}

// 启动变换顶点核函数
void launch_transform_vertices(
    const float* local_x, const float* local_y, const float* local_z,
    const float* matrix,
    float3* world_vertices,
    unsigned int n, cudaStream_t stream
) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    transform_vertices_kernel<<<num_blocks, block_size, 0, stream>>>(local_x, local_y, local_z, matrix, world_vertices, n);
}

// 启动复制结果核函数
void launch_copy_results(
    const float* src,
    float* dst,
    unsigned int n, cudaStream_t stream
) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    copy_results_kernel<<<num_blocks, block_size, 0, stream>>>(src, dst, n);
}

} // extern "C"