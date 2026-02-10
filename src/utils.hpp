// utils.hpp - 使用 DrJit 原生 Matrix
#pragma once
#include <cuda.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <iostream>
#include <vector>
#include <memory>

#include <cuda_runtime.h>

#ifdef USE_DRJIT
#include <drjit/array.h>
#include <drjit/matrix.h>
#include <drjit/jit.h>
#include <drjit/autodiff.h>
#include <drjit/math.h>
#include <drjit/random.h>
#define DIFF_OPTICS_DRJIT 1
#else
#define DIFF_OPTICS_DRJIT 0
#endif

#include <drjit-core/jit.h>

namespace diff_optics {

#if DIFF_OPTICS_DRJIT
    using ScalarType = float;

    // ============= 非微分类型（用于索引、OptiX交互、求交计算）=============
    using FloatC = drjit::CUDAArray<ScalarType>;
    using Int32C = drjit::CUDAArray<int32_t>;
    using UInt32C = drjit::CUDAArray<uint32_t>;
    using MaskC = drjit::CUDAArray<bool>;

    // ============= 微分类型（用于可优化参数、梯度计算）=============
    using Float = drjit::CUDADiffArray<ScalarType>;
    using Int32 = drjit::CUDADiffArray<int32_t>;
    using UInt32 = drjit::CUDADiffArray<uint32_t>;
    using Mask = drjit::CUDADiffArray<bool>;

#else
    using ScalarType = float;
    using Float = float;
    using FloatC = float;
#endif

    // ============= 类型转换辅助函数 =============
    namespace  utils{
        //// 微分 -> 非微分（使用 borrow 获取底层索引的引用）
        inline FloatC detach(const Float& v) {
            return FloatC::borrow(v.index());
        }
    }

    // 标量 -> Float 张量转换
    inline Float from_scalar(ScalarType val) {
#if DIFF_OPTICS_DRJIT
        return Float(val);
#else
        return val;
#endif
    }

    // Float 张量 -> 标量转换
    inline ScalarType to_scalar(const Float& val) {
#if DIFF_OPTICS_DRJIT
        return val[0];
#else
        return val;
#endif
    }

    // ============= Vector 定义 =============
    // 微分版本（用于光线追踪计算、优化参数）
    using Vector2 = drjit::Array<Float, 2>;
    using Vector3 = drjit::Array<Float, 3>;
    using Vector4 = drjit::Array<Float, 4>;

    // 非微分版本（用于 OptiX 交互、索引操作）
    using Vector2C = drjit::Array<FloatC, 2>;
    using Vector3C = drjit::Array<FloatC, 3>;
    using Vector4C = drjit::Array<FloatC, 4>;

    // 使用 DrJit 原生 Matrix
    using Matrix3 = drjit::Matrix<Float, 3>;
    using Matrix4 = drjit::Matrix<Float, 4>;
    using Matrix3C = drjit::Matrix<FloatC, 3>;
    using Matrix4C = drjit::Matrix<FloatC, 4>;

    namespace vec3_utils {
        inline Vector3 from_scalars(ScalarType x, ScalarType y, ScalarType z) {
            return Vector3(from_scalar(x), from_scalar(y), from_scalar(z));
        }

        inline Float get_x(const Vector3& v) { return v[0]; }
        inline Float get_y(const Vector3& v) { return v[1]; }
        inline Float get_z(const Vector3& v) { return v[2]; }

        inline void set_x(Vector3& v, const Float& val) { v[0] = val; }
        inline void set_y(Vector3& v, const Float& val) { v[1] = val; }
        inline void set_z(Vector3& v, const Float& val) { v[2] = val; }

        inline size_t size(const Vector3& v) {
#if DIFF_OPTICS_DRJIT
            return drjit::width(v[0]);
#else
            return 1;
#endif
        }
    }

    namespace vec4_utils {
        inline Vector4 from_scalars(ScalarType x, ScalarType y, ScalarType z, ScalarType w) {
            return Vector4(from_scalar(x), from_scalar(y), from_scalar(z), from_scalar(w));
        }

        inline Float get_x(const Vector4& v) { return v[0]; }
        inline Float get_y(const Vector4& v) { return v[1]; }
        inline Float get_z(const Vector4& v) { return v[2]; }
        inline Float get_w(const Vector4& v) { return v[3]; }

        inline void set_x(Vector4& v, const Float& val) { v[0] = val; }
        inline void set_y(Vector4& v, const Float& val) { v[1] = val; }
        inline void set_z(Vector4& v, const Float& val) { v[2] = val; }
        inline void set_w(Vector4& v, const Float& val) { v[3] = val; }

        inline Vector4 to_homogeneous_point(const Vector3& v) {
            return Vector4(v[0], v[1], v[2], drjit::full<Float>(1.0f, drjit::width(v[0])));
        }

        inline Vector4 to_homogeneous_direction(const Vector3& v) {
            return Vector4(v[0], v[1], v[2], drjit::zeros<Float>(drjit::width(v[0])));
        }

        inline Vector3 to_vector3(const Vector4& v) {
            return Vector3(v[0], v[1], v[2]);
        }

        inline Vector3 to_vector3_perspective(const Vector4& v) {
            Float w = v[3];
            Float eps = from_scalar(1e-10f);
            Float safe_w = drjit::select(drjit::abs(w) < eps, drjit::full<Float>(1e-10f, drjit::width(w)), w);
            return Vector3(v[0] / safe_w, v[1] / safe_w, v[2] / safe_w);
        }
    }

    // ============= 矩阵工具函数 =============
    namespace matrix_utils {
        inline Matrix4 identity() {
            return drjit::identity<Matrix4>();
        }

        inline Matrix4 zeros() {
            return drjit::zeros<Matrix4>();
        }

        inline Matrix4 translation(const Float& tx, const Float& ty, const Float& tz) {
            Matrix4 result = drjit::identity<Matrix4>();
            result(0, 3) = tx;
            result(1, 3) = ty;
            result(2, 3) = tz;
            return result;
        }

        inline Matrix4 translation(ScalarType tx, ScalarType ty, ScalarType tz) {
            return translation(from_scalar(tx), from_scalar(ty), from_scalar(tz));
        }

        inline Matrix4 rotation_x(const Float& angle_deg) {
            Float rad = angle_deg * from_scalar(static_cast<ScalarType>(M_PI / 180.0));
            Float c = drjit::cos(rad);
            Float s = drjit::sin(rad);

            Matrix4 result = drjit::identity<Matrix4>();
            result(1, 1) = c;
            result(1, 2) = -s;
            result(2, 1) = s;
            result(2, 2) = c;
            return result;
        }

        inline Matrix4 rotation_x(ScalarType angle_deg) {
            return rotation_x(from_scalar(angle_deg));
        }

        inline Matrix4 rotation_y(const Float& angle_deg) {
            Float rad = angle_deg * from_scalar(static_cast<ScalarType>(M_PI / 180.0));
            Float c = drjit::cos(rad);
            Float s = drjit::sin(rad);

            Matrix4 result = drjit::identity<Matrix4>();
            result(0, 0) = c;
            result(0, 2) = s;
            result(2, 0) = -s;
            result(2, 2) = c;
            return result;
        }

        inline Matrix4 rotation_y(ScalarType angle_deg) {
            return rotation_y(from_scalar(angle_deg));
        }

        inline Matrix4 rotation_z(const Float& angle_deg) {
            Float rad = angle_deg * from_scalar(static_cast<ScalarType>(M_PI / 180.0));
            Float c = drjit::cos(rad);
            Float s = drjit::sin(rad);

            Matrix4 result = drjit::identity<Matrix4>();
            result(0, 0) = c;
            result(0, 1) = -s;
            result(1, 0) = s;
            result(1, 1) = c;
            return result;
        }

        inline Matrix4 rotation_z(ScalarType angle_deg) {
            return rotation_z(from_scalar(angle_deg));
        }

        inline Matrix4 scale(const Float& sx, const Float& sy, const Float& sz) {
            Matrix4 result = drjit::identity<Matrix4>();
            result(0, 0) = sx;
            result(1, 1) = sy;
            result(2, 2) = sz;
            return result;
        }

        inline Matrix4 scale(ScalarType sx, ScalarType sy, ScalarType sz) {
            return scale(from_scalar(sx), from_scalar(sy), from_scalar(sz));
        }

        inline void print(const Matrix4& m, const std::string& name = "") {
            if (!name.empty()) {
                std::cout << name << ":" << std::endl;
            }
            for (int i = 0; i < 4; ++i) {
                std::cout << "  [";
                for (int j = 0; j < 4; ++j) {
                    std::cout << to_scalar(m(i, j));
                    if (j < 3) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
        }
    }

    struct RNG {
    public:
        RNG(size_t N, uint64_t seed = 1) : rng(N, seed) {}

        FloatC random1d_raw() {
            return rng.next_float32();
        }

        Float random1d() {
            return Float(rng.next_float32());
        }

        Vector2 random2d() {
            return Vector2(
                Float(rng.next_float32()),
                Float(rng.next_float32())
            );
        }

        Vector3 random3d() {
            return Vector3(
                Float(rng.next_float32()),
                Float(rng.next_float32()),
                Float(rng.next_float32())
            );
        }

    private:
        drjit::PCG32<FloatC> rng;
    };

    namespace utils {
        // ============= 基础类型的 gather（微分版本）=============
        inline Float gather(const Float& v, const Int32& indices) {
            return drjit::gather<Float>(v, indices);
        }

        inline Int32 gather(const Int32& v, const Int32& indices) {
            return drjit::gather<Int32>(v, indices);
        }

        inline UInt32 gather(const UInt32& v, const Int32& indices) {
            return drjit::gather<UInt32>(v, indices);
        }

        inline Mask gather(const Mask& v, const Int32& indices) {
            return drjit::gather<Mask>(v, indices);
        }

        // ============= Vector3 的 gather =============
        inline Vector3 gather(const Vector3& v, const Int32& indices) {
            return Vector3(
                drjit::gather<Float>(v[0], indices),
                drjit::gather<Float>(v[1], indices),
                drjit::gather<Float>(v[2], indices)
            );
        }

        // ============= Vector2 的 gather =============
        inline Vector2 gather(const Vector2& v, const Int32& indices) {
            return Vector2(
                drjit::gather<Float>(v[0], indices),
                drjit::gather<Float>(v[1], indices)
            );
        }

        // ============= Vector4 的 gather =============
        inline Vector4 gather(const Vector4& v, const Int32& indices) {
            return Vector4(
                drjit::gather<Float>(v[0], indices),
                drjit::gather<Float>(v[1], indices),
                drjit::gather<Float>(v[2], indices),
                drjit::gather<Float>(v[3], indices)
            );
        }

        // ============= mask_select（使用非微分 Mask 和索引）=============
        template <typename T>
        T mask_select(const T& v, const Mask& mask) {
            Int32 index = drjit::compress(mask);
            return gather(v, index);
        }

        // ============= 基础类型的 scatter（使用非微分 Mask）=============
        inline void scatter(Float& target, const Float& value, const Mask& mask) {
            size_t n = drjit::width(value);
            Int32 indices = drjit::arange<Int32>(static_cast<int>(n));
            drjit::scatter(target, value, indices, mask);
        }


        inline void scatter(Int32& target, const Int32& value, const Mask& mask) {
            size_t n = drjit::width(value);
            Int32 indices = drjit::arange<Int32>(static_cast<int>(n));
            drjit::scatter(target, value, indices, mask);
        }

        inline void scatter(UInt32& target, const UInt32& value, const Mask& mask) {
            size_t n = drjit::width(value);
            Int32 indices = drjit::arange<Int32>(static_cast<int>(n));
            drjit::scatter(target, value, indices, mask);
        }

        inline void scatter(Mask& target, const Mask& value, const Mask& mask) {
            size_t n = drjit::width(value);
            Int32 indices = drjit::arange<Int32>(static_cast<int>(n));
            drjit::scatter(target, value, indices, mask);
        }

        // ============= Vector3 的 scatter =============
        inline void scatter(Vector3& target, const Vector3& value, const Mask& mask) {
            size_t n = drjit::width(value[0]);
            Int32 indices = drjit::arange<Int32>(static_cast<int>(n));
            drjit::scatter(target[0], value[0], indices, mask);
            drjit::scatter(target[1], value[1], indices, mask);
            drjit::scatter(target[2], value[2], indices, mask);
        }

        // ============= Vector2 的 scatter =============
        inline void scatter(Vector2& target, const Vector2& value, const Mask& mask) {
            size_t n = drjit::width(value[0]);
            Int32 indices = drjit::arange<Int32>(static_cast<int>(n));
            drjit::scatter(target[0], value[0], indices, mask);
            drjit::scatter(target[1], value[1], indices, mask);
        }

        // ============= Vector4 的 scatter =============
        inline void scatter(Vector4& target, const Vector4& value, const Mask& mask) {
            size_t n = drjit::width(value[0]);
            Int32 indices = drjit::arange<Int32>(static_cast<int>(n));
            drjit::scatter(target[0], value[0], indices, mask);
            drjit::scatter(target[1], value[1], indices, mask);
            drjit::scatter(target[2], value[2], indices, mask);
            drjit::scatter(target[3], value[3], indices, mask);
        }

        // ============= masked_scatter（指定索引）=============
        inline void masked_scatter(Float& target, const Float& value, const Int32& indices) {
            drjit::scatter(target, value, indices);
        }

        inline void masked_scatter(Int32& target, const Int32& value, const Int32& indices) {
            drjit::scatter(target, value, indices);
        }

        inline void masked_scatter(UInt32& target, const UInt32& value, const Int32& indices) {
            drjit::scatter(target, value, indices);
        }

        inline void masked_scatter(Mask& target, const Mask& value, const Int32& indices) {
            drjit::scatter(target, value, indices);
        }

        inline void masked_scatter(Vector3& target, const Vector3& value, const Int32& indices) {
            drjit::scatter(target[0], value[0], indices);
            drjit::scatter(target[1], value[1], indices);
            drjit::scatter(target[2], value[2], indices);
        }

        inline void masked_scatter(Vector2& target, const Vector2& value, const Int32& indices) {
            drjit::scatter(target[0], value[0], indices);
            drjit::scatter(target[1], value[1], indices);
        }

        inline void masked_scatter(Vector4& target, const Vector4& value, const Int32& indices) {
            drjit::scatter(target[0], value[0], indices);
            drjit::scatter(target[1], value[1], indices);
            drjit::scatter(target[2], value[2], indices);
            drjit::scatter(target[3], value[3], indices);
        }
    }

    // ============= 光线结构体 =============
    struct Ray {
        Vector3 origin;
        Vector3 direction;
        Float wavelength;
        Float radiance;
        Float pdf;

        Ray masked_select(const Mask& mask) const {
            Ray ret;
            Int32 index = drjit::compress(mask);
            ret.origin = utils::gather(this->origin, index);
            ret.direction = utils::gather(this->direction, index);
            ret.wavelength = utils::gather(this->wavelength, index);
            ret.radiance = utils::gather(this->radiance, index);
            ret.pdf = utils::gather(this->pdf, index);
            return ret;
        }

        Ray() :
            origin(Vector3()),
            direction(Vector3()),
            wavelength(from_scalar(550.0f)),
            radiance(from_scalar(0.0f)),
            pdf(from_scalar(0.0f))
        {}

        Ray(const Vector3& origin_, const Vector3& direction_, const Float& wavelength_, const Float& radiance_, const Float& pdf_) :
            origin(origin_),
            direction(direction_),
            wavelength(wavelength_),
            radiance(radiance_),
            pdf(pdf_)
        {}

        static Ray from_scalars(
            ScalarType o_x, ScalarType o_y, ScalarType o_z,
            ScalarType d_x, ScalarType d_y, ScalarType d_z,
            ScalarType wavelength_,
            ScalarType radiance_,
            ScalarType pdf_
        ) {
            return Ray(
                vec3_utils::from_scalars(o_x, o_y, o_z),
                vec3_utils::from_scalars(d_x, d_y, d_z),
                from_scalar(wavelength_),
                from_scalar(radiance_),
                from_scalar(pdf_)
            );
        }

        size_t size() const { return vec3_utils::size(origin); }

        Float get_origin_x() const { return vec3_utils::get_x(origin); }
        Float get_origin_y() const { return vec3_utils::get_y(origin); }
        Float get_origin_z() const { return vec3_utils::get_z(origin); }

        Float get_direction_x() const { return vec3_utils::get_x(direction); }
        Float get_direction_y() const { return vec3_utils::get_y(direction); }
        Float get_direction_z() const { return vec3_utils::get_z(direction); }

        void set_origin_x(const Float& val) { vec3_utils::set_x(origin, val); }
        void set_origin_y(const Float& val) { vec3_utils::set_y(origin, val); }
        void set_origin_z(const Float& val) { vec3_utils::set_z(origin, val); }

        void set_direction_x(const Float& val) { vec3_utils::set_x(direction, val); }
        void set_direction_y(const Float& val) { vec3_utils::set_y(direction, val); }
        void set_direction_z(const Float& val) { vec3_utils::set_z(direction, val); }
    };

    // ============= 表面记录结构体 =============
    struct SurfaceRecord {
        Vector3 position;
        Vector3 normal;
        Int32 surface_indices;
        Mask valid;

        SurfaceRecord masked_select(const Mask& mask) const {
            SurfaceRecord ret;
            Int32 index = drjit::compress(mask);
            ret.position = utils::gather(this->position, index);
            ret.normal = utils::gather(this->normal, index);
            ret.surface_indices = utils::gather(this->surface_indices, index);
            ret.valid = utils::gather(this->valid, index);
            return ret;
        }

        void masked_scatter(const SurfaceRecord& recs, const Int32& index) {
            utils::masked_scatter(position, recs.position, index);
            utils::masked_scatter(normal, recs.normal, index);

            if (drjit::width(recs.surface_indices) > 0)
                utils::masked_scatter(surface_indices, recs.surface_indices, index);
            if (drjit::width(recs.valid) > 0)
                utils::masked_scatter(valid, recs.valid, index);
        }

        std::pair<Vector3, Vector3> getTB() const {
            Float normal_x_c = normal[0];
            Mask mask = drjit::abs(normal_x_c) < Float(0.999f);
            Vector3 helper = drjit::select(mask, Vector3(1, 0, 0), Vector3(0, 1, 0));

            auto tangent = drjit::cross(normal, helper);
            tangent = drjit::normalize(tangent);
            auto bitangent = drjit::cross(normal, tangent);

            return std::make_pair(tangent, drjit::normalize(bitangent));
        }

        Vector3 local_direction_to_world(const Vector3& v) const {
            auto [tangent, bitangent] = getTB();
            Float world_x = v[0] * tangent[0] + v[1] * bitangent[0] + v[2] * normal[0];
            Float world_y = v[0] * tangent[1] + v[1] * bitangent[1] + v[2] * normal[1];
            Float world_z = v[0] * tangent[2] + v[1] * bitangent[2] + v[2] * normal[2];
            return drjit::normalize(Vector3(world_x, world_y, world_z));
        }

        Vector3 world_direction_to_local(const Vector3& v) const {
            auto [tangent, bitangent] = getTB();
            Float local_x = v[0] * tangent[0] + v[1] * tangent[1] + v[2] * tangent[2];
            Float local_y = v[0] * bitangent[0] + v[1] * bitangent[1] + v[2] * bitangent[2];
            Float local_z = v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2];
            return drjit::normalize(Vector3(local_x, local_y, local_z));
        }

        SurfaceRecord() {}
        SurfaceRecord(int N) { init(N); }

        void init(int N) {
            this->position = Vector3(
                drjit::zeros<Float>(N),
                drjit::zeros<Float>(N),
                drjit::zeros<Float>(N)
            );
            this->normal = Vector3(
                drjit::zeros<Float>(N),
                drjit::zeros<Float>(N),
                drjit::zeros<Float>(N)
            );
            this->surface_indices = drjit::full<Int32>(-1, N);
            this->valid = drjit::full<Mask>(false, N);
        }

        void merge(const SurfaceRecord& other) {
            utils::scatter(this->position, other.position, other.valid);
            utils::scatter(this->normal, other.normal, other.valid);
            utils::scatter(this->surface_indices, other.surface_indices, other.valid);
            utils::scatter(this->valid, other.valid, other.valid);
        }

        size_t size() const {
            return drjit::width(surface_indices);
        }
    };
    // 在 Ray 结构体定义之后，SurfaceRecord 之前或之后添加：

// ============= Ray 的标准输出重载 =============
inline std::ostream& operator<<(std::ostream& os, const Ray& ray) {
    size_t n = ray.size();
    os << "Ray(size=" << n << ") {\n";
    
    if (n == 0) {
        os << "  (empty)\n";
    } else if (n == 1) {
        // 单光线，直接输出标量值
        os << "  origin: (" << ray.origin[0][0] << ", " 
           << ray.origin[1][0] << ", " << ray.origin[2][0] << ")\n";
        os << "  direction: (" << ray.direction[0][0] << ", " 
           << ray.direction[1][0] << ", " << ray.direction[2][0] << ")\n";
        os << "  wavelength: " << ray.wavelength[0] << "\n";
        os << "  radiance: " << ray.radiance[0] << "\n";
        os << "  pdf: " << ray.pdf[0] << "\n";
    } else {
        // 多光线，显示前几个和统计信息
        size_t display_count = std::min(n, size_t(3));
        os << "  origin: [";
        for (size_t i = 0; i < display_count; ++i) {
            os << "(" << ray.origin[0][i] << ", " 
               << ray.origin[1][i] << ", " << ray.origin[2][i] << ")";
            if (i < display_count - 1) os << ", ";
        }
        if (n > display_count) os << ", ... (" << n - display_count << " more)";
        os << "]\n";
        
        os << "  direction: [";
        for (size_t i = 0; i < display_count; ++i) {
            os << "(" << ray.direction[0][i] << ", " 
               << ray.direction[1][i] << ", " << ray.direction[2][i] << ")";
            if (i < display_count - 1) os << ", ";
        }
        if (n > display_count) os << ", ... (" << n - display_count << " more)";
        os << "]\n";
        
        os << "  wavelength: [";
        for (size_t i = 0; i < display_count; ++i) {
            os << ray.wavelength[i];
            if (i < display_count - 1) os << ", ";
        }
        if (n > display_count) os << ", ...";
        os << "]\n";
        
        os << "  radiance: [";
        for (size_t i = 0; i < display_count; ++i) {
            os << ray.radiance[i];
            if (i < display_count - 1) os << ", ";
        }
        if (n > display_count) os << ", ...";
        os << "]\n";
        
        os << "  pdf: [";
        for (size_t i = 0; i < display_count; ++i) {
            os << ray.pdf[i];
            if (i < display_count - 1) os << ", ";
        }
        if (n > display_count) os << ", ...";
        os << "]\n";
    }
    
    os << "}";
    return os;
}

// ============= SurfaceRecord 的标准输出重载 =============
inline std::ostream& operator<<(std::ostream& os, const SurfaceRecord& record) {
    size_t n = record.size();
    os << "SurfaceRecord(size=" << n << ") {\n";
    
    if (n == 0) {
        os << "  (empty)\n";
    } else if (n == 1) {
        // 单记录，直接输出标量值
        os << "  position: (" << record.position[0][0] << ", " 
           << record.position[1][0] << ", " << record.position[2][0] << ")\n";
        os << "  normal: (" << record.normal[0][0] << ", " 
           << record.normal[1][0] << ", " << record.normal[2][0] << ")\n";
        os << "  surface_index: " << record.surface_indices[0] << "\n";
        os << "  valid: " << (record.valid[0] ? "true" : "false") << "\n";
    } else {
        // 多记录，显示前几个和统计信息
        size_t display_count = std::min(n, size_t(3));
        
        os << "  position: [";
        for (size_t i = 0; i < display_count; ++i) {
            os << "(" << record.position[0][i] << ", " 
               << record.position[1][i] << ", " << record.position[2][i] << ")";
            if (i < display_count - 1) os << ", ";
        }
        if (n > display_count) os << ", ... (" << n - display_count << " more)";
        os << "]\n";
        
        os << "  normal: [";
        for (size_t i = 0; i < display_count; ++i) {
            os << "(" << record.normal[0][i] << ", " 
               << record.normal[1][i] << ", " << record.normal[2][i] << ")";
            if (i < display_count - 1) os << ", ";
        }
        if (n > display_count) os << ", ...";
        os << "]\n";
        
        os << "  surface_indices: [";
        for (size_t i = 0; i < display_count; ++i) {
            os << record.surface_indices[i];
            if (i < display_count - 1) os << ", ";
        }
        if (n > display_count) os << ", ...";
        os << "]\n";
        
        // 统计有效数量
        size_t valid_count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (record.valid[i]) valid_count++;
        }
        os << "  valid: " << valid_count << "/" << n << " hits\n";
    }
    
    os << "}";
    return os;
}
    // ============= Transform 结构 =============
    struct Transform {
        Float tx, ty, tz;
        Float rx, ry, rz;

        Transform() :
            tx(from_scalar(0.0f)), ty(from_scalar(0.0f)), tz(from_scalar(0.0f)),
            rx(from_scalar(0.0f)), ry(from_scalar(0.0f)), rz(from_scalar(0.0f))
        {}

        Transform(ScalarType tx_, ScalarType ty_, ScalarType tz_,
            ScalarType rx_, ScalarType ry_, ScalarType rz_) :
            tx(from_scalar(tx_)), ty(from_scalar(ty_)), tz(from_scalar(tz_)),
            rx(from_scalar(rx_)), ry(from_scalar(ry_)), rz(from_scalar(rz_))
        {}

        Transform(const std::vector<ScalarType>& t, const std::vector<ScalarType>& r) :
            tx(from_scalar(t[0])), ty(from_scalar(t[1])), tz(from_scalar(t[2])),
            rx(from_scalar(r[0])), ry(from_scalar(r[1])), rz(from_scalar(r[2]))
        {}

        Transform(const Float& tx_, const Float& ty_, const Float& tz_,
            const Float& rx_, const Float& ry_, const Float& rz_) :
            tx(tx_), ty(ty_), tz(tz_),
            rx(rx_), ry(ry_), rz(rz_)
        {}

        const Float& get_tx() const { return tx; }
        const Float& get_ty() const { return ty; }
        const Float& get_tz() const { return tz; }
        const Float& get_rx() const { return rx; }
        const Float& get_ry() const { return ry; }
        const Float& get_rz() const { return rz; }

        ScalarType get_tx_scalar() const { return to_scalar(tx); }
        ScalarType get_ty_scalar() const { return to_scalar(ty); }
        ScalarType get_tz_scalar() const { return to_scalar(tz); }
        ScalarType get_rx_scalar() const { return to_scalar(rx); }
        ScalarType get_ry_scalar() const { return to_scalar(ry); }
        ScalarType get_rz_scalar() const { return to_scalar(rz); }

        void set_translation(ScalarType tx_, ScalarType ty_, ScalarType tz_) {
            tx = from_scalar(tx_);
            ty = from_scalar(ty_);
            tz = from_scalar(tz_);
        }

        void set_translation(const Float& tx_, const Float& ty_, const Float& tz_) {
            tx = tx_;
            ty = ty_;
            tz = tz_;
        }

        void set_rotation(ScalarType rx_, ScalarType ry_, ScalarType rz_) {
            rx = from_scalar(rx_);
            ry = from_scalar(ry_);
            rz = from_scalar(rz_);
        }

        void set_rotation(const Float& rx_, const Float& ry_, const Float& rz_) {
            rx = rx_;
            ry = ry_;
            rz = rz_;
        }

        Matrix4 compute_matrix() const {
            Matrix4 T = matrix_utils::translation(tx, ty, tz);
            Matrix4 Rx = matrix_utils::rotation_x(rx);
            Matrix4 Ry = matrix_utils::rotation_y(ry);
            Matrix4 Rz = matrix_utils::rotation_z(rz);
            return T * Rx * Ry * Rz;
        }

        Matrix4 compute_inverse_matrix() const {
            return drjit::inverse(compute_matrix());
        }

        Vector3 transform_point(const Vector3& p) const {
            Matrix4 M = compute_matrix();
            Vector4 p_homo = vec4_utils::to_homogeneous_point(p);
            Vector4 result = M * p_homo;
            return vec4_utils::to_vector3(result);
        }

        Vector3 transform_direction(const Vector3& d) const {
            Matrix4 M = compute_matrix();
            Vector4 d_homo = vec4_utils::to_homogeneous_direction(d);
            Vector4 result = M * d_homo;
            return vec4_utils::to_vector3(result);
        }

        Vector3 transform_normal(const Vector3& n) const {
            Matrix4 inv_transpose = drjit::transpose(compute_inverse_matrix());
            Vector4 n_homo = vec4_utils::to_homogeneous_direction(n);
            Vector4 result = inv_transpose * n_homo;
            Vector3 n_transformed = vec4_utils::to_vector3(result);
            Float len = drjit::norm(n_transformed);
            Float eps = from_scalar(1e-10f);
            len = drjit::maximum(len, eps);
            return n_transformed / len;
        }

        Vector3 inverse_transform_point(const Vector3& p) const {
            Matrix4 M_inv = compute_inverse_matrix();
            Vector4 p_homo = vec4_utils::to_homogeneous_point(p);
            Vector4 result = M_inv * p_homo;
            return vec4_utils::to_vector3(result);
        }

        Vector3 inverse_transform_direction(const Vector3& d) const {
            Matrix4 M_inv = compute_inverse_matrix();
            Vector4 d_homo = vec4_utils::to_homogeneous_direction(d);
            Vector4 result = M_inv * d_homo;
            return vec4_utils::to_vector3(result);
        }

        Vector3 inverse_transform_normal(const Vector3& n) const {
            Matrix4 mat_transpose = drjit::transpose(compute_matrix());
            Vector4 n_homo = vec4_utils::to_homogeneous_direction(n);
            Vector4 result = mat_transpose * n_homo;
            Vector3 n_transformed = vec4_utils::to_vector3(result);
            Float len = drjit::norm(n_transformed);
            Float eps = from_scalar(1e-10f);
            len = drjit::maximum(len, eps);
            return n_transformed / len;
        }

        Ray transform_ray(const Ray& ray) const {
            Ray result;
            result.origin = transform_point(ray.origin);
            result.direction = transform_direction(ray.direction);
            result.wavelength = ray.wavelength;
            result.radiance = ray.radiance;
            result.pdf = ray.pdf;
            return result;
        }

        Ray inverse_transform_ray(const Ray& ray) const {
            Ray result;
            result.origin = inverse_transform_point(ray.origin);
            result.direction = inverse_transform_direction(ray.direction);
            result.wavelength = ray.wavelength;
            result.radiance = ray.radiance;
            result.pdf = ray.pdf;
            return result;
        }

        Matrix4 operator*(const Transform& other) const {
            return compute_matrix() * other.compute_matrix();
        }

        void print(const std::string& name = "") const {
            if (!name.empty()) {
                std::cout << name << ":" << std::endl;
            }
            std::cout << "  Translation: (" << get_tx_scalar() << ", "
                << get_ty_scalar() << ", " << get_tz_scalar() << ")" << std::endl;
            std::cout << "  Rotation (deg): (" << get_rx_scalar() << ", "
                << get_ry_scalar() << ", " << get_rz_scalar() << ")" << std::endl;
        }
    };

} // namespace diff_optics