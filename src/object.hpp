// object.hpp
#pragma once
#include "utils.hpp"
#include <memory>
#include <vector>
#include <string>
#include <utility>

namespace diff_optics {

// 前向声明
struct Surface;

// ============= Object 类型枚举 =============
enum class ObjectType {
    Surface = 0,
    Shell = 1,
    Solid = 2
};

// ============= 抽象基类 Object =============
struct Object {
protected:
    Transform transform_;        // 局部 -> 父级（或世界）的变换
    Object* parent_ = nullptr;   // 父对象指针（用于层次变换）
    
public:
    virtual ~Object() = default;
    
    // ============= 类型信息 =============
    virtual ObjectType object_type() const = 0;
    virtual std::string type_name() const = 0;
    
    // ============= Transform 访问 =============
    Transform& get_transform() { return transform_; }
    const Transform& get_transform() const { return transform_; }
    void set_transform(const Transform& t) { transform_ = t; }
    
    // ============= 层次关系管理 =============
    void set_parent(Object* p) { parent_ = p; }
    Object* get_parent() const { return parent_; }
    bool has_parent() const { return parent_ != nullptr; }
    
    // ============= 世界变换计算 =============
    Matrix4 get_world_matrix() const {
        if (parent_) {
            return parent_->get_world_matrix() * transform_.compute_matrix();
        }
        return transform_.compute_matrix();
    }
    
    Matrix4 get_inverse_world_matrix() const {
        return drjit::inverse(get_world_matrix());
    }
    
    // ============= 世界坐标变换方法 =============
    Vector3 to_world_point(const Vector3& local_p) const {
        if (!parent_) {
            return transform_.transform_point(local_p);
        }
        Matrix4 M = get_world_matrix();
        Vector4 p_homo = vec4_utils::to_homogeneous_point(local_p);
        return vec4_utils::to_vector3(M * p_homo);
    }
    
    Vector3 to_world_direction(const Vector3& local_d) const {
        if (!parent_) {
            return transform_.transform_direction(local_d);
        }
        Matrix4 M = get_world_matrix();
        Vector4 d_homo = vec4_utils::to_homogeneous_direction(local_d);
        return vec4_utils::to_vector3(M * d_homo);
    }
    
    Vector3 to_world_normal(const Vector3& local_n) const {
        if (!parent_) {
            return transform_.transform_normal(local_n);
        }
        Matrix4 inv_transpose = drjit::transpose(get_inverse_world_matrix());
        Vector4 n_homo = vec4_utils::to_homogeneous_direction(local_n);
        Vector3 n_transformed = vec4_utils::to_vector3(inv_transpose * n_homo);
        Float len = drjit::norm(n_transformed);
        Float eps = from_scalar(1e-10f);
        len = drjit::maximum(len, eps);
        return n_transformed / len;
    }
    
    Vector3 from_world_point(const Vector3& world_p) const {
        if (!parent_) {
            return transform_.inverse_transform_point(world_p);
        }
        Matrix4 M_inv = get_inverse_world_matrix();
        Vector4 p_homo = vec4_utils::to_homogeneous_point(world_p);
        return vec4_utils::to_vector3(M_inv * p_homo);
    }
    
    Vector3 from_world_direction(const Vector3& world_d) const {
        if (!parent_) {
            return transform_.inverse_transform_direction(world_d);
        }
        Matrix4 M_inv = get_inverse_world_matrix();
        Vector4 d_homo = vec4_utils::to_homogeneous_direction(world_d);
        return vec4_utils::to_vector3(M_inv * d_homo);
    }
    
    Vector3 from_world_normal(const Vector3& world_n) const {
        if (!parent_) {
            return transform_.inverse_transform_normal(world_n);
        }
        Matrix4 mat_transpose = drjit::transpose(get_world_matrix());
        Vector4 n_homo = vec4_utils::to_homogeneous_direction(world_n);
        Vector3 n_transformed = vec4_utils::to_vector3(mat_transpose * n_homo);
        Float len = drjit::norm(n_transformed);
        Float eps = from_scalar(1e-10f);
        len = drjit::maximum(len, eps);
        return n_transformed / len;
    }
    
    // ============= Surface 访问接口（用于 OptiX 场景构建）=============
    virtual std::vector<std::pair<std::string, std::shared_ptr<Surface>>> 
        get_surfaces_with_names(const std::string& prefix) = 0;
    
    virtual size_t surface_count() const = 0;
    
    // ============= 网格管理 =============
    virtual void invalidate_all_meshes() = 0;
    
    // ============= 打印信息 =============
    virtual void print(const std::string& name = "") const = 0;
};

} // namespace diff_optics