// light.hpp
#pragma once
#include "utils.hpp"
#include "emitter.hpp"
#include "spectrum.hpp"
#include "surface.hpp"
#include <memory>

namespace diff_optics {

struct Light {
public:
    virtual ~Light() = default;
    virtual Ray sampleRays(const Vector2& uv1, const Vector2& uv2, const Float& u) const = 0;
    
    // Getter/Setter for bindings
    Float& get_power() { return power; }
    const Float& get_power() const { return power; }
    void set_power(const Float& p) { power = p; }
    std::shared_ptr<Emitter> get_emitter() const { return emitter; }
    void set_emitter(std::shared_ptr<Emitter> e) { emitter = e; }
    std::shared_ptr<Spectrum> get_spectrum() const { return spectrum; }
    void set_spectrum(std::shared_ptr<Spectrum> s) { spectrum = s; }
    
    virtual std::string type_name() const = 0;
    virtual void print(const std::string& name = "") const = 0;
    
protected:
    std::shared_ptr<Emitter> emitter;
    std::shared_ptr<Spectrum> spectrum;
    Float power;
};

// ============= 表面光源 =============
struct SurfaceLight : public Light {
public:
    std::shared_ptr<Surface> surface;
    
    SurfaceLight(std::shared_ptr<Surface> sf, std::shared_ptr<Emitter> e, std::shared_ptr<Spectrum> s) {
        surface = sf;
        emitter = e;
        spectrum = s;
        power = from_scalar(1.0f);
        
        // 检查表面类型是否支持作为光源
        if (!supports_surface_light(sf)) {
            throw std::runtime_error("Surface type '" + sf->type_name() + "' is not supported as a light source");
        }
    }
    
    std::string type_name() const override { return "SurfaceLight"; }
    
    // 获取/设置关联的表面
    std::shared_ptr<Surface> get_surface() const { return surface; }
    void set_surface(std::shared_ptr<Surface> sf) { 
        if (!supports_surface_light(sf)) {
            throw std::runtime_error("Surface type '" + sf->type_name() + "' is not supported as a light source");
        }
        surface = sf; 
    }
    
    Ray sampleRays(const Vector2& uv1, const Vector2& uv2, const Float& u) const override {
        size_t n_rays = drjit::width(u);
        
        // ============================================
        // 1. 表面位置采样（局部空间 -> 世界空间）
        // ============================================
        Vector3 world_origin;
        Vector3 world_normal;
        Float pos_pdf;
        
        sample_surface_position(uv1, world_origin, world_normal, pos_pdf);
        
        Float pos_weight = drjit::full<Float>(1.0f, n_rays);
        
        // ============================================
        // 2. 波长采样
        // ============================================
        auto [wavelength, lbd_weight, lbd_pdf] = spectrum->sample(u);
        
        // ============================================
        // 3. 方向采样（发光器局部空间）
        // ============================================
        Float dir_weight, dir_pdf;
        Vector3 local_direction = emitter->sampleDirection(uv2, dir_weight, dir_pdf);
        
        // ============================================
        // 4. 将方向从发光器局部空间变换到世界空间
        // ============================================
        // 发光器局部空间：z 轴为法线方向
        // 需要构建 TBN 矩阵将 local_direction 变换到世界空间
        Vector3 world_direction = local_to_world_direction(local_direction, world_normal);
        
        // 归一化方向（变换可能引入数值误差）
        Float dir_len = drjit::norm(world_direction);
        Float eps = from_scalar(1e-10f);
        dir_len = drjit::maximum(dir_len, eps);
        world_direction = world_direction / dir_len;
        
        // ============================================
        // 5. 计算总 PDF 和总 weight
        // ============================================
        Float total_pdf = pos_pdf * lbd_pdf * dir_pdf;
        Float total_weight = pos_weight * lbd_weight * dir_weight;
        
        Float sum_wp = drjit::sum(total_weight / total_pdf);
        Float Lc = static_cast<Float>(n_rays) * power / sum_wp;
        Float radiance = (drjit::zeros<Float>(n_rays) + Lc) * total_weight;
        
        // ============================================
        // 6. 构造并返回 Ray（世界空间）
        // ============================================
        Ray ret(world_origin, world_direction, wavelength, radiance, total_pdf);
        return ret;
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: SurfaceLight" << std::endl;
        std::cout << "  Power: " << to_scalar(power) << std::endl;
        if (surface) {
            std::cout << "  Surface: " << surface->type_name() << std::endl;
        }
        if (spectrum) {
            std::cout << "  Spectrum: " << spectrum->type_name() << std::endl;
        }
    }
private:
    // 检查表面类型是否支持作为光源
    static bool supports_surface_light(const std::shared_ptr<Surface>& sf) {
        if (!sf) return false;
        std::string type = sf->boundary_type();
        return (type == "Rectangle" || type == "Circle");
    }
    
    // 在表面上采样位置（世界空间）
    void sample_surface_position(
        const Vector2& uv,
        Vector3& world_position,
        Vector3& world_normal,
        Float& pdf
    ) const {
        size_t n = drjit::width(uv[0]);
        
        std::string type = surface->boundary_type();
        
        if (type == "Rectangle") {
            auto rect = std::static_pointer_cast<RectangleSurface>(surface);
            sample_rectangle_surface(rect, uv, world_position, world_normal, pdf);
        } 
        else if (type == "Circle") {
            auto circle = std::static_pointer_cast<CircleSurface>(surface);
            sample_circle_surface(circle, uv, world_position, world_normal, pdf);
        }
        else {
            // 不支持的类型，填充默认值
            world_position = Vector3(
                drjit::zeros<Float>(n),
                drjit::zeros<Float>(n),
                drjit::zeros<Float>(n)
            );
            world_normal = Vector3(
                drjit::zeros<Float>(n),
                drjit::zeros<Float>(n),
                drjit::full<Float>(1.0f, n)
            );
            pdf = drjit::full<Float>(1.0f, n);
        }
    }
    
    // 矩形平面采样
    void sample_rectangle_surface(
        const std::shared_ptr<RectangleSurface>& rect,
        const Vector2& uv,
        Vector3& world_position,
        Vector3& world_normal,
        Float& pdf
    ) const {
        size_t n = drjit::width(uv[0]);
        auto width = rect->width;
        auto height = rect->height;
        ScalarType half_w = width * 0.5f;
        ScalarType half_h = height * 0.5f;
        
        // uv ∈ [0,1]² -> 局部坐标 [-half_w, half_w] x [-half_h, half_h]
        Float local_x = (uv[0] - Float(0.5f)) * Float(width);
        Float local_y = (uv[1] - Float(0.5f)) * Float(height);
        Float local_z = drjit::zeros<Float>(n);
        
        Vector3 local_position(local_x, local_y, local_z);
        
        // 平面的局部法线始终为 +z
        Vector3 local_normal(
            drjit::zeros<Float>(n),
            drjit::zeros<Float>(n),
            drjit::full<Float>(1.0f, n)
        );
        
        // 变换到世界空间
        const Transform& transform = rect->get_transform();
        world_position = transform.transform_point(local_position);
        world_normal = transform.transform_normal(local_normal);
        
        // 归一化法线
        Float normal_len = drjit::norm(world_normal);
        Float eps = from_scalar(1e-10f);
        normal_len = drjit::maximum(normal_len, eps);
        world_normal = world_normal / normal_len;
        
        // PDF = 1 / 面积
        ScalarType area = width * height;
        pdf = drjit::full<Float>(1.0f / area, n);
    }
    
    // 圆形平面采样（使用同心圆盘采样避免中心聚集）
    void sample_circle_surface(
        const std::shared_ptr<CircleSurface>& circle,
        const Vector2& uv,
        Vector3& world_position,
        Vector3& world_normal,
        Float& pdf
    ) const {
        auto radius = circle->radius;
        size_t n = drjit::width(uv[0]);
        
        // 同心圆盘采样 (Concentric Disk Sampling)
        // 将 uv ∈ [0,1]² 映射到单位圆盘
        Float u_mapped = Float(2.0f) * uv[0] - Float(1.0f);  // [-1, 1]
        Float v_mapped = Float(2.0f) * uv[1] - Float(1.0f);  // [-1, 1]
        
        // 处理原点附近
        Float eps = from_scalar(1e-10f);
        Mask at_center = (drjit::abs(u_mapped) < eps) & (drjit::abs(v_mapped) < eps);
        
        // 同心圆盘映射
        Float r, theta;
        
        Mask use_u = drjit::abs(u_mapped) > drjit::abs(v_mapped);
        
        // 当 |u| > |v| 时
        Float r_u = u_mapped;
        Float theta_u = Float(M_PI / 4.0f) * (v_mapped / drjit::select(drjit::abs(u_mapped) > eps, u_mapped, Float(1.0f)));
        
        // 当 |v| >= |u| 时
        Float r_v = v_mapped;
        Float theta_v = Float(M_PI / 2.0f) - Float(M_PI / 4.0f) * (u_mapped / drjit::select(drjit::abs(v_mapped) > eps, v_mapped, Float(1.0f)));
        
        r = drjit::select(use_u, r_u, r_v);
        theta = drjit::select(use_u, theta_u, theta_v);
        
        // 处理原点
        r = drjit::select(at_center, Float(0.0f), r);
        theta = drjit::select(at_center, Float(0.0f), theta);
        
        // 计算局部坐标
        Float local_x = r * drjit::cos(theta) * Float(radius);
        Float local_y = r * drjit::sin(theta) * Float(radius);
        Float local_z = drjit::zeros<Float>(n);
        
        Vector3 local_position(local_x, local_y, local_z);
        
        // 平面的局部法线始终为 +z
        Vector3 local_normal(
            drjit::zeros<Float>(n),
            drjit::zeros<Float>(n),
            drjit::full<Float>(1.0f, n)
        );
        
        // 变换到世界空间
        const Transform& transform = circle->get_transform();
        world_position = transform.transform_point(local_position);
        world_normal = transform.transform_normal(local_normal);
        
        // 归一化法线
        Float normal_len = drjit::norm(world_normal);
        normal_len = drjit::maximum(normal_len, eps);
        world_normal = world_normal / normal_len;
        
        // PDF = 1 / 面积 (圆的面积 = π * r²)
        ScalarType area = static_cast<ScalarType>(M_PI) * radius * radius;
        pdf = drjit::full<Float>(1.0f / area, n);
    }
    
    // 将发光器局部方向变换到世界空间
    // 发光器局部空间：z 轴为法线方向，x/y 为切平面
    Vector3 local_to_world_direction(const Vector3& local_dir, const Vector3& world_normal) const {
        // 构建 TBN 基（切线、副切线、法线）
        // 选择一个与法线不平行的辅助向量来构建切线
        size_t n = drjit::width(local_dir[0]);
        
        Float normal_x = world_normal[0];
        Float normal_y = world_normal[1];
        Float normal_z = world_normal[2];
        
        // 选择辅助向量：如果法线接近 (1,0,0)，使用 (0,1,0)，否则使用 (1,0,0)
        Mask use_y = drjit::abs(normal_x) > Float(0.9f);
        
        Vector3 helper = drjit::select(
            use_y,
            Vector3(drjit::zeros<Float>(n), drjit::full<Float>(1.0f, n), drjit::zeros<Float>(n)),
            Vector3(drjit::full<Float>(1.0f, n), drjit::zeros<Float>(n), drjit::zeros<Float>(n))
        );
        
        // 切线 T = normalize(helper × normal)
        Vector3 tangent = drjit::cross(helper, world_normal);
        Float t_len = drjit::norm(tangent);
        Float eps = from_scalar(1e-10f);
        t_len = drjit::maximum(t_len, eps);
        tangent = tangent / t_len;
        
        // 副切线 B = normal × tangent
        Vector3 bitangent = drjit::cross(world_normal, tangent);
        Float b_len = drjit::norm(bitangent);
        b_len = drjit::maximum(b_len, eps);
        bitangent = bitangent / b_len;
        
        // 变换：world_dir = local_dir.x * T + local_dir.y * B + local_dir.z * N
        Float world_x = local_dir[0] * tangent[0] + local_dir[1] * bitangent[0] + local_dir[2] * world_normal[0];
        Float world_y = local_dir[0] * tangent[1] + local_dir[1] * bitangent[1] + local_dir[2] * world_normal[1];
        Float world_z = local_dir[0] * tangent[2] + local_dir[1] * bitangent[2] + local_dir[2] * world_normal[2];
        
        return Vector3(world_x, world_y, world_z);
    }
};

// ============= 点光源 =============
struct PointLight : public Light {
public:
    Transform transform;  // 局部空间 -> 世界空间的变换
    
    PointLight(std::shared_ptr<Emitter> e, std::shared_ptr<Spectrum> s) {
        power = from_scalar(1.0f);
        emitter = e;
        spectrum = s;
        transform = Transform();
    }
    
    PointLight(const Transform& trans, const Float& pow, std::shared_ptr<Emitter> emit, std::shared_ptr<Spectrum> s) {
        transform = trans;
        power = pow;
        emitter = emit;
        spectrum = s;
    }
    
    std::string type_name() const override { return "PointLight"; }
    
    const Transform& get_transform() const { return transform; }
    Transform& get_transform() { return transform; }
    void set_transform(const Transform& t) { transform = t; }
    
    Ray sampleRays(const Vector2& uv1, const Vector2& uv2, const Float& u) const override {
        size_t n_rays = drjit::width(u);
        
        // ============================================
        // 1. 位置采样（局部空间）
        // ============================================
        Vector3 local_origin(
            drjit::zeros<Float>(n_rays),
            drjit::zeros<Float>(n_rays),
            drjit::zeros<Float>(n_rays)
        );
        
        Float pos_weight = drjit::full<Float>(1.0f, n_rays);
        Float pos_pdf = drjit::full<Float>(1.0f, n_rays);

        // ============================================
        // 2. 波长采样
        // ============================================
        auto [wavelength, lbd_weight, lbd_pdf] = spectrum->sample(u);
        
        // ============================================
        // 3. 方向采样（局部空间）
        // ============================================
        Float dir_weight, dir_pdf;
        Vector3 local_direction = emitter->sampleDirection(uv2, dir_weight, dir_pdf);
        
        // ============================================
        // 4. 变换到世界空间
        // ============================================
        Vector3 world_origin = transform.transform_point(local_origin);
        Vector3 world_direction = transform.transform_direction(local_direction);
        
        Float dir_len = drjit::norm(world_direction);
        Float eps = from_scalar(1e-10f);
        dir_len = drjit::maximum(dir_len, eps);
        world_direction = world_direction / dir_len;

        // ============================================
        // 5. 计算总 PDF 和总 weight
        // ============================================
        Float total_pdf = pos_pdf * lbd_pdf * dir_pdf;
        Float total_weight = pos_weight * lbd_weight * dir_weight;

        Float sum_wp = drjit::sum(total_weight / total_pdf);
        Float Lc = static_cast<Float>(n_rays) * power / sum_wp;
        Float radiance = (drjit::zeros<Float>(n_rays) + Lc) * total_weight;
        
        // ============================================
        // 6. 构造并返回 Ray
        // ============================================
        Ray ret(world_origin, world_direction, wavelength, radiance, total_pdf);
        return ret;
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: PointLight" << std::endl;
        std::cout << "  Power: " << to_scalar(power) << std::endl;
        transform.print("  Transform");
    }
};

} // namespace diff_optics