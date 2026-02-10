// bsdf.hpp
#pragma once
#include "utils.hpp"
#include <memory>

namespace diff_optics {

// ============= BSDF 采样结果 =============
struct BSDFSample {
    Vector3 direction;  // 采样的出射方向（世界空间）
    Float weight;       // 采样权重 = bsdf * cos(theta) / pdf
    Float pdf;          // 采样概率密度
    Mask valid;         // 是否有效（光线是否继续传播）
    
    BSDFSample() = default;
    
    BSDFSample(size_t n) {
        direction = Vector3(
            drjit::zeros<Float>(n),
            drjit::zeros<Float>(n),
            drjit::zeros<Float>(n)
        );
        weight = drjit::zeros<Float>(n);
        pdf = drjit::zeros<Float>(n);
        valid = drjit::full<Mask>(false, n);
    }
};

// ============= 抽象基类 BSDF =============
struct BSDF {
    virtual ~BSDF() = default;
    
    // 获取 BSDF 类型名称
    virtual std::string type_name() const = 0;
    
    // 采样出射方向
    // ray: 入射光线（世界空间）
    // hit: 交点记录（法线为几何法线，未做方向调整）
    // uv: 二维随机数
    // n_outer: 表面外侧（法线正方向）折射率
    // n_inner: 表面内侧（法线负方向）折射率
    // 返回：BSDFSample
    virtual BSDFSample sample(
        const Ray& ray,
        const SurfaceRecord& hit,
        const Vector2& uv,
        const Float& n_outer,
        const Float& n_inner
    ) const = 0;
    
    // 计算 BSDF 值（用于 MIS 等高级算法）
    virtual Float eval(
        const Vector3& wi,
        const Vector3& wo,
        const SurfaceRecord& hit
    ) const = 0;
    
    // 计算采样概率密度
    virtual Float pdf(
        const Vector3& wi,
        const Vector3& wo,
        const SurfaceRecord& hit
    ) const = 0;
    
    // 打印信息
    virtual void print(const std::string& name = "") const = 0;
    
protected:
    // 辅助函数：计算反射方向
    // wi: 入射方向（光线传播方向，指向表面）
    // n: 面向光线的法线
    // 返回：反射方向（离开表面）
    static Vector3 compute_reflect(const Vector3& wi, const Vector3& n, const Float& cos_i) {
        // 反射公式: wr = wi - 2 * (wi · n) * n = wi + 2 * cos_i * n
        // 其中 cos_i = -wi · n（n 面向光线，所以 wi · n < 0）
        return wi + Float(2.0f) * cos_i * n;
    }
    
    // 辅助函数：计算折射方向（使用 Snell 定律）
    // wi: 入射方向（光线传播方向，指向表面）
    // n: 面向光线的法线
    // cos_i: |wi · n|（正值）
    // eta: n_i / n_t（入射侧/透射侧折射率比）
    // 返回：(折射方向, cos_t, 是否发生全反射)
    static std::tuple<Vector3, Float, Mask> compute_refract(
        const Vector3& wi, 
        const Vector3& n, 
        const Float& cos_i,
        const Float& eta
    ) {
        // Snell 定律：n_i * sin_i = n_t * sin_t
        // sin_t = eta * sin_i，其中 eta = n_i / n_t
        Float sin_i_sq = Float(1.0f) - cos_i * cos_i;
        Float sin_t_sq = eta * eta * sin_i_sq;
        
        // 检查全反射条件
        Mask tir = sin_t_sq > Float(1.0f);
        
        // 计算 cos_t，防止负数开方
        Float cos_t_sq = Float(1.0f) - drjit::minimum(sin_t_sq, Float(1.0f));
        Float cos_t = drjit::sqrt(drjit::maximum(cos_t_sq, Float(0.0f)));
        
        // 折射方向：wt = eta * wi + (eta * cos_i - cos_t) * n
        Vector3 wt = eta * wi + (eta * cos_i - cos_t) * n;
        
        return {wt, cos_t, tir};
    }
    
    // 辅助函数：Fresnel 反射率（介电质）
    static Float fresnel_dielectric(const Float& cos_i, const Float& cos_t, const Float& eta) {
        // Fresnel 方程
        // r_s = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t)
        // r_p = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t)
        // 注意：eta = n_i / n_t
        
        Float eps = Float(1e-10f);
        
        // r_s = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
        Float r_s_num = eta * cos_i - cos_t;
        Float r_s_den = eta * cos_i + cos_t;
        Float r_s = r_s_num / drjit::maximum(drjit::abs(r_s_den), eps);
        
        // r_p = (cos_i - eta * cos_t) / (cos_i + eta * cos_t)
        Float r_p_num = cos_i - eta * cos_t;
        Float r_p_den = cos_i + eta * cos_t;
        Float r_p = r_p_num / drjit::maximum(drjit::abs(r_p_den), eps);
        
        Float Fr = Float(0.5f) * (r_s * r_s + r_p * r_p);
        
        return drjit::clamp(Fr, Float(0.0f), Float(1.0f));
    }
};

// ============= 镜面反射 =============
struct SpecularReflector : public BSDF {
    Float reflectance;  // 反射率
    
    SpecularReflector() : reflectance(from_scalar(1.0f)) {}
    
    explicit SpecularReflector(const Float& r) : reflectance(r) {}
    
    explicit SpecularReflector(ScalarType r) : reflectance(from_scalar(r)) {}
    
    std::string type_name() const override { return "SpecularReflector"; }
    
    BSDFSample sample(
        const Ray& ray,
        const SurfaceRecord& hit,
        const Vector2& uv,
        const Float& n_outer,
        const Float& n_inner
    ) const override {
        size_t n = ray.size();
        BSDFSample result(n);
        
        Vector3 wi = ray.direction;
        Vector3 normal = hit.normal;
        
        // 计算入射角余弦
        Float cos_i_signed = drjit::dot(wi, normal);
        
        // 调整法线方向使其面向光线（即与入射方向相对）
        Mask flip = cos_i_signed > Float(0.0f);
        Vector3 n_face = drjit::select(flip, -normal, normal);
        Float cos_i = drjit::abs(cos_i_signed);
        
        // 计算反射方向
        result.direction = compute_reflect(wi, n_face, cos_i);
        result.direction = drjit::normalize(result.direction);
        
        // 镜面反射的 PDF 是 delta 分布，数值上设为 1
        result.pdf = drjit::full<Float>(1.0f, n);
        
        // 权重 = reflectance
        result.weight = drjit::zeros<Float>(n) + reflectance;
        
        result.valid = hit.valid;
        
        return result;
    }
    
    Float eval(
        const Vector3& wi,
        const Vector3& wo,
        const SurfaceRecord& hit
    ) const override {
        // 镜面反射是 delta 分布，eval 返回 0
        return drjit::zeros<Float>(drjit::width(wi[0]));
    }
    
    Float pdf(
        const Vector3& wi,
        const Vector3& wo,
        const SurfaceRecord& hit
    ) const override {
        // delta 分布的 pdf 形式上无穷大，返回 0 表示无法直接采样到
        return drjit::zeros<Float>(drjit::width(wi[0]));
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Reflectance: " << to_scalar(reflectance) << std::endl;
    }
};

// ============= 镜面折射 =============
struct SpecularRefractor : public BSDF {
    Float transmittance;  // 透射率
    
    SpecularRefractor() : transmittance(from_scalar(1.0f)) {}
    
    explicit SpecularRefractor(const Float& t) : transmittance(t) {}
    
    explicit SpecularRefractor(ScalarType t) : transmittance(from_scalar(t)) {}
    
    std::string type_name() const override { return "SpecularRefractor"; }
    
    BSDFSample sample(
        const Ray& ray,
        const SurfaceRecord& hit,
        const Vector2& uv,
        const Float& n_outer,  // 法线正方向（外侧）的折射率
        const Float& n_inner   // 法线负方向（内侧）的折射率
    ) const override {
        size_t n = ray.size();
        BSDFSample result(n);
        
        Vector3 wi = ray.direction;
        Vector3 normal = hit.normal;
        
        // 计算入射角余弦（带符号）
        Float cos_i_signed = drjit::dot(wi, normal);
        
        // 判断光线从哪一侧入射
        // cos_i_signed < 0：从法线正方向入射（outer -> inner）
        // cos_i_signed > 0：从法线负方向入射（inner -> outer）
        Mask from_outside = cos_i_signed < Float(0.0f);
        
        // 调整法线方向使其面向光线
        Vector3 n_face = drjit::select(from_outside, normal, -normal);
        Float cos_i = drjit::abs(cos_i_signed);
        
        // 计算折射率比 eta = n_i / n_t
        // 从外到内：eta = n_outer / n_inner
        // 从内到外：eta = n_inner / n_outer
        Float eps = Float(1e-10f);
        Float eta = drjit::select(
            from_outside,
            n_outer / drjit::maximum(n_inner, eps),
            n_inner / drjit::maximum(n_outer, eps)
        );
        
        // 计算折射方向
        auto [wt, cos_t, tir] = compute_refract(wi, n_face, cos_i, eta);
        wt = drjit::normalize(wt);
        
        // 计算反射方向（用于全反射情况）
        Vector3 wr = compute_reflect(wi, n_face, cos_i);
        wr = drjit::normalize(wr);
        
        // 选择输出方向
        result.direction = drjit::select(tir, wr, wt);
        
        // PDF（delta 分布）
        result.pdf = drjit::full<Float>(1.0f, n);
        
        // 计算权重
        // 折射时需要 (n_t/n_i)² = 1/eta² 校正（光源追踪，radiance 传输）
        Float eta_sq = eta * eta;
        Float weight_refract = (drjit::zeros<Float>(n) + transmittance) / eta_sq;
        Float weight_tir = drjit::full<Float>(1.0f, n);  // 全反射保持能量
        
        result.weight = drjit::select(tir, weight_tir, weight_refract);
        result.valid = hit.valid;
        
        return result;
    }
    
    Float eval(
        const Vector3& wi,
        const Vector3& wo,
        const SurfaceRecord& hit
    ) const override {
        // delta 分布，eval 返回 0
        return drjit::zeros<Float>(drjit::width(wi[0]));
    }
    
    Float pdf(
        const Vector3& wi,
        const Vector3& wo,
        const SurfaceRecord& hit
    ) const override {
        // delta 分布，pdf 返回 0
        return drjit::zeros<Float>(drjit::width(wi[0]));
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Transmittance: " << to_scalar(transmittance) << std::endl;
    }
};

// ============= Fresnel 折射（考虑菲涅尔反射）=============
struct FresnelRefractor : public BSDF {
    Float transmittance;  // 基础透射率
    
    FresnelRefractor() : transmittance(from_scalar(1.0f)) {}
    
    explicit FresnelRefractor(const Float& t) : transmittance(t) {}
    
    explicit FresnelRefractor(ScalarType t) : transmittance(from_scalar(t)) {}
    
    std::string type_name() const override { return "FresnelRefractor"; }
    
    BSDFSample sample(
        const Ray& ray,
        const SurfaceRecord& hit,
        const Vector2& uv,
        const Float& n_outer,
        const Float& n_inner
    ) const override {
        size_t n = ray.size();
        BSDFSample result(n);
        
        Vector3 wi = ray.direction;
        Vector3 normal = hit.normal;
        
        Float cos_i_signed = drjit::dot(wi, normal);
        Mask from_outside = cos_i_signed < Float(0.0f);
        
        Vector3 n_face = drjit::select(from_outside, normal, -normal);
        Float cos_i = drjit::abs(cos_i_signed);
        
        Float eps = Float(1e-10f);
        Float eta = drjit::select(
            from_outside,
            n_outer / drjit::maximum(n_inner, eps),
            n_inner / drjit::maximum(n_outer, eps)
        );
        
        // 计算折射
        auto [wt, cos_t, tir] = compute_refract(wi, n_face, cos_i, eta);
        wt = drjit::normalize(wt);
        
        // 计算反射
        Vector3 wr = compute_reflect(wi, n_face, cos_i);
        wr = drjit::normalize(wr);
        
        // 计算 Fresnel 反射率
        Float Fr = fresnel_dielectric(cos_i, cos_t, eta);
        Fr = drjit::select(tir, Float(1.0f), Fr);  // 全反射时 Fr = 1
        
        // 使用随机数决定反射还是折射
        Float u_fresnel = uv[0];  // 使用第一个随机数
        Mask do_reflect = (u_fresnel < Fr) | tir;
        
        // 选择方向
        result.direction = drjit::select(do_reflect, wr, wt);
        
        // PDF
        result.pdf = drjit::select(do_reflect, Fr, Float(1.0f) - Fr);
        result.pdf = drjit::maximum(result.pdf, eps);
        
        // 权重
        // 反射：weight = Fr / pdf = Fr / Fr = 1
        // 折射：weight = (1-Fr) * transmittance / eta² / pdf = transmittance / eta²
        Float eta_sq = eta * eta;
        Float weight_reflect = drjit::full<Float>(1.0f, n);
        Float weight_refract = (drjit::zeros<Float>(n) + transmittance) / eta_sq;
        
        result.weight = drjit::select(do_reflect, weight_reflect, weight_refract);
        result.valid = hit.valid;
        
        return result;
    }
    
    Float eval(
        const Vector3& wi,
        const Vector3& wo,
        const SurfaceRecord& hit
    ) const override {
        return drjit::zeros<Float>(drjit::width(wi[0]));
    }
    
    Float pdf(
        const Vector3& wi,
        const Vector3& wo,
        const SurfaceRecord& hit
    ) const override {
        return drjit::zeros<Float>(drjit::width(wi[0]));
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Transmittance: " << to_scalar(transmittance) << std::endl;
    }
};

// ============= 吸收器（完全吸收）=============
struct Absorber : public BSDF {
    Absorber() = default;
    
    std::string type_name() const override { return "Absorber"; }
    
    BSDFSample sample(
        const Ray& ray,
        const SurfaceRecord& hit,
        const Vector2& uv,
        const Float& n_outer,
        const Float& n_inner
    ) const override {
        size_t n = ray.size();
        BSDFSample result(n);
        
        // 吸收器不产生有效的出射光线
        result.direction = ray.direction;
        result.weight = drjit::zeros<Float>(n);
        result.pdf = drjit::full<Float>(1.0f, n);
        result.valid = drjit::full<Mask>(false, n);
        
        return result;
    }
    
    Float eval(
        const Vector3& wi,
        const Vector3& wo,
        const SurfaceRecord& hit
    ) const override {
        return drjit::zeros<Float>(drjit::width(wi[0]));
    }
    
    Float pdf(
        const Vector3& wi,
        const Vector3& wo,
        const SurfaceRecord& hit
    ) const override {
        return drjit::zeros<Float>(drjit::width(wi[0]));
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Note: All incident light is absorbed" << std::endl;
    }
};

} // namespace diff_optics