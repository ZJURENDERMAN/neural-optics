#pragma once
#include "utils.hpp"

namespace diff_optics {

struct Emitter {
public:
    Float lower_angle;  // theta_min（弧度）
    Float upper_angle;  // theta_max（弧度）

    Emitter() : 
        lower_angle(from_scalar(0.0f)), 
        upper_angle(from_scalar(M_PI * 0.5f)) 
    {}

    Emitter(const Float& lower, const Float& upper) : 
        lower_angle(lower*M_PI/180), 
        upper_angle(upper*M_PI/180) 
    {}

    virtual ~Emitter() = default;

    // weight: 重要性采样权重
    // pdf: 采样的概率密度（立体角均匀分布）
    virtual Vector3 sampleDirection(const Vector2& uv, Float& weight, Float& pdf) const = 0;

protected:
    // 在立体角范围内均匀采样方向
    // 返回：方向向量、cos(theta)、pdf
    Vector3 sampleDirectionUniform(const Vector2& uv, Float& cos_theta, Float& pdf) const {
    Float u = uv[0];
    Float v = uv[1];
    
    size_t n_rays = drjit::width(u);
    
    Float lower_b = drjit::zeros<Float>(n_rays) + lower_angle;
    Float upper_b = drjit::zeros<Float>(n_rays) + upper_angle;
    
    Float cos_lower = drjit::cos(lower_b);
    Float cos_upper = drjit::cos(upper_b);
    
    // ============= 检测平行光（角度范围为0）=============
    Float angle_diff = drjit::abs(upper_b - lower_b);
    Float eps = from_scalar(1e-6f);
    Mask is_collimated = angle_diff < eps;
    
    // 正常采样
    cos_theta = cos_lower + u * (cos_upper - cos_lower);
    cos_theta = drjit::clamp(cos_theta, from_scalar(-1.0f), from_scalar(1.0f));
    
    Float sin_theta = drjit::sqrt(from_scalar(1.0f) - cos_theta * cos_theta);
    
    Float two_pi = from_scalar(2.0f * M_PI);
    Float phi = v * two_pi;
    
    Float sin_phi = drjit::sin(phi);
    Float cos_phi = drjit::cos(phi);
    
    Float x = sin_theta * cos_phi;
    Float y = sin_theta * sin_phi;
    Float z = cos_theta;
    
    // ============= PDF 计算（特殊处理平行光）=============
    Float solid_angle = two_pi * (cos_lower - cos_upper);
    solid_angle = drjit::maximum(drjit::abs(solid_angle), eps);
    
    // 平行光：pdf = 1（delta分布，归一化后权重为1）
    // 非平行光：pdf = 1 / solid_angle
    pdf = drjit::select(is_collimated, 
                        from_scalar(1.0f), 
                        from_scalar(1.0f) / solid_angle);
    
    // 平行光时强制方向为 (0, 0, 1)
    x = drjit::select(is_collimated, from_scalar(0.0f), x);
    y = drjit::select(is_collimated, from_scalar(0.0f), y);
    z = drjit::select(is_collimated, from_scalar(1.0f), z);
    
    return Vector3(x, y, z);
}
};

// 在立体角范围内均匀采样方向
struct UniformEmitter : public Emitter {
public:
    UniformEmitter() : Emitter() {}

    UniformEmitter(const Float& lower, const Float& upper) : 
        Emitter(lower, upper) 
    {}

    Vector3 sampleDirection(const Vector2& uv, Float& weight, Float& pdf) const override {
        Float cos_theta;
        Vector3 dir = sampleDirectionUniform(uv, cos_theta, pdf);
        
        // 均匀采样：weight = pdf（完美重要性采样）
        weight = pdf;
        
        return dir;
    }
};

// Lambert 分布采样（使用均匀立体角采样 + Lambert 权重）
struct LambertEmitter : public Emitter {
public:
    Float hwhm;  // 半宽半高 (Half Width at Half Maximum)

    LambertEmitter() : 
        Emitter(),
        hwhm(from_scalar(M_PI / 6.0f))  // 默认 30 度
    {}

    LambertEmitter(const Float& lower, const Float& upper, const Float& hwhm_) : 
        Emitter(lower, upper),
        hwhm(hwhm_)
    {}

    Vector3 sampleDirection(const Vector2& uv, Float& weight, Float& pdf) const override {
        Float cos_theta;
        Vector3 dir = sampleDirectionUniform(uv, cos_theta, pdf);
        
        size_t n_rays = drjit::width(uv[0]);
        Float hwhm_b = drjit::zeros<Float>(n_rays) + hwhm;
        
        // ============= Lambert 分布权重 =============
        // Lambert 分布：I(theta) = cos^n(theta)
        // 在 theta = hwhm 时，强度为 0.5
        // cos^n(hwhm) = 0.5 => n = log(0.5) / log(cos(hwhm))
        
        Float eps = from_scalar(1e-6f);
        Float cos_hwhm = drjit::cos(hwhm_b);
        cos_hwhm = drjit::clamp(cos_hwhm, eps, from_scalar(1.0f) - eps);
        
        Float log_half = from_scalar(std::log(0.5f));
        Float log_cos_hwhm = drjit::log(cos_hwhm);
        log_cos_hwhm = drjit::select(
            drjit::abs(log_cos_hwhm) < eps,
            -eps,
            log_cos_hwhm
        );
        
        Float n = log_half / log_cos_hwhm;
        n = drjit::maximum(n, from_scalar(0.0f));
        
        // weight = cos^n(theta)
        Float cos_theta_safe = drjit::maximum(cos_theta, eps);
        weight = drjit::exp(n * drjit::log(cos_theta_safe));
        weight = drjit::maximum(weight, from_scalar(0.0f));
        
        return dir;
    }
};

} // namespace diff_optics