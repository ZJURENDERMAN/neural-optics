// material.hpp
#pragma once
#include "utils.hpp"
#include "spectrum.hpp"
#include <memory>
#include <cmath>

namespace diff_optics {

// // ============= 辅助函数：从 std::vector 创建 Float =============
// inline Float make_float_array(const std::vector<ScalarType>& data) {
//     return drjit::load<Float>(data.data(), data.size());
// }

// ============= 抽象基类 VolumeMaterial =============
struct VolumeMaterial {
    virtual ~VolumeMaterial() = default;
    
    // 获取材料类型名称
    virtual std::string type_name() const = 0;
    
    // 计算体透射率（给定波长和传播距离）
    // transmittance = exp(-absorption_coefficient * distance)
    // 这里存储的是给定测量深度 d 的透射率
    virtual Float transmittance(const Float& wavelength, const Float& distance) const = 0;
    
    // 计算折射率（给定波长）
    virtual Float ior(const Float& wavelength) const = 0;
    
    // 获取测量深度（用于透射率计算）
    virtual ScalarType get_measurement_depth() const = 0;
    
    // 打印信息
    virtual void print(const std::string& name = "") const = 0;
};

// ============= 离散光谱材料 =============
// 使用离散光谱数据定义透射率和折射率
struct DiscreteVolumeMaterial : public VolumeMaterial {
    std::shared_ptr<DiscreteSpectrum> transmittance_spectrum;  // 透射率光谱
    std::shared_ptr<DiscreteSpectrum> ior_spectrum;            // 折射率光谱
    ScalarType measurement_depth;  // 测量深度（mm）
    
    DiscreteVolumeMaterial() : measurement_depth(10.0f) {
        // 默认：完全透明，折射率 1.0
        transmittance_spectrum = std::make_shared<DiscreteSpectrum>(std::vector<ScalarType>{380.0f, 780.0f}, std::vector<ScalarType>{1.0f, 1.0f});
        ior_spectrum = std::make_shared<DiscreteSpectrum>(std::vector<ScalarType>{380.0f, 780.0f}, std::vector<ScalarType>{1.0f, 1.0f});
    }
    
    DiscreteVolumeMaterial(
        std::shared_ptr<DiscreteSpectrum> trans,
        std::shared_ptr<DiscreteSpectrum> ior,
        ScalarType depth
    ) : transmittance_spectrum(trans), ior_spectrum(ior), measurement_depth(depth) {}
    
    std::string type_name() const override { return "DiscreteVolumeMaterial"; }
    
    ScalarType get_measurement_depth() const override { return measurement_depth; }
    
    Float transmittance(const Float& wavelength, const Float& distance) const override {
        // 获取测量深度下的透射率
        Float T_d = transmittance_spectrum->eval(wavelength);
        
        // 计算吸收系数：alpha = -ln(T_d) / d
        Float eps = from_scalar(1e-10f);
        Float T_safe = drjit::maximum(T_d, eps);
        Float alpha = -drjit::log(T_safe) / from_scalar(measurement_depth);
        
        // 计算给定距离的透射率：T = exp(-alpha * distance)
        return drjit::exp(-alpha * distance);
    }
    
    Float ior(const Float& wavelength) const override {
        return ior_spectrum->eval(wavelength);
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Measurement Depth: " << measurement_depth << " mm" << std::endl;
        std::cout << transmittance_spectrum << std::endl;
        std::cout << ior_spectrum << std::endl;
    }
};

// ============= Sellmeier 色散模型 =============
// n²(λ) - 1 = Σ (B_i * λ²) / (λ² - C_i)
// 其中 λ 单位为 μm
struct SellmeierMaterial : public VolumeMaterial {
    // Sellmeier 系数
    std::vector<ScalarType> B;  // B1, B2, B3, ...
    std::vector<ScalarType> C;  // C1, C2, C3, ... (μm²)
    
    // 透射率光谱
    std::shared_ptr<DiscreteSpectrum> transmittance_spectrum;
    ScalarType measurement_depth;
    
    SellmeierMaterial() : measurement_depth(10.0f) {
        // 默认：空气近似
        B = {0.0f};
        C = {0.0f};
        transmittance_spectrum = std::make_shared<DiscreteSpectrum>(
            std::vector<ScalarType>{380.0f, 780.0f}, std::vector<ScalarType>{1.0f, 1.0f}
        );
    }
    
    SellmeierMaterial(
        const std::vector<ScalarType>& b_coeffs,
        const std::vector<ScalarType>& c_coeffs,
        std::shared_ptr<DiscreteSpectrum> trans = nullptr,
        ScalarType depth = 10.0f
    ) : B(b_coeffs), C(c_coeffs), measurement_depth(depth) {
        if (trans) {
            transmittance_spectrum = trans;
        } else {
            transmittance_spectrum = std::make_shared<DiscreteSpectrum>(
                std::vector<ScalarType>{380.0f, 780.0f}, std::vector<ScalarType>{1.0f, 1.0f}
            );
        }
    }
    
    std::string type_name() const override { return "SellmeierMaterial"; }
    
    ScalarType get_measurement_depth() const override { return measurement_depth; }
    
    Float transmittance(const Float& wavelength, const Float& distance) const override {
        Float T_d = transmittance_spectrum->eval(wavelength);
        Float eps = from_scalar(1e-10f);
        Float T_safe = drjit::maximum(T_d, eps);
        Float alpha = -drjit::log(T_safe) / from_scalar(measurement_depth);
        return drjit::exp(-alpha * distance);
    }
    
    Float ior(const Float& wavelength) const override {
        size_t n = drjit::width(wavelength);
        
        // 波长转换为 μm
        Float lambda_um = wavelength * Float(1e-3f);
        Float lambda_sq = lambda_um * lambda_um;
        
        // 计算 n² - 1 = Σ (B_i * λ²) / (λ² - C_i)
        Float n_sq_minus_1 = drjit::zeros<Float>(n);
        
        for (size_t i = 0; i < B.size() && i < C.size(); ++i) {
            Float Bi = from_scalar(B[i]);
            Float Ci = from_scalar(C[i]);
            Float term = (Bi * lambda_sq) / (lambda_sq - Ci);
            n_sq_minus_1 = n_sq_minus_1 + term;
        }
        
        // n = sqrt(1 + n² - 1)
        Float n_sq = Float(1.0f) + n_sq_minus_1;
        n_sq = drjit::maximum(n_sq, Float(1.0f));  // 确保 n >= 1
        
        return drjit::sqrt(n_sq);
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Sellmeier Terms: " << B.size() << std::endl;
        for (size_t i = 0; i < B.size(); ++i) {
            std::cout << "    B" << (i+1) << " = " << B[i] << ", C" << (i+1) << " = " << C[i] << std::endl;
        }
        std::cout << "  Measurement Depth: " << measurement_depth << " mm" << std::endl;
    }
};

// ============= 预定义材料 =============

// 空气（标准大气压，20°C）
inline std::shared_ptr<SellmeierMaterial> create_air() {
    // 简化模型：n ≈ 1.000293（实际有轻微色散）
    auto material = std::make_shared<SellmeierMaterial>(
        std::vector<ScalarType>{0.05792105f, 0.00167917f},
        std::vector<ScalarType>{0.00467914f, 0.0135120f}  // μm²
    );
    return material;
}

// N-BK7 光学玻璃
inline std::shared_ptr<SellmeierMaterial> create_nbk7() {
    auto trans = std::make_shared<DiscreteSpectrum>(
        std::vector<ScalarType>{380.0f, 400.0f, 450.0f, 500.0f, 550.0f, 600.0f, 650.0f, 700.0f, 750.0f, 780.0f}
    , std::vector<ScalarType>{0.92f, 0.96f, 0.995f, 0.997f, 0.998f, 0.998f, 0.998f, 0.998f, 0.997f, 0.996f}
    );
    
    auto material = std::make_shared<SellmeierMaterial>(
        std::vector<ScalarType>{1.03961212f, 0.231792344f, 1.01046945f},
        std::vector<ScalarType>{0.00600069867f, 0.0200179144f, 103.560653f},  // μm²
        trans,
        10.0f  // 10mm 测量深度
    );
    return material;
}

// PMMA（亚克力/有机玻璃）
inline std::shared_ptr<SellmeierMaterial> create_pmma() {
    auto trans = std::make_shared<DiscreteSpectrum>(
        std::vector<ScalarType>{380.0f, 400.0f, 450.0f, 500.0f, 550.0f, 600.0f, 650.0f, 700.0f, 750.0f, 780.0f}
        , std::vector<ScalarType>{0.85f, 0.90f, 0.92f, 0.92f, 0.92f, 0.92f, 0.92f, 0.92f, 0.91f, 0.90f}
    );
    
    // PMMA Sellmeier 系数
    auto material = std::make_shared<SellmeierMaterial>(
        std::vector<ScalarType>{0.99654f, 0.18964f, 0.00411f},
        std::vector<ScalarType>{0.00787f, 0.02191f, 3.85727f},  // μm²
        trans,
        3.0f  // 3mm 测量深度
    );
    return material;
}

// 真空（理想介质）
inline std::shared_ptr<VolumeMaterial> create_vacuum() {
    auto material = std::make_shared<DiscreteVolumeMaterial>();
    return material;
}

// 常量折射率材料
struct ConstantIORMaterial : public VolumeMaterial {
    Float n;  // 折射率
    Float transmittance_value;  // 透射率
    ScalarType measurement_depth;
    
    ConstantIORMaterial(ScalarType ior = 1.5f, ScalarType trans = 1.0f, ScalarType depth = 10.0f)
        : n(from_scalar(ior)), transmittance_value(from_scalar(trans)), measurement_depth(depth) {}
    
    std::string type_name() const override { return "ConstantIORMaterial"; }
    
    ScalarType get_measurement_depth() const override { return measurement_depth; }
    
    Float transmittance(const Float& wavelength, const Float& distance) const override {
        size_t num = drjit::width(wavelength);
        Float T_d = drjit::zeros<Float>(num) + transmittance_value;
        Float eps = from_scalar(1e-10f);
        Float T_safe = drjit::maximum(T_d, eps);
        Float alpha = -drjit::log(T_safe) / from_scalar(measurement_depth);
        return drjit::exp(-alpha * distance);
    }
    
    Float ior(const Float& wavelength) const override {
        size_t num = drjit::width(wavelength);
        return drjit::zeros<Float>(num) + n;
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  IOR: " << to_scalar(n) << std::endl;
        std::cout << "  Transmittance: " << to_scalar(transmittance_value) << std::endl;
    }
};

} // namespace diff_optics