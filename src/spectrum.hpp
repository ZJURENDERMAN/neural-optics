// spectrum.hpp
#pragma once
#include "utils.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace diff_optics {

// ============= 抽象基类 Spectrum =============
struct Spectrum {
    virtual ~Spectrum() = default;
    
    // 获取光谱类型名称
    virtual std::string type_name() const = 0;
    
    // 基于波长张量计算光谱值
    virtual Float eval(const Float& wavelength) const = 0;
    
    // 基于一维随机数张量采样波长
    // 返回：(采样的波长, 光谱值, pdf)
    virtual std::tuple<Float, Float, Float> sample(const Float& u) const = 0;
    
    // 打印信息
    virtual void print(const std::string& name = "") const = 0;
};

// ============= 离散光谱 =============
// 存储离散的 (波长, 值) 对，通过线性插值计算中间值
struct DiscreteSpectrum : public Spectrum {
    Float wavelengths;  // 波长数组（nm）
    Float values;       // 对应的光谱值
    Float cdf;          // 累积分布函数，用于采样
    
    DiscreteSpectrum(const Float& wl, const Float& val) 
        : wavelengths(wl), values(val) {
        size_t n = drjit::width(wl);
        if (n != drjit::width(val) || n == 0) {
            throw std::invalid_argument("DiscreteSpectrum: wavelengths and values must have same non-zero size");
        }
        build_cdf();
    }
    
    // 从 std::vector 构造（便捷接口）
    DiscreteSpectrum(const std::vector<ScalarType>& wl, const std::vector<ScalarType>& val) 
        : DiscreteSpectrum(
            drjit::load<Float>(wl.data(), wl.size()),
            drjit::load<Float>(val.data(), val.size())
        ) {}

    std::string type_name() const override { return "DiscreteSpectrum"; }
    
    size_t num_samples() const { return drjit::width(wavelengths); }
    
    Float wl_min() const { return drjit::gather<Float>(wavelengths, Int32(0)); }
    
    Float wl_max() const { return drjit::gather<Float>(wavelengths, Int32(static_cast<int>(num_samples() - 1))); }
    
    Float total_integral() const { return drjit::gather<Float>(cdf, Int32(static_cast<int>(num_samples() - 1))); }
    
    Float eval(const Float& wavelength) const override {
        size_t n = drjit::width(wavelength);
        size_t ns = num_samples();
        
        if (ns == 0) {
            return drjit::zeros<Float>(n);
        }
        
        // 单波长情况：只有精确匹配才返回值（或者返回常量值）
        if (ns == 1) {
            Float val = drjit::zeros<Float>(n) + drjit::gather<Float>(values, Int32(0));
            Float wl0 = drjit::zeros<Float>(n) + drjit::gather<Float>(wavelengths, Int32(0));
            // 对于 delta 分布，可以选择：精确匹配返回值，否则返回 0
            // 或者简化处理：总是返回该值
            Mask match = drjit::abs(wavelength - wl0) < Float(0.5f);  // 0.5nm 容差
            return drjit::select(match, val, drjit::zeros<Float>(n));
        }
        
        // 广播波长范围
        Float wl_min_b = drjit::zeros<Float>(n) + wl_min();
        Float wl_max_b = drjit::zeros<Float>(n) + wl_max();
        
        // 归一化到 [0, 1] 范围
        Float t = (wavelength - wl_min_b) / (wl_max_b - wl_min_b);
        t = drjit::clamp(t, Float(0.0f), Float(1.0f));
        
        // 计算索引
        Float idx_f = t * Float(static_cast<float>(ns - 1));
        Int32 idx_low = Int32(drjit::floor(idx_f));
        idx_low = drjit::clamp(idx_low, Int32(0), Int32(static_cast<int>(ns - 2)));
        Int32 idx_high = idx_low + Int32(1);
        
        // 线性插值权重
        Float frac = idx_f - Float(idx_low);
        
        // 从数据中 gather
        Float val_low = drjit::gather<Float>(values, idx_low);
        Float val_high = drjit::gather<Float>(values, idx_high);
        
        return val_low * (Float(1.0f) - frac) + val_high * frac;
    }
    
    std::tuple<Float, Float, Float> sample(const Float& u) const override {
    size_t n = drjit::width(u);
    size_t ns = num_samples();
    
    // 特殊情况：单波长（delta 分布）
    if (ns == 1) {
        Float wl = drjit::zeros<Float>(n) + drjit::gather<Float>(wavelengths, Int32(0));
        Float val = drjit::zeros<Float>(n) + drjit::gather<Float>(values, Int32(0));
        Float pdf = drjit::full<Float>(1.0f, n);  // delta 分布的 PDF 概念上是无穷大，但返回 1 表示确定性采样
        return {wl, val, pdf};
    }
    
    Float total = total_integral();
    
    // 使用 CDF 进行逆采样
    Float u_scaled = drjit::clamp(u, Float(0.0f), Float(1.0f - 1e-6f)) * total;
    
    // 二分查找 CDF
    Float idx_f = drjit::zeros<Float>(n);
    for (size_t i = 0; i < ns - 1; ++i) {
        Float cdf_i = drjit::gather<Float>(cdf, Int32(static_cast<int>(i)));
        Float cdf_i1 = drjit::gather<Float>(cdf, Int32(static_cast<int>(i + 1)));
        Mask in_range = (u_scaled >= cdf_i) & (u_scaled < cdf_i1);
        
        // 在区间内线性插值
        Float t = (u_scaled - cdf_i) / drjit::maximum(cdf_i1 - cdf_i, Float(1e-10f));
        Float idx_interp = Float(static_cast<float>(i)) + t;
        idx_f = drjit::select(in_range, idx_interp, idx_f);
    }
    
    // 广播波长范围
    Float wl_min_b = drjit::zeros<Float>(n) + wl_min();
    Float wl_max_b = drjit::zeros<Float>(n) + wl_max();
    
    // 计算采样波长
    Float t_norm = idx_f / Float(static_cast<float>(ns - 1));
    Float sampled_wavelength = wl_min_b + t_norm * (wl_max_b - wl_min_b);
    
    // 计算光谱值和 PDF
    Float value = eval(sampled_wavelength);
    Float total_b = drjit::zeros<Float>(n) + total;
    Float pdf = value / drjit::maximum(total_b, Float(1e-10f));
    
    return {sampled_wavelength, value, pdf};
}
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Samples: " << num_samples() << std::endl;
        if (num_samples() > 0) {
            std::cout << "  Range: [" << to_scalar(wl_min()) << ", " << to_scalar(wl_max()) << "] nm" << std::endl;
        }
    }
    
private:
    void build_cdf() {
        size_t ns = num_samples();
        
        // 使用 CPU 计算 CDF（需要顺序累加）
        std::vector<ScalarType> wl_host(ns), val_host(ns), cdf_host(ns);
        drjit::store(wl_host.data(), wavelengths);
        drjit::store(val_host.data(), values);
        
        cdf_host[0] = 0.0f;
        for (size_t i = 1; i < ns; ++i) {
            ScalarType avg_val = 0.5f * (val_host[i-1] + val_host[i]);
            ScalarType dw = wl_host[i] - wl_host[i-1];
            cdf_host[i] = cdf_host[i-1] + avg_val * dw;
        }
        
        cdf = drjit::load<Float>(cdf_host.data(), ns);
    }
};

// ============= 黑体辐射光谱 =============
struct BlackbodySpectrum : public Spectrum {
    Float temperature;  // 温度（K）
    Float wl_min;       // 可见光范围（nm）
    Float wl_max;
    
    BlackbodySpectrum() 
        : temperature(Float(6500.0f)), wl_min(Float(380.0f)), wl_max(Float(780.0f)) {}
    
    explicit BlackbodySpectrum(const Float& temp) 
        : temperature(temp), wl_min(Float(380.0f)), wl_max(Float(780.0f)) {}
    
    BlackbodySpectrum(const Float& temp, const Float& min_wl, const Float& max_wl) 
        : temperature(temp), wl_min(min_wl), wl_max(max_wl) {}
    
    std::string type_name() const override { return "BlackbodySpectrum"; }
    
    // 普朗克黑体辐射公式
    // B(λ, T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
    Float eval(const Float& wavelength) const override {
        // 物理常数
        constexpr ScalarType h = 6.62607015e-34f;   // 普朗克常数 (J·s)
        constexpr ScalarType c = 2.99792458e8f;     // 光速 (m/s)
        constexpr ScalarType k = 1.380649e-23f;     // 玻尔兹曼常数 (J/K)
        
        // 波长转换为米
        Float lambda_m = wavelength * Float(1e-9f);
        
        // 广播温度
        size_t n = drjit::width(wavelength);
        Float T = drjit::zeros<Float>(n) + temperature;
        
        // 计算 hc / (λkT)
        Float hc_lkt = Float(h * c) / (lambda_m * Float(k) * T);
        
        // 计算分子 2hc²/λ⁵
        Float lambda5 = lambda_m * lambda_m * lambda_m * lambda_m * lambda_m;
        Float numerator = Float(2.0f * h * c * c) / lambda5;
        
        // 计算分母 exp(hc/λkT) - 1
        // 使用安全的 exp 计算，避免溢出
        Float exp_term = drjit::exp(drjit::minimum(hc_lkt, Float(80.0f)));
        Float denominator = drjit::maximum(exp_term - Float(1.0f), Float(1e-30f));
        
        return numerator / denominator;
    }
    
    std::tuple<Float, Float, Float> sample(const Float& u) const override {
        size_t n = drjit::width(u);
        
        // 广播波长范围
        Float wl_min_b = drjit::zeros<Float>(n) + wl_min;
        Float wl_max_b = drjit::zeros<Float>(n) + wl_max;
        
        // 均匀采样波长
        Float sampled_wavelength = wl_min_b + u * (wl_max_b - wl_min_b);
        
        // 计算光谱值
        Float value = eval(sampled_wavelength);
        
        // 均匀采样的 PDF = 1 / (wl_max - wl_min)
        Float pdf = Float(1.0f) / (wl_max_b - wl_min_b);
        
        return {sampled_wavelength, value, pdf};
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Temperature: " << to_scalar(temperature) << " K" << std::endl;
        std::cout << "  Range: [" << to_scalar(wl_min) << ", " << to_scalar(wl_max) << "] nm" << std::endl;
    }
};

// ============= 高斯光谱 =============
struct GaussianSpectrum : public Spectrum {
    Float center;     // 中心波长（nm）
    Float sigma;      // 标准差（nm）
    Float amplitude;  // 振幅
    
    GaussianSpectrum() 
        : center(Float(550.0f)), 
          sigma(Float(30.0f)), 
          amplitude(Float(1.0f)) {}
    
    GaussianSpectrum(const Float& c, const Float& s, const Float& a = Float(1.0f))
        : center(c), sigma(s), amplitude(a) {}
    
    std::string type_name() const override { return "GaussianSpectrum"; }
    
    // 高斯分布：A * exp(-0.5 * ((λ - μ) / σ)²)
    Float eval(const Float& wavelength) const override {
        size_t n = drjit::width(wavelength);
        
        Float c = drjit::zeros<Float>(n) + center;
        Float s = drjit::zeros<Float>(n) + sigma;
        Float a = drjit::zeros<Float>(n) + amplitude;
        
        Float diff = wavelength - c;
        Float exponent = Float(-0.5f) * (diff * diff) / (s * s);
        
        return a * drjit::exp(exponent);
    }
    
    std::tuple<Float, Float, Float> sample(const Float& u) const override {
        size_t n = drjit::width(u);
        
        Float c = drjit::zeros<Float>(n) + center;
        Float s = drjit::zeros<Float>(n) + sigma;
        
        // 在 ±3σ 范围内均匀采样
        Float wl_min = c - Float(3.0f) * s;
        Float wl_max = c + Float(3.0f) * s;
        
        Float sampled_wavelength = wl_min + u * (wl_max - wl_min);
        
        // 计算光谱值
        Float value = eval(sampled_wavelength);
        
        // 均匀采样的 PDF
        Float pdf = Float(1.0f) / (wl_max - wl_min);
        
        return {sampled_wavelength, value, pdf};
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Center: " << to_scalar(center) << " nm" << std::endl;
        std::cout << "  Sigma: " << to_scalar(sigma) << " nm" << std::endl;
        std::cout << "  Amplitude: " << to_scalar(amplitude) << std::endl;
    }
};

// ============= 常量光谱 =============
struct ConstantSpectrum : public Spectrum {
    Float value;
    Float wl_min;
    Float wl_max;
    
    ConstantSpectrum() 
        : value(Float(1.0f)), wl_min(Float(380.0f)), wl_max(Float(780.0f)) {}
    
    explicit ConstantSpectrum(const Float& v) 
        : value(v), wl_min(Float(380.0f)), wl_max(Float(780.0f)) {}
    
    ConstantSpectrum(const Float& v, const Float& min_wl, const Float& max_wl)
        : value(v), wl_min(min_wl), wl_max(max_wl) {}
    
    std::string type_name() const override { return "ConstantSpectrum"; }
    
    Float eval(const Float& wavelength) const override {
        size_t n = drjit::width(wavelength);
        return drjit::zeros<Float>(n) + value;
    }
    
    std::tuple<Float, Float, Float> sample(const Float& u) const override {
        size_t n = drjit::width(u);
        
        // 广播波长范围
        Float wl_min_b = drjit::zeros<Float>(n) + wl_min;
        Float wl_max_b = drjit::zeros<Float>(n) + wl_max;
        
        // 均匀采样波长
        Float sampled_wavelength = wl_min_b + u * (wl_max_b - wl_min_b);
        
        // 光谱值（常量）
        Float spec_value = drjit::zeros<Float>(n) + value;
        
        // 均匀分布的 PDF = 1 / (wl_max - wl_min)
        Float pdf = Float(1.0f) / (wl_max_b - wl_min_b);
        
        return {sampled_wavelength, spec_value, pdf};
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Value: " << to_scalar(value) << std::endl;
        std::cout << "  Range: [" << to_scalar(wl_min) << ", " << to_scalar(wl_max) << "] nm" << std::endl;
    }
};

} // namespace diff_optics