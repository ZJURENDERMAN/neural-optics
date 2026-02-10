// shape_polynomial.hpp
#pragma once
#include "shape_base.hpp"

namespace diff_optics {

// ============= XY 多项式曲面配置 =============
struct XYPolynomialConfig {
    int order = 4;
    ScalarType b = 0.0f;
    std::vector<ScalarType> ai;
    
    XYPolynomialConfig() = default;
    
    XYPolynomialConfig(int J, ScalarType b_coeff = 0.0f)
        : order(J), b(b_coeff) {}
    
    XYPolynomialConfig(int J, ScalarType b_coeff, const std::vector<ScalarType>& coeffs)
        : order(J), b(b_coeff), ai(coeffs) {}
    
    static int compute_num_coeffs(int J) {
        return (J + 1) * (J + 2) / 2;
    }
};

// ============= XY 多项式曲面 =============
struct XYPolynomialSurface : public Shape {
    int order;
    int num_coeffs;
    ScalarType width, height;
    
    Float ai;   // 多项式系数（可微分）
    Float b;    // h(z) 系数（可微分）
    
    std::vector<ScalarType> ai_cpu;
    ScalarType b_cpu;
    
    XYPolynomialSurface() 
        : width(1.0f), height(1.0f), order(4) {
        XYPolynomialConfig config(order);
        init_with_config(config);
    }
    
    XYPolynomialSurface(ScalarType w, ScalarType h, const XYPolynomialConfig& config = XYPolynomialConfig())
        : width(w), height(h), order(config.order) {
        init_with_config(config);
    }
    
    std::string type_name() const override { return "XYPolynomialSurface"; }
    
    // ============= 统一参数接口实现 =============
    
    int get_diff_param_count() const override {
        return num_coeffs;
    }
    
    Float& get_diff_params() override { return ai; }
    const Float& get_diff_params() const override { return ai; }
    
    void set_diff_params(const Float& params) override {
        ai = params;
        sync_to_cpu();
    }
    
    std::vector<ScalarType> get_diff_params_cpu() const override {
        std::vector<ScalarType> result(num_coeffs);
        FloatC ai_c = utils::detach(ai);
        drjit::store(result.data(), ai_c);
        return result;
    }
    
    void set_diff_params_cpu(const std::vector<ScalarType>& params) override {
        if (params.size() != static_cast<size_t>(num_coeffs)) {
            throw std::runtime_error("XYPolynomialSurface: param size mismatch");
        }
        ai = drjit::load<Float>(params.data(), num_coeffs);
        ai_cpu = params;
    }
    
    std::string get_param_config_string() const override {
        return "xy_poly(order=" + std::to_string(order) + ",coeffs=" + std::to_string(num_coeffs) + ")";
    }
    
    std::map<std::string, int> get_param_config() const override {
        return {
            {"order", order},
            {"num_coeffs", num_coeffs}
        };
    }
    
    bool resize_params(const std::map<std::string, int>& new_config) override {
        auto it = new_config.find("order");
        if (it == new_config.end()) {
            return false;
        }
        
        int new_order = it->second;
        if (new_order == order) {
            return false;
        }
        
        resize_order(new_order);
        return true;
    }
    
    bool save_cad(const std::string& filename) const override {
        std::cerr << "[XYPolynomialSurface] CAD export not implemented yet." << std::endl;
        return false;
    }
    
    // ============= 旧接口（保持兼容性）=============
    
    int get_num_coeffs() const { return num_coeffs; }
    int get_order() const { return order; }
    
    Float& get_coefficients() { return ai; }
    const Float& get_coefficients() const { return ai; }
    
    Float& get_b() { return b; }
    const Float& get_b() const { return b; }
    
    void set_coefficients(const Float& coeffs) { 
        ai = coeffs; 
        sync_to_cpu();
    }
    
    void set_b(const Float& b_val) { 
        b = b_val; 
        sync_to_cpu();
    }
    
    Vector3C compute_position_c(const Vector2C& uv) const override {
        FloatC x = uv[0];
        FloatC y = uv[1];
        FloatC z = eval_surface_nondiff(x, y);
        return Vector3C(x, y, z);
    }
    
    Vector3 compute_normal(const Vector2& uv) const override {
        Float x = uv[0];
        Float y = uv[1];
        return eval_normal(x, y);
    }

public:
    std::tuple<int, ScalarType> get_config() const {
        return {order, b_cpu};
    }
    
    void resize_order(int new_order) {
        if (new_order == order) {
            return;
        }
        
        std::vector<ScalarType> old_ai = get_coefficients_as_array();
        int old_order = order;
        int old_num_coeffs = num_coeffs;
        
        order = new_order;
        num_coeffs = XYPolynomialConfig::compute_num_coeffs(order);
        
        std::vector<ScalarType> new_ai(num_coeffs, 0.0f);
        
        int old_idx = 0;
        int new_idx = 0;
        
        for (int j = 0; j <= std::min(old_order, new_order); ++j) {
            for (int i = 0; i <= j; ++i) {
                if (old_idx < old_num_coeffs && new_idx < num_coeffs) {
                    new_ai[new_idx] = old_ai[old_idx];
                }
                old_idx++;
                new_idx++;
            }
        }
        
        ai = drjit::load<Float>(new_ai.data(), num_coeffs);
        ai_cpu = new_ai;
    }
    
    void set_coefficients_from_array(const std::vector<ScalarType>& coeffs) {
        if (coeffs.size() != static_cast<size_t>(num_coeffs)) {
            throw std::runtime_error("coefficients size mismatch");
        }
        ai = drjit::load<Float>(coeffs.data(), num_coeffs);
        ai_cpu = coeffs;
    }
    
    std::vector<ScalarType> get_coefficients_as_array() const {
        std::vector<ScalarType> result(num_coeffs);
        FloatC ai_c = utils::detach(ai);
        drjit::store(result.data(), ai_c);
        return result;
    }
    
    ScalarType get_b_as_scalar() const {
        FloatC b_c = utils::detach(b);
        return b_c[0];
    }
    
    void set_b_from_scalar(ScalarType b_val) {
        b = drjit::full<Float>(b_val, 1);
        b_cpu = b_val;
    }
    
    void reverse() {
        ai = -ai;
        b = -b;
        sync_to_cpu();
    }

private:
    FloatC eval_g_nondiff(const FloatC& x, const FloatC& y) const {
        size_t N = drjit::width(x);
        FloatC c = drjit::zeros<FloatC>(N);
        
        FloatC ai_nondiff = utils::detach(ai);
        
        int idx = 0;
        for (int j = 0; j <= order; ++j) {
            for (int i = 0; i <= j; ++i) {
                FloatC coeff = drjit::gather<FloatC>(ai_nondiff, 
                    drjit::full<UInt32C>(static_cast<uint32_t>(idx), N));
                
                FloatC term = coeff * power(x, i) * power(y, j - i);
                c = c + term;
                
                idx++;
            }
        }
        
        return c;
    }
    
    Float eval_g(const Float& x, const Float& y) const {
        size_t N = drjit::width(x);
        Float c = drjit::zeros<Float>(N);
        
        int idx = 0;
        for (int j = 0; j <= order; ++j) {
            for (int i = 0; i <= j; ++i) {
                Float coeff = drjit::gather<Float>(ai, 
                    drjit::full<UInt32>(static_cast<uint32_t>(idx), N));
                
                Float term = coeff * power_diff(x, i) * power_diff(y, j - i);
                c = c + term;
                
                idx++;
            }
        }
        
        return c;
    }
    
    void eval_g_derivatives(const Float& x, const Float& y,
                            Float& dgdx, Float& dgdy) const {
        size_t N = drjit::width(x);
        dgdx = drjit::zeros<Float>(N);
        dgdy = drjit::zeros<Float>(N);
        
        int idx = 0;
        for (int j = 0; j <= order; ++j) {
            for (int i = 0; i <= j; ++i) {
                Float coeff = drjit::gather<Float>(ai, 
                    drjit::full<UInt32>(static_cast<uint32_t>(idx), N));
                
                if (j > 0) {
                    if (i > 0) {
                        Float term_x = coeff * Float(static_cast<float>(i)) 
                                       * power_diff(x, i - 1) * power_diff(y, j - i);
                        dgdx = dgdx + term_x;
                    }
                    
                    if (j - i > 0) {
                        Float term_y = coeff * Float(static_cast<float>(j - i)) 
                                       * power_diff(x, i) * power_diff(y, j - i - 1);
                        dgdy = dgdy + term_y;
                    }
                }
                
                idx++;
            }
        }
    }
    
    Float eval_h(const Float& z) const {
        size_t N = drjit::width(z);
        Float b_val = drjit::gather<Float>(b, drjit::zeros<UInt32>(N));
        return b_val * z * z - z;
    }
    
    Float eval_dhd(const Float& z) const {
        size_t N = drjit::width(z);
        Float b_val = drjit::gather<Float>(b, drjit::zeros<UInt32>(N));
        return Float(2.0f) * b_val * z - Float(1.0f);
    }
    
    FloatC solve_for_z_nondiff(const FloatC& c) const {
        FloatC b_nondiff = utils::detach(b);
        ScalarType b_val = b_nondiff[0];
        
        if (std::abs(b_val) < 1e-10f) {
            return c;
        } else {
            FloatC discriminant = FloatC(1.0f) - FloatC(4.0f) * b_val * c;
            discriminant = drjit::maximum(discriminant, FloatC(0.0f));
            return (FloatC(1.0f) - drjit::sqrt(discriminant)) / (FloatC(2.0f) * b_val);
        }
    }
    
    Float solve_for_z(const Float& c) const {
        size_t N = drjit::width(c);
        Float b_val = drjit::gather<Float>(b, drjit::zeros<UInt32>(N));
        
        MaskC b_is_zero = drjit::abs(utils::detach(b_val)) < FloatC(1e-10f);
        
        Float discriminant = Float(1.0f) - Float(4.0f) * b_val * c;
        discriminant = drjit::maximum(discriminant, Float(1e-10f));
        
        Float z_nonzero_b = (Float(1.0f) - drjit::sqrt(discriminant)) / 
                           (Float(2.0f) * b_val + Float(1e-10f));
        
        return drjit::select(Mask(b_is_zero), c, z_nonzero_b);
    }
    
    FloatC eval_surface_nondiff(const FloatC& x, const FloatC& y) const {
        FloatC c = eval_g_nondiff(x, y);
        return solve_for_z_nondiff(c);
    }
    
    Vector3 eval_normal(const Float& x, const Float& y) const {
        Float c = eval_g(x, y);
        Float dgdx, dgdy;
        eval_g_derivatives(x, y, dgdx, dgdy);
        
        Float z = solve_for_z(c);
        Float dhdz = eval_dhd(z);
        
        Vector3 normal(-dgdx, -dgdy, -dhdz);
        
        Float nz_sign = drjit::sign(normal[2]);
        normal = normal * Vector3(nz_sign, nz_sign, nz_sign);
        
        return drjit::normalize(normal);
    }
    
    static FloatC power(const FloatC& x, int n) {
        if (n == 0) return drjit::full<FloatC>(1.0f, drjit::width(x));
        if (n == 1) return x;
        if (n == 2) return x * x;
        
        FloatC result = drjit::full<FloatC>(1.0f, drjit::width(x));
        FloatC base = x;
        int exp = n;
        
        while (exp > 0) {
            if (exp & 1) {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }
        
        return result;
    }
    
    static Float power_diff(const Float& x, int n) {
        if (n == 0) return drjit::full<Float>(1.0f, drjit::width(x));
        if (n == 1) return x;
        if (n == 2) return x * x;
        
        Float result = drjit::full<Float>(1.0f, drjit::width(x));
        Float base = x;
        int exp = n;
        
        while (exp > 0) {
            if (exp & 1) {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }
        
        return result;
    }
    
    void init_with_config(const XYPolynomialConfig& config) {
        num_coeffs = XYPolynomialConfig::compute_num_coeffs(order);
        
        if (config.ai.empty()) {
            ai_cpu.resize(num_coeffs, 0.0f);
        } else {
            if (config.ai.size() != static_cast<size_t>(num_coeffs)) {
                throw std::runtime_error("ai size mismatch");
            }
            ai_cpu = config.ai;
        }
        
        b_cpu = config.b;
        
        ai = drjit::load<Float>(ai_cpu.data(), num_coeffs);
        b = drjit::full<Float>(b_cpu, 1);
        
        std::cout << "[XYPolynomialSurface] order=" << order 
                  << ", num_coeffs=" << num_coeffs 
                  << ", b=" << b_cpu << std::endl;
    }
    
    void sync_to_cpu() {
        FloatC ai_c = utils::detach(ai);
        drjit::store(ai_cpu.data(), ai_c);
        
        FloatC b_c = utils::detach(b);
        b_cpu = b_c[0];
    }
};

} // namespace diff_optics