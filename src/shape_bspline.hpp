// shape_bspline.hpp
#pragma once
#include "shape_base.hpp"

#ifdef USE_OPENCASCADE
#include <gp_Pnt.hxx>
#include <TColgp_Array2OfPnt.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <Geom_BSplineSurface.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <TopoDS_Face.hxx>
#include <STEPControl_Writer.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <Standard_Handle.hxx>
#endif

namespace diff_optics {

// ============= B-Spline 配置 =============
struct BSplineConfig {
    int u_degree = 3;
    int v_degree = 3;
    int u_control_points = 6;
    int v_control_points = 6;
    std::vector<ScalarType> control_points_z;
    
    BSplineConfig() = default;
    BSplineConfig(int u_deg, int v_deg, int u_cp, int v_cp, const std::vector<ScalarType>& cpz = {})
        : u_degree(u_deg), v_degree(v_deg),
          u_control_points(u_cp), v_control_points(v_cp), control_points_z(cpz) {}
};

// ============= B-Spline 曲面 =============
struct BSplineSurface : public Shape {
    int u_degree, v_degree;
    int u_num_cp, v_num_cp;
    ScalarType width, height;
    
    // 节点向量（GPU）- 参数域为 [-width/2, width/2] 和 [-height/2, height/2]
    FloatC u_knots, v_knots;
    // 节点向量（CPU）
    std::vector<ScalarType> u_knots_cpu, v_knots_cpu;
    
    // 控制点
    FloatC cp_x, cp_y;  // 非微分
    Float cp_z;          // 可微分
    
    BSplineSurface() : width(1.0f), height(1.0f), u_degree(3), v_degree(3), u_num_cp(6), v_num_cp(6) {
        BSplineConfig config(u_degree, v_degree, u_num_cp, v_num_cp);
        init_with_config(config);
    }
    
    BSplineSurface(ScalarType w, ScalarType h, const BSplineConfig& config = BSplineConfig())
        : width(w), height(h),
          u_degree(config.u_degree), v_degree(config.v_degree),
          u_num_cp(config.u_control_points), v_num_cp(config.v_control_points) {
        init_with_config(config);
    }
    
    std::string type_name() const override { return "BSplineSurface"; }
    
    int num_control_points() const { return u_num_cp * v_num_cp; }
    int cp_index(int i, int j) const { return j * u_num_cp + i; }
    
    // ============= 统一参数接口实现 =============
    
    int get_diff_param_count() const override {
        return u_num_cp * v_num_cp;
    }
    
    Float& get_diff_params() override { return cp_z; }
    const Float& get_diff_params() const override { return cp_z; }
    
    void set_diff_params(const Float& params) override {
        cp_z = params;
    }
    
    std::vector<ScalarType> get_diff_params_cpu() const override {
        std::vector<ScalarType> result(get_diff_param_count());
        FloatC cp_z_c = utils::detach(cp_z);
        drjit::store(result.data(), cp_z_c);
        return result;
    }
    
    void set_diff_params_cpu(const std::vector<ScalarType>& params) override {
        if (params.size() != static_cast<size_t>(get_diff_param_count())) {
            throw std::runtime_error("BSplineSurface: param size mismatch");
        }
        cp_z = drjit::load<Float>(params.data(), params.size());
    }
    
    std::string get_param_config_string() const override {
        return "bspline(" + std::to_string(u_num_cp) + "x" + std::to_string(v_num_cp) + ")";
    }
    
    std::map<std::string, int> get_param_config() const override {
        return {
            {"u_num_cp", u_num_cp},
            {"v_num_cp", v_num_cp},
            {"u_degree", u_degree},
            {"v_degree", v_degree}
        };
    }
    
    bool resize_params(const std::map<std::string, int>& new_config) override {
        auto it_u = new_config.find("u_num_cp");
        auto it_v = new_config.find("v_num_cp");
        
        int new_u = (it_u != new_config.end()) ? it_u->second : u_num_cp;
        int new_v = (it_v != new_config.end()) ? it_v->second : v_num_cp;
        
        if (new_u == u_num_cp && new_v == v_num_cp) {
            return false;
        }
        
        resize_control_points(new_u, new_v);
        return true;
    }
    
    // ============= CAD 导出实现 =============
    
    bool save_cad(const std::string& filename) const override {
#ifdef USE_OPENCASCADE
        try {
            std::vector<ScalarType> cpx_cpu(u_num_cp * v_num_cp);
            std::vector<ScalarType> cpy_cpu(u_num_cp * v_num_cp);
            std::vector<ScalarType> cpz_cpu(u_num_cp * v_num_cp);
            
            drjit::store(cpx_cpu.data(), cp_x);
            drjit::store(cpy_cpu.data(), cp_y);
            FloatC cp_z_c = utils::detach(cp_z);
            drjit::store(cpz_cpu.data(), cp_z_c);
            
            TColgp_Array2OfPnt controlPoints(1, u_num_cp, 1, v_num_cp);
            
            for (int j = 0; j < v_num_cp; ++j) {
                for (int i = 0; i < u_num_cp; ++i) {
                    int idx = cp_index(i, j);
                    controlPoints.SetValue(
                        i + 1, j + 1,
                        gp_Pnt(cpx_cpu[idx], cpy_cpu[idx], cpz_cpu[idx])
                    );
                }
            }
            
            std::vector<ScalarType> u_unique_knots;
            std::vector<int> u_mults;
            compute_unique_knots_and_multiplicities(u_knots_cpu, u_unique_knots, u_mults);
            
            std::vector<ScalarType> v_unique_knots;
            std::vector<int> v_mults;
            compute_unique_knots_and_multiplicities(v_knots_cpu, v_unique_knots, v_mults);
            
            TColStd_Array1OfReal uKnots(1, static_cast<int>(u_unique_knots.size()));
            TColStd_Array1OfInteger uMults(1, static_cast<int>(u_mults.size()));
            for (size_t i = 0; i < u_unique_knots.size(); ++i) {
                uKnots.SetValue(static_cast<int>(i + 1), u_unique_knots[i]);
                uMults.SetValue(static_cast<int>(i + 1), u_mults[i]);
            }
            
            TColStd_Array1OfReal vKnots(1, static_cast<int>(v_unique_knots.size()));
            TColStd_Array1OfInteger vMults(1, static_cast<int>(v_mults.size()));
            for (size_t i = 0; i < v_unique_knots.size(); ++i) {
                vKnots.SetValue(static_cast<int>(i + 1), v_unique_knots[i]);
                vMults.SetValue(static_cast<int>(i + 1), v_mults[i]);
            }
            
            Handle(Geom_BSplineSurface) bsplineSurface = new Geom_BSplineSurface(
                controlPoints,
                uKnots,
                vKnots,
                uMults,
                vMults,
                u_degree,
                v_degree,
                Standard_False,
                Standard_False
            );
            
            TopoDS_Face face = BRepBuilderAPI_MakeFace(bsplineSurface, 1e-6);
            
            STEPControl_Writer stepWriter;
            IFSelect_ReturnStatus status = stepWriter.Transfer(face, STEPControl_AsIs);
            
            if (status != IFSelect_RetDone) {
                std::cerr << "[BSplineSurface] Failed to transfer surface to STEP writer." << std::endl;
                return false;
            }
            
            if (stepWriter.Write(filename.c_str()) == IFSelect_RetDone) {
                std::cout << "[BSplineSurface] Successfully exported to: " << filename << std::endl;
                return true;
            } else {
                std::cerr << "[BSplineSurface] Failed to write STEP file: " << filename << std::endl;
                return false;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "[BSplineSurface] CAD export error: " << e.what() << std::endl;
            return false;
        }
#else
        std::cerr << "[BSplineSurface] OpenCASCADE not available. Rebuild with USE_OPENCASCADE." << std::endl;
        return false;
#endif
    }
    
    // ============= 旧接口（保持兼容性）=============
    
    Float& get_parameters() { return cp_z; }
    void set_parameters(const Float& z) { cp_z = z; }
    Float& get_control_points_z() { return cp_z; }
    const Float& get_control_points_z() const { return cp_z; }
    void set_control_points_z(const Float& z) { cp_z = z; }
    
    std::pair<ScalarType, ScalarType> get_u_domain() const {
        return {-width / 2.0f, width / 2.0f};
    }
    
    std::pair<ScalarType, ScalarType> get_v_domain() const {
        return {-height / 2.0f, height / 2.0f};
    }

    // ========== 非微分版本：用于生成三角网格 ==========
    Vector3C compute_position_c(const Vector2C& uv) const override {
        FloatC u = uv[0], v = uv[1];
        FloatC sx, sy, sz;
        eval_position_nondiff(u, v, sx, sy, sz);
        return Vector3C(sx, sy, sz);
    }
    
    // ========== 可微分版本：计算法线，保持 uv 梯度连接 ==========
    Vector3 compute_normal(const Vector2& uv) const override {
        return eval_normal_diff(uv[0], uv[1]);
    }

public:
    // ========== 多尺度控制点重采样 ==========
    
    std::tuple<int, int, int, int> get_config() const {
        return {u_degree, v_degree, u_num_cp, v_num_cp};
    }
    
    void resize_control_points(int new_u_num_cp, int new_v_num_cp) {
        if (new_u_num_cp == u_num_cp && new_v_num_cp == v_num_cp) {
            return;
        }
        
        std::vector<ScalarType> old_cp_z_cpu(u_num_cp * v_num_cp);
        FloatC old_cp_z_c = utils::detach(cp_z);
        drjit::store(old_cp_z_cpu.data(), old_cp_z_c);
        
        int old_u_num_cp = u_num_cp;
        int old_v_num_cp = v_num_cp;
        
        u_num_cp = new_u_num_cp;
        v_num_cp = new_v_num_cp;
        
        u_knots_cpu = generate_clamped_knots_physical(u_num_cp, u_degree, -width/2.0f, width/2.0f);
        v_knots_cpu = generate_clamped_knots_physical(v_num_cp, v_degree, -height/2.0f, height/2.0f);
        u_knots = drjit::load<FloatC>(u_knots_cpu.data(), u_knots_cpu.size());
        v_knots = drjit::load<FloatC>(v_knots_cpu.data(), v_knots_cpu.size());
        
        auto new_u_greville = compute_greville_physical(u_knots_cpu, u_degree, u_num_cp);
        auto new_v_greville = compute_greville_physical(v_knots_cpu, v_degree, v_num_cp);
        
        int new_total = u_num_cp * v_num_cp;
        std::vector<ScalarType> new_cpx(new_total);
        std::vector<ScalarType> new_cpy(new_total);
        std::vector<ScalarType> new_cpz(new_total);
        
        for (int j = 0; j < v_num_cp; ++j) {
            for (int i = 0; i < u_num_cp; ++i) {
                int new_idx = j * u_num_cp + i;
                
                ScalarType u_norm = (u_num_cp > 1) ? 
                    static_cast<ScalarType>(i) / (u_num_cp - 1) : 0.5f;
                ScalarType v_norm = (v_num_cp > 1) ? 
                    static_cast<ScalarType>(j) / (v_num_cp - 1) : 0.5f;
                
                new_cpz[new_idx] = interpolate_old_cp_z_uniform(
                    old_cp_z_cpu, old_u_num_cp, old_v_num_cp, u_norm, v_norm
                );
                
                new_cpx[new_idx] = new_u_greville[i];
                new_cpy[new_idx] = new_v_greville[j];
            }
        }
        
        cp_x = drjit::load<FloatC>(new_cpx.data(), new_total);
        cp_y = drjit::load<FloatC>(new_cpy.data(), new_total);
        cp_z = drjit::load<Float>(new_cpz.data(), new_total);
    }
    
private:
    // ============= CAD 导出辅助函数 =============
    
    static void compute_unique_knots_and_multiplicities(
        const std::vector<ScalarType>& knots,
        std::vector<ScalarType>& unique_knots,
        std::vector<int>& multiplicities
    ) {
        unique_knots.clear();
        multiplicities.clear();
        
        if (knots.empty()) return;
        
        ScalarType current = knots[0];
        int count = 1;
        
        for (size_t i = 1; i < knots.size(); ++i) {
            if (std::abs(knots[i] - current) < 1e-10f) {
                count++;
            } else {
                unique_knots.push_back(current);
                multiplicities.push_back(count);
                current = knots[i];
                count = 1;
            }
        }
        
        unique_knots.push_back(current);
        multiplicities.push_back(count);
    }
    
    ScalarType interpolate_old_cp_z_uniform(
        const std::vector<ScalarType>& old_cp_z,
        int old_u_num_cp, int old_v_num_cp,
        ScalarType u_norm, ScalarType v_norm
    ) const {
        ScalarType fi = u_norm * (old_u_num_cp - 1);
        ScalarType fj = v_norm * (old_v_num_cp - 1);
        
        fi = std::max(0.0f, std::min(static_cast<ScalarType>(old_u_num_cp - 1), fi));
        fj = std::max(0.0f, std::min(static_cast<ScalarType>(old_v_num_cp - 1), fj));
        
        int i0 = static_cast<int>(std::floor(fi));
        int j0 = static_cast<int>(std::floor(fj));
        
        i0 = std::min(i0, old_u_num_cp - 2);
        j0 = std::min(j0, old_v_num_cp - 2);
        i0 = std::max(i0, 0);
        j0 = std::max(j0, 0);
        
        int i1 = i0 + 1;
        int j1 = j0 + 1;
        
        ScalarType wi = fi - i0;
        ScalarType wj = fj - j0;
        
        auto get_val = [&](int ii, int jj) -> ScalarType {
            ii = std::max(0, std::min(old_u_num_cp - 1, ii));
            jj = std::max(0, std::min(old_v_num_cp - 1, jj));
            return old_cp_z[jj * old_u_num_cp + ii];
        };
        
        ScalarType v00 = get_val(i0, j0);
        ScalarType v10 = get_val(i1, j0);
        ScalarType v01 = get_val(i0, j1);
        ScalarType v11 = get_val(i1, j1);
        
        ScalarType val0 = v00 * (1 - wi) + v10 * wi;
        ScalarType val1 = v01 * (1 - wi) + v11 * wi;
        
        return val0 * (1 - wj) + val1 * wj;
    }

    // ============= Span 查找（非微分，用于离散索引操作）=============
    
    UInt32C find_span_binary(const FloatC& t, const FloatC& knots, 
                              int degree, int num_cp, 
                              ScalarType t_min, ScalarType t_max) const {
        size_t N = drjit::width(t);
        
        int low_init = degree;
        int high_init = num_cp - 1;
        
        MaskC at_end = (t >= FloatC(t_max - 1e-6f));
        
        UInt32C low = drjit::full<UInt32C>(static_cast<uint32_t>(low_init), N);
        UInt32C high = drjit::full<UInt32C>(static_cast<uint32_t>(high_init), N);
        
        int max_iter = 0;
        int range = high_init - low_init;
        while (range > 0) {
            max_iter++;
            range >>= 1;
        }
        max_iter = std::max(max_iter, 1);
        
        for (int iter = 0; iter < max_iter; ++iter) {
            UInt32C mid = (low + high + UInt32C(1)) / UInt32C(2);
            FloatC k_mid = drjit::gather<FloatC>(knots, mid);
            
            MaskC go_left = (t < k_mid);
            high = drjit::select(go_left, mid - UInt32C(1), high);
            low = drjit::select(go_left, low, mid);
        }
        
        UInt32C span = drjit::select(at_end, UInt32C(static_cast<uint32_t>(num_cp - 1)), low);
        span = drjit::clamp(span, UInt32C(static_cast<uint32_t>(degree)), 
                           UInt32C(static_cast<uint32_t>(num_cp - 1)));
        
        return span;
    }

    // ============= 非微分基函数计算 =============
    
    void eval_basis_functions(const FloatC& t, const UInt32C& span,
                              const FloatC& knots, int degree,
                              std::vector<FloatC>& N_out) const {
        size_t n = drjit::width(t);
        N_out.resize(degree + 1);
        
        std::vector<FloatC> left(degree + 1);
        std::vector<FloatC> right(degree + 1);
        
        N_out[0] = drjit::full<FloatC>(1.0f, n);
        
        for (int j = 1; j <= degree; ++j) {
            UInt32C left_idx = span + UInt32C(1) - UInt32C(j);
            left[j] = t - drjit::gather<FloatC>(knots, left_idx);
            
            UInt32C right_idx = span + UInt32C(j);
            right[j] = drjit::gather<FloatC>(knots, right_idx) - t;
            
            FloatC saved = drjit::zeros<FloatC>(n);
            
            for (int r = 0; r < j; ++r) {
                FloatC denom = right[r + 1] + left[j - r];
                MaskC valid = denom > FloatC(1e-10f);
                FloatC safe_denom = drjit::select(valid, denom, FloatC(1.0f));
                
                FloatC temp = drjit::select(valid, N_out[r] / safe_denom, FloatC(0.0f));
                N_out[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            N_out[j] = saved;
        }
    }

    // ============= 非微分基函数及导数计算 =============
    
    void eval_basis_and_derivatives(const FloatC& t, const UInt32C& span,
                                    const FloatC& knots, int degree,
                                    std::vector<FloatC>& N_out,
                                    std::vector<FloatC>& dN_out) const {
        size_t n = drjit::width(t);
        N_out.resize(degree + 1);
        dN_out.resize(degree + 1);
        
        std::vector<std::vector<FloatC>> ndu(degree + 1, 
            std::vector<FloatC>(degree + 1));
        
        std::vector<FloatC> left(degree + 1);
        std::vector<FloatC> right(degree + 1);
        
        ndu[0][0] = drjit::full<FloatC>(1.0f, n);
        
        for (int j = 1; j <= degree; ++j) {
            UInt32C left_idx = span + UInt32C(1) - UInt32C(j);
            left[j] = t - drjit::gather<FloatC>(knots, left_idx);
            
            UInt32C right_idx = span + UInt32C(j);
            right[j] = drjit::gather<FloatC>(knots, right_idx) - t;
            
            FloatC saved = drjit::zeros<FloatC>(n);
            
            for (int r = 0; r < j; ++r) {
                FloatC denom = right[r + 1] + left[j - r];
                MaskC valid = denom > FloatC(1e-10f);
                FloatC safe_denom = drjit::select(valid, denom, FloatC(1.0f));
                
                ndu[j][r] = drjit::select(valid, FloatC(1.0f) / safe_denom, FloatC(0.0f));
                
                FloatC temp = ndu[r][j-1] * ndu[j][r];
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }
        
        for (int r = 0; r <= degree; ++r) {
            N_out[r] = ndu[r][degree];
        }
        
        for (int r = 0; r <= degree; ++r) {
            FloatC d = drjit::zeros<FloatC>(n);
            int rk = r - 1;
            int pk = degree - 1;
            
            if (r >= 1) {
                FloatC a1 = ndu[degree][rk];
                d = a1 * ndu[rk][pk];
            }
            
            if (r <= degree - 1) {
                FloatC a0 = ndu[degree][r];
                d = d - a0 * ndu[r][pk];
            }
            
            dN_out[r] = d * FloatC(static_cast<float>(degree));
        }
    }

    // ============= 可微分基函数及导数计算（保持 t 的梯度）=============
    
    void eval_basis_and_derivatives_diff(const Float& t, const UInt32C& span,
                                         const FloatC& knots, int degree,
                                         std::vector<Float>& N_out,
                                         std::vector<Float>& dN_out) const {
        size_t n = drjit::width(t);
        N_out.resize(degree + 1);
        dN_out.resize(degree + 1);
        
        std::vector<std::vector<Float>> ndu(degree + 1, 
            std::vector<Float>(degree + 1));
        
        std::vector<Float> left(degree + 1);
        std::vector<Float> right(degree + 1);
        
        ndu[0][0] = drjit::full<Float>(1.0f, n);
        
        for (int j = 1; j <= degree; ++j) {
            UInt32C left_idx = span + UInt32C(1) - UInt32C(j);
            FloatC knot_left = drjit::gather<FloatC>(knots, left_idx);
            left[j] = t - Float(knot_left);
            
            UInt32C right_idx = span + UInt32C(j);
            FloatC knot_right = drjit::gather<FloatC>(knots, right_idx);
            right[j] = Float(knot_right) - t;
            
            Float saved = drjit::zeros<Float>(n);
            
            for (int r = 0; r < j; ++r) {
                Float denom = right[r + 1] + left[j - r];
                FloatC denom_c = utils::detach(denom);
                MaskC valid = denom_c > FloatC(1e-10f);
                Float safe_denom = drjit::select(Mask(valid), denom, Float(1.0f));
                
                ndu[j][r] = drjit::select(Mask(valid), Float(1.0f) / safe_denom, Float(0.0f));
                
                Float temp = ndu[r][j-1] * ndu[j][r];
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }
        
        for (int r = 0; r <= degree; ++r) {
            N_out[r] = ndu[r][degree];
        }
        
        for (int r = 0; r <= degree; ++r) {
            Float d = drjit::zeros<Float>(n);
            int rk = r - 1;
            int pk = degree - 1;
            
            if (r >= 1) {
                Float a1 = ndu[degree][rk];
                d = a1 * ndu[rk][pk];
            }
            
            if (r <= degree - 1) {
                Float a0 = ndu[degree][r];
                d = d - a0 * ndu[r][pk];
            }
            
            dN_out[r] = d * Float(static_cast<float>(degree));
        }
    }

    // ============= 控制点索引计算 =============
    
    void compute_control_point_indices(const UInt32C& u_span, const UInt32C& v_span,
                                       std::vector<UInt32C>& cp_indices) const {
        int num_cp_needed = (u_degree + 1) * (v_degree + 1);
        cp_indices.resize(num_cp_needed);
        
        UInt32C u_start = u_span - UInt32C(u_degree);
        UInt32C v_start = v_span - UInt32C(v_degree);
        
        int idx = 0;
        for (int jj = 0; jj <= v_degree; ++jj) {
            UInt32C cp_j = v_start + UInt32C(jj);
            UInt32C row_offset = cp_j * UInt32C(u_num_cp);
            
            for (int ii = 0; ii <= u_degree; ++ii) {
                UInt32C cp_i = u_start + UInt32C(ii);
                cp_indices[idx++] = row_offset + cp_i;
            }
        }
    }

    // ============= 非微分位置计算 =============
    
    void eval_position_nondiff(const FloatC& u, const FloatC& v,
                               FloatC& sx, FloatC& sy, FloatC& sz) const {
        size_t N = drjit::width(u);
        
        UInt32C u_span = find_span_binary(u, u_knots, u_degree, u_num_cp, 
                                          -width/2.0f, width/2.0f);
        UInt32C v_span = find_span_binary(v, v_knots, v_degree, v_num_cp,
                                          -height/2.0f, height/2.0f);
        
        std::vector<FloatC> Nu, Nv;
        eval_basis_functions(u, u_span, u_knots, u_degree, Nu);
        eval_basis_functions(v, v_span, v_knots, v_degree, Nv);
        
        std::vector<UInt32C> cp_indices;
        compute_control_point_indices(u_span, v_span, cp_indices);
        
        FloatC cp_z_nondiff = utils::detach(cp_z);
        
        sx = drjit::zeros<FloatC>(N);
        sy = drjit::zeros<FloatC>(N);
        sz = drjit::zeros<FloatC>(N);
        
        int idx = 0;
        for (int jj = 0; jj <= v_degree; ++jj) {
            for (int ii = 0; ii <= u_degree; ++ii) {
                FloatC w = Nu[ii] * Nv[jj];
                
                FloatC cpx = drjit::gather<FloatC>(cp_x, cp_indices[idx]);
                FloatC cpy = drjit::gather<FloatC>(cp_y, cp_indices[idx]);
                FloatC cpz = drjit::gather<FloatC>(cp_z_nondiff, cp_indices[idx]);
                
                sx = drjit::fmadd(w, cpx, sx);
                sy = drjit::fmadd(w, cpy, sy);
                sz = drjit::fmadd(w, cpz, sz);
                
                idx++;
            }
        }
    }

    // ============= 可微分法线计算（保持 uv 梯度连接）=============
    
    Vector3 eval_normal_diff(const Float& u, const Float& v) const {
        size_t N = drjit::width(u);
        
        FloatC u_c = utils::detach(u);
        FloatC v_c = utils::detach(v);
        
        UInt32C u_span = find_span_binary(u_c, u_knots, u_degree, u_num_cp,
                                          -width/2.0f, width/2.0f);
        UInt32C v_span = find_span_binary(v_c, v_knots, v_degree, v_num_cp,
                                          -height/2.0f, height/2.0f);
        
        std::vector<Float> Nu, dNu, Nv, dNv;
        eval_basis_and_derivatives_diff(u, u_span, u_knots, u_degree, Nu, dNu);
        eval_basis_and_derivatives_diff(v, v_span, v_knots, v_degree, Nv, dNv);
        
        std::vector<UInt32C> cp_indices;
        compute_control_point_indices(u_span, v_span, cp_indices);
        
        Float Su_x = drjit::zeros<Float>(N);
        Float Su_y = drjit::zeros<Float>(N);
        Float Su_z = drjit::zeros<Float>(N);
        Float Sv_x = drjit::zeros<Float>(N);
        Float Sv_y = drjit::zeros<Float>(N);
        Float Sv_z = drjit::zeros<Float>(N);
        
        int idx = 0;
        for (int jj = 0; jj <= v_degree; ++jj) {
            for (int ii = 0; ii <= u_degree; ++ii) {
                Float w_du = dNu[ii] * Nv[jj];
                Float w_dv = Nu[ii] * dNv[jj];
                
                FloatC cpx_c = drjit::gather<FloatC>(cp_x, cp_indices[idx]);
                FloatC cpy_c = drjit::gather<FloatC>(cp_y, cp_indices[idx]);
                Float cpx = Float(cpx_c);
                Float cpy = Float(cpy_c);
                
                UInt32 idx_diff = UInt32(cp_indices[idx]);
                Float cpz = drjit::gather<Float>(cp_z, idx_diff);
                
                Su_x = drjit::fmadd(w_du, cpx, Su_x);
                Su_y = drjit::fmadd(w_du, cpy, Su_y);
                Su_z = drjit::fmadd(w_du, cpz, Su_z);
                
                Sv_x = drjit::fmadd(w_dv, cpx, Sv_x);
                Sv_y = drjit::fmadd(w_dv, cpy, Sv_y);
                Sv_z = drjit::fmadd(w_dv, cpz, Sv_z);
                
                idx++;
            }
        }
        
        Vector3 Su(Su_x, Su_y, Su_z);
        Vector3 Sv(Sv_x, Sv_y, Sv_z);
        
        Vector3 normal = drjit::cross(Su, Sv);
        
        return drjit::normalize(normal);
    }

    // ============= 初始化 =============
    
    void init_with_config(const BSplineConfig& config) {
        u_knots_cpu = generate_clamped_knots_physical(u_num_cp, u_degree, -width/2.0f, width/2.0f);
        v_knots_cpu = generate_clamped_knots_physical(v_num_cp, v_degree, -height/2.0f, height/2.0f);
        u_knots = drjit::load<FloatC>(u_knots_cpu.data(), u_knots_cpu.size());
        v_knots = drjit::load<FloatC>(v_knots_cpu.data(), v_knots_cpu.size());
        
        auto u_greville = compute_greville_physical(u_knots_cpu, u_degree, u_num_cp);
        auto v_greville = compute_greville_physical(v_knots_cpu, v_degree, v_num_cp);
        
        int total = num_control_points();
        std::vector<ScalarType> cpx(total), cpy(total), cpz(total, 0.0f);
        
        for (int j = 0; j < v_num_cp; ++j) {
            for (int i = 0; i < u_num_cp; ++i) {
                int idx = cp_index(i, j);
                cpx[idx] = u_greville[i];
                cpy[idx] = v_greville[j];
            }
        }
        
        if (!config.control_points_z.empty()) {
            if (config.control_points_z.size() == static_cast<size_t>(total)) {
                cpz = config.control_points_z;
                std::cout << "[BSplineSurface] Loaded " << total << " control points from config" << std::endl;
            } else {
                std::cerr << "[BSplineSurface] Warning: control_points_z size mismatch" << std::endl;
            }
        }

        cp_x = drjit::load<FloatC>(cpx.data(), total);
        cp_y = drjit::load<FloatC>(cpy.data(), total);
        cp_z = drjit::load<Float>(cpz.data(), total);
        
        std::cout << "[BSplineSurface] Physical domain: u=[" << -width/2 << ", " << width/2 
                  << "], v=[" << -height/2 << ", " << height/2 << "]" << std::endl;
    }
    
    static std::vector<ScalarType> generate_clamped_knots_physical(
        int num_cp, int degree, ScalarType t_min, ScalarType t_max) {
        
        int n = num_cp + degree + 1;
        std::vector<ScalarType> knots(n);
        int num_internal = n - 2 * (degree + 1);
        
        for (int i = 0; i < n; ++i) {
            if (i <= degree) {
                knots[i] = t_min;
            } else if (i >= n - degree - 1) {
                knots[i] = t_max;
            } else {
                ScalarType t = static_cast<ScalarType>(i - degree) / (num_internal + 1);
                knots[i] = t_min + t * (t_max - t_min);
            }
        }
        return knots;
    }
    
    static std::vector<ScalarType> compute_greville_physical(
        const std::vector<ScalarType>& knots, int degree, int num_cp) {
        
        std::vector<ScalarType> greville(num_cp);
        for (int i = 0; i < num_cp; ++i) {
            ScalarType sum = 0.0f;
            for (int k = 1; k <= degree; ++k) {
                sum += knots[i + k];
            }
            greville[i] = sum / degree;
        }
        return greville;
    }
};

} // namespace diff_optics