// shape_displacement.hpp
#pragma once
#include "shape_base.hpp"

namespace diff_optics {

// ============= 位移网格曲面配置 =============
struct DisplacementMeshConfig {
    int width = 64;
    int height = 64;
    std::vector<ScalarType> heightmap;
    
    DisplacementMeshConfig() = default;
    
    DisplacementMeshConfig(int w, int h)
        : width(w), height(h) {}
    
    DisplacementMeshConfig(int w, int h, const std::vector<ScalarType>& data)
        : width(w), height(h), heightmap(data) {}
    
    int compute_num_pixels() const {
        return width * height;
    }
};

enum class BoundaryMode {
    Zero,
    Replicate,
    Mirror,
    Wrap
};

enum class InterpolationMode {
    Nearest,
    Bilinear
};

// ============= 位移网格曲面 =============
struct DisplacementMeshSurface : public Shape {
    int grid_width;
    int grid_height;
    int num_pixels;
    ScalarType surface_width;
    ScalarType surface_height;
    
    Float heightmap;
    
    std::vector<ScalarType> heightmap_cpu;
    
    BoundaryMode boundary_mode = BoundaryMode::Replicate;
    InterpolationMode interp_mode = InterpolationMode::Bilinear;
    
    DisplacementMeshSurface() 
        : grid_width(64), grid_height(64), 
          surface_width(50.0f), surface_height(50.0f) {
        DisplacementMeshConfig config(grid_width, grid_height);
        init_with_config(config);
    }
    
    DisplacementMeshSurface(ScalarType w, ScalarType h, 
                            const DisplacementMeshConfig& config = DisplacementMeshConfig())
        : grid_width(config.width), grid_height(config.height),
          surface_width(w), surface_height(h) {
        init_with_config(config);
    }
    
    std::string type_name() const override { return "DisplacementMeshSurface"; }
    
    // ============= 统一参数接口实现 =============
    
    int get_diff_param_count() const override {
        return num_pixels;
    }
    
    Float& get_diff_params() override { return heightmap; }
    const Float& get_diff_params() const override { return heightmap; }
    
    void set_diff_params(const Float& params) override {
        heightmap = params;
        sync_to_cpu();
    }
    
    std::vector<ScalarType> get_diff_params_cpu() const override {
        std::vector<ScalarType> result(num_pixels);
        FloatC hmap_c = utils::detach(heightmap);
        drjit::store(result.data(), hmap_c);
        return result;
    }
    
    void set_diff_params_cpu(const std::vector<ScalarType>& params) override {
        if (params.size() != static_cast<size_t>(num_pixels)) {
            throw std::runtime_error("DisplacementMeshSurface: param size mismatch");
        }
        heightmap = drjit::load<Float>(params.data(), num_pixels);
        heightmap_cpu = params;
    }
    
    std::string get_param_config_string() const override {
        return "displacement(" + std::to_string(grid_width) + "x" + std::to_string(grid_height) + ")";
    }
    
    std::map<std::string, int> get_param_config() const override {
        return {
            {"grid_width", grid_width},
            {"grid_height", grid_height}
        };
    }
    
    bool resize_params(const std::map<std::string, int>& new_config) override {
        auto it_w = new_config.find("grid_width");
        auto it_h = new_config.find("grid_height");
        
        int new_w = (it_w != new_config.end()) ? it_w->second : grid_width;
        int new_h = (it_h != new_config.end()) ? it_h->second : grid_height;
        
        if (new_w == grid_width && new_h == grid_height) {
            return false;
        }
        
        resize_grid(new_w, new_h);
        return true;
    }
    
    bool save_cad(const std::string& filename) const override {
        std::cerr << "[DisplacementMeshSurface] CAD export not implemented yet." << std::endl;
        return false;
    }
    
    // ============= 旧接口（保持兼容性）=============
    
    std::pair<int, int> get_grid_size() const { 
        return {grid_width, grid_height}; 
    }
    
    std::pair<ScalarType, ScalarType> get_surface_size() const {
        return {surface_width, surface_height};
    }
    
    Float& get_heightmap() { return heightmap; }
    const Float& get_heightmap() const { return heightmap; }
    
    void set_heightmap(const Float& hmap) { 
        heightmap = hmap; 
        sync_to_cpu();
    }
    
    Vector3C compute_position_c(const Vector2C& uv) const override {
        FloatC x = uv[0];
        FloatC y = uv[1];
        FloatC z = sample_heightmap_nondiff(x, y);
        return Vector3C(x, y, z);
    }
    
    Vector3 compute_normal(const Vector2& uv) const override {
        Float x = uv[0];
        Float y = uv[1];
        return compute_normal_at(x, y);
    }

public:
    std::pair<int, int> get_recommended_tessellation() const {
        return {grid_width - 1, grid_height - 1};
    }
    
    std::pair<ScalarType, ScalarType> pixel_to_physical(int i, int j) const {
        ScalarType dx = surface_width / static_cast<float>(grid_width - 1);
        ScalarType dy = surface_height / static_cast<float>(grid_height - 1);
        
        ScalarType x = -surface_width * 0.5f + i * dx;
        ScalarType y = -surface_height * 0.5f + j * dy;
        return {x, y};
    }
    
    void resize_grid(int new_width, int new_height) {
        if (new_width == grid_width && new_height == grid_height) {
            return;
        }
        
        std::vector<ScalarType> old_heightmap = get_heightmap_as_array();
        int old_width = grid_width;
        int old_height = grid_height;
        
        grid_width = new_width;
        grid_height = new_height;
        num_pixels = grid_width * grid_height;
        
        std::vector<ScalarType> new_heightmap(num_pixels);
        
        for (int j = 0; j < grid_height; ++j) {
            for (int i = 0; i < grid_width; ++i) {
                float u = static_cast<float>(i) / (grid_width - 1) * (old_width - 1);
                float v = static_cast<float>(j) / (grid_height - 1) * (old_height - 1);
                
                int u0 = std::min(static_cast<int>(u), old_width - 2);
                int v0 = std::min(static_cast<int>(v), old_height - 2);
                float alpha = u - u0;
                float beta = v - v0;
                
                ScalarType c00 = old_heightmap[v0 * old_width + u0];
                ScalarType c10 = old_heightmap[v0 * old_width + u0 + 1];
                ScalarType c01 = old_heightmap[(v0 + 1) * old_width + u0];
                ScalarType c11 = old_heightmap[(v0 + 1) * old_width + u0 + 1];
                
                new_heightmap[j * grid_width + i] = 
                    (1 - alpha) * (1 - beta) * c00 +
                    alpha * (1 - beta) * c10 +
                    (1 - alpha) * beta * c01 +
                    alpha * beta * c11;
            }
        }
        
        heightmap = drjit::load<Float>(new_heightmap.data(), num_pixels);
        heightmap_cpu = new_heightmap;
        
        std::cout << "[DisplacementMeshSurface] Resized from " 
                  << old_width << "x" << old_height << " to "
                  << grid_width << "x" << grid_height << std::endl;
    }
    
    void set_heightmap_from_array(const std::vector<ScalarType>& data) {
        if (data.size() != static_cast<size_t>(num_pixels)) {
            throw std::runtime_error("heightmap size mismatch");
        }
        heightmap = drjit::load<Float>(data.data(), num_pixels);
        heightmap_cpu = data;
    }
    
    std::vector<ScalarType> get_heightmap_as_array() const {
        std::vector<ScalarType> result(num_pixels);
        FloatC hmap_c = utils::detach(heightmap);
        drjit::store(result.data(), hmap_c);
        return result;
    }
    
    std::vector<std::vector<ScalarType>> get_heightmap_as_2d() const {
        std::vector<ScalarType> flat = get_heightmap_as_array();
        std::vector<std::vector<ScalarType>> result(grid_height, 
            std::vector<ScalarType>(grid_width));
        
        for (int j = 0; j < grid_height; ++j) {
            for (int i = 0; i < grid_width; ++i) {
                result[j][i] = flat[j * grid_width + i];
            }
        }
        return result;
    }
    
    void set_heightmap_from_2d(const std::vector<std::vector<ScalarType>>& data) {
        if (data.size() != static_cast<size_t>(grid_height) ||
            data[0].size() != static_cast<size_t>(grid_width)) {
            throw std::runtime_error("2D heightmap size mismatch");
        }
        
        std::vector<ScalarType> flat(num_pixels);
        for (int j = 0; j < grid_height; ++j) {
            for (int i = 0; i < grid_width; ++i) {
                flat[j * grid_width + i] = data[j][i];
            }
        }
        set_heightmap_from_array(flat);
    }
    
    void reverse() {
        heightmap = -heightmap;
        sync_to_cpu();
    }
    
    void set_pixel(int i, int j, ScalarType value) {
        if (i < 0 || i >= grid_width || j < 0 || j >= grid_height) {
            return;
        }
        int idx = j * grid_width + i;
        heightmap_cpu[idx] = value;
        heightmap = drjit::load<Float>(heightmap_cpu.data(), num_pixels);
    }
    
    ScalarType get_pixel(int i, int j) const {
        if (i < 0 || i >= grid_width || j < 0 || j >= grid_height) {
            return 0.0f;
        }
        return heightmap_cpu[j * grid_width + i];
    }

private:
    template<typename T>
    void physical_to_texture(const T& x, const T& y, T& u, T& v) const {
        ScalarType dx = surface_width / static_cast<float>(grid_width - 1);
        ScalarType dy = surface_height / static_cast<float>(grid_height - 1);
        
        u = (x + T(surface_width * 0.5f)) / T(dx);
        v = (y + T(surface_height * 0.5f)) / T(dy);
    }
    
    ScalarType get_dx() const {
        return surface_width / static_cast<float>(grid_width - 1);
    }
    
    ScalarType get_dy() const {
        return surface_height / static_cast<float>(grid_height - 1);
    }
    
    template<typename T>
    T handle_boundary(const T& idx, int size) const {
        switch (boundary_mode) {
            case BoundaryMode::Zero:
                return drjit::select(idx >= 0 && idx < size, idx, T(-1));
                
            case BoundaryMode::Replicate:
                return drjit::clamp(idx, T(0), T(size - 1));
                
            case BoundaryMode::Mirror: {
                T result = idx;
                result = drjit::select(idx < 0, -idx - 1, result);
                result = drjit::select(idx >= size, T(2 * size) - idx - 1, result);
                return drjit::clamp(result, T(0), T(size - 1));
            }
                
            case BoundaryMode::Wrap:
                return ((idx % size) + size) % size;
                
            default:
                return drjit::clamp(idx, T(0), T(size - 1));
        }
    }
    
    FloatC tex_sample_nondiff(const Int32C& i, const Int32C& j) const {
        Int32C i_safe = handle_boundary(i, grid_width);
        Int32C j_safe = handle_boundary(j, grid_height);
        
        UInt32C idx = UInt32C(j_safe * grid_width + i_safe);
        
        FloatC hmap_nondiff = utils::detach(heightmap);
        FloatC result = drjit::gather<FloatC>(hmap_nondiff, idx);
        
        if (boundary_mode == BoundaryMode::Zero) {
            MaskC valid = (i >= 0) && (i < grid_width) && 
                         (j >= 0) && (j < grid_height);
            result = drjit::select(valid, result, FloatC(0.0f));
        }
        
        return result;
    }
    
    void tex4_sample_nondiff(const Int32C& i, const Int32C& j,
                             FloatC& c00, FloatC& c10, 
                             FloatC& c01, FloatC& c11) const {
        c00 = tex_sample_nondiff(i, j);
        c10 = tex_sample_nondiff(i + 1, j);
        c01 = tex_sample_nondiff(i, j + 1);
        c11 = tex_sample_nondiff(i + 1, j + 1);
    }
    
    FloatC sample_heightmap_nondiff(const FloatC& x, const FloatC& y) const {
        FloatC u, v;
        physical_to_texture(x, y, u, v);
        
        if (interp_mode == InterpolationMode::Nearest) {
            Int32C i = Int32C(drjit::round(u));
            Int32C j = Int32C(drjit::round(v));
            return tex_sample_nondiff(i, j);
        } else {
            Int32C i0 = Int32C(drjit::floor(u));
            Int32C j0 = Int32C(drjit::floor(v));
            
            FloatC alpha = u - FloatC(i0);
            FloatC beta = v - FloatC(j0);
            
            FloatC c00, c10, c01, c11;
            tex4_sample_nondiff(i0, j0, c00, c10, c01, c11);
            
            FloatC result = (FloatC(1.0f) - alpha) * (FloatC(1.0f) - beta) * c00 +
                           alpha * (FloatC(1.0f) - beta) * c10 +
                           (FloatC(1.0f) - alpha) * beta * c01 +
                           alpha * beta * c11;
            
            return result;
        }
    }
    
    Float tex_sample(const Int32& i, const Int32& j) const {
        Int32 i_safe = handle_boundary(i, grid_width);
        Int32 j_safe = handle_boundary(j, grid_height);
        
        UInt32 idx = UInt32(j_safe * grid_width + i_safe);
        
        Float result = drjit::gather<Float>(heightmap, idx);
        
        if (boundary_mode == BoundaryMode::Zero) {
            MaskC valid_c = (utils::detach(i) >= 0) && 
                           (utils::detach(i) < grid_width) && 
                           (utils::detach(j) >= 0) && 
                           (utils::detach(j) < grid_height);
            result = drjit::select(Mask(valid_c), result, Float(0.0f));
        }
        
        return result;
    }
    
    void tex4_sample(const Int32& i, const Int32& j,
                     Float& c00, Float& c10, 
                     Float& c01, Float& c11) const {
        c00 = tex_sample(i, j);
        c10 = tex_sample(i + 1, j);
        c01 = tex_sample(i, j + 1);
        c11 = tex_sample(i + 1, j + 1);
    }
    
    Float sample_heightmap(const Float& x, const Float& y) const {
        Float u, v;
        physical_to_texture(x, y, u, v);
        
        if (interp_mode == InterpolationMode::Nearest) {
            Int32 i = Int32(drjit::round(utils::detach(u)));
            Int32 j = Int32(drjit::round(utils::detach(v)));
            return tex_sample(i, j);
        } else {
            Int32C i0_c = Int32C(drjit::floor(utils::detach(u)));
            Int32C j0_c = Int32C(drjit::floor(utils::detach(v)));
            Int32 i0(i0_c);
            Int32 j0(j0_c);
            
            Float alpha = u - Float(FloatC(i0_c));
            Float beta = v - Float(FloatC(j0_c));
            
            Float c00, c10, c01, c11;
            tex4_sample(i0, j0, c00, c10, c01, c11);
            
            Float result = (Float(1.0f) - alpha) * (Float(1.0f) - beta) * c00 +
                          alpha * (Float(1.0f) - beta) * c10 +
                          (Float(1.0f) - alpha) * beta * c01 +
                          alpha * beta * c11;
            
            return result;
        }
    }
    
    void compute_derivatives_bilinear(const Float& x, const Float& y,
                                      Float& dzdx, Float& dzdy) const {
        Float u, v;
        physical_to_texture(x, y, u, v);
        
        FloatC u_c = utils::detach(u);
        FloatC v_c = utils::detach(v);
        Int32C i0_c = Int32C(drjit::floor(u_c));
        Int32C j0_c = Int32C(drjit::floor(v_c));
        Int32 i0(i0_c);
        Int32 j0(j0_c);
        
        Float alpha = u - Float(FloatC(i0_c));
        Float beta = v - Float(FloatC(j0_c));
        
        Float c00, c10, c01, c11;
        tex4_sample(i0, j0, c00, c10, c01, c11);
        
        Float dzdu = (Float(1.0f) - beta) * (c10 - c00) + beta * (c11 - c01);
        Float dzdv = (Float(1.0f) - alpha) * (c01 - c00) + alpha * (c11 - c10);
        
        ScalarType dx = get_dx();
        ScalarType dy = get_dy();
        
        dzdx = dzdu / Float(dx);
        dzdy = dzdv / Float(dy);
    }
    
    Vector3 compute_normal_at(const Float& x, const Float& y) const {
        Float dzdx, dzdy;
        compute_derivatives_bilinear(x, y, dzdx, dzdy);
        
        Vector3 normal(-dzdx, -dzdy, Float(1.0f));
        return drjit::normalize(normal);
    }
    
    Float g(const Float& x, const Float& y) const {
        return sample_heightmap(x, y);
    }
    
    Float h(const Float& z) const {
        return -z;
    }
    
    Float dhd(const Float& z) const {
        return -drjit::full<Float>(1.0, drjit::width(z));
    }
    
    void init_with_config(const DisplacementMeshConfig& config) {
        num_pixels = grid_width * grid_height;
        
        if (config.heightmap.empty()) {
            heightmap_cpu.resize(num_pixels, 0.0f);
        } else {
            if (config.heightmap.size() != static_cast<size_t>(num_pixels)) {
                throw std::runtime_error("heightmap size mismatch");
            }
            heightmap_cpu = config.heightmap;
        }
        
        heightmap = drjit::load<Float>(heightmap_cpu.data(), num_pixels);
        
        std::cout << "[DisplacementMeshSurface] grid=" << grid_width << "x" << grid_height
                  << ", surface=" << surface_width << "x" << surface_height
                  << ", pixels=" << num_pixels << std::endl;
    }
    
    void sync_to_cpu() {
        FloatC hmap_c = utils::detach(heightmap);
        drjit::store(heightmap_cpu.data(), hmap_c);
    }
};

} // namespace diff_optics