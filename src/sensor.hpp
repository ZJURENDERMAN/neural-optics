// sensor.hpp - 简化版，移除金字塔功能
#pragma once
#include "utils.hpp"
#include "surface.hpp"
#include <memory>

namespace diff_optics {

// ============= 传感器数据 =============
struct SensorData {
    std::string sensor_name;
    int width = 0;
    int height = 0;
    
    Float data;
    UInt32 hit_count;
    
    // ============= 构造函数 =============
    SensorData() = default;
    
    SensorData(const std::string& name, int w, int h)
        : sensor_name(name), width(w), height(h) {
        size_t size = static_cast<size_t>(w) * static_cast<size_t>(h);
        data = drjit::zeros<Float>(size);
        hit_count = drjit::zeros<UInt32>(size);
    }
    
    // ============= 基本操作 =============
    void clear() {
        size_t size = static_cast<size_t>(width) * static_cast<size_t>(height);
        data = drjit::zeros<Float>(size);
        hit_count = drjit::zeros<UInt32>(size);
    }
    
    size_t size() const {
        return static_cast<size_t>(width) * static_cast<size_t>(height);
    }
    
    size_t total_hits() const {
        if (drjit::width(hit_count) == 0) return 0;
        return static_cast<size_t>(drjit::sum(hit_count)[0]);
    }
    
    std::pair<int, int> get_resolution() const {
        return {width, height};
    }
    
    // ============= 累加操作 =============
    
    // Box filter: 累加到单个像素
    void accumulate(const UInt32& pixel_indices, const Float& values, const Mask& valid) {
        Float masked_values = drjit::select(valid, values, Float(0.0f));
        UInt32 masked_counts = drjit::select(valid, UInt32(1), UInt32(0));
        
        drjit::scatter_add(data, masked_values, pixel_indices);
        drjit::scatter_add(hit_count, masked_counts, pixel_indices);
    }
    
    // Bilinear filter: 累加到四个相邻像素
    void accumulate_bilinear(const Float& px, const Float& py, 
                             const Float& values, const Mask& valid) {
        Float fx = drjit::clamp(px * Float(width), Float(0.0f), Float(width - 1));
        Float fy = drjit::clamp(py * Float(height), Float(0.0f), Float(height - 1));
        
        UInt32 int_px = UInt32(drjit::floor(fx));
        UInt32 int_py = UInt32(drjit::floor(fy));
        
        Float wx = fx - Float(int_px);
        Float wy = fy - Float(int_py);
        
        UInt32 int_px_1 = drjit::minimum(int_px + 1, UInt32(width - 1));
        UInt32 int_py_1 = drjit::minimum(int_py + 1, UInt32(height - 1));
        
        UInt32 idx_tl = int_py * UInt32(width) + int_px;
        UInt32 idx_tr = int_py * UInt32(width) + int_px_1;
        UInt32 idx_bl = int_py_1 * UInt32(width) + int_px;
        UInt32 idx_br = int_py_1 * UInt32(width) + int_px_1;
        
        Float w_tl = (Float(1.0f) - wx) * (Float(1.0f) - wy);
        Float w_tr = wx * (Float(1.0f) - wy);
        Float w_bl = (Float(1.0f) - wx) * wy;
        Float w_br = wx * wy;
        
        Float v_tl = drjit::select(valid, values * w_tl, Float(0.0f));
        Float v_tr = drjit::select(valid, values * w_tr, Float(0.0f));
        Float v_bl = drjit::select(valid, values * w_bl, Float(0.0f));
        Float v_br = drjit::select(valid, values * w_br, Float(0.0f));
        
        drjit::scatter_add(data, v_tl, idx_tl);
        drjit::scatter_add(data, v_tr, idx_tr);
        drjit::scatter_add(data, v_bl, idx_bl);
        drjit::scatter_add(data, v_br, idx_br);
        
        UInt32 cnt_tl = drjit::select(valid, UInt32(1), UInt32(0));
        drjit::scatter_add(hit_count, cnt_tl, idx_tl);
    }
    
    // ============= 打印信息 =============
    void print(const std::string& prefix = "") const {
        std::cout << prefix << "SensorData '" << sensor_name << "':" << std::endl;
        std::cout << prefix << "  resolution: " << width << " x " << height << std::endl;
        std::cout << prefix << "  total_hits: " << total_hits() << std::endl;
        std::cout << prefix << "  data_sum: " << drjit::sum(data)[0] << std::endl;
    }
};

// ============= 滤波器类型 =============
enum class FilterType {
    Box,
    Bilinear
};

// ============= IES 测光类型 =============
enum class IESType {
    TypeA = 0,
    TypeB = 1,
    TypeC = 2
};

// ============= 抽象基类 Sensor =============
struct Sensor {
    std::shared_ptr<Surface> surface;
    
    std::array<ScalarType,2> u_range;
    std::array<ScalarType,2> v_range;
    
    int width;
    int height;
    
    Float pixel_area;
    
    FilterType filter_type = FilterType::Box;
    
    virtual ~Sensor() = default;
    
    virtual std::string type_name() const = 0;
    
    bool has_surface() const { return surface != nullptr; }
    
    // Getters
    std::shared_ptr<Surface> get_surface() const { return surface; }
    std::array<ScalarType, 2> get_u_range() const { return u_range; }
    std::array<ScalarType, 2> get_v_range() const { return v_range; }
    int get_width() const { return width; }
    int get_height() const { return height; }
    FilterType get_filter_type() const { return filter_type; }
    Float get_pixel_area() const { return pixel_area; }
    
    // Setters
    void set_surface(std::shared_ptr<Surface> s) { surface = s; }
    void set_u_range(const std::array<ScalarType, 2>& range) { u_range = range; update_pixel_area(); }
    void set_v_range(const std::array<ScalarType, 2>& range) { v_range = range; update_pixel_area(); }
    void set_width(int w) { width = w; update_pixel_area(); }
    void set_height(int h) { height = h; update_pixel_area(); }
    void set_resolution(int w, int h) { width = w; height = h; update_pixel_area(); }
    void set_filter_type(FilterType type) { filter_type = type; }
    
    void set_filter(const std::string& type_name) {
        if (type_name == "Box" || type_name == "box") {
            filter_type = FilterType::Box;
        } else if (type_name == "Bilinear" || type_name == "bilinear") {
            filter_type = FilterType::Bilinear;
        }
    }
    
    std::string get_filter_name() const {
        switch (filter_type) {
            case FilterType::Box: return "Box";
            case FilterType::Bilinear: return "Bilinear";
            default: return "Unknown";
        }
    }
    
    // 创建对应的 SensorData
    SensorData create_data(const std::string& name) const {
        return SensorData(name, width, height);
    }
    
    // ============= 核心收集方法 =============
    virtual void collect(const Ray& ray, const SurfaceRecord& hit, 
                        SensorData& data, uint32_t N) const = 0;
    
    virtual void update_pixel_area() {
        size_t size = static_cast<size_t>(width) * static_cast<size_t>(height);
        float u_size = to_scalar(u_range[1]) - to_scalar(u_range[0]);
        float v_size = to_scalar(v_range[1]) - to_scalar(v_range[0]);
        float area = (u_size / static_cast<float>(width)) * (v_size / static_cast<float>(height));
        pixel_area = drjit::full<Float>(area, size);
    }
    
    virtual void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Resolution: " << width << " x " << height << std::endl;
        std::cout << "  U Range: [" << to_scalar(u_range[0]) << ", " << to_scalar(u_range[1]) << "]" << std::endl;
        std::cout << "  V Range: [" << to_scalar(v_range[0]) << ", " << to_scalar(v_range[1]) << "]" << std::endl;
        std::cout << "  Filter: " << get_filter_name() << std::endl;
        std::cout << "  Has Surface: " << (has_surface() ? "Yes" : "No") << std::endl;
    }
    void normalize_image(SensorData& data, uint32_t N) const {
        data.data = data.data / (Float(static_cast<float>(N)) * pixel_area);
    }
protected:
    //void init_defaults() {
    //    //u_range = Vector2(from_scalar(-5.0f), from_scalar(5.0f));
    //    //v_range = Vector2(from_scalar(-5.0f), from_scalar(5.0f));
    //    width = 256;
    //    height = 256;
    //    filter_type = FilterType::Bilinear;
    //    update_pixel_area();
    //}
    
    void uv_to_normalized(const Float& u, const Float& v, Float& px, Float& py) const {
        px = (u - u_range[0]) / (u_range[1] - u_range[0]);
        py = (v - v_range[0]) / (v_range[1] - v_range[0]);
    }
    
    UInt32 uv_to_pixel_index(const Float& u, const Float& v) const {
        Float px, py;
        uv_to_normalized(u, v, px, py);
        
        Float fx = drjit::clamp(px * Float(width), Float(0.0f), Float(width - 1));
        Float fy = drjit::clamp(py * Float(height), Float(0.0f), Float(height - 1));
        
        UInt32 int_px = UInt32(drjit::floor(fx));
        UInt32 int_py = UInt32(drjit::floor(fy));
        
        return int_py * UInt32(width) + int_px;
    }
    
    Mask uv_in_range(const Float& u, const Float& v) const {
        return (u >= u_range[0]) & (u <= u_range[1]) &
               (v >= v_range[0]) & (v <= v_range[1]);
    }
    
    void filter_accumulate(const Float& u, const Float& v, 
                           const Float& values, const Mask& valid,
                           SensorData& data) const {
        if (filter_type == FilterType::Box) {
            UInt32 pixel_idx = uv_to_pixel_index(u, v);
            data.accumulate(pixel_idx, values, valid);
        } else {
            Float px, py;
            uv_to_normalized(u, v, px, py);
            data.accumulate_bilinear(px, py, values, valid);
        }
    }
    
    
};

// ============= 辐照度传感器 =============
struct IrradianceSensor : public Sensor {
    IrradianceSensor() = delete;
    
    IrradianceSensor(std::shared_ptr<Surface> s, 
                     int w, int h,
                     FilterType filter) {
        if (!s) {
            throw std::invalid_argument("IrradianceSensor requires a valid Surface");
        }
        surface = s;
        width = w;
        height = h;
        u_range=surface->get_u_range();
        v_range=surface->get_v_range();
        filter_type = filter;
        update_pixel_area();
    }
    
    std::string type_name() const override { return "IrradianceSensor"; }
    
    void collect(const Ray& ray, const SurfaceRecord& hit, 
                SensorData& data, uint32_t N) const override {
        auto transform = surface->get_transform();
        auto local_pos = transform.inverse_transform_point(hit.position);
        Mask valid = hit.valid;
        
        Float u = local_pos[0];
        Float v = local_pos[1];
        valid &= uv_in_range(u, v);
        
        Float contribution = ray.radiance / ray.pdf;
        
        filter_accumulate(u, v, contribution, valid, data);
        //normalize_image(data, N);
    }
    
    void print(const std::string& name = "") const override {
        Sensor::print(name);
        if (surface) {
            std::cout << "  Surface Type: " << surface->type_name() << std::endl;
        }
    }
};

// ============= 强度传感器 =============
struct IntensitySensor : public Sensor {
    IESType ies_type;
    bool is_farfield;  // 新增：标记是否为远场模式
    
    // 近场构造函数（绑定Surface）
    IntensitySensor(std::shared_ptr<Surface> s,
                    IESType type,
                    int w, int h,
                    float u_min, float u_max,
                    float v_min, float v_max,
                    FilterType filter) {
        if (!s) {
            throw std::invalid_argument("Near-field IntensitySensor requires a valid Surface");
        }
        surface = s;
        ies_type = type;
        is_farfield = false;
        width = w;
        height = h;
        u_range = {u_min, u_max};
        v_range = { v_min, v_max };
        filter_type = filter;
        update_pixel_area();
    }
    
    // 远场构造函数（无Surface）
    IntensitySensor(IESType type,
                    int w, int h,
                    float u_min, float u_max,
                    float v_min, float v_max,
                    FilterType filter) {
        surface = nullptr;
        ies_type = type;
        is_farfield = true;
        width = w;
        height = h;
        u_range = { u_min, u_max };
        v_range = { v_min, v_max };
        filter_type = filter;
        update_pixel_area();
    }
    
    std::string type_name() const override { 
        std::string mode = is_farfield ? "FarField" : "NearField";
        return "IntensitySensor (" + get_ies_type_name() + ", " + mode + ")"; 
    }
    
    IESType get_ies_type() const { return ies_type; }
    std::string get_ies_type_name() const {
        switch (ies_type) {
            case IESType::TypeA: return "IES-A";
            case IESType::TypeB: return "IES-B";
            case IESType::TypeC: return "IES-C";
            default: return "Unknown";
        }
    }
    
    void set_ies_type(IESType type) { 
        ies_type = type; 
        update_pixel_area();
    }
    
     void collect(const Ray& ray, const SurfaceRecord& hit, 
                SensorData& data, uint32_t N) const override {
        Float u, v;
        Mask valid;
        
        if (is_farfield) {
            // 远场：从光线方向计算角度
            direction_to_ies_coords(ray.direction, u, v);
            valid = uv_in_range(u, v);
        } else {
            // 近场：从交点位置计算角度
            auto transform = surface->get_transform();
            auto local_pos = transform.inverse_transform_point(hit.position);
            u = local_pos[0];
            v = local_pos[1];
            valid = hit.valid & uv_in_range(u, v);
        }
        
        Float contribution = ray.radiance / ray.pdf;
        filter_accumulate(u, v, contribution, valid, data);
        //normalize_image(data, N);
    }
    
    void print(const std::string& name = "") const override {
        Sensor::print(name);
        std::cout << "  Mode: " << (is_farfield ? "Far Field" : "Near Field") << std::endl;
        std::cout << "  IES Type: " << get_ies_type_name() << std::endl;
    }
    
private:
    Float compute_solid_angle_array() const {
    size_t size = static_cast<size_t>(width) * static_cast<size_t>(height);
    
    float u_min = to_scalar(u_range[0]);
    float u_max = to_scalar(u_range[1]);
    float v_min = to_scalar(v_range[0]);
    float v_max = to_scalar(v_range[1]);
    
    float du = (u_max - u_min) / static_cast<float>(width);
    float dv = (v_max - v_min) / static_cast<float>(height);
    
    std::vector<float> solid_angles(size);
    
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            size_t idx = static_cast<size_t>(j) * static_cast<size_t>(width) + static_cast<size_t>(i);
            
            float u_center = u_min + (static_cast<float>(i) + 0.5f) * du;
            float v_center = v_min + (static_cast<float>(j) + 0.5f) * dv;
            
            solid_angles[idx] = compute_pixel_solid_angle_at(u_center, v_center, du, dv);
        }
    }
    
    return drjit::load<Float>(solid_angles.data(), size);
}

float compute_pixel_solid_angle_at(float u, float v, float du, float dv) const {
    constexpr float deg2rad = M_PI / 180.0f;
    
    float u_min_rad = (u - du * 0.5f) * deg2rad;
    float u_max_rad = (u + du * 0.5f) * deg2rad;
    float v_min_rad = (v - dv * 0.5f) * deg2rad;
    float v_max_rad = (v + dv * 0.5f) * deg2rad;
    
    float du_rad = du * deg2rad;
    float dv_rad = dv * deg2rad;
    
    float solid_angle = 0.0f;
    
    switch (ies_type) {
        case IESType::TypeA:
            solid_angle = du_rad * std::abs(std::sin(v_max_rad) - std::sin(v_min_rad));
            break;
        case IESType::TypeB:
            solid_angle = dv_rad * std::abs(std::sin(u_max_rad) - std::sin(u_min_rad));
            break;
        case IESType::TypeC:
            solid_angle = du_rad * std::abs(std::cos(v_min_rad) - std::cos(v_max_rad));
            break;
    }
    
    return std::max(solid_angle, 1e-10f);
}

void direction_to_ies_coords(const Vector3& dir, Float& u, Float& v) const {
    Float x = dir[0];
    Float y = dir[1];
    Float z = dir[2];
    
    constexpr float rad2deg = 180.0f / M_PI;
    
    switch (ies_type) {
        case IESType::TypeA:
            v = drjit::asin(drjit::clamp(y, Float(-1.0f), Float(1.0f))) * Float(rad2deg);
            u = drjit::atan2(-x, z) * Float(rad2deg);
            break;
        case IESType::TypeB:
            u = drjit::asin(drjit::clamp(-x, Float(-1.0f), Float(1.0f))) * Float(rad2deg);
            v = drjit::atan2(y, z) * Float(rad2deg);
            break;
        case IESType::TypeC: {
            Float theta = drjit::acos(drjit::clamp(z, Float(-1.0f), Float(1.0f)));
            Float phi = drjit::atan2(x, y);
            phi = drjit::select(phi < Float(0.0f), phi + Float(2.0f * M_PI), phi);
            u = phi * Float(rad2deg);
            v = theta * Float(rad2deg);
            break;
        }
    }
}
};

} // namespace diff_optics