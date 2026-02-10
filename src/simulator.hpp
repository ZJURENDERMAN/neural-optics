// simulator.hpp - 修复非序列化追踪逻辑
#pragma once
#include "scene.hpp"
#include "sensor.hpp"
#include "optix_scene.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

// 调试开关：取消注释以启用调试输出
// #define SIMULATOR_DEBUG

namespace diff_optics {

// ============= 序列化追踪配置 =============
struct SequenceConfig {
    std::string light_name;
    std::vector<std::string> surfaces_name;

    SequenceConfig() = default;
    
    SequenceConfig(const std::string& light, const std::vector<std::string>& surfaces)
        : light_name(light), surfaces_name(surfaces) {}
    
    SequenceConfig(const std::string& light, std::initializer_list<std::string> surfaces)
        : light_name(light), surfaces_name(surfaces) {}
};

// ============= 非序列化追踪配置 =============
struct NonSeqConfig {
    std::vector<std::string> light_names;
    std::vector<std::string> surface_names;
    
    NonSeqConfig() = default;
    
    NonSeqConfig(const std::vector<std::string>& lights, const std::vector<std::string>& surfaces)
        : light_names(lights), surface_names(surfaces) {}
    
    NonSeqConfig(std::initializer_list<std::string> lights, std::initializer_list<std::string> surfaces)
        : light_names(lights), surface_names(surfaces) {}
};

// ============= 仿真配置 =============
struct SimulatorConfig {
    size_t num_rays = 1000000;
    int max_depth = 10;
    ScalarType min_radiance = 0.0f;
    
    int random_type = 0;
    int seed = 0;
    int sim_type = 1;
    int trace_type = 0;
    
    bool use_optix = true;
    std::string ptx_dir = "";
    
    SequenceConfig seq_config;
    NonSeqConfig non_seq_config;
    
    SimulatorConfig() = default;
    
    SimulatorConfig(size_t rays, int depth, ScalarType min_rad)
        : num_rays(rays), max_depth(depth), min_radiance(min_rad) {}
    
    void set_sequence(const std::string& light_name, 
                      const std::vector<std::string>& surfaces) {
        seq_config.light_name = light_name;
        seq_config.surfaces_name = surfaces;
        sim_type = 0;
    }
    
    void set_non_sequential(const std::vector<std::string>& light_names,
                            const std::vector<std::string>& surface_names) {
        non_seq_config.light_names = light_names;
        non_seq_config.surface_names = surface_names;
        sim_type = 1;
    }
    
    void set_optix(bool enable, const std::string& ptx_path = "") {
        use_optix = enable;
        ptx_dir = ptx_path;
    }
    
    void print(const std::string& name = "SimulatorConfig") const {
        std::cout << name << ":" << std::endl;
        std::cout << "  num_rays: " << num_rays << std::endl;
        std::cout << "  max_depth: " << max_depth << std::endl;
        std::cout << "  min_radiance: " << min_radiance << std::endl;
        std::cout << "  sim_type: " << (sim_type == 0 ? "sequential" : "non-sequential") << std::endl;
        std::cout << "  trace_type: " << trace_type << std::endl;
        std::cout << "  use_optix: " << (use_optix ? "true" : "false") << std::endl;
        if (!ptx_dir.empty()) {
            std::cout << "  ptx_dir: " << ptx_dir << std::endl;
        }
        if (sim_type == 1) {
            std::cout << "  non_seq_config:" << std::endl;
            std::cout << "    lights: " << non_seq_config.light_names.size() << std::endl;
            std::cout << "    surfaces: " << non_seq_config.surface_names.size() << std::endl;
        }
    }
};

// ============= 仿真结果 =============
struct SimulationResult {
    bool success = false;
    std::string error_message;

    std::unordered_map<std::string, SensorData> sensor_data;
    
    // 出射光线（用于准直优化）
    Ray exit_rays;
    bool has_exit_rays = false;

    double optix_build_time_ms = 0.0;
    double optix_trace_time_ms = 0.0;
    double total_time_ms = 0.0;
    
    SimulationResult() = default;
    
    bool has_sensor_data(const std::string& name) const {
        return sensor_data.find(name) != sensor_data.end();
    }
    
    SensorData& get_sensor_data(const std::string& name) {
        auto it = sensor_data.find(name);
        if (it == sensor_data.end()) {
            throw std::runtime_error("SensorData '" + name + "' not found");
        }
        return it->second;
    }
    
    const SensorData& get_sensor_data(const std::string& name) const {
        auto it = sensor_data.find(name);
        if (it == sensor_data.end()) {
            throw std::runtime_error("SensorData '" + name + "' not found");
        }
        return it->second;
    }
    
    std::vector<std::string> get_sensor_names() const {
        std::vector<std::string> names;
        names.reserve(sensor_data.size());
        for (const auto& [name, _] : sensor_data) {
            names.push_back(name);
        }
        return names;
    }
    
    void print(const std::string& name = "SimulationResult") const {
        std::cout << name << ":" << std::endl;
        std::cout << "  success: " << (success ? "true" : "false") << std::endl;
        if (!success && !error_message.empty()) {
            std::cout << "  error: " << error_message << std::endl;
        }
        for (const auto& [sname, sdata] : sensor_data) {
            sdata.print("    ");
        }
        if (optix_build_time_ms > 0 || optix_trace_time_ms > 0) {
            std::cout << "  timing:" << std::endl;
            std::cout << "    optix_build: " << optix_build_time_ms << " ms" << std::endl;
            std::cout << "    optix_trace: " << optix_trace_time_ms << " ms" << std::endl;
            std::cout << "    total: " << total_time_ms << " ms" << std::endl;
        }
    }
};

// ============= 抽象基类 Simulator =============
struct Simulator {
public:
    virtual ~Simulator() = default;
    
    virtual SimulationResult simulate(
        const Scene& scene,
        const SimulatorConfig& config
    ) = 0;
    
protected:
    std::shared_ptr<RNG> rng;
    
    bool m_optix_initialized = false;
    bool m_optix_available = false;
};

// ============= 正向仿真器 =============
struct ForwardSimulator : public Simulator {
    
    ForwardSimulator() = default;
    
    SimulationResult simulate(
        const Scene& scene,
        const SimulatorConfig& config
    ) override {
        rng = std::make_shared<RNG>(config.num_rays, config.seed);
        
        if (config.use_optix && !m_optix_initialized) {
            init_optix(config.ptx_dir);
        }
        if (config.sim_type == 0) {
            return trace_sequential(scene, config);
        } else {
            return trace_non_sequential(scene, config);
        }
    }
    
private:
    void init_optix(const std::string& ptx_dir) {
        m_optix_initialized = true;
        
        std::string actual_ptx_dir = ptx_dir;
        if (actual_ptx_dir.empty()) {
            #ifdef PTX_DIR
            actual_ptx_dir = PTX_DIR;
            #else
            actual_ptx_dir = ".";
            #endif
        }
        
        try {
            m_optix_available = init_optix_manager(actual_ptx_dir);
        } catch (const std::exception& e) {
            std::cerr << "[ForwardSimulator] OptiX init error: " << e.what() << std::endl;
            m_optix_available = false;
        }
    }
    
    // ============= 单表面相交（序列化追踪用）=============
    SurfaceRecord intersect_with_optix(
        const Ray& ray,
        const std::string& surface_name,
        const std::shared_ptr<Surface>& surface,
        const Float& tmax
    ) {
        size_t ray_count = ray.size();
        
        if (ray_count == 0) {
            SurfaceRecord record;
            record.valid = Mask(false);
            return record;
        }
        
        auto optix_result = optix_trace_single_surface(ray, surface_name);
        
        SurfaceRecord record;
        record.init(static_cast<int>(ray_count));
        
        Mask hit_valid = optix_result.valid & (optix_result.t > Float(0.0f)) & (optix_result.t < tmax);
        record.valid = hit_valid;

        if (drjit::any(hit_valid)) {
            Float t_hit = drjit::select(hit_valid, Float(optix_result.t), Float(tmax));
            surface->compute_surface_record_from_t(ray, t_hit, record);
        }
        
        return record;
    }
    
    // ============= 光线合并辅助函数 =============
    
    Float concat_floats(const std::vector<Float>& arrays) {
        if (arrays.empty()) {
            return Float();
        }
        if (arrays.size() == 1) {
            return arrays[0];
        }
        
        std::vector<Float> current = arrays;
        while (current.size() > 1) {
            std::vector<Float> next;
            for (size_t i = 0; i < current.size(); i += 2) {
                if (i + 1 < current.size()) {
                    next.push_back(drjit::concat(current[i], current[i + 1]));
                } else {
                    next.push_back(current[i]);
                }
            }
            current = std::move(next);
        }
        return current[0];
    }
    
    Mask concat_masks(const std::vector<Mask>& arrays) {
        if (arrays.empty()) {
            return Mask();
        }
        if (arrays.size() == 1) {
            return arrays[0];
        }
        
        std::vector<Mask> current = arrays;
        while (current.size() > 1) {
            std::vector<Mask> next;
            for (size_t i = 0; i < current.size(); i += 2) {
                if (i + 1 < current.size()) {
                    next.push_back(drjit::concat(current[i], current[i + 1]));
                } else {
                    next.push_back(current[i]);
                }
            }
            current = std::move(next);
        }
        return current[0];
    }
    
    Ray concat_rays(const std::vector<Ray>& rays) {
        if (rays.empty()) {
            return Ray();
        }
        if (rays.size() == 1) {
            return rays[0];
        }
        
        std::vector<Float> ox, oy, oz, dx, dy, dz, wavelength, radiance, pdf;
        for (const auto& r : rays) {
            if (r.size() == 0) continue;
            ox.push_back(r.origin[0]);
            oy.push_back(r.origin[1]);
            oz.push_back(r.origin[2]);
            dx.push_back(r.direction[0]);
            dy.push_back(r.direction[1]);
            dz.push_back(r.direction[2]);
            wavelength.push_back(r.wavelength);
            radiance.push_back(r.radiance);
            pdf.push_back(r.pdf);
        }
        
        if (ox.empty()) {
            return Ray();
        }
        
        Ray result;
        result.origin[0] = concat_floats(ox);
        result.origin[1] = concat_floats(oy);
        result.origin[2] = concat_floats(oz);
        result.direction[0] = concat_floats(dx);
        result.direction[1] = concat_floats(dy);
        result.direction[2] = concat_floats(dz);
        result.wavelength = concat_floats(wavelength);
        result.radiance = concat_floats(radiance);
        result.pdf = concat_floats(pdf);
        
        return result;
    }
    
    // ============= 非序列化追踪辅助结构 =============
    
    struct RayHitInfo {
        Float t;
        Int32 surface_id;
        Mask valid;
        
        RayHitInfo(size_t n = 0) {
            if (n > 0) {
                t = drjit::full<Float>(1e10f, n);
                surface_id = drjit::full<Int32>(-1, n);
                valid = drjit::full<Mask>(false, n);
            }
        }
    };
    
    RayHitInfo intersect_all_surfaces_optix(
        const Ray& ray,
        const Float& tmax
    ) {
        size_t ray_count = ray.size();
        RayHitInfo result(ray_count);
        
        if (ray_count == 0) {
            return result;
        }
        
        auto optix_result = optix_trace_rays(ray);
        
        result.valid = optix_result.valid & (optix_result.t > Float(0.0f)) & (optix_result.t < tmax);
        result.t = drjit::select(result.valid, optix_result.t, drjit::full<Float>(1e10f, ray_count));
        result.surface_id = drjit::select(result.valid, optix_result.surface_id, drjit::full<Int32>(-1, ray_count));
        
        return result;
    }
    
    std::vector<std::string> get_active_lights(const Scene& scene, const NonSeqConfig& config) {
        if (config.light_names.empty()) {
            return scene.get_light_names();
        }
        return config.light_names;
    }
    
    std::vector<std::string> get_active_surfaces(const Scene& scene, const NonSeqConfig& config) {
        if (config.surface_names.empty()) {
            return scene.get_surface_names();
        }
        return config.surface_names;
    }
    
    Ray sample_rays_from_lights(
        const Scene& scene,
        const std::vector<std::string>& light_names,
        size_t total_rays
    ) {
        size_t num_lights = light_names.size();
        if (num_lights == 0) {
            return Ray();
        }
        
        size_t rays_per_light = total_rays / num_lights;
        size_t remaining_rays = total_rays % num_lights;
        
        std::vector<Ray> all_rays;
        all_rays.reserve(num_lights);
        
        for (size_t i = 0; i < num_lights; ++i) {
            size_t num_rays_for_this_light = rays_per_light + (i < remaining_rays ? 1 : 0);
            
            if (num_rays_for_this_light == 0) continue;
            
            auto light = scene.get_light(light_names[i]);
            Ray light_rays = light->sampleRays(
                rng->random2d(),
                rng->random2d(),
                rng->random1d()
            );
            
            all_rays.push_back(light_rays);
        }
        
        return concat_rays(all_rays);
    }
    
    // ============= 非序列化追踪主逻辑 =============
    
    SimulationResult trace_non_sequential(
        const Scene& scene,
        const SimulatorConfig& config
    ) {
        SimulationResult result;
        result.success = true;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        init_sensor_data(scene, result);
        
        auto active_lights = get_active_lights(scene, config.non_seq_config);
        auto active_surfaces = get_active_surfaces(scene, config.non_seq_config);
        
        if (active_lights.empty()) {
            result.success = false;
            result.error_message = "No active lights found";
            return result;
        }
        
        for (const auto& light_name : active_lights) {
            if (!scene.has_light(light_name)) {
                result.success = false;
                result.error_message = "Light '" + light_name + "' not found";
                return result;
            }
        }
        
        for (const auto& surface_name : active_surfaces) {
            if (!scene.has_surface(surface_name)) {
                result.success = false;
                result.error_message = "Surface '" + surface_name + "' not found";
                return result;
            }
        }
        
        std::unordered_set<std::string> active_surface_set(active_surfaces.begin(), active_surfaces.end());
        
        bool use_optix_for_trace = config.use_optix && m_optix_available;
        
        if (use_optix_for_trace) {
            auto build_start = std::chrono::high_resolution_clock::now();
            
            if (!build_optix_scene(scene)) {
                use_optix_for_trace = false;
            }
            
            auto build_end = std::chrono::high_resolution_clock::now();
            result.optix_build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
        }
        
        if (!use_optix_for_trace) {
            result.success = false;
            result.error_message = "Non-sequential tracing requires OptiX acceleration";
            return result;
        }
        
        Ray ray = sample_rays_from_lights(scene, active_lights, config.num_rays);

        auto trace_start = std::chrono::high_resolution_clock::now();
        
        auto farfield_sensors = scene.get_farfield_sensors();
        
        for (int depth = 0; depth < config.max_depth && ray.size() > 0; ++depth) {
            size_t current_ray_count = ray.size();
            
#ifdef SIMULATOR_DEBUG
            std::cout << "[DEBUG] Depth " << depth << ", Active rays: " << current_ray_count << std::endl;
#endif
            
            Float tmax = drjit::full<Float>(1e10f, current_ray_count);
            
            RayHitInfo hit_info = intersect_all_surfaces_optix(ray, tmax);
            
            // 确保数据同步
            drjit::eval(hit_info.valid, hit_info.surface_id, hit_info.t);
            drjit::sync_thread();
            
            Mask miss_mask = !hit_info.valid;
            Mask hit_mask = hit_info.valid;
            
            // 处理未命中的光线（发送到远场传感器）
            if (drjit::any(miss_mask)) {
                Ray miss_rays = ray.masked_select(miss_mask);
                
                for (const auto& sensor_name : farfield_sensors) {
                    auto& sensor_data = result.sensor_data[sensor_name];
                    auto sensor = scene.get_sensor(sensor_name);
                    sensor->collect(miss_rays, SurfaceRecord(), sensor_data, config.num_rays);
                }
            }
            
            // 没有命中任何表面，结束追踪
            if (!drjit::any(hit_mask)) {
                break;
            }
            
            // 收集下一轮的光线
            std::vector<Ray> next_rays;
            
            // 按表面处理命中的光线
            for (const auto& surface_name : active_surfaces) {
                auto surface = scene.get_surface(surface_name);
                int surface_id = get_optix_manager()->get_surface_id(surface_name);
                
                if (surface_id < 0) continue;
                
                // 找出命中当前表面的光线
                Mask surface_hit_mask = hit_mask & (hit_info.surface_id == Int32(surface_id));
                
                if (!drjit::any(surface_hit_mask)) continue;
                
                // 压缩光线数据：只保留命中当前表面的光线
                Ray surface_rays = ray.masked_select(surface_hit_mask);
                
                if (surface_rays.size() == 0) continue;
                
                // 同步压缩 t 值
                Float surface_t = utils::mask_select(hit_info.t, surface_hit_mask);
                
#ifdef SIMULATOR_DEBUG
                std::cout << "[DEBUG]   Surface: " << surface_name 
                          << ", Hits: " << surface_rays.size() << std::endl;
#endif
                
                // 计算表面记录
                SurfaceRecord surface_hit;
                surface_hit.init(static_cast<int>(surface_rays.size()));
                surface_hit.valid = drjit::full<Mask>(true, surface_rays.size());
                surface->compute_surface_record_from_t(surface_rays, surface_t, surface_hit);
                
                // 收集传感器数据
                auto sensors = scene.get_sensors_for_surface(surface_name);
                for (const auto& sensor_name : sensors) {
                    // std::cout<<"collect on"<<surface_name<<std::endl;
                    auto& sensor_data = result.sensor_data[sensor_name];
                    auto sensor = scene.get_sensor(sensor_name);
                    sensor->collect(surface_rays, surface_hit, sensor_data, config.num_rays);
                }
                
                // 采样新的出射光线
                Vector2 uv_sample = rng->random2d();
                // 需要调整随机数尺寸以匹配压缩后的光线数量
                if (drjit::width(uv_sample[0]) != surface_rays.size()) {
                    // 重新采样匹配尺寸的随机数
                    // 这里使用简单方法：从现有随机数中选取
                    uv_sample = Vector2(
                        utils::mask_select(uv_sample[0], surface_hit_mask),
                        utils::mask_select(uv_sample[1], surface_hit_mask)
                    );
                }
                
                Ray new_rays = surface->sample_ray(surface_rays, surface_hit, uv_sample);
                
                if (new_rays.size() > 0) {
                    // 过滤低能量光线
                    Mask valid_mask = new_rays.radiance > Float(config.min_radiance);
                    
                    if (drjit::any(valid_mask)) {
                        Ray valid_rays = new_rays.masked_select(valid_mask);
                        if (valid_rays.size() > 0) {
                            next_rays.push_back(valid_rays);
                        }
                    }
                }
            }
            
            // 合并所有下一轮光线
            ray = concat_rays(next_rays);
            
#ifdef SIMULATOR_DEBUG
            std::cout << "[DEBUG]   Next round rays: " << ray.size() << std::endl;
#endif
        }
        
        // 处理最终未命中的光线
        if (ray.size() > 0) {
            result.exit_rays = ray;
            result.has_exit_rays = true;
            
            for (const auto& sensor_name : farfield_sensors) {
                auto& sensor_data = result.sensor_data[sensor_name];
                auto sensor = scene.get_sensor(sensor_name);
                sensor->collect(ray, SurfaceRecord(), sensor_data, config.num_rays);
            }
        }
        // 在 trace_non_sequential 返回前添加
        for (auto& [sensor_name, sensor_data] : result.sensor_data) {
            auto sensor = scene.get_sensor(sensor_name);
            if (sensor) {
                sensor->normalize_image(sensor_data, config.num_rays);
            }
        }
        auto trace_end = std::chrono::high_resolution_clock::now();
        result.optix_trace_time_ms = std::chrono::duration<double, std::milli>(trace_end - trace_start).count();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        return result;
    }
    
    // ============= 序列化追踪 =============
    
    SimulationResult trace_sequential(
        const Scene& scene,
        const SimulatorConfig& config
    ) {
        SimulationResult result;
        result.success = true;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        init_sensor_data(scene, result);
        
        if (!scene.has_light(config.seq_config.light_name)) {
            result.success = false;
            result.error_message = "Light '" + config.seq_config.light_name + "' not found";
            return result;
        }
        
        for (const auto& surface_name : config.seq_config.surfaces_name) {
            if (!scene.has_surface(surface_name)) {
                result.success = false;
                result.error_message = "Surface '" + surface_name + "' not found in sequence";
                return result;
            }
        }
        
        bool use_optix_for_trace = config.use_optix && m_optix_available;
        
        if (use_optix_for_trace) {
            auto build_start = std::chrono::high_resolution_clock::now();
            
            if (!build_optix_scene(scene)) {
                use_optix_for_trace = false;
            }
            
            auto build_end = std::chrono::high_resolution_clock::now();
            result.optix_build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
        }
        
        auto light = scene.get_light(config.seq_config.light_name);
        auto ray = light->sampleRays(rng->random2d(), rng->random2d(), rng->random1d());
        
        Float tmax = drjit::full<Float>(10000.0f, ray.size());
        
        auto trace_start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < config.seq_config.surfaces_name.size(); i++) {
            const auto& surface_name = config.seq_config.surfaces_name[i];
            auto surface = scene.get_surface(surface_name);
            
            size_t current_ray_count = ray.size();
            
            if (current_ray_count == 0) {
                break;
            }
            
            size_t tmax_count = drjit::width(tmax);
            if (tmax_count != current_ray_count) {
                tmax = drjit::full<Float>(10000.0f, current_ray_count);
            }
            
            SurfaceRecord hit;
            if (use_optix_for_trace) {
                hit = intersect_with_optix(ray, surface_name, surface, tmax);
            }
            
#ifdef SIMULATOR_DEBUG
            std::cout << "[DEBUG] Surface: " << surface_name 
                      << ", Rays: " << current_ray_count << std::endl;
#endif
            
            auto sensors = scene.get_sensors_for_surface(surface_name);
            for (size_t s = 0; s < sensors.size(); s++) {
                auto& sensor_data = result.sensor_data[sensors[s]];
                auto sensor = scene.get_sensor(sensors[s]);
                sensor->collect(ray, hit, sensor_data, config.num_rays);
            }
            
            ray = surface->sample_ray(ray, hit, rng->random2d());
            
            if (ray.size() == 0) {
                break;
            }
            
            tmax = drjit::full<Float>(10000.0f, ray.size());
        }
        
        auto trace_end = std::chrono::high_resolution_clock::now();
        result.optix_trace_time_ms = std::chrono::duration<double, std::milli>(trace_end - trace_start).count();
        
        if (ray.size() > 0) {
            result.exit_rays = ray;
            result.has_exit_rays = (ray.size() > 0);
            
            auto farfield_sensors = scene.get_farfield_sensors();
            for (size_t i = 0; i < farfield_sensors.size(); i++) {
                auto& sensor_data = result.sensor_data[farfield_sensors[i]];
                auto sensor = scene.get_sensor(farfield_sensors[i]);
                sensor->collect(ray, SurfaceRecord(), sensor_data, config.num_rays);
            }
        }
        
        // 在 trace_non_sequential 返回前添加
        for (auto& [sensor_name, sensor_data] : result.sensor_data) {
            auto sensor = scene.get_sensor(sensor_name);
            if (sensor) {
                sensor->normalize_image(sensor_data, config.num_rays);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        return result;
    }
    
    void init_sensor_data(const Scene& scene, SimulationResult& result) {
        for (const auto& sensor_name : scene.get_sensor_names()) {
            auto sensor = scene.get_sensor(sensor_name);
            if (sensor) {
                result.sensor_data.emplace(
                    sensor_name,
                    SensorData(sensor_name, sensor->get_width(), sensor->get_height())
                );
            }
        }
    }
};

} // namespace diff_optics