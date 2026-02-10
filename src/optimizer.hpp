// optimizer.hpp - 修复后的完整版本

// optimizer.hpp - 优化器，管理金字塔配置、Adam优化和迭代
#pragma once
#include "simulator.hpp"
#include <functional>
#include <cmath>
#include <iomanip>

namespace diff_optics {

// 调试宏 - 可以通过定义 OPTIM_DEBUG 来启用详细调试
#ifndef OPTIM_DEBUG
#define OPTIM_DEBUG 1  // 默认开启，发布时设为 0
#endif

#if OPTIM_DEBUG
#define OPTIM_LOG(msg) std::cout << "[Optimizer] " << msg << std::endl
#define OPTIM_LOG_VERBOSE(msg) std::cout << "[Optimizer::Debug] " << msg << std::endl
#else
#define OPTIM_LOG(msg)
#define OPTIM_LOG_VERBOSE(msg)
#endif

// ============= Adam 优化器配置 =============
struct AdamConfig {
    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.0f;
    bool amsgrad = false;
    
    AdamConfig() = default;
    AdamConfig(float learning_rate) : lr(learning_rate) {}
    
    AdamConfig& set_lr(float learning_rate) { lr = learning_rate; return *this; }
    AdamConfig& set_betas(float b1, float b2) { beta1 = b1; beta2 = b2; return *this; }
    AdamConfig& set_weight_decay(float wd) { weight_decay = wd; return *this; }
    AdamConfig& set_amsgrad(bool use_amsgrad) { amsgrad = use_amsgrad; return *this; }
    
    void print(const std::string& prefix = "") const {
        std::cout << prefix << "AdamConfig: lr=" << lr 
                  << ", beta1=" << beta1 << ", beta2=" << beta2
                  << ", eps=" << epsilon << ", wd=" << weight_decay 
                  << ", amsgrad=" << (amsgrad ? "true" : "false") << std::endl;
    }
};

// ============= Adam 优化器状态（单个参数） =============
struct AdamState {
    Float m;
    Float v;
    Float v_max;
    int step = 0;
    
    AdamState() = default;
    
    void init(size_t size) {
        OPTIM_LOG_VERBOSE("AdamState::init size=" << size);
        m = drjit::zeros<Float>(size);
        v = drjit::zeros<Float>(size);
        v_max = drjit::zeros<Float>(size);
        step = 0;
    }
    
    void reset() {
        size_t size = drjit::width(m);
        if (size > 0) {
            OPTIM_LOG_VERBOSE("AdamState::reset size=" << size);
            m = drjit::zeros<Float>(size);
            v = drjit::zeros<Float>(size);
            v_max = drjit::zeros<Float>(size);
        }
        step = 0;
    }
};

// ============= 金字塔配置 =============
struct PyramidConfig {
    bool enabled = false;
    int num_levels = 4;
    PyramidScheduleType schedule_type = PyramidScheduleType::Exponential;
    std::vector<int> custom_iterations;
    
    PyramidConfig() = default;
    
    PyramidConfig& set_levels(int levels) { num_levels = levels; return *this; }
    PyramidConfig& set_schedule(PyramidScheduleType schedule) { schedule_type = schedule; return *this; }
    PyramidConfig& set_custom_iterations(const std::vector<int>& iters) {
        custom_iterations = iters;
        schedule_type = PyramidScheduleType::Custom;
        return *this;
    }
    
    void print(const std::string& prefix = "") const {
        std::cout << prefix << "PyramidConfig: enabled=" << (enabled ? "true" : "false");
        if (enabled) {
            std::cout << ", levels=" << num_levels << ", schedule=";
            switch (schedule_type) {
                case PyramidScheduleType::Linear: std::cout << "Linear"; break;
                case PyramidScheduleType::Exponential: std::cout << "Exponential"; break;
                case PyramidScheduleType::Custom: std::cout << "Custom"; break;
            }
        }
        std::cout << std::endl;
    }
};

// ============= 回调配置 =============
struct CallbackConfig {
    int save_interval = 10;
    int print_interval = 1;
    std::string save_dir = ".";
    std::string save_prefix = "result";
    bool save_on_level_change = true;
    bool save_final = true;
    bool verbose = true;
    
    CallbackConfig() = default;
    
    CallbackConfig& set_save_interval(int interval) { save_interval = interval; return *this; }
    CallbackConfig& set_print_interval(int interval) { print_interval = interval; return *this; }
    CallbackConfig& set_save_dir(const std::string& dir) { save_dir = dir; return *this; }
    CallbackConfig& set_save_prefix(const std::string& prefix) { save_prefix = prefix; return *this; }
    CallbackConfig& set_verbose(bool v) { verbose = v; return *this; }
    
    void print(const std::string& prefix = "") const {
        std::cout << prefix << "CallbackConfig: save_interval=" << save_interval
                  << ", print_interval=" << print_interval
                  << ", save_dir=" << save_dir
                  << ", verbose=" << (verbose ? "true" : "false") << std::endl;
    }
};

// ============= 损失函数类型 =============
enum class LossType { L1, L2, L1_L2, Custom };

// ============= 损失配置 =============
struct LossConfig {
    LossType type = LossType::L1;
    float l2_weight = 0.1f;
    bool normalize = true;
    
    LossConfig() = default;
    
    LossConfig& set_type(LossType t) { type = t; return *this; }
    LossConfig& set_l2_weight(float w) { l2_weight = w; return *this; }
    LossConfig& set_normalize(bool n) { normalize = n; return *this; }
    
    void print(const std::string& prefix = "") const {
        std::cout << prefix << "LossConfig: type=";
        switch (type) {
            case LossType::L1: std::cout << "L1"; break;
            case LossType::L2: std::cout << "L2"; break;
            case LossType::L1_L2: std::cout << "L1_L2(w=" << l2_weight << ")"; break;
            case LossType::Custom: std::cout << "Custom"; break;
        }
        std::cout << ", normalize=" << (normalize ? "true" : "false") << std::endl;
    }
};

// ============= 优化器配置 =============
struct OptimizerConfig {
    int total_iterations = 100;
    AdamConfig adam;
    PyramidConfig pyramid;
    CallbackConfig callback;
    LossConfig loss;
    
    OptimizerConfig() = default;
    explicit OptimizerConfig(int iterations) : total_iterations(iterations) {}
    
    OptimizerConfig& set_iterations(int iters) { total_iterations = iters; return *this; }
    OptimizerConfig& set_lr(float lr) { adam.lr = lr; return *this; }
    OptimizerConfig& set_adam(const AdamConfig& cfg) { adam = cfg; return *this; }
    OptimizerConfig& enable_pyramid(int num_levels, PyramidScheduleType schedule = PyramidScheduleType::Exponential) {
        pyramid.enabled = true;
        pyramid.num_levels = num_levels;
        pyramid.schedule_type = schedule;
        return *this;
    }
    OptimizerConfig& disable_pyramid() { pyramid.enabled = false; return *this; }
    OptimizerConfig& set_pyramid(const PyramidConfig& cfg) { pyramid = cfg; return *this; }
    OptimizerConfig& set_callback(const CallbackConfig& cfg) { callback = cfg; return *this; }
    OptimizerConfig& set_loss(const LossConfig& cfg) { loss = cfg; return *this; }
    OptimizerConfig& set_loss_type(LossType type) { loss.type = type; return *this; }
    
    void print(const std::string& name = "OptimizerConfig") const {
        std::cout << name << ":" << std::endl;
        std::cout << "  total_iterations: " << total_iterations << std::endl;
        adam.print("  ");
        pyramid.print("  ");
        callback.print("  ");
        loss.print("  ");
    }
};

// ============= 优化参数 =============
struct OptimParam {
    std::string name;
    Float* data;
    AdamState adam_state;
    bool enabled = true;
    bool has_lower_bound = false;
    bool has_upper_bound = false;
    float lower_bound = 0.0f;
    float upper_bound = 1.0f;
    
    OptimParam() : data(nullptr) {}
    
    OptimParam(const std::string& n, Float* d) : name(n), data(d) {
        if (data) {
            size_t size = drjit::width(*data);
            OPTIM_LOG("OptimParam '" << n << "' created with size=" << size);
            if (size > 0) {
                adam_state.init(size);
                drjit::enable_grad(*data);
                OPTIM_LOG_VERBOSE("  Gradient enabled for '" << n << "'");
            } else {
                OPTIM_LOG("  WARNING: Parameter '" << n << "' has zero size!");
            }
        } else {
            OPTIM_LOG("  WARNING: Parameter '" << n << "' has null data pointer!");
        }
    }
    
    void set_bounds(float lower, float upper) {
        has_lower_bound = true; has_upper_bound = true;
        lower_bound = lower; upper_bound = upper;
    }
    void set_lower_bound(float lower) { has_lower_bound = true; lower_bound = lower; }
    void set_upper_bound(float upper) { has_upper_bound = true; upper_bound = upper; }
    
    void apply_constraints() {
        if (!data) return;
        if (has_lower_bound && has_upper_bound) {
            *data = drjit::clamp(*data, Float(lower_bound), Float(upper_bound));
        } else if (has_lower_bound) {
            *data = drjit::maximum(*data, Float(lower_bound));
        } else if (has_upper_bound) {
            *data = drjit::minimum(*data, Float(upper_bound));
        }
    }
    
    size_t size() const { return data ? drjit::width(*data) : 0; }
};

// ============= 优化状态 =============
struct OptimizationState {
    int iteration = 0;
    int current_level = 0;
    bool level_changed = false;
    bool initialized = false;
    bool running = false;
    bool completed = false;
    
    std::unordered_map<std::string, SensorData> sensor_data;
    std::unordered_map<std::string, SensorData> target_data;
    std::unordered_map<std::string, OptimParam> params;
    std::vector<float> loss_history;
    
    double last_sim_time_ms = 0.0;
    double total_sim_time_ms = 0.0;
    double last_loss = 0.0;
    double best_loss = std::numeric_limits<double>::max();
    int best_iteration = 0;
    
    OptimizationState() = default;
    
    void reset() {
        OPTIM_LOG("OptimizationState::reset()");
        iteration = 0;
        current_level = 0;
        level_changed = false;
        initialized = false;
        running = false;
        completed = false;
        sensor_data.clear();
        for (auto& [_, param] : params) {
            param.adam_state.reset();
        }
        loss_history.clear();
        last_sim_time_ms = 0.0;
        total_sim_time_ms = 0.0;
        last_loss = 0.0;
        best_loss = std::numeric_limits<double>::max();
        best_iteration = 0;
    }
    
    bool has_sensor(const std::string& name) const { return sensor_data.find(name) != sensor_data.end(); }
    SensorData& get_sensor(const std::string& name) {
        auto it = sensor_data.find(name);
        if (it == sensor_data.end()) throw std::runtime_error("SensorData '" + name + "' not found");
        return it->second;
    }
    const SensorData& get_sensor(const std::string& name) const {
        auto it = sensor_data.find(name);
        if (it == sensor_data.end()) throw std::runtime_error("SensorData '" + name + "' not found");
        return it->second;
    }
    
    bool has_target(const std::string& name) const { return target_data.find(name) != target_data.end(); }
    SensorData& get_target(const std::string& name) {
        auto it = target_data.find(name);
        if (it == target_data.end()) throw std::runtime_error("Target '" + name + "' not found");
        return it->second;
    }
    const SensorData& get_target(const std::string& name) const {
        auto it = target_data.find(name);
        if (it == target_data.end()) throw std::runtime_error("Target '" + name + "' not found");
        return it->second;
    }
    
    bool has_param(const std::string& name) const { return params.find(name) != params.end(); }
    OptimParam& get_param(const std::string& name) {
        auto it = params.find(name);
        if (it == params.end()) throw std::runtime_error("Param '" + name + "' not found");
        return it->second;
    }
    const OptimParam& get_param(const std::string& name) const {
        auto it = params.find(name);
        if (it == params.end()) throw std::runtime_error("Param '" + name + "' not found");
        return it->second;
    }
    
    std::vector<std::string> get_sensor_names() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : sensor_data) names.push_back(name);
        return names;
    }
    std::vector<std::string> get_target_names() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : target_data) names.push_back(name);
        return names;
    }
    std::vector<std::string> get_param_names() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : params) names.push_back(name);
        return names;
    }
    
    void clear_current_level_data() {
        for (auto& [_, data] : sensor_data) {
            data.clear_current_level();
        }
    }
    
    void print(const std::string& prefix = "") const {
        std::cout << prefix << "OptimizationState:" << std::endl;
        std::cout << prefix << "  initialized=" << initialized << ", running=" << running 
                  << ", completed=" << completed << std::endl;
        std::cout << prefix << "  iteration=" << iteration << ", level=" << current_level << std::endl;
        std::cout << prefix << "  sensors=" << sensor_data.size() << ", targets=" << target_data.size()
                  << ", params=" << params.size() << std::endl;
        std::cout << prefix << "  last_loss=" << last_loss << ", best_loss=" << best_loss 
                  << " (iter " << best_iteration << ")" << std::endl;
    }
};

// ============= 回调函数类型 =============
using IterationCallback = std::function<void(int iteration, float loss, const OptimizationState& state)>;
using SaveCallback = std::function<void(const std::string& sensor_name, const SensorData& data, const std::string& filepath)>;

// ============= 优化器 =============
struct Optimizer {
    std::shared_ptr<Simulator> simulator;
    OptimizerConfig config;
    OptimizationState state;
    
    IterationCallback on_iteration;
    SaveCallback on_save;
    std::function<void(const OptimizationState&)> on_complete;
    std::function<void(int old_level, int new_level)> on_level_change;
    std::function<Float(const OptimizationState&)> custom_loss_fn;
    
    // 场景引用，用于更新 BSpline
    const Scene* m_scene = nullptr;
    
    Optimizer() : simulator(std::make_shared<ForwardSimulator>()) {
        OPTIM_LOG("Optimizer created with default ForwardSimulator");
    }
    
    explicit Optimizer(std::shared_ptr<Simulator> sim) 
        : simulator(sim ? sim : std::make_shared<ForwardSimulator>()) {
        OPTIM_LOG("Optimizer created with custom simulator");
    }
    
    explicit Optimizer(const OptimizerConfig& cfg)
        : simulator(std::make_shared<ForwardSimulator>()), config(cfg) {
        OPTIM_LOG("Optimizer created with config (iterations=" << cfg.total_iterations << ")");
    }
    
    Optimizer(std::shared_ptr<Simulator> sim, const OptimizerConfig& cfg)
        : simulator(sim ? sim : std::make_shared<ForwardSimulator>()), config(cfg) {
        OPTIM_LOG("Optimizer created with simulator and config");
    }
    
    // ============= 配置 =============
    void set_config(const OptimizerConfig& cfg) { config = cfg; state.initialized = false; }
    OptimizerConfig& get_config() { return config; }
    const OptimizerConfig& get_config() const { return config; }
    void set_simulator(std::shared_ptr<Simulator> sim) { if (sim) simulator = sim; }
    std::shared_ptr<Simulator> get_simulator() const { return simulator; }
    OptimizationState& get_state() { return state; }
    const OptimizationState& get_state() const { return state; }
    
    // ============= 参数注册 =============
    void register_param(const std::string& name, Float* data) {
        OPTIM_LOG("register_param: '" << name << "'");
        if (!data) {
            OPTIM_LOG("  ERROR: null data pointer!");
            throw std::runtime_error("Cannot register null parameter");
        }
        OPTIM_LOG("  data size: " << drjit::width(*data));
        state.params[name] = OptimParam(name, data);
    }
    
    void register_param(const std::string& name, Float* data, float lower, float upper) {
        register_param(name, data);
        state.params[name].set_bounds(lower, upper);
        OPTIM_LOG("  bounds: [" << lower << ", " << upper << "]");
    }
    
    void unregister_param(const std::string& name) { state.params.erase(name); }
    void clear_params() { state.params.clear(); }
    
    // ============= 目标设置 =============
    void set_target(const std::string& name, const Float& data, int width, int height) {
        OPTIM_LOG("set_target: '" << name << "' (" << width << "x" << height << ")");
        OPTIM_LOG("  data size: " << drjit::width(data));
        
        if (config.pyramid.enabled) {
            state.target_data.emplace(name, SensorData(data, width, height, config.pyramid.num_levels));
        } else {
            state.target_data.emplace(name, SensorData(name, width, height));
            state.target_data[name].pyramid_levels[0].data = data;
        }
    }
    
    void set_target(const std::string& name, const SensorData& target) {
        OPTIM_LOG("set_target: '" << name << "' from SensorData");
        state.target_data[name] = target;
    }
    
    // ============= 回调设置 =============
    void set_iteration_callback(IterationCallback cb) { on_iteration = std::move(cb); }
    void set_save_callback(SaveCallback cb) { on_save = std::move(cb); }
    void set_complete_callback(std::function<void(const OptimizationState&)> cb) { on_complete = std::move(cb); }
    void set_level_change_callback(std::function<void(int, int)> cb) { on_level_change = std::move(cb); }
    void set_custom_loss(std::function<Float(const OptimizationState&)> fn) {
        custom_loss_fn = std::move(fn);
        config.loss.type = LossType::Custom;
    }
    
    // ============= 初始化 =============
    void init(const Scene& scene) {
        OPTIM_LOG("========================================");
        OPTIM_LOG("Optimizer::init() BEGIN");
        OPTIM_LOG("========================================");
        
        m_scene = &scene;
        
        // 保存用户设置的参数和目标
        auto saved_targets = std::move(state.target_data);
        auto saved_params = std::move(state.params);
        
        state.reset();
        
        // 恢复用户设置的参数和目标
        state.target_data = std::move(saved_targets);
        state.params = std::move(saved_params);
        
        // 打印场景信息
        auto sensor_names = scene.get_sensor_names();
        OPTIM_LOG("Scene sensors: " << sensor_names.size());
        for (const auto& name : sensor_names) {
            auto sensor = scene.get_sensor(name);
            if (sensor) {
                OPTIM_LOG("  - " << name << " (" << sensor->get_width() << "x" << sensor->get_height() << ")");
            }
        }
        
        // 为每个 sensor 创建 SensorData
        for (const auto& sensor_name : sensor_names) {
            auto sensor = scene.get_sensor(sensor_name);
            if (sensor) {
                OPTIM_LOG("Creating SensorData for '" << sensor_name << "'");
                state.sensor_data.emplace(
                    sensor_name,
                    SensorData(sensor_name, sensor->get_width(), sensor->get_height())
                );
                
                if (config.pyramid.enabled) {
                    auto& sdata = state.sensor_data[sensor_name];
                    OPTIM_LOG("  Initializing pyramid with " << config.pyramid.num_levels << " levels");
                    sdata.init_pyramid(
                        config.pyramid.num_levels,
                        config.total_iterations,
                        config.pyramid.schedule_type,
                        config.pyramid.custom_iterations
                    );
                    sensor->init_pixel_area_pyramid(config.pyramid.num_levels);
                }
            }
        }
        
        // 检查并初始化目标的金字塔
        OPTIM_LOG("Targets registered: " << state.target_data.size());
        for (auto& [name, target] : state.target_data) {
            OPTIM_LOG("  - " << name << " (" << target.width << "x" << target.height << ")");
            
            if (config.pyramid.enabled && target.pyramid_levels.size() <= 1) {
                OPTIM_LOG("    Initializing target pyramid...");
                Float original_data = target.pyramid_levels[0].data;
                target = SensorData(original_data, target.width, target.height, config.pyramid.num_levels);
            }
        }
        
        // 重新初始化 Adam 状态
        OPTIM_LOG("Initializing Adam states for " << state.params.size() << " parameters:");
        for (auto& [name, param] : state.params) {
            if (param.data) {
                size_t size = drjit::width(*param.data);
                OPTIM_LOG("  - " << name << ": size=" << size);
                if (size > 0) {
                    param.adam_state.init(size);
                    drjit::enable_grad(*param.data);
                    OPTIM_LOG("    Gradient enabled");
                } else {
                    OPTIM_LOG("    WARNING: zero size, skipping");
                }
            } else {
                OPTIM_LOG("  - " << name << ": WARNING null data!");
            }
        }
        
        if (config.pyramid.enabled && !state.sensor_data.empty()) {
            state.current_level = state.sensor_data.begin()->second.get_current_level();
            OPTIM_LOG("Initial pyramid level: " << state.current_level);
        }
        
        state.initialized = true;
        
        OPTIM_LOG("========================================");
        OPTIM_LOG("Optimizer::init() COMPLETE");
        OPTIM_LOG("========================================");
    }
    
    // ============= 运行优化 =============
    void run(const Scene& scene, const SimulatorConfig& sim_config) {
        OPTIM_LOG("========================================");
        OPTIM_LOG("Optimizer::run() BEGIN");
        OPTIM_LOG("========================================");
        
        OPTIM_LOG("Config summary:");
        OPTIM_LOG("  total_iterations: " << config.total_iterations);
        OPTIM_LOG("  learning_rate: " << config.adam.lr);
        OPTIM_LOG("  pyramid_enabled: " << config.pyramid.enabled);
        OPTIM_LOG("  loss_type: " << static_cast<int>(config.loss.type));
        
        if (!state.initialized) {
            OPTIM_LOG("Not initialized, calling init()...");
            init(scene);
        }
        
        OPTIM_LOG("Pre-run validation:");
        OPTIM_LOG("  sensors: " << state.sensor_data.size());
        OPTIM_LOG("  targets: " << state.target_data.size());
        OPTIM_LOG("  params: " << state.params.size());
        
        if (state.params.empty()) {
            OPTIM_LOG("WARNING: No parameters registered! Optimization will have no effect.");
        }
        
        if (state.target_data.empty()) {
            OPTIM_LOG("WARNING: No targets set! Loss will be zero.");
        }
        
        int matched_pairs = 0;
        for (const auto& [sensor_name, _] : state.sensor_data) {
            if (state.has_target(sensor_name)) {
                matched_pairs++;
                OPTIM_LOG("  Sensor-Target pair: " << sensor_name);
            }
        }
        OPTIM_LOG("  Matched sensor-target pairs: " << matched_pairs);
        
        if (matched_pairs == 0) {
            OPTIM_LOG("ERROR: No matching sensor-target pairs! Check that target names match sensor names.");
        }
        
        state.running = true;
        state.completed = false;
        
        OPTIM_LOG("Starting optimization loop...");
        OPTIM_LOG("----------------------------------------");
        
        int step_count = 0;
        while (!is_complete()) {
            try {
                step(scene, sim_config);
                step_count++;
                
                if (step_count == 1 || step_count % 100 == 0) {
                    OPTIM_LOG_VERBOSE("Completed " << step_count << " steps, iteration=" << state.iteration);
                }
            } catch (const std::exception& e) {
                OPTIM_LOG("ERROR in step " << state.iteration << ": " << e.what());
                throw;
            }
        }
        
        OPTIM_LOG("----------------------------------------");
        OPTIM_LOG("Optimization loop finished after " << step_count << " steps");
        
        state.running = false;
        state.completed = true;
        
        if (config.callback.save_final) {
            OPTIM_LOG("Saving final results...");
            save_results("final");
        }
        
        if (on_complete) {
            OPTIM_LOG("Calling on_complete callback...");
            on_complete(state);
        }
        
        OPTIM_LOG("========================================");
        OPTIM_LOG("Optimizer::run() COMPLETE");
        OPTIM_LOG("  Total iterations: " << state.iteration);
        OPTIM_LOG("  Best loss: " << state.best_loss << " at iteration " << state.best_iteration);
        OPTIM_LOG("  Total time: " << state.total_sim_time_ms << " ms");
        OPTIM_LOG("========================================");
        
        if (config.callback.verbose) {
            std::cout << "\n=== Optimization Complete ===" << std::endl;
            std::cout << "Total iterations: " << state.iteration << std::endl;
            std::cout << "Best loss: " << state.best_loss << " at iteration " << state.best_iteration << std::endl;
            std::cout << "Total time: " << state.total_sim_time_ms << " ms" << std::endl;
        }
    }
    
    // 单步执行
    SimulationResult step(const Scene& scene, const SimulatorConfig& sim_config) {
        SimulationResult result;
        
        bool verbose_step = (state.iteration == 0) || (state.iteration % 100 == 0);
        
        if (verbose_step) {
            OPTIM_LOG_VERBOSE("--- Step " << state.iteration << " BEGIN ---");
        }
        
        if (!state.initialized) {
            OPTIM_LOG("ERROR: step() called but not initialized!");
            result.success = false;
            result.error_message = "Optimizer not initialized. Call init() first.";
            return result;
        }
        
        if (state.iteration >= config.total_iterations) {
            OPTIM_LOG("WARNING: step() called but already complete!");
            result.success = false;
            result.error_message = "Optimization already complete.";
            return result;
        }
        
        int old_level = state.current_level;
        
        update_pyramid_level();
        
        if (state.level_changed) {
            OPTIM_LOG("Pyramid level changed: " << old_level << " -> " << state.current_level);
            if (on_level_change) {
                on_level_change(old_level, state.current_level);
            }
            state.clear_current_level_data();
            if (config.callback.save_on_level_change) {
                save_results("level_" + std::to_string(old_level) + "_end");
            }
        }
        
        // 清除当前层级的 sensor data（每次迭代都需要清除）
        state.clear_current_level_data();
        
        // 更新 BSpline 表面的网格
        update_bspline_meshes(scene);
        
        if (verbose_step) {
            OPTIM_LOG_VERBOSE("  Running forward simulation...");
        }
        
        try {
            simulator->simulate_into(scene, sim_config, state.sensor_data, result);
        } catch (const std::exception& e) {
            OPTIM_LOG("ERROR in simulate_into: " << e.what());
            throw;
        }
        
        if (!result.success) {
            OPTIM_LOG("WARNING: Simulation failed: " << result.error_message);
        }
        
        if (verbose_step) {
            OPTIM_LOG_VERBOSE("  Simulation time: " << result.total_time_ms << " ms");
        }
        
        if (verbose_step) {
            OPTIM_LOG_VERBOSE("  Computing loss...");
        }
        
        Float loss;
        try {
            loss = compute_loss();
        } catch (const std::exception& e) {
            OPTIM_LOG("ERROR in compute_loss: " << e.what());
            throw;
        }
        
        // 确保 loss 被求值
        drjit::eval(loss);
        
        float loss_val = 0.0f;
        try {
            if (drjit::width(loss) > 0) {
                loss_val = drjit::slice(loss, 0);
            } else {
                OPTIM_LOG("WARNING: Loss has zero width!");
            }
        } catch (const std::exception& e) {
            OPTIM_LOG("ERROR extracting loss value: " << e.what());
            throw;
        }
        
        state.last_loss = loss_val;
        state.loss_history.push_back(loss_val);
        
        if (verbose_step) {
            OPTIM_LOG_VERBOSE("  Loss: " << loss_val);
        }
        
        if (loss_val < state.best_loss) {
            state.best_loss = loss_val;
            state.best_iteration = state.iteration;
            if (verbose_step) {
                OPTIM_LOG_VERBOSE("  New best loss!");
            }
        }
        
        if (verbose_step) {
            OPTIM_LOG_VERBOSE("  Running backward pass...");
        }
        
        try {
            drjit::backward(loss);
        } catch (const std::exception& e) {
            OPTIM_LOG("ERROR in backward: " << e.what());
            throw;
        }
        
        if (verbose_step) {
            OPTIM_LOG_VERBOSE("  Running Adam step...");
        }
        
        try {
            adam_step();
        } catch (const std::exception& e) {
            OPTIM_LOG("ERROR in adam_step: " << e.what());
            throw;
        }
        
        try {
            apply_constraints();
        } catch (const std::exception& e) {
            OPTIM_LOG("ERROR in apply_constraints: " << e.what());
            throw;
        }
        
        state.last_sim_time_ms = result.total_time_ms;
        state.total_sim_time_ms += result.total_time_ms;
        
        if (config.callback.verbose && 
            (state.iteration % config.callback.print_interval == 0 || state.iteration == 0)) {
            print_progress();
        }
        
        if (on_iteration) {
            try {
                on_iteration(state.iteration, loss_val, state);
            } catch (const std::exception& e) {
                OPTIM_LOG("ERROR in on_iteration callback: " << e.what());
            }
        }
        
        if (config.callback.save_interval > 0 && 
            state.iteration % config.callback.save_interval == 0) {
            save_results("step_" + std::to_string(state.iteration));
        }
        
        result.pyramid_enabled = config.pyramid.enabled;
        result.pyramid_current_level = state.current_level;
        result.pyramid_level_changed = state.level_changed;
        
        state.iteration++;
        
        if (verbose_step) {
            OPTIM_LOG_VERBOSE("--- Step " << (state.iteration - 1) << " END ---");
        }
        
        return result;
    }
    
    // ============= 状态查询 =============
    bool is_complete() const { return state.iteration >= config.total_iterations; }
    bool is_initialized() const { return state.initialized; }
    bool is_running() const { return state.running; }
    int get_current_iteration() const { return state.iteration; }
    int get_current_level() const { return state.current_level; }
    bool is_level_changed() const { return state.level_changed; }
    
    float progress() const {
        if (config.total_iterations <= 0) return 1.0f;
        return static_cast<float>(state.iteration) / static_cast<float>(config.total_iterations);
    }
    
    int remaining_iterations() const {
        return std::max(0, config.total_iterations - state.iteration);
    }
    
    // ============= 损失计算 =============
    Float compute_loss() const {
        OPTIM_LOG_VERBOSE("compute_loss()");
        
        if (config.loss.type == LossType::Custom && custom_loss_fn) {
            OPTIM_LOG_VERBOSE("  Using custom loss function");
            return custom_loss_fn(state);
        }
        
        Float total_loss = drjit::zeros<Float>(1);
        int count = 0;
        
        for (const auto& [name, sdata] : state.sensor_data) {
            OPTIM_LOG_VERBOSE("  Checking sensor '" << name << "'...");
            
            if (state.has_target(name)) {
                const auto& target = state.get_target(name);
                OPTIM_LOG_VERBOSE("    Found matching target");
                
                const Float& sim = sdata.get_current_data();
                const Float& ref = target.get_data_at_level(sdata.get_current_level());
                
                // 确保数据被求值
                drjit::eval(sim, ref);
                
                Float sim_sum = drjit::sum(sim);
                Float ref_sum = drjit::sum(ref);
                
                drjit::eval(sim_sum, ref_sum);
                
                float sim_sum_val = drjit::slice(sim_sum, 0);
                float ref_sum_val = drjit::slice(ref_sum, 0);
                OPTIM_LOG_VERBOSE("    sim_sum: " << sim_sum_val << ", ref_sum: " << ref_sum_val);
                
                // 避免除零，使用更安全的归一化
                Float eps = Float(1e-10f);
                Float sim_norm = drjit::select(sim_sum > eps, sim / sim_sum, sim);
                Float ref_norm = drjit::select(ref_sum > eps, ref / ref_sum, ref);
                
                // 计算损失
                Float diff = sim_norm - ref_norm;
                Float sensor_loss;
                
                switch (config.loss.type) {
                    case LossType::L1:
                        sensor_loss = drjit::mean(drjit::abs(diff));
                        break;
                    case LossType::L2:
                        sensor_loss = drjit::mean(drjit::sqr(diff));
                        break;
                    case LossType::L1_L2:
                        sensor_loss = drjit::mean(drjit::abs(diff)) + 
                                      Float(config.loss.l2_weight) * drjit::mean(drjit::sqr(diff));
                        break;
                    default:
                        sensor_loss = drjit::mean(drjit::abs(diff));
                }
                
                drjit::eval(sensor_loss);
                float loss_val = drjit::slice(sensor_loss, 0);
                OPTIM_LOG_VERBOSE("    Sensor loss: " << loss_val);
                
                total_loss = total_loss + sensor_loss;
                count++;
            } else {
                OPTIM_LOG_VERBOSE("    No matching target");
            }
        }
        
        OPTIM_LOG_VERBOSE("  Matched pairs: " << count);
        
        if (count > 1) {
            total_loss = total_loss / Float(static_cast<float>(count));
        }
        
        return total_loss;
    }
    
    Float compute_l1_loss(bool normalize = true) const {
        Float total_loss = drjit::zeros<Float>(1);
        int count = 0;
        for (const auto& [name, sdata] : state.sensor_data) {
            if (state.has_target(name)) {
                total_loss = total_loss + sdata.l1_loss(state.get_target(name), normalize);
                count++;
            }
        }
        if (count > 1) total_loss = total_loss / Float(static_cast<float>(count));
        return total_loss;
    }
    
    Float compute_l2_loss(bool normalize = true) const {
        Float total_loss = drjit::zeros<Float>(1);
        int count = 0;
        for (const auto& [name, sdata] : state.sensor_data) {
            if (state.has_target(name)) {
                total_loss = total_loss + sdata.l2_loss(state.get_target(name), normalize);
                count++;
            }
        }
        if (count > 1) total_loss = total_loss / Float(static_cast<float>(count));
        return total_loss;
    }
    
    // ============= 打印 =============
    void print(const std::string& name = "Optimizer") const {
        std::cout << name << ":" << std::endl;
        std::cout << "  progress: " << state.iteration << "/" << config.total_iterations 
                  << " (" << (progress() * 100.0f) << "%)" << std::endl;
        config.print("  ");
        state.print("  ");
    }
    
private:
    // ============= 更新 BSpline 网格 =============
    void update_bspline_meshes(const Scene& scene) {
        // 遍历所有表面，找到 BSpline 表面并更新
        auto surface_names = scene.get_surface_names();
        
        for (const auto& name : surface_names) {
            auto surface = scene.get_surface(name);
            if (!surface) continue;
            
            // 检查是否是 BSpline 表面
            auto bspline_surface = std::dynamic_pointer_cast<RectangleBSplineSurface>(surface);
            if (bspline_surface) {
                // 标记网格需要更新
                bspline_surface->invalidate_mesh();
                
                // 更新 OptiX 加速结构
                auto optix_mgr = get_optix_manager();
                if (optix_mgr && optix_mgr->is_initialized()) {
                    optix_mgr->update_surface_mesh(name, scene);
                }
            }
        }
    }
    
    // ============= Adam 更新 =============
    void adam_step() {
        OPTIM_LOG_VERBOSE("adam_step() - " << state.params.size() << " parameters");
        
        for (auto& [name, param] : state.params) {
            OPTIM_LOG_VERBOSE("  Processing param '" << name << "'");
            
            if (!param.enabled) {
                OPTIM_LOG_VERBOSE("    Skipped: disabled");
                continue;
            }
            if (!param.data) {
                OPTIM_LOG_VERBOSE("    Skipped: null data");
                continue;
            }
            
            Float& x = *param.data;
            size_t x_size = drjit::width(x);
            OPTIM_LOG_VERBOSE("    Param size: " << x_size);
            
            Float grad = drjit::grad(x);
            drjit::eval(grad);
            
            size_t grad_size = drjit::width(grad);
            OPTIM_LOG_VERBOSE("    Grad size: " << grad_size);
            
            if (grad_size == 0) {
                OPTIM_LOG_VERBOSE("    Skipped: zero gradient width");
                continue;
            }
            
            // 检查梯度是否全为零
            Float grad_abs = drjit::abs(grad);
            float grad_max = drjit::slice(drjit::max(grad_abs), 0);
            float grad_mean = drjit::slice(drjit::mean(grad_abs), 0);
            
            if (state.iteration == 0 || state.iteration % 100 == 0) {
                OPTIM_LOG_VERBOSE("    Grad stats: max=" << grad_max << ", mean=" << grad_mean);
            }
            
            if (grad_max < 1e-12f) {
                OPTIM_LOG_VERBOSE("    WARNING: gradient is effectively zero!");
            }
            
            param.adam_state.step++;
            int t = param.adam_state.step;
            
            // 分离梯度追踪
            x = drjit::detach(x);
            
            // Weight decay
            if (config.adam.weight_decay > 0) {
                x = x * Float(1.0f - config.adam.lr * config.adam.weight_decay);
            }
            
            // 更新动量
            param.adam_state.m = Float(config.adam.beta1) * param.adam_state.m + 
                                  Float(1.0f - config.adam.beta1) * grad;
            param.adam_state.v = Float(config.adam.beta2) * param.adam_state.v + 
                                  Float(1.0f - config.adam.beta2) * grad * grad;
            
            // 偏差校正
            float bc1 = 1.0f - std::pow(config.adam.beta1, t);
            float bc2 = 1.0f - std::pow(config.adam.beta2, t);
            
            Float m_hat = param.adam_state.m / Float(bc1);
            Float v_hat = param.adam_state.v / Float(bc2);
            
            // AMSGrad
            if (config.adam.amsgrad) {
                param.adam_state.v_max = drjit::maximum(param.adam_state.v_max, v_hat);
                v_hat = param.adam_state.v_max;
            }
            
            // 更新参数
            Float update = Float(config.adam.lr) * m_hat / (drjit::sqrt(v_hat) + Float(config.adam.epsilon));
            x = x - update;
            
            // 检查更新幅度
            if (state.iteration == 0 || state.iteration % 100 == 0) {
                Float update_abs = drjit::abs(update);
                float update_max = drjit::slice(drjit::max(update_abs), 0);
                float update_mean = drjit::slice(drjit::mean(update_abs), 0);
                OPTIM_LOG_VERBOSE("    Update stats: max=" << update_max << ", mean=" << update_mean);
            }
            
            // 重新启用梯度
            drjit::enable_grad(x);
            
            // 确保更新被执行
            drjit::eval(x);
            
            OPTIM_LOG_VERBOSE("    Adam step complete");
        }
    }
    
    void apply_constraints() {
        for (auto& [name, param] : state.params) {
            param.apply_constraints();
            if (param.data) {
                drjit::eval(*param.data);
            }
        }
    }
    
    // ============= 金字塔更新 =============
    void update_pyramid_level() {
        if (!config.pyramid.enabled) {
            state.level_changed = false;
            return;
        }
        
        bool any_level_changed = false;
        int new_level = state.current_level;
        
        for (auto& [name, sdata] : state.sensor_data) {
            bool changed = sdata.set_level_by_iteration(state.iteration);
            if (changed) {
                any_level_changed = true;
                new_level = sdata.get_current_level();
            }
        }
        
        state.level_changed = any_level_changed;
        state.current_level = new_level;
        
        for (auto& [name, tdata] : state.target_data) {
            tdata.set_current_level(state.current_level);
        }
    }
    
    // ============= 保存结果 =============
    void save_results(const std::string& suffix) {
        if (!on_save) {
            OPTIM_LOG_VERBOSE("save_results: no save callback set");
            return;
        }
        
        OPTIM_LOG("Saving results with suffix '" << suffix << "'");
        
        for (const auto& [name, sdata] : state.sensor_data) {
            std::string filepath = config.callback.save_dir + "/" + 
                                   config.callback.save_prefix + "_" + 
                                   name + "_" + suffix + ".exr";
            OPTIM_LOG_VERBOSE("  Saving: " << filepath);
            try {
                on_save(name, sdata, filepath);
            } catch (const std::exception& e) {
                OPTIM_LOG("ERROR saving " << filepath << ": " << e.what());
            }
        }
    }
    
    // ============= 进度打印 =============
    void print_progress() const {
        std::cout << "[" << std::setw(4) << state.iteration << "/" << config.total_iterations << "] ";
        std::cout << "loss: " << std::scientific << std::setprecision(4) << state.last_loss;
        if (config.pyramid.enabled) {
            std::cout << " | level: " << state.current_level;
        }
        std::cout << " | time: " << std::fixed << std::setprecision(1) << state.last_sim_time_ms << "ms";
        if (state.best_iteration == state.iteration - 1) {
            std::cout << " *";
        }
        std::cout << std::endl;
    }
};

} // namespace diff_optics