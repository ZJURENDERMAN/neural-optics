// shape_base.hpp
#pragma once
#include "utils.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <iostream>

namespace diff_optics {

// ============= 抽象基类 Shape =============
struct Shape {
    virtual ~Shape() = default;
    virtual std::string type_name() const = 0;
    
    // 非微分版本，用于生成三角网格
    virtual Vector3C compute_position_c(const Vector2C& uv) const = 0;
    
    // 可微分版本，用于着色计算（仅返回法线）
    virtual Vector3 compute_normal(const Vector2& uv) const = 0;
    
    // ============= 统一的可微分参数接口 =============
    
    /// 获取可微分参数数量
    virtual int get_diff_param_count() const { return 0; }
    
    /// 获取可微分参数（返回引用，用于优化器）
    virtual Float& get_diff_params() {
        static Float empty;
        return empty;
    }
    
    /// 获取可微分参数（const 版本）
    virtual const Float& get_diff_params() const {
        static Float empty;
        return empty;
    }
    
    /// 设置可微分参数
    virtual void set_diff_params(const Float& params) {}
    
    /// 获取参数为 CPU 数组
    virtual std::vector<ScalarType> get_diff_params_cpu() const {
        return {};
    }
    
    /// 从 CPU 数组设置参数
    virtual void set_diff_params_cpu(const std::vector<ScalarType>& params) {}
    
    /// 获取参数配置描述（用于 Python 侧了解当前配置）
    virtual std::string get_param_config_string() const {
        return "no_params";
    }
    
    /// 调整参数尺寸/分辨率
    /// @param new_config 配置字典（由子类解释）
    /// @return 是否进行了调整
    virtual bool resize_params(const std::map<std::string, int>& new_config) {
        return false;
    }
    
    /// 获取当前参数配置
    virtual std::map<std::string, int> get_param_config() const {
        return {};
    }
    
    // ============= CAD 导出接口 =============
    
    /// 导出为 STEP 文件
    /// @param filename 输出文件路径（.stp 或 .step）
    /// @return 是否成功
    virtual bool save_cad(const std::string& filename) const {
        std::cerr << "[" << type_name() << "] CAD export not supported for this shape type." << std::endl;
        return false;
    }
};

// ============= 平面 =============
struct Plane : public Shape {
    std::string type_name() const override { return "Plane"; }
    
    Vector3C compute_position_c(const Vector2C& uv) const override {
        size_t n = drjit::width(uv[0]);
        return Vector3C(uv[0], uv[1], drjit::zeros<FloatC>(n));
    }
    
    Vector3 compute_normal(const Vector2& uv) const override {
        size_t n = drjit::width(uv[0]);
        return Vector3(
            drjit::zeros<Float>(n),
            drjit::zeros<Float>(n),
            drjit::full<Float>(1.0f, n)
        );
    }
    
    // Plane 没有可微分参数
    std::string get_param_config_string() const override {
        return "plane(no_params)";
    }
    
    bool save_cad(const std::string& filename) const override {
        std::cerr << "[Plane] CAD export not implemented yet." << std::endl;
        return false;
    }
};

} // namespace diff_optics