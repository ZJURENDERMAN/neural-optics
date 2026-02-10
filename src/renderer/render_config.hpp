// render_config.hpp - 渲染配置
#pragma once
#include <string>

namespace diff_optics {

struct RenderConfig {
    // 分辨率
    int width = 1024;
    int height = 768;
    
    // 采样
    int samples_per_pixel = 64;
    int max_depth = 8;
    
    // 环境贴图
    std::string environment_map_path = "";  // 空则使用默认天空
    float environment_intensity = 1.0f;
    float environment_rotation = 0.0f;      // 环境贴图旋转（度）
    
    // 默认天空颜色（无环境贴图时使用）
    float sky_color_top[3] = {0.5f, 0.7f, 1.0f};
    float sky_color_bottom[3] = {1.0f, 1.0f, 1.0f};
    
    // 玻璃材质默认IOR
    float glass_ior = 1.5f;
    
    // 漫反射默认颜色
    float diffuse_color[3] = {0.8f, 0.8f, 0.8f};
    
    // 渐进式渲染
    bool progressive = false;
    
    // Gamma校正
    float gamma = 2.2f;
    
    // 曝光
    float exposure = 1.0f;
    
    RenderConfig() = default;
    
    RenderConfig& set_resolution(int w, int h) {
        width = w; height = h;
        return *this;
    }
    
    RenderConfig& set_samples(int spp) {
        samples_per_pixel = spp;
        return *this;
    }
    
    RenderConfig& set_max_depth(int depth) {
        max_depth = depth;
        return *this;
    }
    
    RenderConfig& set_environment(const std::string& path, float intensity = 1.0f) {
        environment_map_path = path;
        environment_intensity = intensity;
        return *this;
    }
    
    RenderConfig& set_sky_gradient(
        float top_r, float top_g, float top_b,
        float bottom_r, float bottom_g, float bottom_b
    ) {
        sky_color_top[0] = top_r; sky_color_top[1] = top_g; sky_color_top[2] = top_b;
        sky_color_bottom[0] = bottom_r; sky_color_bottom[1] = bottom_g; sky_color_bottom[2] = bottom_b;
        return *this;
    }
    
    RenderConfig& set_glass_ior(float ior) {
        glass_ior = ior;
        return *this;
    }
    
    RenderConfig& set_diffuse_color(float r, float g, float b) {
        diffuse_color[0] = r; diffuse_color[1] = g; diffuse_color[2] = b;
        return *this;
    }
};

} // namespace diff_optics