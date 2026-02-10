// scene_parser.hpp
#pragma once

#include "scene.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <optional>
#include <filesystem>

namespace diff_optics {

using json = nlohmann::json;
namespace fs = std::filesystem;

// ============= NPY 文件加载器 =============
struct NpyLoader {
    // NPY 文件头信息
    struct NpyHeader {
        bool fortran_order = false;
        std::vector<size_t> shape;
        char dtype = 'f';  // 'f' for float, 'd' for double, 'i' for int
        size_t word_size = 4;
        bool valid = false;
    };
    
    // 解析 NPY 头
    static NpyHeader parse_header(std::ifstream& file) {
        NpyHeader header;
        
        // 读取魔数 "\x93NUMPY"
        char magic[6];
        file.read(magic, 6);
        if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
            std::cerr << "[NpyLoader] Invalid NPY magic number" << std::endl;
            return header;
        }
        
        // 读取版本
        uint8_t major, minor;
        file.read(reinterpret_cast<char*>(&major), 1);
        file.read(reinterpret_cast<char*>(&minor), 1);
        
        // 读取头长度
        uint32_t header_len = 0;
        if (major == 1) {
            uint16_t len16;
            file.read(reinterpret_cast<char*>(&len16), 2);
            header_len = len16;
        } else if (major == 2 || major == 3) {
            file.read(reinterpret_cast<char*>(&header_len), 4);
        } else {
            std::cerr << "[NpyLoader] Unsupported NPY version: " << (int)major << "." << (int)minor << std::endl;
            return header;
        }
        
        // 读取头字符串
        std::string header_str(header_len, '\0');
        file.read(&header_str[0], header_len);
        
        // 解析头字典
        // 格式类似: {'descr': '<f4', 'fortran_order': False, 'shape': (64, 64), }
        
        // 解析 descr
        size_t descr_pos = header_str.find("'descr'");
        if (descr_pos != std::string::npos) {
            size_t quote1 = header_str.find("'", descr_pos + 7);
            size_t quote2 = header_str.find("'", quote1 + 1);
            if (quote1 != std::string::npos && quote2 != std::string::npos) {
                std::string descr = header_str.substr(quote1 + 1, quote2 - quote1 - 1);
                // 解析如 "<f4", ">f8", "|f4" 等
                if (descr.length() >= 3) {
                    header.dtype = descr[1];
                    header.word_size = std::stoul(descr.substr(2));
                }
            }
        }
        
        // 解析 fortran_order
        if (header_str.find("True") != std::string::npos && 
            header_str.find("fortran_order") != std::string::npos) {
            // 简单检查，实际应该更精确
            size_t fo_pos = header_str.find("fortran_order");
            size_t true_pos = header_str.find("True", fo_pos);
            size_t false_pos = header_str.find("False", fo_pos);
            if (true_pos != std::string::npos && 
                (false_pos == std::string::npos || true_pos < false_pos)) {
                header.fortran_order = true;
            }
        }
        
        // 解析 shape
        size_t shape_pos = header_str.find("'shape'");
        if (shape_pos != std::string::npos) {
            size_t paren1 = header_str.find("(", shape_pos);
            size_t paren2 = header_str.find(")", paren1);
            if (paren1 != std::string::npos && paren2 != std::string::npos) {
                std::string shape_str = header_str.substr(paren1 + 1, paren2 - paren1 - 1);
                // 解析逗号分隔的数字
                size_t pos = 0;
                while (pos < shape_str.length()) {
                    size_t comma = shape_str.find(",", pos);
                    if (comma == std::string::npos) comma = shape_str.length();
                    std::string num_str = shape_str.substr(pos, comma - pos);
                    // 去除空格
                    num_str.erase(std::remove_if(num_str.begin(), num_str.end(), ::isspace), num_str.end());
                    if (!num_str.empty()) {
                        header.shape.push_back(std::stoul(num_str));
                    }
                    pos = comma + 1;
                }
            }
        }
        
        header.valid = true;
        return header;
    }
    
    // 加载 NPY 文件为 float vector
    static std::optional<std::vector<float>> load_float(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[NpyLoader] Cannot open file: " << filepath << std::endl;
            return std::nullopt;
        }
        
        NpyHeader header = parse_header(file);
        if (!header.valid) {
            return std::nullopt;
        }
        
        // 计算元素总数
        size_t total_elements = 1;
        for (size_t dim : header.shape) {
            total_elements *= dim;
        }
        
        if (total_elements == 0) {
            std::cerr << "[NpyLoader] Empty array" << std::endl;
            return std::nullopt;
        }
        
        std::vector<float> result(total_elements);
        
        // 根据数据类型读取
        if (header.dtype == 'f' && header.word_size == 4) {
            // float32，直接读取
            file.read(reinterpret_cast<char*>(result.data()), total_elements * sizeof(float));
        } else if (header.dtype == 'f' && header.word_size == 8) {
            // float64，转换为 float32
            std::vector<double> temp(total_elements);
            file.read(reinterpret_cast<char*>(temp.data()), total_elements * sizeof(double));
            for (size_t i = 0; i < total_elements; ++i) {
                result[i] = static_cast<float>(temp[i]);
            }
        } else if (header.dtype == 'i' && header.word_size == 4) {
            // int32，转换为 float32
            std::vector<int32_t> temp(total_elements);
            file.read(reinterpret_cast<char*>(temp.data()), total_elements * sizeof(int32_t));
            for (size_t i = 0; i < total_elements; ++i) {
                result[i] = static_cast<float>(temp[i]);
            }
        } else {
            std::cerr << "[NpyLoader] Unsupported dtype: " << header.dtype 
                      << header.word_size << std::endl;
            return std::nullopt;
        }
        
        std::cout << "[NpyLoader] Loaded " << filepath << ": " << total_elements << " elements" << std::endl;
        if (!result.empty()) {
            float min_val = *std::min_element(result.begin(), result.end());
            float max_val = *std::max_element(result.begin(), result.end());
            std::cout << "  z range: [" << min_val << ", " << max_val << "]" << std::endl;
        }
        
        return result;
    }
    
    // 加载文本文件
    static std::optional<std::vector<float>> load_txt(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[NpyLoader] Cannot open file: " << filepath << std::endl;
            return std::nullopt;
        }
        
        std::vector<float> result;
        float value;
        while (file >> value) {
            result.push_back(value);
        }
        
        if (result.empty()) {
            return std::nullopt;
        }
        
        std::cout << "[NpyLoader] Loaded " << filepath << ": " << result.size() << " elements" << std::endl;
        float min_val = *std::min_element(result.begin(), result.end());
        float max_val = *std::max_element(result.begin(), result.end());
        std::cout << "  z range: [" << min_val << ", " << max_val << "]" << std::endl;
        
        return result;
    }
    
    // 加载二进制文件
    static std::optional<std::vector<float>> load_bin(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "[NpyLoader] Cannot open file: " << filepath << std::endl;
            return std::nullopt;
        }
        
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        size_t num_floats = file_size / sizeof(float);
        if (num_floats == 0) {
            return std::nullopt;
        }
        
        std::vector<float> result(num_floats);
        file.read(reinterpret_cast<char*>(result.data()), file_size);
        
        std::cout << "[NpyLoader] Loaded " << filepath << ": " << result.size() << " elements" << std::endl;
        float min_val = *std::min_element(result.begin(), result.end());
        float max_val = *std::max_element(result.begin(), result.end());
        std::cout << "  z range: [" << min_val << ", " << max_val << "]" << std::endl;
        
        return result;
    }
    
    // 自动检测格式并加载
    static std::optional<std::vector<float>> load(const std::string& filepath) {
        fs::path path(filepath);
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        if (ext == ".npy") {
            return load_float(filepath);
        } else if (ext == ".txt" || ext == ".csv") {
            return load_txt(filepath);
        } else if (ext == ".bin") {
            return load_bin(filepath);
        } else {
            std::cerr << "[NpyLoader] Unsupported file format: " << ext << std::endl;
            return std::nullopt;
        }
    }
};

// ============= 场景解析器 =============
class SceneParser {
public:
    SceneParser() = default;
    
    // 从 JSON 文件加载场景
    std::shared_ptr<Scene> load(const std::string& scene_path) {
        base_dir_ = fs::path(scene_path).parent_path();
        if (base_dir_.empty()) {
            base_dir_ = ".";
        }
        
        // 读取 JSON 文件
        std::ifstream file(scene_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open scene file: " + scene_path);
        }
        
        json scene_cfg;
        try {
            file >> scene_cfg;
        } catch (const json::parse_error& e) {
            throw std::runtime_error("JSON parse error in " + scene_path + ": " + e.what());
        }
        
        return build_scene(scene_cfg);
    }
    
    // 从 JSON 字符串解析场景
    std::shared_ptr<Scene> parse_string(const std::string& json_str) {
        base_dir_ = ".";
        
        json scene_cfg;
        try {
            scene_cfg = json::parse(json_str);
        } catch (const json::parse_error& e) {
            throw std::runtime_error(std::string("JSON parse error: ") + e.what());
        }
        
        return build_scene(scene_cfg);
    }
    
    // 从 JSON 对象构建场景
    std::shared_ptr<Scene> build_scene(const json& scene_cfg) {
        auto scene = std::make_shared<Scene>();
        
        // 按顺序解析各组件（顺序重要，因为有引用关系）
        
        // 1. Spectrums
        if (scene_cfg.contains("spectrums")) {
            for (const auto& spec_cfg : scene_cfg["spectrums"]) {
                create_spectrum(*scene, spec_cfg);
            }
        }
        
        // 2. Volume Materials
        if (scene_cfg.contains("volume_materials")) {
            for (const auto& mat_cfg : scene_cfg["volume_materials"]) {
                create_volume_material(*scene, mat_cfg);
            }
        }
        
        // 3. BSDFs
        if (scene_cfg.contains("bsdfs")) {
            for (const auto& bsdf_cfg : scene_cfg["bsdfs"]) {
                create_bsdf(*scene, bsdf_cfg);
            }
        }
        
        // 4. Emitters
        if (scene_cfg.contains("emitters")) {
            for (const auto& emitter_cfg : scene_cfg["emitters"]) {
                create_emitter(*scene, emitter_cfg);
            }
        }
        
        // 5. Surfaces
        if (scene_cfg.contains("surfaces")) {
            for (const auto& surface_cfg : scene_cfg["surfaces"]) {
                create_surface(*scene, surface_cfg);
            }
        }
        
        // 6. Lights
        if (scene_cfg.contains("lights")) {
            for (const auto& light_cfg : scene_cfg["lights"]) {
                create_light(*scene, light_cfg);
            }
        }
        
        // 7. Sensors
        if (scene_cfg.contains("sensors")) {
            for (const auto& sensor_cfg : scene_cfg["sensors"]) {
                create_sensor(*scene, sensor_cfg);
            }
        }
        
        return scene;
    }
    
    // 设置基础目录（用于相对路径解析）
    void set_base_dir(const std::string& dir) {
        base_dir_ = dir;
    }

private:
    fs::path base_dir_ = ".";
    
    // ============= 文件查找 =============
    std::optional<fs::path> find_file(const std::string& filename) const {
        std::vector<fs::path> search_paths = {
            fs::path(filename),
            base_dir_ / filename,
            fs::path("experiments") / filename,
        };
        
        for (const auto& p : search_paths) {
            if (fs::exists(p)) {
                return p;
            }
        }
        return std::nullopt;
    }
    
    // ============= 控制点加载 =============
    std::optional<std::vector<float>> load_control_points_z(
        const std::string& filename, int u_num_cp, int v_num_cp) const 
    {
        auto file_path = find_file(filename);
        if (!file_path) {
            std::cerr << "[SceneParser] Warning: Cannot find control points file '" 
                      << filename << "', using default 0" << std::endl;
            return std::nullopt;
        }
        
        size_t expected_count = static_cast<size_t>(u_num_cp) * static_cast<size_t>(v_num_cp);
        
        auto data = NpyLoader::load(file_path->string());
        if (!data) {
            return std::nullopt;
        }
        
        // 检查大小
        if (data->size() != expected_count) {
            std::cerr << "[SceneParser] Warning: Control point count mismatch, expected " 
                      << expected_count << ", got " << data->size() << std::endl;
            if (data->size() > expected_count) {
                data->resize(expected_count);
            } else {
                data->resize(expected_count, 0.0f);
            }
        }
        
        return data;
    }
    
    // ============= Spectrum 创建 =============
    void create_spectrum(Scene& scene, const json& cfg) {
        std::string name = cfg.at("name").get<std::string>();
        std::string spec_type = cfg.value("type", "constant");
        
        if (spec_type == "discrete") {
            auto wavelengths = cfg.at("wavelengths").get<std::vector<float>>();
            auto values = cfg.at("values").get<std::vector<float>>();
            
            Float wl = drjit::load<Float>(wavelengths.data(), wavelengths.size());
            Float val = drjit::load<Float>(values.data(), values.size());
            scene.create_discrete_spectrum(name, wl, val);
        }
        else if (spec_type == "blackbody") {
            float temperature = cfg.value("temperature", 6500.0f);
            float wl_min = cfg.value("wl_min", 380.0f);
            float wl_max = cfg.value("wl_max", 780.0f);
            scene.create_blackbody_spectrum(name, Float(temperature), Float(wl_min), Float(wl_max));
        }
        else if (spec_type == "gaussian") {
            float center = cfg.value("center", 550.0f);
            float sigma = cfg.value("sigma", 30.0f);
            float amplitude = cfg.value("amplitude", 1.0f);
            scene.create_gaussian_spectrum(name, Float(center), Float(sigma), Float(amplitude));
        }
        else if (spec_type == "constant") {
            float value = cfg.value("value", 1.0f);
            float wl_min = cfg.value("wl_min", 380.0f);
            float wl_max = cfg.value("wl_max", 780.0f);
            scene.create_constant_spectrum(name, Float(value), Float(wl_min), Float(wl_max));
        }
        else {
            throw std::runtime_error("Unknown spectrum type: " + spec_type);
        }
    }
    
    // ============= Volume Material 创建 =============
    void create_volume_material(Scene& scene, const json& cfg) {
        std::string name = cfg.at("name").get<std::string>();
        std::string mat_type = cfg.value("type", "constant_ior");
        
        if (mat_type == "air") {
            scene.create_air(name);
        }
        else if (mat_type == "nbk7") {
            scene.create_nbk7(name);
        }
        else if (mat_type == "pmma") {
            scene.create_pmma(name);
        }
        else if (mat_type == "vacuum") {
            scene.create_vacuum(name);
        }
        else if (mat_type == "constant_ior") {
            float ior = cfg.value("ior", 1.5f);
            float transmittance = cfg.value("transmittance", 1.0f);
            float measurement_depth = cfg.value("measurement_depth", 10.0f);
            scene.create_constant_ior_material(name, ior, transmittance, measurement_depth);
        }
        else {
            throw std::runtime_error("Unknown volume material type: " + mat_type);
        }
    }
    
    // ============= BSDF 创建 =============
    void create_bsdf(Scene& scene, const json& cfg) {
        std::string name = cfg.at("name").get<std::string>();
        std::string bsdf_type = cfg.value("type", "specular_refractor");
        
        if (bsdf_type == "specular_reflector") {
            float reflectance = cfg.value("reflectance", 1.0f);
            scene.create_specular_reflector(name, reflectance);
        }
        else if (bsdf_type == "specular_refractor") {
            float transmittance = cfg.value("transmittance", 1.0f);
            scene.create_specular_refractor(name, transmittance);
        }
        else if (bsdf_type == "absorber") {
            scene.create_absorber(name);
        }
        else {
            throw std::runtime_error("Unknown BSDF type: " + bsdf_type);
        }
    }
    
    // ============= Emitter 创建 =============
    void create_emitter(Scene& scene, const json& cfg) {
        std::string name = cfg.at("name").get<std::string>();
        std::string emitter_type = cfg.value("type", "uniform");
        
        if (emitter_type == "uniform") {
            float upper_angle = cfg.value("upper_angle", 30.0f);
            // 角度转换为弧度
            Float upper_rad = Float(upper_angle * static_cast<float>(M_PI) / 180.0f);
            scene.create_uniform_emitter(name, upper_rad);
        }
        else if (emitter_type == "lambert") {
            scene.create_lambert_emitter(name);
        }
        else {
            throw std::runtime_error("Unknown emitter type: " + emitter_type);
        }
    }
    
    // ============= Surface 创建 =============
    void create_surface(Scene& scene, const json& cfg) {
        std::string name = cfg.at("name").get<std::string>();
        std::string surface_type = cfg.value("type", "rectangle_plane");
        
        // 获取材质引用
        std::shared_ptr<VolumeMaterial> inner_material = nullptr;
        std::shared_ptr<VolumeMaterial> outer_material = nullptr;
        std::shared_ptr<BSDF> bsdf = nullptr;
        
        if (cfg.contains("inner_material") && !cfg["inner_material"].is_null()) {
            std::string mat_name = cfg["inner_material"].get<std::string>();
            if (!mat_name.empty()) {
                inner_material = scene.get_volume_material(mat_name);
            }
        }
        if (cfg.contains("outer_material") && !cfg["outer_material"].is_null()) {
            std::string mat_name = cfg["outer_material"].get<std::string>();
            if (!mat_name.empty()) {
                outer_material = scene.get_volume_material(mat_name);
            }
        }
        if (cfg.contains("bsdf") && !cfg["bsdf"].is_null()) {
            std::string bsdf_name = cfg["bsdf"].get<std::string>();
            if (!bsdf_name.empty()) {
                bsdf = scene.get_bsdf(bsdf_name);
            }
        }
        
        // 创建表面
        std::shared_ptr<Surface> surface;
        
        if (surface_type == "rectangle_plane") {
            auto size = cfg.value("size", std::array<float, 2>{1.0f, 1.0f});
            surface = scene.create_rectangle_plane(name, size, inner_material, outer_material, bsdf);
        }
        else if (surface_type == "rectangle_bspline") {
            auto size = cfg.value("size", std::array<float, 2>{1.0f, 1.0f});
            int u_degree = cfg.value("u_degree", 3);
            int v_degree = cfg.value("v_degree", 3);
            
            auto resolution = cfg.value("resolution", std::array<int, 2>{8, 8});
            int u_num_cp = resolution[0];
            int v_num_cp = resolution[1];
            
            // 加载控制点 z 值
            std::vector<float> control_points_z;
            if (cfg.contains("control_points_z") && !cfg["control_points_z"].is_null()) {
                std::string cpz_file = cfg["control_points_z"].get<std::string>();
                auto cpz_data = load_control_points_z(cpz_file, u_num_cp, v_num_cp);
                if (cpz_data) {
                    control_points_z = *cpz_data;
                }
            }
            
            surface = scene.create_rectangle_bspline(
                name, size, u_degree, v_degree, u_num_cp, v_num_cp,
                control_points_z, inner_material, outer_material, bsdf
            );
        }
        else if (surface_type == "rectangle_xy") {
            auto size = cfg.value("size", std::array<float, 2>{1.0f, 1.0f});
            int order = cfg.value("order", 4);
            float b = cfg.value("b", 0.0f);
            
            std::vector<float> coefficients;
            if (cfg.contains("coefficients") && !cfg["coefficients"].is_null()) {
                coefficients = cfg["coefficients"].get<std::vector<float>>();
            }
            
            surface = scene.create_rectangle_xy(
                name, size, order, b, coefficients,
                inner_material, outer_material, bsdf
            );
        }
        else if (surface_type == "rectangle_heightmap") {
            auto size = cfg.value("size", std::array<float, 2>{1.0f, 1.0f});
            auto resolution = cfg.value("resolution", std::array<int, 2>{64, 64});
            int grid_width = resolution[0];
            int grid_height = resolution[1];
            
            std::vector<float> heightmap;
            if (cfg.contains("heightmap") && !cfg["heightmap"].is_null()) {
                // 可以是文件路径或直接数组
                if (cfg["heightmap"].is_string()) {
                    std::string hmap_file = cfg["heightmap"].get<std::string>();
                    auto file_path = find_file(hmap_file);
                    if (file_path) {
                        auto data = NpyLoader::load(file_path->string());
                        if (data) {
                            heightmap = *data;
                        }
                    }
                } else if (cfg["heightmap"].is_array()) {
                    heightmap = cfg["heightmap"].get<std::vector<float>>();
                }
            }
            
            surface = scene.create_rectangle_displacement(
                name, size, grid_width, grid_height, heightmap,
                inner_material, outer_material, bsdf
            );
        }
        else if (surface_type == "circle_plane") {
            float radius = cfg.value("radius", 1.0f);
            surface = scene.create_circle_plane(name, radius);
        }
        else {
            throw std::runtime_error("Unknown surface type: " + surface_type);
        }
        
        // 设置变换
        if (cfg.contains("transform")) {
            const auto& transform_cfg = cfg["transform"];
            auto pos = transform_cfg.value("position", std::array<float, 3>{0, 0, 0});
            auto rot = transform_cfg.value("rotation", std::array<float, 3>{0, 0, 0});
            Transform transform(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]);
            surface->set_transform(transform);
        }
    }
    
    // ============= Light 创建 =============
    void create_light(Scene& scene, const json& cfg) {
        std::string name = cfg.at("name").get<std::string>();
        std::string light_type = cfg.value("type", "point");
        
        // 获取 emitter 和 spectrum 引用
        std::shared_ptr<Emitter> emitter = nullptr;
        std::shared_ptr<Spectrum> spectrum = nullptr;
        
        if (cfg.contains("emitter")) {
            emitter = scene.get_emitter(cfg["emitter"].get<std::string>());
        }
        if (cfg.contains("spectrum")) {
            spectrum = scene.get_spectrum(cfg["spectrum"].get<std::string>());
        }
        
        if (light_type == "point") {
            auto light = scene.create_point_light(name, emitter, spectrum);
            
            // 设置变换
            if (cfg.contains("transform")) {
                const auto& transform_cfg = cfg["transform"];
                auto pos = transform_cfg.value("position", std::array<float, 3>{0, 0, 0});
                auto rot = transform_cfg.value("rotation", std::array<float, 3>{0, 0, 0});
                Transform transform(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]);
                light->set_transform(transform);
            }
        }
        else if (light_type == "surface") {
            std::string surface_name = cfg.at("surface").get<std::string>();
            scene.create_surface_light(name, surface_name, emitter, spectrum);
        }
        else {
            throw std::runtime_error("Unknown light type: " + light_type);
        }
    }
    
    // ============= Sensor 创建 =============
    void create_sensor(Scene& scene, const json& cfg) {
        std::string name = cfg.at("name").get<std::string>();
        std::string sensor_type = cfg.value("type", "irradiance");
        
        auto resolution = cfg.value("resolution", std::array<int, 2>{256, 256});
        std::string filter_type = cfg.value("filter", "bilinear");
        
        if (sensor_type == "irradiance") {
            std::string surface_name = cfg.at("surface").get<std::string>();
            scene.create_irradiance_sensor(name, surface_name, resolution, filter_type);
        }
        else if (sensor_type == "intensity") {
            std::string ies_type = cfg.value("ies_type", "TypeA");
            auto u_range = cfg.value("u_range", std::array<float, 2>{-5.0f, 5.0f});
            auto v_range = cfg.value("v_range", std::array<float, 2>{-5.0f, 5.0f});
            
            if (cfg.contains("surface") && !cfg["surface"].is_null()) {
                // 近场：绑定 Surface
                std::string surface_name = cfg["surface"].get<std::string>();
                scene.create_intensity_sensor(
                    name, surface_name, ies_type, resolution, u_range, v_range, filter_type
                );
            } else {
                // 远场：无 Surface
                scene.create_far_field_sensor(
                    name, ies_type, resolution, u_range, v_range, filter_type
                );
            }
        }
        else {
            throw std::runtime_error("Unknown sensor type: " + sensor_type);
        }
    }
};

// ============= 便捷函数 =============

// 从文件加载场景
inline std::shared_ptr<Scene> load_scene(const std::string& scene_path) {
    SceneParser parser;
    return parser.load(scene_path);
}

// 从 JSON 字符串解析场景
inline std::shared_ptr<Scene> parse_scene_string(const std::string& json_str) {
    SceneParser parser;
    return parser.parse_string(json_str);
}

} // namespace diff_optics