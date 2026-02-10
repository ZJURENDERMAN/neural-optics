#pragma once
#include "utils.hpp"
#include "light.hpp"
#include "emitter.hpp"
#include "surface.hpp"
#include "sensor.hpp"
#include "shell.hpp"
#include "solid.hpp"
#include "spectrum.hpp"
#include "material.hpp"
#include "bsdf.hpp"
#include <memory>
#include <string>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>

namespace diff_optics {

    // 辅助函数：解析滤波器类型字符串
    inline FilterType parse_filter_type(const std::string& filter_name) {
        if (filter_name == "Box" || filter_name == "box") {
            return FilterType::Box;
        } else if (filter_name == "Bilinear" || filter_name == "bilinear") {
            return FilterType::Bilinear;
        }
        throw std::invalid_argument("Unknown filter type: " + filter_name);
    }

    // 辅助函数：解析IES类型字符串
    inline IESType parse_ies_type(const std::string& ies_name) {
        if (ies_name == "A" || ies_name == "a" || ies_name == "TypeA") {
            return IESType::TypeA;
        } else if (ies_name == "B" || ies_name == "b" || ies_name == "TypeB") {
            return IESType::TypeB;
        } else if (ies_name == "C" || ies_name == "c" || ies_name == "TypeC") {
            return IESType::TypeC;
        }
        throw std::invalid_argument("Unknown IES type: " + ies_name);
    }

    class Scene {
    public:
        Scene() = default;
        ~Scene() = default;

        // ============= Emitter 管理 =============
        std::string add_emitter(const std::string& name, std::shared_ptr<Emitter> emitter) {
            std::string final_name = generate_unique_name(emitters, name);
            emitters[final_name] = emitter;
            return final_name;
        }

        std::shared_ptr<Emitter> get_emitter(const std::string& name) const {
            auto it = emitters.find(name);
            if (it == emitters.end()) {
                throw std::runtime_error("Emitter with name '" + name + "' not found");
            }
            return it->second;
        }

        bool has_emitter(const std::string& name) const {
            return emitters.find(name) != emitters.end();
        }

        void remove_emitter(const std::string& name) {
            auto it = emitters.find(name);
            if (it == emitters.end()) {
                throw std::runtime_error("Emitter with name '" + name + "' not found");
            }
            emitters.erase(it);
        }

        std::vector<std::string> get_emitter_names() const {
            std::vector<std::string> names;
            names.reserve(emitters.size());
            for (const auto& pair : emitters) {
                names.push_back(pair.first);
            }
            return names;
        }

        size_t emitter_count() const { return emitters.size(); }

        // ============= Light 管理 =============
        std::string add_light(const std::string& name, std::shared_ptr<Light> light) {
            std::string final_name = generate_unique_name(lights, name);
            lights[final_name] = light;
            return final_name;
        }

        std::shared_ptr<Light> get_light(const std::string& name) const {
            auto it = lights.find(name);
            if (it == lights.end()) {
                throw std::runtime_error("Light with name '" + name + "' not found");
            }
            return it->second;
        }

        template<typename T>
        std::shared_ptr<T> get_light_as(const std::string& name) const {
            auto light = get_light(name);
            auto casted = std::dynamic_pointer_cast<T>(light);
            if (!casted) {
                throw std::runtime_error("Light '" + name + "' is not of the requested type");
            }
            return casted;
        }

        bool has_light(const std::string& name) const {
            return lights.find(name) != lights.end();
        }

        void remove_light(const std::string& name) {
            auto it = lights.find(name);
            if (it == lights.end()) {
                throw std::runtime_error("Light with name '" + name + "' not found");
            }
            lights.erase(it);
        }

        std::vector<std::string> get_light_names() const {
            std::vector<std::string> names;
            names.reserve(lights.size());
            for (const auto& pair : lights) {
                names.push_back(pair.first);
            }
            return names;
        }

        size_t light_count() const { return lights.size(); }

        // ============= Shell 管理 =============
    std::string add_shell(const std::string& name, std::shared_ptr<Shell> shell) {
        std::string final_name = generate_unique_name(shells, name);
        shells[final_name] = shell;
        return final_name;
    }
    
    std::shared_ptr<Shell> get_shell(const std::string& name) const {
        auto it = shells.find(name);
        if (it == shells.end()) {
            throw std::runtime_error("Shell with name '" + name + "' not found");
        }
        return it->second;
    }
    
    bool has_shell(const std::string& name) const {
        return shells.find(name) != shells.end();
    }
    
    void remove_shell(const std::string& name) {
        auto it = shells.find(name);
        if (it == shells.end()) {
            throw std::runtime_error("Shell with name '" + name + "' not found");
        }
        shells.erase(it);
    }
    
    std::vector<std::string> get_shell_names() const {
        std::vector<std::string> names;
        for (const auto& pair : shells) {
            names.push_back(pair.first);
        }
        return names;
    }
    
    size_t shell_count() const { return shells.size(); }
    
    // ============= Solid 管理 =============
    std::string add_solid(const std::string& name, std::shared_ptr<Solid> solid) {
        std::string final_name = generate_unique_name(solids, name);
        solids[final_name] = solid;
        return final_name;
    }
    
    std::shared_ptr<Solid> get_solid(const std::string& name) const {
        auto it = solids.find(name);
        if (it == solids.end()) {
            throw std::runtime_error("Solid with name '" + name + "' not found");
        }
        return it->second;
    }
    
    bool has_solid(const std::string& name) const {
        return solids.find(name) != solids.end();
    }
    
    void remove_solid(const std::string& name) {
        auto it = solids.find(name);
        if (it == solids.end()) {
            throw std::runtime_error("Solid with name '" + name + "' not found");
        }
        solids.erase(it);
    }
    
    std::vector<std::string> get_solid_names() const {
        std::vector<std::string> names;
        for (const auto& pair : solids) {
            names.push_back(pair.first);
        }
        return names;
    }
    
    size_t solid_count() const { return solids.size(); }
    
    // ============= 统一 Surface 访问（用于 OptiX）=============
    // 返回所有需要参与光线追踪的 Surface 及其唯一名称
    std::vector<std::pair<std::string, std::shared_ptr<Surface>>> 
        get_all_surfaces_for_optix() const {
        std::vector<std::pair<std::string, std::shared_ptr<Surface>>> result;
        
        // 1. 独立的 Surface
        for (const auto& [name, surf] : surfaces) {
            result.emplace_back(name, surf);
        }
        
        // 2. Shell 的子 Surface
        for (const auto& [shell_name, shell] : shells) {
            auto shell_surfaces = shell->get_surfaces_with_names(shell_name);
            result.insert(result.end(), shell_surfaces.begin(), shell_surfaces.end());
        }
        
        // 3. Solid 的子 Surface
        for (const auto& [solid_name, solid] : solids) {
            auto solid_surfaces = solid->get_surfaces_with_names(solid_name);
            result.insert(result.end(), solid_surfaces.begin(), solid_surfaces.end());
        }
        
        return result;
    }
    
    // 获取 Surface 总数（包括 Shell 和 Solid 的子 Surface）
    size_t total_surface_count() const {
        size_t count = surfaces.size();
        for (const auto& [_, shell] : shells) {
            count += shell->surface_count();
        }
        for (const auto& [_, solid] : solids) {
            count += solid->surface_count();
        }
        return count;
    }
    
    // ============= 便捷创建方法 - Shell =============
    /*std::shared_ptr<RectanglePlaneShell> create_rectangle_plane_shell(
        const std::string& name,
        int rows, int cols,
        ScalarType cell_width, ScalarType cell_height
    ) {
        auto shell = std::make_shared<RectanglePlaneShell>(rows, cols, cell_width, cell_height);
        add_shell(name, shell);
        return shell;
    }
    
    std::shared_ptr<RectangleBSplineShell> create_rectangle_bspline_shell(
        const std::string& name,
        int rows, int cols,
        ScalarType cell_width, ScalarType cell_height,
        const BSplineConfig& config = BSplineConfig()
    ) {
        auto shell = std::make_shared<RectangleBSplineShell>(
            rows, cols, cell_width, cell_height, config
        );
        add_shell(name, shell);
        return shell;
    }*/
    
    // ============= 便捷创建方法 - Solid =============
    std::shared_ptr<Cube> create_cube(
        const std::string& name,
        ScalarType size_x, ScalarType size_y, ScalarType size_z
    ) {
        auto cube = std::make_shared<Cube>(size_x, size_y, size_z);
        add_solid(name, cube);
        return cube;
    }
    
    std::shared_ptr<RectangleLens> create_rectangle_lens(
        const std::string& name,
        ScalarType width, ScalarType height, ScalarType thickness,
        std::shared_ptr<VolumeMaterial> material = nullptr
    ) {
        auto lens = std::make_shared<RectangleLens>(
            width, height, thickness
        );
        if (material) {
            lens->set_lens_material(material);
        }
        add_solid(name, lens);
        return lens;
    }
    
    std::shared_ptr<CircleLens> create_circle_lens(
        const std::string& name,
        ScalarType radius, ScalarType thickness,
        std::shared_ptr<VolumeMaterial> material = nullptr
    ) {
        auto lens = std::make_shared<CircleLens>(radius, thickness);
        if (material) {
            lens->set_lens_material(material);
        }
        add_solid(name, lens);
        return lens;
    }

        // ============= Surface 管理 =============
        std::string add_surface(const std::string& name, std::shared_ptr<Surface> surface) {
            std::string final_name = generate_unique_name(surfaces, name);
            surfaces[final_name] = surface;
            return final_name;
        }

        std::shared_ptr<Surface> get_surface(const std::string& name) const {
            auto it = surfaces.find(name);
            if (it == surfaces.end()) {
                throw std::runtime_error("Surface with name '" + name + "' not found");
            }
            return it->second;
        }

        template<typename T>
        std::shared_ptr<T> get_surface_as(const std::string& name) const {
            auto surface = get_surface(name);
            auto casted = std::dynamic_pointer_cast<T>(surface);
            if (!casted) {
                throw std::runtime_error("Surface '" + name + "' is not of the requested type");
            }
            return casted;
        }

        bool has_surface(const std::string& name) const {
            return surfaces.find(name) != surfaces.end();
        }

        void remove_surface(const std::string& name) {
            auto it = surfaces.find(name);
            if (it == surfaces.end()) {
                throw std::runtime_error("Surface with name '" + name + "' not found");
            }
            // 同时移除该 surface 关联的所有 sensor 映射
            surface_to_sensors.erase(name);
            surfaces.erase(it);
        }

        std::vector<std::string> get_surface_names() const {
            std::vector<std::string> names;
            names.reserve(surfaces.size());
            for (const auto& pair : surfaces) {
                names.push_back(pair.first);
            }
            return names;
        }

        size_t surface_count() const { return surfaces.size(); }

        // ============= Sensor 管理 =============
        std::string add_sensor(const std::string& name, std::shared_ptr<Sensor> sensor) {
            std::string final_name = generate_unique_name(sensors, name);
            sensors[final_name] = sensor;
            return final_name;
        }

        std::shared_ptr<Sensor> get_sensor(const std::string& name) const {
            auto it = sensors.find(name);
            if (it == sensors.end()) {
                throw std::runtime_error("Sensor with name '" + name + "' not found");
            }
            return it->second;
        }

        template<typename T>
        std::shared_ptr<T> get_sensor_as(const std::string& name) const {
            auto sensor = get_sensor(name);
            auto casted = std::dynamic_pointer_cast<T>(sensor);
            if (!casted) {
                throw std::runtime_error("Sensor '" + name + "' is not of the requested type");
            }
            return casted;
        }

        bool has_sensor(const std::string& name) const {
            return sensors.find(name) != sensors.end();
        }

        void remove_sensor(const std::string& name) {
            auto it = sensors.find(name);
            if (it == sensors.end()) {
                throw std::runtime_error("Sensor with name '" + name + "' not found");
            }
            // 从 surface_to_sensors 映射中移除该 sensor
            for (auto& pair : surface_to_sensors) {
                auto& sensor_list = pair.second;
                sensor_list.erase(
                    std::remove(sensor_list.begin(), sensor_list.end(), name),
                    sensor_list.end()
                );
            }
            // 从远场传感器集合中移除
            farfield_sensors.erase(name);
            sensors.erase(it);
        }

        std::vector<std::string> get_sensor_names() const {
            std::vector<std::string> names;
            names.reserve(sensors.size());
            for (const auto& pair : sensors) {
                names.push_back(pair.first);
            }
            return names;
        }

        size_t sensor_count() const { return sensors.size(); }

        // ============= 远场传感器管理 =============
        
        std::vector<std::string> get_farfield_sensors() const {
            return std::vector<std::string>(farfield_sensors.begin(), farfield_sensors.end());
        }

        bool is_farfield_sensor(const std::string& name) const {
            return farfield_sensors.find(name) != farfield_sensors.end();
        }

        size_t farfield_sensor_count() const {
            return farfield_sensors.size();
        }

        // ============= Spectrum 管理 =============
        std::string add_spectrum(const std::string& name, std::shared_ptr<Spectrum> spectrum) {
            std::string final_name = generate_unique_name(spectra, name);
            spectra[final_name] = spectrum;
            return final_name;
        }

        std::shared_ptr<Spectrum> get_spectrum(const std::string& name) const {
            auto it = spectra.find(name);
            if (it == spectra.end()) {
                throw std::runtime_error("Spectrum with name '" + name + "' not found");
            }
            return it->second;
        }

        template<typename T>
        std::shared_ptr<T> get_spectrum_as(const std::string& name) const {
            auto spectrum = get_spectrum(name);
            auto casted = std::dynamic_pointer_cast<T>(spectrum);
            if (!casted) {
                throw std::runtime_error("Spectrum '" + name + "' is not of the requested type");
            }
            return casted;
        }

        bool has_spectrum(const std::string& name) const {
            return spectra.find(name) != spectra.end();
        }

        void remove_spectrum(const std::string& name) {
            auto it = spectra.find(name);
            if (it == spectra.end()) {
                throw std::runtime_error("Spectrum with name '" + name + "' not found");
            }
            spectra.erase(it);
        }

        std::vector<std::string> get_spectrum_names() const {
            std::vector<std::string> names;
            names.reserve(spectra.size());
            for (const auto& pair : spectra) {
                names.push_back(pair.first);
            }
            return names;
        }

        size_t spectrum_count() const { return spectra.size(); }

        // ============= VolumeMaterial 管理 =============
        std::string add_volume_material(const std::string& name, std::shared_ptr<VolumeMaterial> material) {
            std::string final_name = generate_unique_name(volume_materials, name);
            volume_materials[final_name] = material;
            return final_name;
        }

        std::shared_ptr<VolumeMaterial> get_volume_material(const std::string& name) const {
            auto it = volume_materials.find(name);
            if (it == volume_materials.end()) {
                throw std::runtime_error("VolumeMaterial with name '" + name + "' not found");
            }
            return it->second;
        }

        template<typename T>
        std::shared_ptr<T> get_volume_material_as(const std::string& name) const {
            auto material = get_volume_material(name);
            auto casted = std::dynamic_pointer_cast<T>(material);
            if (!casted) {
                throw std::runtime_error("VolumeMaterial '" + name + "' is not of the requested type");
            }
            return casted;
        }

        bool has_volume_material(const std::string& name) const {
            return volume_materials.find(name) != volume_materials.end();
        }

        void remove_volume_material(const std::string& name) {
            auto it = volume_materials.find(name);
            if (it == volume_materials.end()) {
                throw std::runtime_error("VolumeMaterial with name '" + name + "' not found");
            }
            volume_materials.erase(it);
        }

        std::vector<std::string> get_volume_material_names() const {
            std::vector<std::string> names;
            names.reserve(volume_materials.size());
            for (const auto& pair : volume_materials) {
                names.push_back(pair.first);
            }
            return names;
        }

        size_t volume_material_count() const { return volume_materials.size(); }

        // ============= BSDF (SurfaceProperty) 管理 =============
        std::string add_bsdf(const std::string& name, std::shared_ptr<BSDF> bsdf) {
            std::string final_name = generate_unique_name(bsdfs, name);
            bsdfs[final_name] = bsdf;
            return final_name;
        }

        std::shared_ptr<BSDF> get_bsdf(const std::string& name) const {
            auto it = bsdfs.find(name);
            if (it == bsdfs.end()) {
                throw std::runtime_error("BSDF with name '" + name + "' not found");
            }
            return it->second;
        }

        template<typename T>
        std::shared_ptr<T> get_bsdf_as(const std::string& name) const {
            auto bsdf = get_bsdf(name);
            auto casted = std::dynamic_pointer_cast<T>(bsdf);
            if (!casted) {
                throw std::runtime_error("BSDF '" + name + "' is not of the requested type");
            }
            return casted;
        }

        bool has_bsdf(const std::string& name) const {
            return bsdfs.find(name) != bsdfs.end();
        }

        void remove_bsdf(const std::string& name) {
            auto it = bsdfs.find(name);
            if (it == bsdfs.end()) {
                throw std::runtime_error("BSDF with name '" + name + "' not found");
            }
            bsdfs.erase(it);
        }

        std::vector<std::string> get_bsdf_names() const {
            std::vector<std::string> names;
            names.reserve(bsdfs.size());
            for (const auto& pair : bsdfs) {
                names.push_back(pair.first);
            }
            return names;
        }

        size_t bsdf_count() const { return bsdfs.size(); }

        // ============= 便捷创建方法 - Spectrum =============
        
        std::shared_ptr<DiscreteSpectrum> create_discrete_spectrum(
            const std::string& name,
            const Float& wavelengths,
            const Float& values
        ) {
            auto spectrum = std::make_shared<DiscreteSpectrum>(wavelengths, values);
            add_spectrum(name, spectrum);
            return spectrum;
        }

        std::shared_ptr<BlackbodySpectrum> create_blackbody_spectrum(
            const std::string& name,
            Float temperature,
            Float wl_min = 380.0f,
            Float wl_max = 780.0f
        ) {
            auto spectrum = std::make_shared<BlackbodySpectrum>(temperature, wl_min, wl_max);
            add_spectrum(name, spectrum);
            return spectrum;
        }

        std::shared_ptr<GaussianSpectrum> create_gaussian_spectrum(
            const std::string& name,
            Float center,
            Float sigma,
            Float amplitude = 1.0f
        ) {
            auto spectrum = std::make_shared<GaussianSpectrum>(center, sigma, amplitude);
            add_spectrum(name, spectrum);
            return spectrum;
        }

        std::shared_ptr<ConstantSpectrum> create_constant_spectrum(
            const std::string& name,
            Float value,
            Float wl_min = 380.0f,
            Float wl_max = 780.0f
        ) {
            auto spectrum = std::make_shared<ConstantSpectrum>(value, wl_min, wl_max);
            add_spectrum(name, spectrum);
            return spectrum;
        }

        // ============= 便捷创建方法 - VolumeMaterial =============

        std::shared_ptr<VolumeMaterial> create_air(const std::string& name = "Air") {
            auto material = diff_optics::create_air();
            add_volume_material(name, material);
            return material;
        }

        std::shared_ptr<VolumeMaterial> create_nbk7(const std::string& name = "NBK7") {
            auto material = diff_optics::create_nbk7();
            add_volume_material(name, material);
            return material;
        }

        std::shared_ptr<VolumeMaterial> create_pmma(const std::string& name = "PMMA") {
            auto material = diff_optics::create_pmma();
            add_volume_material(name, material);
            return material;
        }

        std::shared_ptr<VolumeMaterial> create_vacuum(const std::string& name = "Vacuum") {
            auto material = diff_optics::create_vacuum();
            add_volume_material(name, material);
            return material;
        }

        std::shared_ptr<ConstantIORMaterial> create_constant_ior_material(
            const std::string& name,
            ScalarType ior,
            ScalarType transmittance = 1.0f,
            ScalarType measurement_depth = 10.0f
        ) {
            auto material = std::make_shared<ConstantIORMaterial>(ior, transmittance, measurement_depth);
            add_volume_material(name, material);
            return material;
        }

        // ============= 便捷创建方法 - BSDF =============

        std::shared_ptr<SpecularReflector> create_specular_reflector(
            const std::string& name,
            ScalarType reflectance = 1.0f
        ) {
            auto bsdf = std::make_shared<SpecularReflector>(reflectance);
            add_bsdf(name, bsdf);
            return bsdf;
        }

        std::shared_ptr<SpecularRefractor> create_specular_refractor(
            const std::string& name,
            ScalarType transmittance = 1.0f
        ) {
            auto bsdf = std::make_shared<SpecularRefractor>(transmittance);
            add_bsdf(name, bsdf);
            return bsdf;
        }

        std::shared_ptr<Absorber> create_absorber(const std::string& name) {
            auto bsdf = std::make_shared<Absorber>();
            add_bsdf(name, bsdf);
            return bsdf;
        }

        // ============= 便捷方法：为 Surface 设置材料 =============

        void set_surface_bsdf(const std::string& surface_name, const std::string& bsdf_name) {
            auto surface = get_surface(surface_name);
            auto bsdf = get_bsdf(bsdf_name);
            surface->set_bsdf(bsdf);
        }

        void set_surface_inner_material(const std::string& surface_name, const std::string& material_name) {
            auto surface = get_surface(surface_name);
            auto material = get_volume_material(material_name);
            surface->set_inner_material(material);
        }

        void set_surface_outer_material(const std::string& surface_name, const std::string& material_name) {
            auto surface = get_surface(surface_name);
            auto material = get_volume_material(material_name);
            surface->set_outer_material(material);
        }

        void set_surface_materials(
            const std::string& surface_name,
            const std::string& bsdf_name,
            const std::string& inner_material_name,
            const std::string& outer_material_name
        ) {
            set_surface_bsdf(surface_name, bsdf_name);
            set_surface_inner_material(surface_name, inner_material_name);
            set_surface_outer_material(surface_name, outer_material_name);
        }

        // ============= Surface-Sensor 映射管理 =============
        
        std::vector<std::string> get_sensors_for_surface(const std::string& surface_name) const {
            auto it = surface_to_sensors.find(surface_name);
            if (it == surface_to_sensors.end()) {
                return {};
            }
            return it->second;
        }

        bool surface_has_sensors(const std::string& surface_name) const {
            auto it = surface_to_sensors.find(surface_name);
            return it != surface_to_sensors.end() && !it->second.empty();
        }

        size_t get_sensor_count_for_surface(const std::string& surface_name) const {
            auto it = surface_to_sensors.find(surface_name);
            if (it == surface_to_sensors.end()) {
                return 0;
            }
            return it->second.size();
        }

        // ============= 便捷创建方法 - Emitter =============

        std::shared_ptr<UniformEmitter> create_uniform_emitter(const std::string& name = "UniformEmitter",const Float& upper_angle=M_PI * 0.5f) {
            auto emitter = std::make_shared<UniformEmitter>(0.0,upper_angle);
            add_emitter(name, emitter);
            return emitter;
        }

        std::shared_ptr<LambertEmitter> create_lambert_emitter(const std::string& name = "LambertEmitter") {
            auto emitter = std::make_shared<LambertEmitter>();
            add_emitter(name, emitter);
            return emitter;
        }

        // ============= 便捷创建方法 - Light =============

        std::shared_ptr<PointLight> create_point_light(const std::string& name, std::shared_ptr<Emitter> emitter, std::shared_ptr<Spectrum> spectrum) {
            auto light = std::make_shared<PointLight>(emitter, spectrum);
            add_light(name, light);
            return light;
        }
// 在 Scene 类中添加以下方法（在 create_point_light 之后）

// ============= 便捷创建方法 - SurfaceLight =============
std::shared_ptr<SurfaceLight> create_surface_light(
    const std::string& name,
    const std::string& surface_name,
    std::shared_ptr<Emitter> emitter,
    std::shared_ptr<Spectrum> spectrum
) {
    auto surface = get_surface(surface_name);
    auto light = std::make_shared<SurfaceLight>(surface, emitter, spectrum);
    add_light(name, light);
    return light;
}
        // ============= 便捷创建方法 - Surface =============

        std::shared_ptr<RectangleSurface> create_rectangle_bspline(
    const std::string& name = "BSplinePlane",
    const std::array<float, 2>& size = {1.0f, 1.0f},
    int u_degree = 3,
    int v_degree = 3,
    int u_num_cp = 8,
    int v_num_cp = 8,
    const std::vector<float>& control_points_z = {},  // 新增参数
    std::shared_ptr<VolumeMaterial> inner_material = nullptr,
    std::shared_ptr<VolumeMaterial> outer_material = nullptr,
    std::shared_ptr<BSDF> bsdf = nullptr
) {
    BSplineConfig config;
    config.u_degree = u_degree;
    config.v_degree = v_degree;
    config.u_control_points = u_num_cp;
    config.v_control_points = v_num_cp;
    config.control_points_z = control_points_z;  // 传递控制点
    
    auto surface = make_rectangle_bspline(size[0], size[1], config);
    surface->set_bsdf(bsdf);
    surface->set_inner_material(inner_material);
    surface->set_outer_material(outer_material);
    surface->set_tessellation(u_num_cp*4,v_num_cp*4);
    add_surface(name, surface);
    return surface;
}
/// 创建 XY 多项式曲面
std::shared_ptr<RectangleSurface> create_rectangle_xy(
    const std::string& name,
    const std::array<float, 2>& size,
    int order = 4,
    float b = 0.0f,
    const std::vector<float>& coefficients = {},
    std::shared_ptr<VolumeMaterial> inner_material = nullptr,
    std::shared_ptr<VolumeMaterial> outer_material = nullptr,
    std::shared_ptr<BSDF> bsdf = nullptr)
{
    XYPolynomialConfig config(order, b);
    if (!coefficients.empty()) {
        config.ai = coefficients;
    }
    
    auto shape = std::make_shared<XYPolynomialSurface>(size[0], size[1], config);
    auto surface = std::make_shared<RectangleSurface>(size[0], size[1], shape);
    
    if (inner_material) {
        surface->set_inner_material(inner_material);
    }
    if (outer_material) {
        surface->set_outer_material(outer_material);
    }
    if (bsdf) {
        surface->set_bsdf(bsdf);
    }
    
    add_surface(name, surface);
    return surface;
}

/// 创建位移网格曲面
std::shared_ptr<RectangleSurface> create_rectangle_displacement(
    const std::string& name,
    const std::array<float, 2>& size,
    int grid_width = 64,
    int grid_height = 64,
    const std::vector<float>& heightmap = {},
    std::shared_ptr<VolumeMaterial> inner_material = nullptr,
    std::shared_ptr<VolumeMaterial> outer_material = nullptr,
    std::shared_ptr<BSDF> bsdf = nullptr)
{
    DisplacementMeshConfig config(grid_width, grid_height);
    if (!heightmap.empty()) {
        config.heightmap = heightmap;
    }
    
    auto shape = std::make_shared<DisplacementMeshSurface>(size[0], size[1], config);
    auto surface = std::make_shared<RectangleSurface>(size[0], size[1], shape);
    
    if (inner_material) {
        surface->set_inner_material(inner_material);
    }
    if (outer_material) {
        surface->set_outer_material(outer_material);
    }
    if (bsdf) {
        surface->set_bsdf(bsdf);
    }
    
    add_surface(name, surface);
    return surface;
}
        std::shared_ptr<RectangleSurface> create_rectangle_plane(
            const std::string& name="RectanglePlane", 
            const std::array<float, 2>& size={1.0,1.0},
            std::shared_ptr<VolumeMaterial> inner_mat=nullptr,
            std::shared_ptr<VolumeMaterial> outer_mat = nullptr,
            std::shared_ptr<BSDF> bsdf = nullptr
        ) {
            auto surface = make_rectangle_plane(size[0], size[1]);
            surface->set_bsdf(bsdf);
            surface->set_inner_material(inner_mat);
            surface->set_outer_material(outer_mat);
            add_surface(name, surface);
            return surface;
        }

        std::shared_ptr<CircleSurface> create_circle_plane(
            const std::string& name, 
            ScalarType radius = 1.0f
        ) {
            auto surface = make_circle_plane(radius);
            add_surface(name, surface);
            return surface;
        }

        // ============= 便捷创建方法 - Sensor =============

        // IrradianceSensor
        // create_irradiance_sensor("IrradianceSensor", "RectanglePlane", [512, 512], [-5,5], [-5,5], "Box")
        std::shared_ptr<IrradianceSensor> create_irradiance_sensor(
            const std::string& name,
            const std::string& surface_name,
            const std::array<int, 2>& resolution,
            const std::string& filter_name
        ) {
            auto surface = get_surface(surface_name);
            FilterType filter = parse_filter_type(filter_name);
            
            auto sensor = std::make_shared<IrradianceSensor>(
                surface,
                resolution[0], resolution[1],
                filter
            );
            
            std::string sensor_final_name = add_sensor(name, sensor);
            surface_to_sensors[surface_name].push_back(sensor_final_name);
            
            return sensor;
        }

        // IntensitySensor
        // create_intensity_sensor("IntensitySensor", "RectanglePlane", "A", [360, 180], [-180,180], [-90,90], "Box")
        std::shared_ptr<IntensitySensor> create_intensity_sensor(
            const std::string& name,
            const std::string& surface_name,
            const std::string& ies_type_name,
            const std::array<int, 2>& resolution,
            const std::array<float, 2>& u_range,
            const std::array<float, 2>& v_range,
            const std::string& filter_name
        ) {
            auto surface = get_surface(surface_name);
            IESType ies_type = parse_ies_type(ies_type_name);
            FilterType filter = parse_filter_type(filter_name);
            
            // 将角度转换为弧度
            float u_min_rad = u_range[0] * M_PI / 180.0f;
            float u_max_rad = u_range[1] * M_PI / 180.0f;
            float v_min_rad = v_range[0] * M_PI / 180.0f;
            float v_max_rad = v_range[1] * M_PI / 180.0f;
            
            auto sensor = std::make_shared<IntensitySensor>(
                surface,
                ies_type,
                resolution[0], resolution[1],
                u_min_rad, u_max_rad,
                v_min_rad, v_max_rad,
                filter
            );
            
            std::string sensor_final_name = add_sensor(name, sensor);
            surface_to_sensors[surface_name].push_back(sensor_final_name);
            
            return sensor;
        }

        // FarFieldSensor
        // create_far_field_sensor("FarfieldSensor", "C", [360, 180], [0,360], [0,180], "Bilinear")
        std::shared_ptr<IntensitySensor> create_far_field_sensor(
            const std::string& name,
            const std::string& ies_type_name,
            const std::array<int, 2>& resolution,
            const std::array<float, 2>& u_range,
            const std::array<float, 2>& v_range,
            const std::string& filter_name
        ) {
            IESType ies_type = parse_ies_type(ies_type_name);
            FilterType filter = parse_filter_type(filter_name);
            
            // 使用远场构造函数
            auto sensor = std::make_shared<IntensitySensor>(
                ies_type,
                resolution[0], resolution[1],
                u_range[0], u_range[1],
                v_range[0], v_range[1],
                filter
            );
            
            std::string sensor_final_name = add_sensor(name, sensor);
            farfield_sensors.insert(sensor_final_name);
            
            return sensor;
        }

        // ============= 场景操作 =============
        void clear() {
            emitters.clear();
        lights.clear();
        surfaces.clear();
        shells.clear();   // 新增
        solids.clear();   // 新增
        sensors.clear();
        surface_to_sensors.clear();
        farfield_sensors.clear();
        spectra.clear();
        volume_materials.clear();
        bsdfs.clear();
        }

        void print(const std::string& name = "Scene") const {
            std::cout << name << ":" << std::endl;
            
            std::cout << "  Emitters (" << emitters.size() << "):" << std::endl;
            for (const auto& pair : emitters) {
                std::cout << "    - " << pair.first << std::endl;
            }
            
            std::cout << "  Lights (" << lights.size() << "):" << std::endl;
            for (const auto& pair : lights) {
                std::cout << "    - " << pair.first << std::endl;
            }
            
            std::cout << "  Surfaces (" << surfaces.size() << "):" << std::endl;
            for (const auto& pair : surfaces) {
                std::cout << "    - " << pair.first << " (" << pair.second->type_name() << ")";
                if (pair.second->get_bsdf()) {
                    std::cout << " [bsdf: " << pair.second->get_bsdf()->type_name() << "]";
                }
                auto sensors_it = surface_to_sensors.find(pair.first);
                if (sensors_it != surface_to_sensors.end() && !sensors_it->second.empty()) {
                    std::cout << " -> sensors: [";
                    for (size_t i = 0; i < sensors_it->second.size(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << sensors_it->second[i];
                    }
                    std::cout << "]";
                }
                std::cout << std::endl;
            }

            std::cout << "  Sensors (" << sensors.size() << "):" << std::endl;
            for (const auto& pair : sensors) {
                std::cout << "    - " << pair.first << " (" << pair.second->type_name() << ")";
                if (is_farfield_sensor(pair.first)) {
                    std::cout << " [farfield]";
                }
                std::cout << std::endl;
            }

            std::cout << "  Spectra (" << spectra.size() << "):" << std::endl;
            for (const auto& pair : spectra) {
                std::cout << "    - " << pair.first << " (" << pair.second->type_name() << ")" << std::endl;
            }

            std::cout << "  Volume Materials (" << volume_materials.size() << "):" << std::endl;
            for (const auto& pair : volume_materials) {
                std::cout << "    - " << pair.first << " (" << pair.second->type_name() << ")" << std::endl;
            }

            std::cout << "  BSDFs (" << bsdfs.size() << "):" << std::endl;
            for (const auto& pair : bsdfs) {
                std::cout << "    - " << pair.first << " (" << pair.second->type_name() << ")" << std::endl;
            }

            if (!farfield_sensors.empty()) {
                std::cout << "  Far-field Sensors (" << farfield_sensors.size() << "):" << std::endl;
                for (const auto& name : farfield_sensors) {
                    std::cout << "    - " << name << std::endl;
                }
            }
            std::cout << "  Shells (" << shells.size() << "):" << std::endl;
        for (const auto& pair : shells) {
            std::cout << "    - " << pair.first << " (" << pair.second->type_name() 
                      << ", " << pair.second->surface_count() << " surfaces)" << std::endl;
        }
        
        std::cout << "  Solids (" << solids.size() << "):" << std::endl;
        for (const auto& pair : solids) {
            std::cout << "    - " << pair.first << " (" << pair.second->type_name() 
                      << ", " << pair.second->surface_count() << " surfaces)" << std::endl;
        }
        
        std::cout << "  Total Surfaces for OptiX: " << total_surface_count() << std::endl;
        

        }

    private:
        std::unordered_map<std::string, std::shared_ptr<Emitter>> emitters;
        std::unordered_map<std::string, std::shared_ptr<Light>> lights;
        std::unordered_map<std::string, std::shared_ptr<Surface>> surfaces;
        std::unordered_map<std::string, std::shared_ptr<Sensor>> sensors;
        std::unordered_map<std::string, std::shared_ptr<Shell>> shells;   // 新增
        std::unordered_map<std::string, std::shared_ptr<Solid>> solids;   // 新增

        // 新增资源
        std::unordered_map<std::string, std::shared_ptr<Spectrum>> spectra;
        std::unordered_map<std::string, std::shared_ptr<VolumeMaterial>> volume_materials;
        std::unordered_map<std::string, std::shared_ptr<BSDF>> bsdfs;

        std::unordered_map<std::string, std::vector<std::string>> surface_to_sensors;
        std::unordered_set<std::string> farfield_sensors;

        template<typename T>
        std::string generate_unique_name(
            const std::unordered_map<std::string, std::shared_ptr<T>>& map,
            const std::string& base_name
        ) const {
            if (map.find(base_name) == map.end()) {
                return base_name;
            }
            int counter = 1;
            std::string new_name;
            do {
                new_name = base_name + "_" + std::to_string(counter);
                counter++;
            } while (map.find(new_name) != map.end());
            return new_name;
        }
    };

} // namespace diff_optics