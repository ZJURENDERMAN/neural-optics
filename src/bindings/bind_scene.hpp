#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include "../scene.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_scene(nb::module_& m) {
    nb::class_<Scene>(m, "Scene")
        .def(nb::init<>(), "Create an empty scene")

        // ============= Emitter 管理 =============
        .def("add_emitter", &Scene::add_emitter, "name"_a, "emitter"_a)
        .def("get_emitter", &Scene::get_emitter, "name"_a)
        .def("has_emitter", &Scene::has_emitter, "name"_a)
        .def("remove_emitter", &Scene::remove_emitter, "name"_a)
        .def("get_emitter_names", &Scene::get_emitter_names)
        .def("emitter_count", &Scene::emitter_count)

        // ============= Light 管理 =============
        .def("add_light", &Scene::add_light, "name"_a, "light"_a)
        .def("get_light", &Scene::get_light, "name"_a)
        .def("has_light", &Scene::has_light, "name"_a)
        .def("remove_light", &Scene::remove_light, "name"_a)
        .def("get_light_names", &Scene::get_light_names)
        .def("light_count", &Scene::light_count)

        // ============= Surface 管理 =============
        .def("add_surface", &Scene::add_surface, "name"_a, "surface"_a)
        .def("get_surface", &Scene::get_surface, "name"_a)
        .def("has_surface", &Scene::has_surface, "name"_a)
        .def("remove_surface", &Scene::remove_surface, "name"_a)
        .def("get_surface_names", &Scene::get_surface_names)
        .def("surface_count", &Scene::surface_count)

        // ============= Sensor 管理 =============
        .def("add_sensor", &Scene::add_sensor, "name"_a, "sensor"_a)
        .def("get_sensor", &Scene::get_sensor, "name"_a)
        .def("has_sensor", &Scene::has_sensor, "name"_a)
        .def("remove_sensor", &Scene::remove_sensor, "name"_a)
        .def("get_sensor_names", &Scene::get_sensor_names)
        .def("sensor_count", &Scene::sensor_count)

        // ============= Spectrum 管理 =============
        .def("add_spectrum", &Scene::add_spectrum, "name"_a, "spectrum"_a,
             "Add a spectrum to the scene, returns final name")
        .def("get_spectrum", &Scene::get_spectrum, "name"_a,
             "Get a spectrum by name")
        .def("has_spectrum", &Scene::has_spectrum, "name"_a,
             "Check if a spectrum exists")
        .def("remove_spectrum", &Scene::remove_spectrum, "name"_a,
             "Remove a spectrum by name")
        .def("get_spectrum_names", &Scene::get_spectrum_names,
             "Get all spectrum names")
        .def("spectrum_count", &Scene::spectrum_count,
             "Get number of spectra")

        // ============= VolumeMaterial 管理 =============
        .def("add_volume_material", &Scene::add_volume_material, "name"_a, "material"_a,
             "Add a volume material to the scene, returns final name")
        .def("get_volume_material", &Scene::get_volume_material, "name"_a,
             "Get a volume material by name")
        .def("has_volume_material", &Scene::has_volume_material, "name"_a,
             "Check if a volume material exists")
        .def("remove_volume_material", &Scene::remove_volume_material, "name"_a,
             "Remove a volume material by name")
        .def("get_volume_material_names", &Scene::get_volume_material_names,
             "Get all volume material names")
        .def("volume_material_count", &Scene::volume_material_count,
             "Get number of volume materials")

        // ============= BSDF 管理 =============
        .def("add_bsdf", &Scene::add_bsdf, "name"_a, "bsdf"_a,
             "Add a BSDF to the scene, returns final name")
        .def("get_bsdf", &Scene::get_bsdf, "name"_a,
             "Get a BSDF by name")
        .def("has_bsdf", &Scene::has_bsdf, "name"_a,
             "Check if a BSDF exists")
        .def("remove_bsdf", &Scene::remove_bsdf, "name"_a,
             "Remove a BSDF by name")
        .def("get_bsdf_names", &Scene::get_bsdf_names,
             "Get all BSDF names")
        .def("bsdf_count", &Scene::bsdf_count,
             "Get number of BSDFs")

        // ============= Surface-Sensor 映射 =============
        .def("get_sensors_for_surface", &Scene::get_sensors_for_surface, "surface_name"_a)
        .def("surface_has_sensors", &Scene::surface_has_sensors, "surface_name"_a)
        .def("get_sensor_count_for_surface", &Scene::get_sensor_count_for_surface, "surface_name"_a)

        // ============= 远场传感器查询 =============
        .def("get_farfield_sensors", &Scene::get_farfield_sensors)
        .def("is_farfield_sensor", &Scene::is_farfield_sensor, "name"_a)
        .def("farfield_sensor_count", &Scene::farfield_sensor_count)

        // ============= 便捷创建方法 - Emitter =============
        .def("create_uniform_emitter", &Scene::create_uniform_emitter,
            "name"_a = "UniformEmitter","upper_angle"_a=0.1)
        .def("create_lambert_emitter", &Scene::create_lambert_emitter,
            "name"_a = "LambertEmitter")

        // ============= 便捷创建方法 - Light =============
        .def("create_point_light", &Scene::create_point_light,
            "name"_a="PointLight", "emitter"_a,"spectrum"_a)
            // 在 .def("create_point_light", ...) 之后添加

        .def("create_surface_light", &Scene::create_surface_light,
            "name"_a="SurfaceLight", "surface_name"_a, "emitter"_a, "spectrum"_a,
            "Create a surface light from an existing surface in the scene")

        // ============= 便捷创建方法 - Surface =============
        .def("create_rectangle_plane", &Scene::create_rectangle_plane,
            "name"_a = "RectanglePlane",
            "size"_a = std::array<float, 2>{1.0f, 1.0f},
            "inner_material"_a = nullptr,
            "outer_material"_a = nullptr,
            "bsdf"_a = nullptr)
        .def("create_circle_plane", &Scene::create_circle_plane,
            "name"_a = "CirclePlane", "radius"_a = 1.0f)
        // 在 bind_scene.hpp 或类似文件中
        // 替换原来的 create_rectangle_bspline 绑定
.def("create_rectangle_bspline", 
    [](Scene& self, 
       const std::string& name, 
       const std::array<float, 2>& size,
       int u_degree, int v_degree,
       int u_num_cp, int v_num_cp,
       nb::object control_points_z_obj,  // 使用 nb::object 接收
       std::shared_ptr<VolumeMaterial> inner_material,
       std::shared_ptr<VolumeMaterial> outer_material,
       std::shared_ptr<BSDF> bsdf) {
        
        // 手动处理 None -> 空 vector
        std::vector<float> control_points_z;
        if (!control_points_z_obj.is_none()) {
            control_points_z = nb::cast<std::vector<float>>(control_points_z_obj);
        }
        
        return self.create_rectangle_bspline(
            name, size, 
            u_degree, v_degree, 
            u_num_cp, v_num_cp, 
            control_points_z,  // 传递空 vector 或实际数据
            inner_material, outer_material, bsdf
        );
    },
    "name"_a, "size"_a,
    "u_degree"_a = 3, "v_degree"_a = 3,
    "u_num_cp"_a = 8, "v_num_cp"_a = 8,
    "control_points_z"_a = nb::none(),
    "inner_material"_a = nullptr, "outer_material"_a = nullptr, "bsdf"_a = nullptr,
    "Create a rectangle B-spline surface")

    .def("create_rectangle_xy", 
    [](Scene& self, 
       const std::string& name, 
       const std::array<float, 2>& size,
       int order,
       float b,
       nb::object coefficients_obj,
       std::shared_ptr<VolumeMaterial> inner_material,
       std::shared_ptr<VolumeMaterial> outer_material,
       std::shared_ptr<BSDF> bsdf) {
        
        std::vector<float> coefficients;
        if (!coefficients_obj.is_none()) {
            coefficients = nb::cast<std::vector<float>>(coefficients_obj);
        }
        
        return self.create_rectangle_xy(
            name, size, order, b, coefficients,
            inner_material, outer_material, bsdf
        );
    },
    "name"_a, "size"_a,
    "order"_a = 4, "b"_a = 0.0f,
    "coefficients"_a = nb::none(),
    "inner_material"_a = nullptr, "outer_material"_a = nullptr, "bsdf"_a = nullptr,
    "Create a rectangle XY polynomial surface")

.def("create_rectangle_heightmap", 
    [](Scene& self, 
       const std::string& name, 
       const std::array<float, 2>& size,
       int grid_width,
       int grid_height,
       nb::object heightmap_obj,
       std::shared_ptr<VolumeMaterial> inner_material,
       std::shared_ptr<VolumeMaterial> outer_material,
       std::shared_ptr<BSDF> bsdf) {
        
        std::vector<float> heightmap;
        if (!heightmap_obj.is_none()) {
            heightmap = nb::cast<std::vector<float>>(heightmap_obj);
        }
        
        return self.create_rectangle_displacement(
            name, size, grid_width, grid_height, heightmap,
            inner_material, outer_material, bsdf
        );
    },
    "name"_a, "size"_a,
    "grid_width"_a = 64, "grid_height"_a = 64,
    "heightmap"_a = nb::none(),
    "inner_material"_a = nullptr, "outer_material"_a = nullptr, "bsdf"_a = nullptr,
    "Create a rectangle displacement mesh surface")
    
        // ============= 便捷创建方法 - Sensor =============
        .def("create_irradiance_sensor", &Scene::create_irradiance_sensor,
            "name"_a, "surface_name"_a, "resolution"_a, "filter"_a,
            "Create an irradiance sensor on a surface")
        .def("create_intensity_sensor", &Scene::create_intensity_sensor,
            "name"_a, "surface_name"_a, "ies_type"_a, "resolution"_a, "u_range"_a, "v_range"_a, "filter"_a,
            "Create an intensity sensor on a surface")
        .def("create_far_field_sensor", &Scene::create_far_field_sensor,
            "name"_a, "ies_type"_a, "resolution"_a, "u_range"_a, "v_range"_a, "filter"_a,
            "Create a far-field sensor")

        // ============= 便捷创建方法 - Spectrum =============
        .def("create_discrete_spectrum", &Scene::create_discrete_spectrum,
            "name"_a, "wavelengths"_a, "values"_a,
            "Create a discrete spectrum from wavelength and value arrays")
        .def("create_blackbody_spectrum", &Scene::create_blackbody_spectrum,
            "name"_a, "temperature"_a, "wl_min"_a = 380.0f, "wl_max"_a = 780.0f,
            "Create a blackbody spectrum with specified temperature (K)")
        .def("create_gaussian_spectrum", &Scene::create_gaussian_spectrum,
            "name"_a, "center"_a, "sigma"_a, "amplitude"_a = 1.0f,
            "Create a Gaussian spectrum")
        .def("create_constant_spectrum", &Scene::create_constant_spectrum,
            "name"_a, "value"_a, "wl_min"_a = 380.0f, "wl_max"_a = 780.0f,
            "Create a constant spectrum")

        // ============= 便捷创建方法 - VolumeMaterial =============
        .def("create_air", &Scene::create_air,
            "name"_a = "Air",
            "Create an air material")
        .def("create_nbk7", &Scene::create_nbk7,
            "name"_a = "NBK7",
            "Create an N-BK7 optical glass material")
        .def("create_pmma", &Scene::create_pmma,
            "name"_a = "PMMA",
            "Create a PMMA (acrylic) material")
        .def("create_vacuum", &Scene::create_vacuum,
            "name"_a = "Vacuum",
            "Create a vacuum material")
        .def("create_constant_ior_material", &Scene::create_constant_ior_material,
            "name"_a, "ior"_a, "transmittance"_a = 1.0f, "measurement_depth"_a = 10.0f,
            "Create a material with constant IOR")

        // ============= 便捷创建方法 - BSDF =============
        .def("create_specular_reflector", &Scene::create_specular_reflector,
            "name"_a="SpecularReflector", "reflectance"_a = 1.0f,
            "Create a specular reflector")
        .def("create_specular_refractor", &Scene::create_specular_refractor,
            "name"_a="SpecularRefractor", "transmittance"_a = 1.0f,
            "Create a specular refractor")
        .def("create_absorber", &Scene::create_absorber,
            "name"_a,
            "Create an absorber")

        // ============= 便捷方法：为 Surface 设置材料 =============
        .def("set_surface_bsdf", &Scene::set_surface_bsdf,
            "surface_name"_a, "bsdf_name"_a,
            "Set the BSDF for a surface")
        .def("set_surface_inner_material", &Scene::set_surface_inner_material,
            "surface_name"_a, "material_name"_a,
            "Set the inner volume material for a surface")
        .def("set_surface_outer_material", &Scene::set_surface_outer_material,
            "surface_name"_a, "material_name"_a,
            "Set the outer volume material for a surface")
        .def("set_surface_materials", &Scene::set_surface_materials,
            "surface_name"_a, "bsdf_name"_a, "inner_material_name"_a, "outer_material_name"_a,
            "Set all materials for a surface at once")

        // ============= 全局操作 =============
        .def("clear", &Scene::clear)
        .def("print", &Scene::print, "name"_a = "Scene")

        .def("__repr__", [](const Scene& s) {
            return "Scene(emitters=" + std::to_string(s.emitter_count()) +
                   ", lights=" + std::to_string(s.light_count()) +
                   ", surfaces=" + std::to_string(s.surface_count()) +
                   ", sensors=" + std::to_string(s.sensor_count()) +
                   ", spectra=" + std::to_string(s.spectrum_count()) +
                   ", materials=" + std::to_string(s.volume_material_count()) +
                   ", bsdfs=" + std::to_string(s.bsdf_count()) + ")";
        });
}