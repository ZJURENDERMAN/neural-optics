// bindings/bind_shape.hpp
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/map.h>
#include "../shape.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_shape(nb::module_& m) {
    
    // ============= BSplineConfig =============
    nb::class_<BSplineConfig>(m, "BSplineConfig")
        .def(nb::init<>())
        .def(nb::init<int, int, int, int, const std::vector<ScalarType>&>(),
             "u_degree"_a, "v_degree"_a, "u_num_cp"_a, "v_num_cp"_a, "control_points_z"_a = std::vector<ScalarType>())
        .def_rw("u_degree", &BSplineConfig::u_degree)
        .def_rw("v_degree", &BSplineConfig::v_degree)
        .def_rw("u_control_points", &BSplineConfig::u_control_points)
        .def_rw("v_control_points", &BSplineConfig::v_control_points)
        .def_rw("control_points_z", &BSplineConfig::control_points_z);

    // ============= Shape 基类 =============
    nb::class_<Shape>(m, "Shape")
        .def("type_name", &Shape::type_name)
        // 统一的可微分参数接口
        .def("get_diff_param_count", &Shape::get_diff_param_count,
             "Get number of differentiable parameters")
        .def("get_diff_params", 
             [](Shape& s) -> Float& { return s.get_diff_params(); },
             nb::rv_policy::reference_internal,
             "Get differentiable parameters (GPU array)")
        .def("set_diff_params", &Shape::set_diff_params, "params"_a,
             "Set differentiable parameters (GPU array)")
        .def("get_diff_params_cpu", &Shape::get_diff_params_cpu,
             "Get differentiable parameters as CPU list")
        .def("set_diff_params_cpu", &Shape::set_diff_params_cpu, "params"_a,
             "Set differentiable parameters from CPU list")
        .def("get_param_config_string", &Shape::get_param_config_string,
             "Get parameter configuration as string")
        .def("get_param_config", &Shape::get_param_config,
             "Get parameter configuration as dict")
        .def("resize_params", &Shape::resize_params, "new_config"_a,
             "Resize parameters according to config dict")
          .def("save_cad", &Shape::save_cad, "filename"_a,
         "Export shape to STEP file (requires OpenCASCADE)");

    // ============= Plane =============
    nb::class_<Plane, Shape>(m, "Plane")
        .def(nb::init<>())
        .def("__repr__", [](const Plane&) { return "Plane()"; });

    // ============= BSplineSurface =============
    nb::class_<BSplineSurface, Shape>(m, "BSplineSurface")
        .def(nb::init<>())
        .def(nb::init<ScalarType, ScalarType, const BSplineConfig&>(),
             "width"_a, "height"_a, "config"_a = BSplineConfig())
        .def_ro("u_degree", &BSplineSurface::u_degree)
        .def_ro("v_degree", &BSplineSurface::v_degree)
        .def_ro("u_num_cp", &BSplineSurface::u_num_cp)
        .def_ro("v_num_cp", &BSplineSurface::v_num_cp)
        .def_ro("width", &BSplineSurface::width)
        .def_ro("height", &BSplineSurface::height)
        .def("num_control_points", &BSplineSurface::num_control_points)
        .def("get_config", &BSplineSurface::get_config)
        .def("resize_control_points", &BSplineSurface::resize_control_points,
             "new_u_num_cp"_a, "new_v_num_cp"_a)
        // 旧接口（保持兼容）
        .def("get_parameters", 
             [](BSplineSurface& s) -> Float& { return s.get_parameters(); },
             nb::rv_policy::reference_internal)
        .def("set_parameters", &BSplineSurface::set_parameters, "z"_a)
        .def("__repr__", [](const BSplineSurface& s) {
            return "BSplineSurface(" + std::to_string(s.u_num_cp) + "x" + 
                   std::to_string(s.v_num_cp) + ")";
        });

    // ============= XYPolynomialConfig =============
    nb::class_<XYPolynomialConfig>(m, "XYPolynomialConfig")
        .def(nb::init<>())
        .def(nb::init<int, ScalarType>(), "order"_a, "b"_a = 0.0f)
        .def(nb::init<int, ScalarType, const std::vector<ScalarType>&>(),
             "order"_a, "b"_a, "ai"_a)
        .def_rw("order", &XYPolynomialConfig::order)
        .def_rw("b", &XYPolynomialConfig::b)
        .def_rw("ai", &XYPolynomialConfig::ai);
    
    m.def("xy_polynomial_num_coeffs", &XYPolynomialConfig::compute_num_coeffs,
          "order"_a, "Compute number of coefficients for given order");

    // ============= XYPolynomialSurface =============
    nb::class_<XYPolynomialSurface, Shape>(m, "XYPolynomialSurface")
        .def(nb::init<>())
        .def(nb::init<ScalarType, ScalarType, const XYPolynomialConfig&>(),
             "width"_a, "height"_a, "config"_a = XYPolynomialConfig())
        .def_ro("order", &XYPolynomialSurface::order)
        .def_ro("num_coeffs", &XYPolynomialSurface::num_coeffs)
        .def_ro("width", &XYPolynomialSurface::width)
        .def_ro("height", &XYPolynomialSurface::height)
        .def("get_order", &XYPolynomialSurface::get_order)
        .def("get_num_coeffs", &XYPolynomialSurface::get_num_coeffs)
        .def("get_config", &XYPolynomialSurface::get_config)
        .def("resize_order", &XYPolynomialSurface::resize_order, "new_order"_a)
        // 旧接口
        .def("get_coefficients", 
             [](XYPolynomialSurface& s) -> Float& { return s.get_coefficients(); },
             nb::rv_policy::reference_internal)
        .def("set_coefficients", &XYPolynomialSurface::set_coefficients, "coeffs"_a)
        .def("get_coefficients_as_array", &XYPolynomialSurface::get_coefficients_as_array)
        .def("set_coefficients_from_array", &XYPolynomialSurface::set_coefficients_from_array, "coeffs"_a)
        .def("get_b", 
             [](XYPolynomialSurface& s) -> Float& { return s.get_b(); },
             nb::rv_policy::reference_internal)
        .def("set_b", &XYPolynomialSurface::set_b, "b"_a)
        .def("get_b_as_scalar", &XYPolynomialSurface::get_b_as_scalar)
        .def("set_b_from_scalar", &XYPolynomialSurface::set_b_from_scalar, "b"_a)
        .def("reverse", &XYPolynomialSurface::reverse)
        .def("__repr__", [](const XYPolynomialSurface& s) {
            return "XYPolynomialSurface(order=" + std::to_string(s.order) + ")";
        });

    // ============= DisplacementMeshConfig =============
    nb::class_<DisplacementMeshConfig>(m, "DisplacementMeshConfig")
        .def(nb::init<>())
        .def(nb::init<int, int>(), "width"_a, "height"_a)
        .def(nb::init<int, int, const std::vector<ScalarType>&>(),
             "width"_a, "height"_a, "heightmap"_a)
        .def_rw("width", &DisplacementMeshConfig::width)
        .def_rw("height", &DisplacementMeshConfig::height)
        .def_rw("heightmap", &DisplacementMeshConfig::heightmap)
        .def("compute_num_pixels", &DisplacementMeshConfig::compute_num_pixels);

    // ============= DisplacementMeshSurface =============
    nb::class_<DisplacementMeshSurface, Shape>(m, "DisplacementMeshSurface")
        .def(nb::init<>())
        .def(nb::init<ScalarType, ScalarType, const DisplacementMeshConfig&>(),
             "width"_a, "height"_a, "config"_a = DisplacementMeshConfig())
        .def_ro("grid_width", &DisplacementMeshSurface::grid_width)
        .def_ro("grid_height", &DisplacementMeshSurface::grid_height)
        .def_ro("num_pixels", &DisplacementMeshSurface::num_pixels)
        .def_ro("surface_width", &DisplacementMeshSurface::surface_width)
        .def_ro("surface_height", &DisplacementMeshSurface::surface_height)
        .def("get_grid_size", &DisplacementMeshSurface::get_grid_size)
        .def("get_surface_size", &DisplacementMeshSurface::get_surface_size)
        .def("resize_grid", &DisplacementMeshSurface::resize_grid,
             "new_width"_a, "new_height"_a)
        // 旧接口
        .def("get_heightmap", 
             [](DisplacementMeshSurface& s) -> Float& { return s.get_heightmap(); },
             nb::rv_policy::reference_internal)
        .def("set_heightmap", &DisplacementMeshSurface::set_heightmap, "hmap"_a)
        .def("get_heightmap_as_array", &DisplacementMeshSurface::get_heightmap_as_array)
        .def("set_heightmap_from_array", &DisplacementMeshSurface::set_heightmap_from_array, "data"_a)
        .def("get_heightmap_as_2d", &DisplacementMeshSurface::get_heightmap_as_2d)
        .def("set_heightmap_from_2d", &DisplacementMeshSurface::set_heightmap_from_2d, "data"_a)
        .def("reverse", &DisplacementMeshSurface::reverse)
        .def("get_pixel", &DisplacementMeshSurface::get_pixel, "i"_a, "j"_a)
        .def("set_pixel", &DisplacementMeshSurface::set_pixel, "i"_a, "j"_a, "value"_a)
        .def("__repr__", [](const DisplacementMeshSurface& s) {
            return "DisplacementMeshSurface(" + std::to_string(s.grid_width) + 
                   "x" + std::to_string(s.grid_height) + ")";
        });
}