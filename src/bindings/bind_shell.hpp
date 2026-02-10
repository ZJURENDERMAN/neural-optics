// bindings/bind_shell.hpp
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include "../shell.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_shell(nb::module_& m) {
    
    // ============= Shell 基类 =============
    nb::class_<Shell, Object>(m, "Shell")
        .def("get_num_rows", &Shell::get_num_rows)
        .def("get_num_cols", &Shell::get_num_cols)
        .def("get_surface", &Shell::get_surface, "row"_a, "col"_a)
        .def("surface_count", &Shell::surface_count);
    
    // ============= RectangleShell =============
    nb::class_<RectangleShell, Shell>(m, "RectangleShell")
        .def("get_width", &RectangleShell::get_width)
        .def("get_height", &RectangleShell::get_height)
        .def("get_cell_width", &RectangleShell::get_cell_width)
        .def("get_cell_height", &RectangleShell::get_cell_height)
        .def("get_u_gaps", &RectangleShell::get_u_gaps)
        .def("get_v_gaps", &RectangleShell::get_v_gaps)
        .def("set_u_gap", &RectangleShell::set_u_gap, "index"_a, "gap"_a)
        .def("set_v_gap", &RectangleShell::set_v_gap, "index"_a, "gap"_a);
    
    // ============= RectanglePlaneShell =============
    //nb::class_<RectanglePlaneShell, RectangleShell>(m, "RectanglePlaneShell")
    //    .def(nb::init<int, int, ScalarType, ScalarType>(),
    //         "rows"_a, "cols"_a, "cell_width"_a, "cell_height"_a)
    //    .def("__repr__", [](const RectanglePlaneShell& s) {
    //        return "RectanglePlaneShell(" + std::to_string(s.get_num_rows()) + "x" +
    //               std::to_string(s.get_num_cols()) + ", cell=" +
    //               std::to_string(s.get_cell_width()) + "x" + 
    //               std::to_string(s.get_cell_height()) + ")";
    //    });
    //
    //// ============= RectangleBSplineShell =============
    //nb::class_<RectangleBSplineShell, RectangleShell>(m, "RectangleBSplineShell")
    //    .def(nb::init<int, int, ScalarType, ScalarType, const BSplineConfig&>(),
    //         "rows"_a, "cols"_a, "cell_width"_a, "cell_height"_a,
    //         "config"_a = BSplineConfig())
    //    .def("get_global_cp_z", 
    //         [](RectangleBSplineShell& s) -> Float& { return s.get_global_cp_z(); },
    //         nb::rv_policy::reference_internal)
    //    .def("set_global_cp_z", &RectangleBSplineShell::set_global_cp_z, "z"_a)
    //    .def("get_global_u_cp", &RectangleBSplineShell::get_global_u_cp)
    //    .def("get_global_v_cp", &RectangleBSplineShell::get_global_v_cp)
    //    .def("get_total_control_points", &RectangleBSplineShell::get_total_control_points)
    //    .def("get_bspline_surface", &RectangleBSplineShell::get_bspline_surface,
    //         "row"_a, "col"_a)
    //    .def("__repr__", [](const RectangleBSplineShell& s) {
    //        return "RectangleBSplineShell(" + std::to_string(s.get_num_rows()) + "x" +
    //               std::to_string(s.get_num_cols()) + ", global_cp=" +
    //               std::to_string(s.get_total_control_points()) + ")";
    //    });
}