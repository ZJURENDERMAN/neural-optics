// bindings/bind_surface.hpp
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/pair.h>
#include "../surface.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_surface(nb::module_& m) {

    // ============= TriangleMesh =============
    nb::class_<TriangleMesh>(m, "TriangleMesh")
        .def(nb::init<>())
        .def_ro("num_vertices", &TriangleMesh::num_vertices)
        .def_ro("num_triangles", &TriangleMesh::num_triangles)
        .def("is_valid", &TriangleMesh::is_valid)
        .def("__repr__", [](const TriangleMesh& m) {
            return "TriangleMesh(vertices=" + std::to_string(m.num_vertices) +
                   ", triangles=" + std::to_string(m.num_triangles) + ")";
        });

    // ============= 抽象基类 Surface（继承自 Object）=============
    nb::class_<Surface, Object>(m, "Surface")
        .def("type_name", &Surface::type_name,
             "Get the surface type name")
        .def("boundary_type", &Surface::boundary_type,
             "Get the boundary type name")
        .def("get_shape", &Surface::get_shape,
             "Get the shape")
        // BSDF 属性
        .def_prop_rw("bsdf", &Surface::get_bsdf, &Surface::set_bsdf,
                     "Surface BSDF (bidirectional scattering distribution function)")
        // 材料属性
        .def_prop_rw("inner_material", &Surface::get_inner_material, &Surface::set_inner_material,
                     "Inner volume material (opposite to normal direction)")
        .def_prop_rw("outer_material", &Surface::get_outer_material, &Surface::set_outer_material,
                     "Outer volume material (normal direction)")
        // 细分参数
        .def("set_tessellation", &Surface::set_tessellation,
             "tess_u"_a, "tess_v"_a,
             "Set tessellation parameters")
        .def("get_tess_u", &Surface::get_tess_u,
             "Get tessellation parameter u")
        .def("get_tess_v", &Surface::get_tess_v,
             "Get tessellation parameter v")
        .def("get_mesh", &Surface::get_mesh,
             "Get the triangle mesh (generates if needed)")
        .def("invalidate_mesh", &Surface::invalidate_mesh,
             "Invalidate the cached mesh")
        .def("update_mesh_vertices", &Surface::update_mesh_vertices,
             "Invalidate the cached mesh")
        .def("print", &Surface::print, "name"_a = "",
             "Print surface info");
    
    // ============= RectangleSurface=============
    nb::class_<RectangleSurface, Surface>(m, "RectagleSurface")
        .def("get_width", &RectangleSurface::get_width,
             "Get the width")
        .def("get_height", &RectangleSurface::get_height,
             "Get the height")
        .def("set_width", &RectangleSurface::set_width,
             "val"_a, "Set the width")
        .def("set_height", &RectangleSurface::set_height,
             "val"_a, "Set the height");
    
    // ============= CircleSurface=============
    nb::class_<CircleSurface, Surface>(m, "CircleSurface")
        .def("get_radius", &CircleSurface::get_radius,
             "Get the radius")
        .def("set_radius", &CircleSurface::set_radius,
             "val"_a, "Set the radius");
    
//     // ============= RectanglePlaneSurface =============
//     nb::class_<RectanglePlaneSurface, RectangleSurface>(m, "RectanglePlaneSurface")
//         .def(nb::init<>(), 
//              "Create a rectangle plane surface with default size (1x1)")
//         .def(nb::init<ScalarType, ScalarType>(),
//              "width"_a, "height"_a,
//              "Create a rectangle plane surface with specified dimensions")
//         .def("__repr__", [](const RectanglePlaneSurface& s) {
//             std::string repr = "RectanglePlaneSurface(width=" + 
//                    std::to_string(s.get_width()) + 
//                    ", height=" + std::to_string(s.get_height()) +
//                    ", tess=" + std::to_string(s.get_tess_u()) + "x" + std::to_string(s.get_tess_v());
//             if (s.get_bsdf()) {
//                 repr += ", bsdf=" + s.get_bsdf()->type_name();
//             }
//             repr += ")";
//             return repr;
//         });
    
//     // ============= RectangleBSplineSurface =============
//     nb::class_<RectangleBSplineSurface, RectangleSurface>(m, "RectangleBSplineSurface")
//         .def(nb::init<>(),
//              "Create a B-spline surface with default size (1x1)")
//         .def(nb::init<ScalarType, ScalarType, const BSplineConfig&>(),
//              "width"_a, "height"_a, "config"_a = BSplineConfig(),
//              "Create a B-spline surface with specified dimensions and config")
//         .def("num_control_points", &RectangleBSplineSurface::num_control_points,
//              "Get total number of control points")
//         .def("get_control_points_z", 
//              [](RectangleBSplineSurface& s) -> Float& { return s.get_control_points_z(); },
//              nb::rv_policy::reference_internal,
//              "Get control points Z coordinates")
//         .def("set_control_points_z", &RectangleBSplineSurface::set_control_points_z,
//              "z"_a, "Set control points Z coordinates")
//           .def("get_bspline_config", &RectangleBSplineSurface::get_bspline_config,
//          "Get (u_degree, v_degree, u_num_cp, v_num_cp)")
//     .def("get_u_num_cp", &RectangleBSplineSurface::get_u_num_cp)
//     .def("get_v_num_cp", &RectangleBSplineSurface::get_v_num_cp)
//     .def("resize_control_points", &RectangleBSplineSurface::resize_control_points,
//          "new_u_num_cp"_a, "new_v_num_cp"_a,
//          "Resize control points with bilinear interpolation")
//     .def("set_control_points_z_from_array", &RectangleBSplineSurface::set_control_points_z_from_array,
//          "z_values"_a, "Set control points z from Python list")
//     .def("get_control_points_z_as_array", &RectangleBSplineSurface::get_control_points_z_as_array,
//          "Get control points z as Python list")
//         .def("__repr__", [](const RectangleBSplineSurface& s) {
//             auto bspline = s.get_bspline_shape();
//             std::string repr = "RectangleBSplineSurface(width=" + 
//                    std::to_string(s.get_width()) + 
//                    ", height=" + std::to_string(s.get_height()) +
//                    ", degree=" + std::to_string(bspline->u_degree) + "x" + 
//                    std::to_string(bspline->v_degree) +
//                    ", control_points=" + std::to_string(bspline->u_num_cp) + "x" + 
//                    std::to_string(bspline->v_num_cp) +
//                    ", tess=" + std::to_string(s.get_tess_u()) + "x" + std::to_string(s.get_tess_v());
//             if (s.get_bsdf()) {
//                 repr += ", bsdf=" + s.get_bsdf()->type_name();
//             }
//             repr += ")";
//             return repr;
//         });
    
//     // ============= CirclePlaneSurface =============
//     nb::class_<CirclePlaneSurface, CircleSurface>(m, "CirclePlaneSurface")
//         .def(nb::init<>(), 
//              "Create a circle plane surface with default radius (1)")
//         .def(nb::init<ScalarType>(),
//              "radius"_a,
//              "Create a circle plane surface with specified radius")
//         .def("__repr__", [](const CirclePlaneSurface& s) {
//             std::string repr = "CirclePlaneSurface(radius=" + 
//                    std::to_string(s.get_radius()) +
//                    ", tess=" + std::to_string(s.get_tess_u()) + "x" + std::to_string(s.get_tess_v());
//             if (s.get_bsdf()) {
//                 repr += ", bsdf=" + s.get_bsdf()->type_name();
//             }
//             repr += ")";
//             return repr;
//         });
}