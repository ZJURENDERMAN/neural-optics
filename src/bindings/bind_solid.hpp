// bindings/bind_solid.hpp
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include "../solid.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_solid(nb::module_& m) {
    
    // ============= Solid 基类 =============
    nb::class_<Solid, Object>(m, "Solid")
        .def("get_surface", &Solid::get_surface, "index"_a)
        .def("surface_count", &Solid::surface_count);
    
    // ============= Cube =============
    nb::class_<Cube, Solid>(m, "Cube")
        .def(nb::init<>())
        .def(nb::init<ScalarType>(), "size"_a)
        .def(nb::init<ScalarType, ScalarType, ScalarType>(),
             "size_x"_a, "size_y"_a, "size_z"_a)
        .def("get_size_x", &Cube::get_size_x)
        .def("get_size_y", &Cube::get_size_y)
        .def("get_size_z", &Cube::get_size_z)
        .def("set_size", &Cube::set_size, "sx"_a, "sy"_a, "sz"_a)
        .def("get_face", &Cube::get_face, "face_index"_a)
        .def_ro_static("FACE_FRONT", &Cube::FACE_FRONT)
        .def_ro_static("FACE_BACK", &Cube::FACE_BACK)
        .def_ro_static("FACE_LEFT", &Cube::FACE_LEFT)
        .def_ro_static("FACE_RIGHT", &Cube::FACE_RIGHT)
        .def_ro_static("FACE_TOP", &Cube::FACE_TOP)
        .def_ro_static("FACE_BOTTOM", &Cube::FACE_BOTTOM)
        .def("__repr__", [](const Cube& c) {
            return "Cube(" + std::to_string(c.get_size_x()) + "x" +
                   std::to_string(c.get_size_y()) + "x" +
                   std::to_string(c.get_size_z()) + ")";
        });
    
    // ============= Lens 基类 =============
    nb::class_<Lens, Solid>(m, "Lens")
        .def("get_front_surface", &Lens::get_front_surface)
        .def("get_back_surface", &Lens::get_back_surface)
        .def("get_lens_material", &Lens::get_lens_material)
        .def("set_lens_material", &Lens::set_lens_material, "material"_a)
        .def("get_thickness", &Lens::get_thickness)
        .def("set_thickness", &Lens::set_thickness, "thickness"_a);
    
    // ============= RectangleLens =============
    nb::class_<RectangleLens, Lens>(m, "RectangleLens")
        .def(nb::init<>())
        .def(nb::init<ScalarType, ScalarType, ScalarType>(),
             "width"_a, "height"_a, "thickness"_a)
        .def("get_width", &RectangleLens::get_width)
        .def("get_height", &RectangleLens::get_height)
        .def("__repr__", [](const RectangleLens& l) {
            return "RectangleLens(" + std::to_string(l.get_width()) + "x" +
                   std::to_string(l.get_height()) + ", t=" +
                   std::to_string(l.get_thickness()) + ")";
        });
    
    // ============= CircleLens =============
    nb::class_<CircleLens, Lens>(m, "CircleLens")
        .def(nb::init<>())
        .def(nb::init<ScalarType, ScalarType>(),
             "radius"_a, "thickness"_a)
        .def("get_radius", &CircleLens::get_radius)
        .def("__repr__", [](const CircleLens& l) {
            return "CircleLens(r=" + std::to_string(l.get_radius()) + 
                   ", t=" + std::to_string(l.get_thickness()) + ")";
        });
}