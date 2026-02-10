// bindings/bind_object.hpp
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/pair.h>
#include "../object.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_object(nb::module_& m) {
    // ============= ObjectType 枚举 =============
    nb::enum_<ObjectType>(m, "ObjectType")
        .value("Surface", ObjectType::Surface)
        .value("Shell", ObjectType::Shell)
        .value("Solid", ObjectType::Solid);
    
    // ============= Object 抽象基类 =============
    nb::class_<Object>(m, "Object")
        .def("object_type", &Object::object_type,
             "Get the object type")
        .def("type_name", &Object::type_name,
             "Get the object type name")
        .def_prop_rw("transform",
            [](Object& o) -> Transform& { return o.get_transform(); },
            &Object::set_transform,
            "Transform from local to parent/world space")
        .def("get_parent", &Object::get_parent,
             nb::rv_policy::reference,
             "Get parent object")
        .def("has_parent", &Object::has_parent,
             "Check if object has a parent")
        .def("get_world_matrix", &Object::get_world_matrix,
             "Get world transformation matrix")
        .def("to_world_point", &Object::to_world_point,
             "point"_a, "Transform point from local to world space")
        .def("to_world_direction", &Object::to_world_direction,
             "direction"_a, "Transform direction from local to world space")
        .def("to_world_normal", &Object::to_world_normal,
             "normal"_a, "Transform normal from local to world space")
        .def("from_world_point", &Object::from_world_point,
             "point"_a, "Transform point from world to local space")
        .def("from_world_direction", &Object::from_world_direction,
             "direction"_a, "Transform direction from world to local space")
        .def("from_world_normal", &Object::from_world_normal,
             "normal"_a, "Transform normal from world to local space")
        .def("surface_count", &Object::surface_count,
             "Get number of surfaces")
        .def("invalidate_all_meshes", &Object::invalidate_all_meshes,
             "Invalidate all cached meshes")
        .def("print", &Object::print, "name"_a = "",
             "Print object info");
}