// bindings/bind_utils.hpp
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include "../utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_utils(nb::module_& m) {
    // Transform
    nb::class_<Transform>(m, "Transform")
        .def(nb::init<>(), "Create identity transform")
        .def(nb::init<ScalarType, ScalarType, ScalarType, ScalarType, ScalarType, ScalarType>(),
             nb::arg("tx"), nb::arg("ty"), nb::arg("tz"),
             nb::arg("rx"), nb::arg("ry"), nb::arg("rz"),
             "Create transform with translation (tx, ty, tz) and rotation (rx, ry, rz) in degrees")
        .def(nb::init<const std::vector<ScalarType>&, const std::vector<ScalarType>&>(),
             nb::arg("translation"), nb::arg("rotation"),
             "Create transform from translation [tx, ty, tz] and rotation [rx, ry, rz] in degrees")
        .def_prop_rw("tx",
            [](const Transform& t) { return t.get_tx_scalar(); },
            [](Transform& t, ScalarType val) { t.tx = from_scalar(val); })
        .def_prop_rw("ty",
            [](const Transform& t) { return t.get_ty_scalar(); },
            [](Transform& t, ScalarType val) { t.ty = from_scalar(val); })
        .def_prop_rw("tz",
            [](const Transform& t) { return t.get_tz_scalar(); },
            [](Transform& t, ScalarType val) { t.tz = from_scalar(val); })
        .def_prop_rw("rx",
            [](const Transform& t) { return t.get_rx_scalar(); },
            [](Transform& t, ScalarType val) { t.rx = from_scalar(val); })
        .def_prop_rw("ry",
            [](const Transform& t) { return t.get_ry_scalar(); },
            [](Transform& t, ScalarType val) { t.ry = from_scalar(val); })
        .def_prop_rw("rz",
            [](const Transform& t) { return t.get_rz_scalar(); },
            [](Transform& t, ScalarType val) { t.rz = from_scalar(val); })
        .def("set_translation", 
            static_cast<void (Transform::*)(ScalarType, ScalarType, ScalarType)>(&Transform::set_translation),
            "tx"_a, "ty"_a, "tz"_a)
        .def("set_rotation",
            static_cast<void (Transform::*)(ScalarType, ScalarType, ScalarType)>(&Transform::set_rotation),
            "rx"_a, "ry"_a, "rz"_a, "Set rotation in degrees")
        .def("transform_point", &Transform::transform_point, "point"_a, "Transform a point from local to world space")
        .def("transform_direction", &Transform::transform_direction, "direction"_a, "Transform a direction from local to world space")
        .def("transform_normal", &Transform::transform_normal, "normal"_a, "Transform a normal from local to world space")
        .def("inverse_transform_point", &Transform::inverse_transform_point, "point"_a, "Transform a point from world to local space")
        .def("inverse_transform_direction", &Transform::inverse_transform_direction, "direction"_a, "Transform a direction from world to local space")
        .def("print", &Transform::print, nb::arg("name") = "", "Print transform info");

    // Ray
    nb::class_<Ray>(m, "Ray")
        .def(nb::init<>())
        .def(nb::init<const Vector3&, const Vector3&, const Float&, const Float&, const Float&>(),
             nb::arg("origin"), nb::arg("direction"), nb::arg("wavelength"), nb::arg("radiance"), nb::arg("pdf"))
        .def_static("from_scalars", &Ray::from_scalars)
        .def_prop_rw("origin", 
            [](const Ray& r) { return r.origin; }, 
            [](Ray& r, const Vector3& val) { r.origin = val; })
        .def_prop_rw("direction", 
            [](const Ray& r) { return r.direction; }, 
            [](Ray& r, const Vector3& val) { r.direction = val; })
        .def_prop_rw("wavelength", 
            [](const Ray& r) { return r.wavelength; }, 
            [](Ray& r, const Float& val) { r.wavelength = val; })
        .def_prop_rw("radiance", 
            [](const Ray& r) { return r.radiance; }, 
            [](Ray& r, const Float& val) { r.radiance = val; })
        .def_prop_rw("pdf", 
            [](const Ray& r) { return r.pdf; }, 
            [](Ray& r, const Float& val) { r.pdf = val; })
        .def("size", &Ray::size);
}