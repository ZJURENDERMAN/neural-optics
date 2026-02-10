// bindings/bind_emitter.hpp
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include "../emitter.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_emitter(nb::module_& m) {
    // Emitter 基类（抽象类，不能直接实例化）
    nb::class_<Emitter>(m, "Emitter")
        .def("sample_direction", [](const Emitter& self, const Vector2& uv) {
            Float weight, pdf;
            Vector3 dir = self.sampleDirection(uv, weight, pdf);
            return std::make_tuple(dir, weight, pdf);
        }, "uv"_a, "Sample direction from UV random numbers, returns (direction, weight, pdf)");

    // UniformEmitter
    nb::class_<UniformEmitter, Emitter>(m, "UniformEmitter")
        .def(nb::init<>())
        .def(nb::init<const Float&, const Float&>(), 
             nb::arg("lower_angle"), nb::arg("upper_angle"),
             "Create uniform emitter with angle range in radians")
        .def_static("from_scalars", [](ScalarType lower, ScalarType upper) {
            return UniformEmitter(from_scalar(lower), from_scalar(upper));
        }, "lower_angle"_a, "upper_angle"_a,
           "Create uniform emitter from scalar angle values (radians)")
        .def_prop_rw("lower_angle", 
            [](const UniformEmitter& e) { return e.lower_angle; },
            [](UniformEmitter& e, const Float& val) { e.lower_angle = val; })
        .def_prop_rw("upper_angle",
            [](const UniformEmitter& e) { return e.upper_angle; },
            [](UniformEmitter& e, const Float& val) { e.upper_angle = val; })
        .def("sample_direction", [](const UniformEmitter& self, const Vector2& uv) {
            Float weight, pdf;
            Vector3 dir = self.sampleDirection(uv, weight, pdf);
            return std::make_tuple(dir, weight, pdf);
        }, "uv"_a, "Sample direction uniformly within angle range, returns (direction, weight, pdf)");

    // LambertEmitter
    nb::class_<LambertEmitter, Emitter>(m, "LambertEmitter")
        .def(nb::init<>())
        .def(nb::init<const Float&, const Float&, const Float&>(),
             nb::arg("lower_angle"), nb::arg("upper_angle"), nb::arg("hwhm"),
             "Create Lambert emitter with angle range and HWHM in radians")
        .def_static("from_scalars", [](ScalarType lower, ScalarType upper, ScalarType hwhm) {
            return LambertEmitter(from_scalar(lower), from_scalar(upper), from_scalar(hwhm));
        }, "lower_angle"_a, "upper_angle"_a, "hwhm"_a,
           "Create Lambert emitter from scalar values (radians)")
        .def_prop_rw("lower_angle",
            [](const LambertEmitter& e) { return e.lower_angle; },
            [](LambertEmitter& e, const Float& val) { e.lower_angle = val; })
        .def_prop_rw("upper_angle",
            [](const LambertEmitter& e) { return e.upper_angle; },
            [](LambertEmitter& e, const Float& val) { e.upper_angle = val; })
        .def_prop_rw("hwhm",
            [](const LambertEmitter& e) { return e.hwhm; },
            [](LambertEmitter& e, const Float& val) { e.hwhm = val; })
        .def("sample_direction", [](const LambertEmitter& self, const Vector2& uv) {
            Float weight, pdf;
            Vector3 dir = self.sampleDirection(uv, weight, pdf);
            return std::make_tuple(dir, weight, pdf);
        }, "uv"_a, "Sample direction with Lambert distribution, returns (direction, weight, pdf)");
}