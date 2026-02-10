#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include "../bsdf.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_bsdf(nb::module_& m) {
    // ============= BSDFSample 结构体 =============
    nb::class_<BSDFSample>(m, "BSDFSample")
        .def(nb::init<>(),
             "Create an empty BSDF sample")
        .def(nb::init<size_t>(),
             "n"_a,
             "Create a BSDF sample with specified size")
        .def_ro("direction", &BSDFSample::direction,
                "Sampled direction (world space)")
        .def_ro("weight", &BSDFSample::weight,
                "Sample weight = bsdf * cos(theta) / pdf")
        .def_ro("pdf", &BSDFSample::pdf,
                "Probability density")
        .def_ro("valid", &BSDFSample::valid,
                "Validity mask")
        .def("__repr__", [](const BSDFSample& s) {
            return "BSDFSample()";
        });

    // ============= 抽象基类 BSDF =============
    nb::class_<BSDF>(m, "BSDF")
        .def("type_name", &BSDF::type_name,
             "Get the BSDF type name")
        .def("print", &BSDF::print, "name"_a = "",
             "Print BSDF info");

    // ============= SpecularReflector =============
    nb::class_<SpecularReflector, BSDF>(m, "SpecularReflector")
        .def(nb::init<>(),
             "Create a perfect specular reflector")
        .def(nb::init<ScalarType>(),
             "reflectance"_a,
             "Create a specular reflector with specified reflectance")
        .def_prop_rw("reflectance",
            [](const SpecularReflector& b) { return to_scalar(b.reflectance); },
            [](SpecularReflector& b, ScalarType val) { b.reflectance = from_scalar(val); },
            "Reflectance value")
        .def("__repr__", [](const SpecularReflector& b) {
            return "SpecularReflector(reflectance=" + 
                   std::to_string(to_scalar(b.reflectance)) + ")";
        });

    // ============= SpecularRefractor =============
    nb::class_<SpecularRefractor, BSDF>(m, "SpecularRefractor")
        .def(nb::init<>(),
             "Create a perfect specular refractor")
        .def(nb::init<ScalarType>(),
             "transmittance"_a,
             "Create a specular refractor with specified transmittance")
        .def_prop_rw("transmittance",
            [](const SpecularRefractor& b) { return to_scalar(b.transmittance); },
            [](SpecularRefractor& b, ScalarType val) { b.transmittance = from_scalar(val); },
            "Transmittance value")
        .def("__repr__", [](const SpecularRefractor& b) {
            return "SpecularRefractor(transmittance=" + 
                   std::to_string(to_scalar(b.transmittance)) + ")";
        });

    // ============= Absorber =============
    nb::class_<Absorber, BSDF>(m, "Absorber")
        .def(nb::init<>(),
             "Create a perfect absorber (black body)")
        .def("__repr__", [](const Absorber&) {
            return "Absorber()";
        });

}