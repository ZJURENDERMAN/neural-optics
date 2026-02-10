#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include "../material.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_material(nb::module_& m) {
    // ============= 抽象基类 VolumeMaterial =============
    nb::class_<VolumeMaterial>(m, "VolumeMaterial")
        .def("type_name", &VolumeMaterial::type_name,
             "Get the material type name")
        .def("transmittance", &VolumeMaterial::transmittance, 
             "wavelength"_a, "distance"_a,
             "Calculate transmittance for given wavelength(s) and propagation distance")
        .def("ior", &VolumeMaterial::ior, "wavelength"_a,
             "Calculate index of refraction for given wavelength(s)")
        .def("get_measurement_depth", &VolumeMaterial::get_measurement_depth,
             "Get the measurement depth used for transmittance calculation (mm)")
        .def("print", &VolumeMaterial::print, "name"_a = "",
             "Print material info");

    // ============= DiscreteVolumeMaterial =============
    nb::class_<DiscreteVolumeMaterial, VolumeMaterial>(m, "DiscreteVolumeMaterial")
        .def(nb::init<>(),
             "Create a default discrete volume material (transparent, n=1)")
        .def(nb::init<std::shared_ptr<DiscreteSpectrum>, 
                      std::shared_ptr<DiscreteSpectrum>, 
                      ScalarType>(),
             "transmittance_spectrum"_a, "ior_spectrum"_a, "measurement_depth"_a,
             "Create a discrete volume material with transmittance and IOR spectra")
        .def_prop_ro("transmittance_spectrum", 
            [](const DiscreteVolumeMaterial& m) { return m.transmittance_spectrum; },
            "Transmittance spectrum")
        .def_prop_ro("ior_spectrum",
            [](const DiscreteVolumeMaterial& m) { return m.ior_spectrum; },
            "Index of refraction spectrum")
        .def_prop_rw("measurement_depth",
            [](const DiscreteVolumeMaterial& m) { return m.measurement_depth; },
            [](DiscreteVolumeMaterial& m, ScalarType val) { m.measurement_depth = val; },
            "Measurement depth (mm)")
        .def("__repr__", [](const DiscreteVolumeMaterial& m) {
            return "DiscreteVolumeMaterial(depth=" + 
                   std::to_string(m.measurement_depth) + "mm)";
        });

    // ============= SellmeierMaterial =============
    nb::class_<SellmeierMaterial, VolumeMaterial>(m, "SellmeierMaterial")
        .def(nb::init<>(),
             "Create a default Sellmeier material")
        .def(nb::init<const std::vector<ScalarType>&, 
                      const std::vector<ScalarType>&,
                      std::shared_ptr<DiscreteSpectrum>,
                      ScalarType>(),
             "B"_a, "C"_a, "transmittance"_a = nullptr, "depth"_a = 10.0f,
             "Create a Sellmeier material with B and C coefficients")
        .def_ro("B", &SellmeierMaterial::B,
                "Sellmeier B coefficients")
        .def_ro("C", &SellmeierMaterial::C,
                "Sellmeier C coefficients (μm²)")
        .def_prop_ro("transmittance_spectrum",
            [](const SellmeierMaterial& m) { return m.transmittance_spectrum; },
            "Transmittance spectrum")
        .def_prop_rw("measurement_depth",
            [](const SellmeierMaterial& m) { return m.measurement_depth; },
            [](SellmeierMaterial& m, ScalarType val) { m.measurement_depth = val; },
            "Measurement depth (mm)")
        .def("__repr__", [](const SellmeierMaterial& m) {
            return "SellmeierMaterial(terms=" + std::to_string(m.B.size()) + 
                   ", depth=" + std::to_string(m.measurement_depth) + "mm)";
        });

    // ============= ConstantIORMaterial =============
    nb::class_<ConstantIORMaterial, VolumeMaterial>(m, "ConstantIORMaterial")
        .def(nb::init<ScalarType, ScalarType, ScalarType>(),
             "ior"_a = 1.5f, "transmittance"_a = 1.0f, "depth"_a = 10.0f,
             "Create a material with constant IOR and transmittance")
        .def_prop_rw("n",
            [](const ConstantIORMaterial& m) { return to_scalar(m.n); },
            [](ConstantIORMaterial& m, ScalarType val) { m.n = from_scalar(val); },
            "Index of refraction")
        .def_prop_rw("transmittance_value",
            [](const ConstantIORMaterial& m) { return to_scalar(m.transmittance_value); },
            [](ConstantIORMaterial& m, ScalarType val) { m.transmittance_value = from_scalar(val); },
            "Transmittance value")
        .def_prop_rw("measurement_depth",
            [](const ConstantIORMaterial& m) { return m.measurement_depth; },
            [](ConstantIORMaterial& m, ScalarType val) { m.measurement_depth = val; },
            "Measurement depth (mm)")
        .def("__repr__", [](const ConstantIORMaterial& m) {
            return "ConstantIORMaterial(n=" + std::to_string(to_scalar(m.n)) + 
                   ", T=" + std::to_string(to_scalar(m.transmittance_value)) + ")";
        });

    // ============= 预定义材料创建函数 =============
    m.def("create_air", &create_air,
          "Create an air material (standard atmosphere, 20°C)");
    
    m.def("create_nbk7", &create_nbk7,
          "Create an N-BK7 optical glass material");
    
    m.def("create_pmma", &create_pmma,
          "Create a PMMA (acrylic) material");
    
    m.def("create_vacuum", &create_vacuum,
          "Create a vacuum material (n=1, perfect transmittance)");
}