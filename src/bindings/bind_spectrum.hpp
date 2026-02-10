#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include "../spectrum.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_spectrum(nb::module_& m) {
    // ============= 抽象基类 Spectrum =============
    nb::class_<Spectrum>(m, "Spectrum")
        .def("type_name", &Spectrum::type_name,
             "Get the spectrum type name")
        .def("eval", &Spectrum::eval, "wavelength"_a,
             "Evaluate spectrum value at given wavelength(s)")
        .def("sample", &Spectrum::sample, "u"_a,
             "Sample wavelength from spectrum, returns (wavelength, value, pdf)")
        .def("print", &Spectrum::print, "name"_a = "",
             "Print spectrum info");

    // ============= DiscreteSpectrum =============
    nb::class_<DiscreteSpectrum, Spectrum>(m, "DiscreteSpectrum")
        .def(nb::init<const Float&, const Float&>(),
             "wavelengths"_a, "values"_a,
             "Create a discrete spectrum from wavelength and value arrays")
        .def_ro("wavelengths", &DiscreteSpectrum::wavelengths,
                "Wavelength array (nm)")
        .def_ro("values", &DiscreteSpectrum::values,
                "Value array")
        .def_ro("cdf", &DiscreteSpectrum::cdf,
                "Cumulative distribution function for sampling")
        .def("num_samples", &DiscreteSpectrum::num_samples,
             "Get the number of samples")
        .def("wl_min", &DiscreteSpectrum::wl_min,
             "Get minimum wavelength (nm)")
        .def("wl_max", &DiscreteSpectrum::wl_max,
             "Get maximum wavelength (nm)")
        .def("total_integral", &DiscreteSpectrum::total_integral,
             "Get total integral of the spectrum")
        .def("__repr__", [](const DiscreteSpectrum& s) {
            size_t n = s.num_samples();
            if (n == 0) {
                return std::string("DiscreteSpectrum(empty)");
            }
            return "DiscreteSpectrum(samples=" + std::to_string(n) +
                   ", range=[" + std::to_string(to_scalar(s.wl_min())) + ", " + 
                   std::to_string(to_scalar(s.wl_max())) + "] nm)";
        });

    // ============= BlackbodySpectrum =============
    nb::class_<BlackbodySpectrum, Spectrum>(m, "BlackbodySpectrum")
        .def(nb::init<>(),
             "Create a blackbody spectrum with default temperature (6500K)")
        .def(nb::init<const Float&>(),
             "temperature"_a,
             "Create a blackbody spectrum with specified temperature in Kelvin")
        .def(nb::init<const Float&, const Float&, const Float&>(),
             "temperature"_a, "wl_min"_a, "wl_max"_a,
             "Create a blackbody spectrum with temperature and wavelength range")
        .def_rw("temperature", &BlackbodySpectrum::temperature,
                "Temperature in Kelvin")
        .def_rw("wl_min", &BlackbodySpectrum::wl_min,
                "Minimum wavelength (nm)")
        .def_rw("wl_max", &BlackbodySpectrum::wl_max,
                "Maximum wavelength (nm)")
        .def("__repr__", [](const BlackbodySpectrum& s) {
            return "BlackbodySpectrum(T=" + std::to_string(to_scalar(s.temperature)) + 
                   "K, range=[" + std::to_string(to_scalar(s.wl_min)) + ", " + 
                   std::to_string(to_scalar(s.wl_max)) + "] nm)";
        });

    // ============= GaussianSpectrum =============
    nb::class_<GaussianSpectrum, Spectrum>(m, "GaussianSpectrum")
        .def(nb::init<>(),
             "Create a Gaussian spectrum with default parameters (center=550nm, sigma=30nm)")
        .def(nb::init<const Float&, const Float&, const Float&>(),
             "center"_a, "sigma"_a, "amplitude"_a = Float(1.0f),
             "Create a Gaussian spectrum with center wavelength, sigma, and amplitude")
        .def_rw("center", &GaussianSpectrum::center,
                "Center wavelength (nm)")
        .def_rw("sigma", &GaussianSpectrum::sigma,
                "Standard deviation (nm)")
        .def_rw("amplitude", &GaussianSpectrum::amplitude,
                "Amplitude")
        .def("__repr__", [](const GaussianSpectrum& s) {
            return "GaussianSpectrum(center=" + std::to_string(to_scalar(s.center)) + 
                   "nm, sigma=" + std::to_string(to_scalar(s.sigma)) + 
                   "nm, amplitude=" + std::to_string(to_scalar(s.amplitude)) + ")";
        });

    // ============= ConstantSpectrum =============
    nb::class_<ConstantSpectrum, Spectrum>(m, "ConstantSpectrum")
        .def(nb::init<>(),
             "Create a constant spectrum with value 1.0")
        .def(nb::init<const Float&>(),
             "value"_a,
             "Create a constant spectrum with specified value")
        .def(nb::init<const Float&, const Float&, const Float&>(),
             "value"_a, "wl_min"_a, "wl_max"_a,
             "Create a constant spectrum with value and wavelength range")
        .def_rw("value", &ConstantSpectrum::value,
                "Constant value")
        .def_rw("wl_min", &ConstantSpectrum::wl_min,
                "Minimum wavelength (nm)")
        .def_rw("wl_max", &ConstantSpectrum::wl_max,
                "Maximum wavelength (nm)")
        .def("__repr__", [](const ConstantSpectrum& s) {
            return "ConstantSpectrum(value=" + std::to_string(to_scalar(s.value)) + 
                   ", range=[" + std::to_string(to_scalar(s.wl_min)) + ", " + 
                   std::to_string(to_scalar(s.wl_max)) + "] nm)";
        });
}