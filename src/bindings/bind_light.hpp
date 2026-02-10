// bindings/bind_light.hpp
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include "../light.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_light(nb::module_& m) {
    // Light 基类（抽象类）
    nb::class_<Light>(m, "Light")
        .def("sample_rays", [](const Light& self, const Vector2& uv1, const Vector2& uv2, const Float& u) {
            return self.sampleRays(uv1, uv2, u);
        }, "uv1"_a, "uv2"_a, "u"_a, 
           "Sample rays from the light source. uv1 for position, uv2 for direction, u for wavelength")
        .def("type_name", &Light::type_name)
        .def_prop_rw("power", 
            [](Light& l) -> Float& { return l.get_power(); },
            &Light::set_power)
        .def_prop_rw("emitter", &Light::get_emitter, &Light::set_emitter)
        .def_prop_rw("spectrum", &Light::get_spectrum, &Light::set_spectrum);

    // PointLight
    nb::class_<PointLight, Light>(m, "PointLight")
        .def(nb::init<std::shared_ptr<Emitter>, std::shared_ptr<Spectrum>>(), 
             "emitter"_a, "spectrum"_a,
             "Create a point light at origin with default settings")
        .def(nb::init<const Transform&, const Float&, std::shared_ptr<Emitter>, std::shared_ptr<Spectrum>>(),
             "transform"_a, "power"_a, "emitter"_a, "spectrum"_a,
             "Create a point light with specified transform, power, emitter, and spectrum")
        .def_prop_rw("transform",
            [](const PointLight& l) -> const Transform& { return l.get_transform(); },
            [](PointLight& l, const Transform& t) { l.set_transform(t); })
        .def("sample_rays", [](const PointLight& self, const Vector2& uv1, const Vector2& uv2, const Float& u) {
            return self.sampleRays(uv1, uv2, u);
        }, "uv1"_a, "uv2"_a, "u"_a,
           "Sample rays from the point light")
        .def("print", &PointLight::print, "name"_a = "", "Print light info")
        .def("__repr__", [](const PointLight& l) {
            return "PointLight(power=" + std::to_string(to_scalar(l.get_power())) + ")";
        });

    // SurfaceLight
    nb::class_<SurfaceLight, Light>(m, "SurfaceLight")
        .def(nb::init<std::shared_ptr<Surface>, std::shared_ptr<Emitter>, std::shared_ptr<Spectrum>>(),
             "surface"_a, "emitter"_a, "spectrum"_a,
             "Create a surface light from an existing surface")
        .def_prop_rw("surface", &SurfaceLight::get_surface, &SurfaceLight::set_surface,
             "The surface used as light source")
        .def("sample_rays", [](const SurfaceLight& self, const Vector2& uv1, const Vector2& uv2, const Float& u) {
            return self.sampleRays(uv1, uv2, u);
        }, "uv1"_a, "uv2"_a, "u"_a,
           "Sample rays from the surface light")
        .def("print", &SurfaceLight::print, "name"_a = "", "Print light info")
        .def("__repr__", [](const SurfaceLight& l) {
            std::string surface_type = l.get_surface() ? l.get_surface()->type_name() : "None";
            return "SurfaceLight(surface=" + surface_type + 
                   ", power=" + std::to_string(to_scalar(l.get_power())) + ")";
        });
}