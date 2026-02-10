// binding.cpp
#include <nanobind/nanobind.h>

#include "bindings/bind_utils.hpp"
#include "bindings/bind_shape.hpp"
#include "bindings/bind_emitter.hpp"
#include "bindings/bind_light.hpp"
#include "bindings/bind_sensor.hpp"
#include "bindings/bind_simulator.hpp"
#include "bindings/bind_material.hpp"
#include "bindings/bind_spectrum.hpp"
#include "bindings/bind_bsdf.hpp"
#include "bindings/bind_optix.hpp"
#include "bindings/bind_object.hpp"
#include "bindings/bind_surface.hpp"
#include "bindings/bind_shell.hpp"
#include "bindings/bind_solid.hpp"
#include "bindings/bind_scene.hpp"
#include "bindings/bind_parser.hpp"  // 新增

namespace nb = nanobind;

NB_MODULE(diff_optics, m) {
    m.doc() = "Differentiable optics library with OptiX support";
    bind_utils(m);
    bind_object(m);
    bind_shape(m);
    
    bind_surface(m);
    bind_shell(m);
    bind_solid(m);

    bind_emitter(m);
    bind_light(m);
    bind_sensor(m);
    bind_simulator(m);
    bind_material(m);
    bind_bsdf(m);
    bind_spectrum(m);
    bind_optix(m);
    bind_scene(m);
    bind_parser(m);  // 新增
}