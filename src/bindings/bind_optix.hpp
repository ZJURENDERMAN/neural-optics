// ==================== bindings/bind_optix.hpp ====================
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include "../optix_scene.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_optix(nb::module_& m) {
    
    // IntersectionResult
    nb::class_<IntersectionResult>(m, "IntersectionResult")
        .def(nb::init<>())
        .def(nb::init<size_t>(), "n"_a)
        .def_ro("t", &IntersectionResult::t, "Hit distance (negative if miss)")
        .def_ro("surface_id", &IntersectionResult::surface_id, "Surface ID (-1 if miss)")
        .def_ro("prim_id", &IntersectionResult::prim_id, "Primitive (triangle) ID")
        .def_ro("bary_u", &IntersectionResult::bary_u, "Barycentric u coordinate")
        .def_ro("bary_v", &IntersectionResult::bary_v, "Barycentric v coordinate")
        .def_ro("valid", &IntersectionResult::valid, "Valid mask")
        .def("__repr__", [](const IntersectionResult& r) {
            size_t n = drjit::width(r.t);
            size_t num_valid = drjit::count(r.valid)[0];
            return "IntersectionResult(size=" + std::to_string(n) + 
                   ", valid=" + std::to_string(num_valid) + ")";
        });
    
    // OptiXSceneManager
    nb::class_<OptiXSceneManager>(m, "OptiXSceneManager")
        .def(nb::init<>())
        .def("initialize", &OptiXSceneManager::initialize, "ptx_dir"_a,
             "Initialize OptiX with PTX directory")
        .def("build_from_scene", &OptiXSceneManager::build_from_scene, "scene"_a,
             "Build acceleration structure from scene")
        .def("update_surface_mesh", &OptiXSceneManager::update_surface_mesh,
             "surface_name"_a, "scene"_a,
             "Update surface mesh (for BSpline control point changes)")
        .def("trace_rays", &OptiXSceneManager::trace_rays, "rays"_a,
             "Trace rays against all surfaces")
        .def("trace_rays_single_surface", &OptiXSceneManager::trace_rays_single_surface,
             "rays"_a, "surface_name"_a,
             "Trace rays against a single surface")
        .def("clear", &OptiXSceneManager::clear,
             "Clear scene data")
        .def("is_initialized", &OptiXSceneManager::is_initialized,
             "Check if OptiX is initialized")
        .def("surface_count", &OptiXSceneManager::surface_count,
             "Get number of surfaces")
        .def("get_surface_id", &OptiXSceneManager::get_surface_id,
             "name"_a, "Get surface ID by name")
        .def("__repr__", [](const OptiXSceneManager& m) {
            return "OptiXSceneManager(initialized=" + 
                   std::string(m.is_initialized() ? "true" : "false") +
                   ", surfaces=" + std::to_string(m.surface_count()) + ")";
        });
    
    // 全局函数
    m.def("init_optix_manager", &init_optix_manager, "ptx_dir"_a,
          "Initialize global OptiX manager");
    
    m.def("get_optix_manager", &get_optix_manager, nb::rv_policy::reference,
          "Get global OptiX manager");
    
    m.def("build_optix_scene", &build_optix_scene, "scene"_a,
          "Build OptiX acceleration structure from scene");
    
    m.def("optix_trace_rays", &optix_trace_rays, "rays"_a,
          "Trace rays using OptiX");
    
    m.def("optix_trace_single_surface", &optix_trace_single_surface,
          "rays"_a, "surface_name"_a,
          "Trace rays against a single surface using OptiX");
}