// bindings/bind_simulator.hpp - 简化版
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include "../simulator.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_simulator(nb::module_& m) {
    
    // ============= SequenceConfig =============
    nb::class_<SequenceConfig>(m, "SequenceConfig")
        .def(nb::init<>(), "Create an empty sequence config")
        .def(nb::init<const std::string&, const std::vector<std::string>&>(),
             "light"_a, "surfaces"_a,
             "Create a sequence config with light name and surface names")
        .def_rw("light_name", &SequenceConfig::light_name,
                "Name of the light source")
        .def_rw("surfaces_name", &SequenceConfig::surfaces_name,
                "List of surface names in propagation order")
        .def("__repr__", [](const SequenceConfig& c) {
            std::string surfaces_str;
            for (size_t i = 0; i < c.surfaces_name.size(); ++i) {
                surfaces_str += c.surfaces_name[i];
                if (i < c.surfaces_name.size() - 1) surfaces_str += " -> ";
            }
            return "SequenceConfig(" + c.light_name + " -> " + surfaces_str + ")";
        });
    
    // ============= SimulatorConfig =============
    nb::class_<SimulatorConfig>(m, "SimulatorConfig")
        .def(nb::init<>(), "Create a default simulator config")
        .def(nb::init<size_t, int, ScalarType>(),
             "num_rays"_a, "max_depth"_a, "min_radiance"_a,
             "Create a simulator config with specified parameters")
        .def_rw("num_rays", &SimulatorConfig::num_rays,
                "Number of rays to sample")
        .def_rw("max_depth", &SimulatorConfig::max_depth,
                "Maximum path depth")
        .def_rw("min_radiance", &SimulatorConfig::min_radiance,
                "Minimum radiance threshold")
        .def_rw("sim_type", &SimulatorConfig::sim_type,
                "Simulation type: 0=sequential, 1=non-sequential")
        .def_rw("trace_type", &SimulatorConfig::trace_type,
                "Trace type: 0=forward, 1=backward")
        .def_rw("seed", &SimulatorConfig::seed, "Random seed")
        .def_rw("use_optix", &SimulatorConfig::use_optix, "Whether to use OptiX")
        .def_rw("ptx_dir", &SimulatorConfig::ptx_dir, "PTX directory path")
        .def_rw("seq_config", &SimulatorConfig::seq_config, "Sequence configuration")
        .def("set_sequence", 
             nb::overload_cast<const std::string&, const std::vector<std::string>&>(
                 &SimulatorConfig::set_sequence),
             "light_name"_a, "surfaces"_a,
             "Set tracing sequence with light name and surface names")
        .def("set_optix", &SimulatorConfig::set_optix,
             "enable"_a, "ptx_path"_a = "",
             "Configure OptiX acceleration")
        .def("print", &SimulatorConfig::print, "name"_a = "SimulatorConfig",
             "Print config info")
        .def("__repr__", [](const SimulatorConfig& c) {
            return "SimulatorConfig(num_rays=" + std::to_string(c.num_rays) +
                   ", max_depth=" + std::to_string(c.max_depth) +
                   ", optix=" + (c.use_optix ? "true" : "false") + ")";
        });
    
    // ============= SimulationResult =============
    nb::class_<SimulationResult>(m, "SimulationResult")
        .def(nb::init<>(), "Create empty simulation result")
        .def_ro("success", &SimulationResult::success,
                "Whether simulation succeeded")
        .def_ro("error_message", &SimulationResult::error_message,
                "Error message if failed")
        .def_ro("sensor_data", &SimulationResult::sensor_data,
                "Dictionary of sensor data by name")
        .def_ro("optix_build_time_ms", &SimulationResult::optix_build_time_ms,
                "OptiX build time in milliseconds")
        .def_ro("optix_trace_time_ms", &SimulationResult::optix_trace_time_ms,
                "OptiX trace time in milliseconds")
        .def_ro("total_time_ms", &SimulationResult::total_time_ms,
                "Total simulation time in milliseconds")
        .def_ro("exit_rays", &SimulationResult::exit_rays,
                "Exit rays after last surface (for collimation)")
        .def_ro("has_exit_rays", &SimulationResult::has_exit_rays,
                "Whether exit rays are available")
        .def("has_sensor_data", &SimulationResult::has_sensor_data, "name"_a,
             "Check if sensor data exists")
        .def("get_sensor_data", 
             static_cast<SensorData& (SimulationResult::*)(const std::string&)>(
                 &SimulationResult::get_sensor_data),
             "name"_a, nb::rv_policy::reference_internal,
             "Get sensor data by name")
        .def("get_sensor_names", &SimulationResult::get_sensor_names,
             "Get all sensor names")
        .def("print", &SimulationResult::print, "name"_a = "SimulationResult",
             "Print result info")
        .def("__repr__", [](const SimulationResult& r) {
            std::string status = r.success ? "success" : "failed";
            return "SimulationResult(" + status + 
                   ", sensors=" + std::to_string(r.sensor_data.size()) + ")";
        })
        .def("__contains__", &SimulationResult::has_sensor_data)
        .def("__len__", [](const SimulationResult& r) {
            return r.sensor_data.size();
        })
        .def("__getitem__", [](SimulationResult& r, const std::string& name) -> SensorData& {
            return r.get_sensor_data(name);
        }, nb::rv_policy::reference_internal)
        .def("keys", &SimulationResult::get_sensor_names,
             "Get sensor names (dict-like interface)");
    
    // ============= Simulator (Abstract Base) =============
        nb::class_<Simulator>(m, "Simulator")
            .def("simulate", &Simulator::simulate,
                "scene"_a, "config"_a,
                "Run simulation on the scene with given config");
    
    // ============= ForwardSimulator =============
    nb::class_<ForwardSimulator, Simulator>(m, "ForwardSimulator")
        .def(nb::init<>(), "Create a forward ray tracing simulator")
        .def("simulate", &ForwardSimulator::simulate,
             "scene"_a, "config"_a,
             "Run forward simulation on the scene")
        .def("__repr__", [](const ForwardSimulator&) {
            return "ForwardSimulator()";
        });
}