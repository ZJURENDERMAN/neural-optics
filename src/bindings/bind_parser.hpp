// bindings/bind_parser.hpp
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include "../scene_parser.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_parser(nb::module_& m) {
    // NpyLoader 类（可选暴露，用于调试）
    nb::class_<NpyLoader>(m, "NpyLoader")
        .def_static("load_float", [](const std::string& filepath) -> nb::object {
            auto result = NpyLoader::load(filepath);
            if (result) {
                return nb::cast(*result);
            }
            return nb::none();
        }, "filepath"_a, "Load a numpy/text/binary file as float array")
        .def_static("load", [](const std::string& filepath) -> nb::object {
            auto result = NpyLoader::load(filepath);
            if (result) {
                return nb::cast(*result);
            }
            return nb::none();
        }, "filepath"_a, "Load a file (auto-detect format)");
    
    // SceneParser 类
    nb::class_<SceneParser>(m, "SceneParser")
        .def(nb::init<>(), "Create a scene parser")
        .def("load", &SceneParser::load, "scene_path"_a,
             "Load scene from JSON file")
        .def("parse_string", &SceneParser::parse_string, "json_str"_a,
             "Parse scene from JSON string")
        .def("set_base_dir", &SceneParser::set_base_dir, "dir"_a,
             "Set base directory for relative path resolution")
        .def("__repr__", [](const SceneParser&) {
            return "SceneParser()";
        });
    
    // 便捷函数
    m.def("load_scene", &load_scene, "scene_path"_a,
          "Load scene from JSON file (convenience function)");
    
    m.def("parse_scene_string", &parse_scene_string, "json_str"_a,
          "Parse scene from JSON string (convenience function)");
}