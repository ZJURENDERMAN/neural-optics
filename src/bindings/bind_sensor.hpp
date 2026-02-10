// bindings/bind_sensor.hpp - 简化版
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include "../sensor.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace diff_optics;

inline void bind_sensor(nb::module_& m) {
    // ============= FilterType 枚举 =============
    nb::enum_<FilterType>(m, "FilterType")
        .value("Box", FilterType::Box)
        .value("Bilinear", FilterType::Bilinear);

    // ============= IESType 枚举 =============
    nb::enum_<IESType>(m, "IESType")
        .value("TypeA", IESType::TypeA)
        .value("TypeB", IESType::TypeB)
        .value("TypeC", IESType::TypeC);

    // ============= SensorData =============
    nb::class_<SensorData>(m, "SensorData")
        .def(nb::init<>(), "Create empty sensor data")
        .def(nb::init<const std::string&, int, int>(),
             "name"_a, "width"_a, "height"_a,
             "Create sensor data with name and resolution")
        .def_ro("sensor_name", &SensorData::sensor_name, "Name of the sensor")
        .def_ro("width", &SensorData::width, "Width in pixels")
        .def_ro("height", &SensorData::height, "Height in pixels")
        .def_rw("data", &SensorData::data, "Measurement data")
        .def_rw("hit_count", &SensorData::hit_count, "Hit count per pixel")
        
        .def("clear", &SensorData::clear, "Clear data to zero")
        .def("size", &SensorData::size, "Get total pixel count")
        .def("total_hits", &SensorData::total_hits, "Get total hit count")
        .def("get_resolution", &SensorData::get_resolution,
             "Get resolution as (width, height)")
        .def("print", &SensorData::print, "prefix"_a = "", "Print sensor data info")
        
        .def("__repr__", [](const SensorData& d) {
            return "SensorData('" + d.sensor_name + "', " +
                   std::to_string(d.width) + "x" + std::to_string(d.height) +
                   ", hits=" + std::to_string(d.total_hits()) + ")";
        });

    // ============= 抽象基类 Sensor =============
    nb::class_<Sensor>(m, "Sensor")
        .def("type_name", &Sensor::type_name)
        .def("has_surface", &Sensor::has_surface)
        .def_prop_rw("surface", &Sensor::get_surface, &Sensor::set_surface)
        .def_prop_rw("u_range", &Sensor::get_u_range, &Sensor::set_u_range)
        .def_prop_rw("v_range", &Sensor::get_v_range, &Sensor::set_v_range)
        .def_prop_rw("width", &Sensor::get_width, &Sensor::set_width)
        .def_prop_rw("height", &Sensor::get_height, &Sensor::set_height)
        .def("set_resolution", &Sensor::set_resolution, "width"_a, "height"_a)
        .def_prop_ro("filter_name", &Sensor::get_filter_name)
        .def("set_filter", &Sensor::set_filter, "filter_name"_a)
        .def("get_pixel_area", &Sensor::get_pixel_area, "Get pixel area array")
        .def("create_data", &Sensor::create_data, "name"_a,
             "Create a SensorData object for this sensor")
        .def("print", &Sensor::print, "name"_a = "");
    
    // ============= IrradianceSensor =============
    nb::class_<IrradianceSensor, Sensor>(m, "IrradianceSensor")
        .def("__repr__", [](const IrradianceSensor& s) {
            return "IrradianceSensor(resolution=" + std::to_string(s.get_width()) + 
                   "x" + std::to_string(s.get_height()) + 
                   ", filter=" + s.get_filter_name() + ")";
        });
    
    // ============= IntensitySensor =============
    nb::class_<IntensitySensor, Sensor>(m, "IntensitySensor")
        .def_prop_ro("ies_type", &IntensitySensor::get_ies_type)
        .def_prop_ro("ies_type_name", &IntensitySensor::get_ies_type_name)
        .def("__repr__", [](const IntensitySensor& s) {
            return "IntensitySensor(ies_type=" + s.get_ies_type_name() +
                   ", resolution=" + std::to_string(s.get_width()) + 
                   "x" + std::to_string(s.get_height()) + 
                   ", filter=" + s.get_filter_name() + ")";
        });
}