cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_ARCHITECTURES 75)

project(diff_optics LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Python
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()
find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit: ${CUDAToolkit_LIBRARY_DIR}")

# OpenCASCADE - 禁用
set(USE_OPENCASCADE OFF)
message(STATUS "OpenCASCADE CAD Export: DISABLED")

# OptiX
set(OptiX_INSTALL_DIR "$ENV{HOME}/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64" CACHE PATH "Path to OptiX SDK")
if(NOT EXISTS ${OptiX_INSTALL_DIR})
    message(FATAL_ERROR "OptiX not found: ${OptiX_INSTALL_DIR}")
endif()

# PTX 编译
set(OPTIX_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/optix_shaders")
set(PTX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")
file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

set(OPTIX_KERNEL "${OPTIX_SHADER_DIR}/optix_kernels.cu")
set(PTX_FILE "${PTX_OUTPUT_DIR}/optix_kernels.ptx")

add_custom_command(
    OUTPUT ${PTX_FILE}
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            -ptx
            -arch=sm_75
            "-I${OptiX_INSTALL_DIR}/include"
            "-I${CUDAToolkit_INCLUDE_DIRS}"
            --use_fast_math
            -O3
            -lineinfo
            -o "${PTX_FILE}"
            "${OPTIX_KERNEL}"
    MAIN_DEPENDENCY ${OPTIX_KERNEL}
    COMMENT "Compiling OptiX kernels to PTX"
    VERBATIM
)

add_custom_target(optix_shaders ALL DEPENDS ${PTX_FILE})

# DrJit
set(DRJIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/drjit")
set(DRJIT_BUILD_LLVM ON CACHE BOOL "" FORCE)
set(DRJIT_BUILD_CUDA ON CACHE BOOL "" FORCE)
set(DRJIT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(DRJIT_BUILD_EXTRA ON CACHE BOOL "" FORCE)
set(DRJIT_OPTIX_VERSION 81 CACHE STRING "" FORCE)

add_subdirectory(${DRJIT_DIR} drjit)
add_subdirectory(${DRJIT_DIR}/ext/nanobind nanobind)

# CUDA 核函数库
add_library(cuda_kernels STATIC
    src/cuda_kernels.cu
)

set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(cuda_kernels PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ==================== Python 模块 ====================
nanobind_add_module(
    diff_optics
    NB_DOMAIN drjit
    src/binding.cpp
    src/optix_scene.cpp
)

add_dependencies(diff_optics optix_shaders)

target_include_directories(diff_optics PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${DRJIT_DIR}/include
    ${DRJIT_DIR}/ext/drjit-core/include
    ${DRJIT_DIR}/ext/drjit-core/ext/nanothread/include
    ${DRJIT_DIR}/ext/nanobind/include
    "${OptiX_INSTALL_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(diff_optics PRIVATE 
    cuda_kernels
    drjit
    drjit-core
    drjit-extra
    nanothread
    CUDA::cudart
    CUDA::cuda_driver
)

target_compile_definitions(diff_optics PRIVATE 
    USE_DRJIT
    USE_CUDA
    DRJIT_ENABLE_CUDA
    DRJIT_ENABLE_JIT
    DRJIT_ENABLE_OPTIX
    PTX_DIR="${PTX_OUTPUT_DIR}"
)

set_target_properties(diff_optics PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
)

add_custom_command(TARGET diff_optics POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PTX_FILE}
        $<TARGET_FILE_DIR:diff_optics>/optix_kernels.ptx
    COMMENT "Copying PTX to output directory"
)