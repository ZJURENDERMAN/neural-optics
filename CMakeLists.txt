# ==================== CMakeLists.txt ====================
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

if(WIN32)
    add_definitions(-DNOMINMAX)
    add_definitions(-D_USE_MATH_DEFINES)
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

# OpenCASCADE (可选)
set(OpenCASCADE_DIR "D:/opencascade-7.9.3-vc14-64/cmake")
find_package(OpenCASCADE QUIET)
link_directories("D:/opencascade-7.9.3-vc14-64/win64/vc14/lib")
set(USE_OPENCASCADE OFF)

if(OpenCASCADE_FOUND)
    set(USE_OPENCASCADE ON)
    message(STATUS "OpenCASCADE found: ${OpenCASCADE_INCLUDE_DIR}")
    message(STATUS "OpenCASCADE libraries: ${OpenCASCADE_LIBRARIES}")
else()
    message(STATUS "OpenCASCADE not found - CAD export will be disabled")
    message(STATUS "To enable CAD export, install OpenCASCADE and set OpenCASCADE_DIR")
endif()

# OptiX
set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0" CACHE PATH "Path to OptiX SDK")
if(NOT EXISTS ${OptiX_INSTALL_DIR})
    message(FATAL_ERROR "OptiX not found: ${OptiX_INSTALL_DIR}")
endif()

# PTX 编译目录
set(OPTIX_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/optix_shaders")
set(PTX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")
file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

# 仿真用 OptiX PTX
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

# 渲染用 OptiX PTX
set(RENDER_OPTIX_KERNEL "${OPTIX_SHADER_DIR}/optix_render_kernels.cu")
set(RENDER_PTX_FILE "${PTX_OUTPUT_DIR}/optix_render_kernels.ptx")

add_custom_command(
    OUTPUT ${RENDER_PTX_FILE}
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            -ptx
            -arch=sm_75
            "-I${OptiX_INSTALL_DIR}/include"
            "-I${CUDAToolkit_INCLUDE_DIRS}"
            --use_fast_math
            -O3
            -lineinfo
            -o "${RENDER_PTX_FILE}"
            "${RENDER_OPTIX_KERNEL}"
    MAIN_DEPENDENCY ${RENDER_OPTIX_KERNEL}
    COMMENT "Compiling OptiX render kernels to PTX"
    VERBATIM
)

add_custom_target(optix_render_shaders ALL DEPENDS ${RENDER_PTX_FILE})

# DrJit
set(DRJIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/drjit")
set(DRJIT_BUILD_LLVM ON CACHE BOOL "" FORCE)
set(DRJIT_BUILD_CUDA ON CACHE BOOL "" FORCE)
set(DRJIT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(DRJIT_BUILD_EXTRA ON CACHE BOOL "" FORCE)
set(DRJIT_OPTIX_VERSION 81 CACHE STRING "" FORCE)

add_subdirectory(${DRJIT_DIR} drjit)
add_subdirectory(${DRJIT_DIR}/ext/nanobind nanobind)

# CUDA 核函数库（必须用 nvcc 编译）
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

# ==================== 设置 OpenCASCADE 相关变量 ====================
if(USE_OPENCASCADE)
    set(OCCT_INCLUDE_DIRS ${OpenCASCADE_INCLUDE_DIR})
    set(OCCT_LIB_DIR "D:/opencascade-7.9.3-vc14-64/win64/vc14/lib")
    set(OCCT_LIBRARIES 
        ${OCCT_LIB_DIR}/TKernel.lib
        ${OCCT_LIB_DIR}/TKMath.lib
        ${OCCT_LIB_DIR}/TKG2d.lib
        ${OCCT_LIB_DIR}/TKG3d.lib
        ${OCCT_LIB_DIR}/TKGeomBase.lib
        ${OCCT_LIB_DIR}/TKBRep.lib
        ${OCCT_LIB_DIR}/TKGeomAlgo.lib
        ${OCCT_LIB_DIR}/TKTopAlgo.lib
        ${OCCT_LIB_DIR}/TKPrim.lib
        ${OCCT_LIB_DIR}/TKBO.lib
        ${OCCT_LIB_DIR}/TKShHealing.lib
        ${OCCT_LIB_DIR}/TKBool.lib
        ${OCCT_LIB_DIR}/TKFillet.lib
        ${OCCT_LIB_DIR}/TKOffset.lib
        ${OCCT_LIB_DIR}/TKFeat.lib
        ${OCCT_LIB_DIR}/TKHLR.lib
        ${OCCT_LIB_DIR}/TKMesh.lib
        ${OCCT_LIB_DIR}/TKDESTEP.lib
        ${OCCT_LIB_DIR}/TKDE.lib
        ${OCCT_LIB_DIR}/TKXSBase.lib
    )
    set(OCCT_COMPILE_DEFS USE_OPENCASCADE)
else()
    set(OCCT_INCLUDE_DIRS "")
    set(OCCT_LIBRARIES "")
    set(OCCT_COMPILE_DEFS "")
endif()

set(STB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb")
set(JSON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/single_include")

# ==================== 渲染器源文件 ====================
set(RENDERER_SOURCES
    src/renderer/stb_impl.cpp
    src/renderer/scene_renderer.cpp
)
# ==================== GLAD 2.0（手动生成）====================
set(GLAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glad")

if(EXISTS ${GLAD_DIR}/src/gl.c)
    add_library(glad STATIC ${GLAD_DIR}/src/gl.c)
    target_include_directories(glad PUBLIC ${GLAD_DIR}/include)
    set(GLAD_FOUND TRUE)
    message(STATUS "GLAD 2.0 found at ${GLAD_DIR}")
else()
    set(GLAD_FOUND FALSE)
    message(STATUS "GLAD not found. To generate:")
    message(STATUS "  pip install glad2")
    message(STATUS "  glad --api gl:core=4.6 --out-path third_party/glad c")
endif()

# ==================== C++ 可执行文件（调试用）====================
add_executable(diff_optics_debug
    src/main.cpp
    src/optix_scene.cpp
    ${RENDERER_SOURCES}
)

add_dependencies(diff_optics_debug optix_shaders optix_render_shaders)

target_include_directories(diff_optics_debug PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${DRJIT_DIR}/include
    ${DRJIT_DIR}/ext/drjit-core/include
    ${DRJIT_DIR}/ext/drjit-core/ext/nanothread/include
    "${OptiX_INSTALL_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OCCT_INCLUDE_DIRS}
    ${STB_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
)

target_link_libraries(diff_optics_debug PRIVATE 
    ${OPENGL_LIBRARIES}
    cuda_kernels
    drjit
    drjit-core
    drjit-extra
    nanothread
    CUDA::cudart
    CUDA::cuda_driver
    ${OCCT_LIBRARIES}
)

target_compile_definitions(diff_optics_debug PRIVATE 
    NOMINMAX
    USE_DRJIT
    USE_CUDA
    DRJIT_ENABLE_CUDA
    DRJIT_ENABLE_JIT
    DRJIT_ENABLE_OPTIX
    _USE_MATH_DEFINES
    PTX_DIR="${PTX_OUTPUT_DIR}"
    ${OCCT_COMPILE_DEFS}
)

set_target_properties(diff_optics_debug PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
)

# ==================== Python 模块 ====================
nanobind_add_module(
    diff_optics
    NB_DOMAIN drjit
    src/binding.cpp
    src/optix_scene.cpp
    ${RENDERER_SOURCES}
)

add_dependencies(diff_optics optix_shaders optix_render_shaders)

target_include_directories(diff_optics PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${DRJIT_DIR}/include
    ${DRJIT_DIR}/ext/drjit-core/include
    ${DRJIT_DIR}/ext/drjit-core/ext/nanothread/include
    ${DRJIT_DIR}/ext/nanobind/include
    "${OptiX_INSTALL_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OCCT_INCLUDE_DIRS}
    ${STB_INCLUDE_DIR}
)

target_link_libraries(diff_optics PRIVATE 
    cuda_kernels
    drjit
    drjit-core
    drjit-extra
    nanothread
    CUDA::cudart
    CUDA::cuda_driver
    ${OCCT_LIBRARIES}
)

target_compile_definitions(diff_optics PRIVATE 
    NOMINMAX
    USE_DRJIT
    USE_CUDA
    DRJIT_ENABLE_CUDA
    DRJIT_ENABLE_JIT
    DRJIT_ENABLE_OPTIX
    _USE_MATH_DEFINES
    PTX_DIR="${PTX_OUTPUT_DIR}"
    ${OCCT_COMPILE_DEFS}
)

set_target_properties(diff_optics PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
)

# ==================== 复制 PTX 文件 ====================
# 复制仿真 PTX 到 Python 模块目录
add_custom_command(TARGET diff_optics POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PTX_FILE}
        $<TARGET_FILE_DIR:diff_optics>/optix_kernels.ptx
    COMMENT "Copying simulation PTX to output directory"
)

# 复制渲染 PTX 到 Python 模块目录
add_custom_command(TARGET diff_optics POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${RENDER_PTX_FILE}
        $<TARGET_FILE_DIR:diff_optics>/optix_render_kernels.ptx
    COMMENT "Copying render PTX to output directory"
)

# 复制仿真 PTX 到调试可执行文件目录
add_custom_command(TARGET diff_optics_debug POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PTX_FILE}
        $<TARGET_FILE_DIR:diff_optics_debug>/optix_kernels.ptx
    COMMENT "Copying simulation PTX to debug executable directory"
)

# 复制渲染 PTX 到调试可执行文件目录
add_custom_command(TARGET diff_optics_debug POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${RENDER_PTX_FILE}
        $<TARGET_FILE_DIR:diff_optics_debug>/optix_render_kernels.ptx
    COMMENT "Copying render PTX to debug executable directory"
)

# 复制 pyd 到 build 根目录
add_custom_command(TARGET diff_optics POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:diff_optics>
        ${CMAKE_CURRENT_SOURCE_DIR}/build/
    COMMENT "Copying pyd to build root"
)

# ==================== Windows 交互窗口 ====================
if(WIN32)
    find_package(glfw3 CONFIG QUIET)
    if(glfw3_FOUND AND GLAD_FOUND)
        target_sources(diff_optics_debug PRIVATE src/interactive/interactive_viewer.cpp)
        target_link_libraries(diff_optics_debug PRIVATE glfw glad)
        target_compile_definitions(diff_optics_debug PRIVATE HAS_INTERACTIVE_VIEWER)
        message(STATUS "GLFW + GLAD found - Interactive viewer enabled")
    else()
        if(NOT glfw3_FOUND)
            message(STATUS "GLFW not found - Interactive viewer disabled")
        endif()
        if(NOT GLAD_FOUND)
            message(STATUS "GLAD not found - Interactive viewer disabled")
        endif()
    endif()
endif()

# ==================== OpenEXR 支持（可选）====================
find_package(OpenEXR QUIET)
if(OpenEXR_FOUND)
    target_link_libraries(diff_optics PRIVATE OpenEXR::OpenEXR)
    target_link_libraries(diff_optics_debug PRIVATE OpenEXR::OpenEXR)
    target_compile_definitions(diff_optics PRIVATE USE_OPENEXR)
    target_compile_definitions(diff_optics_debug PRIVATE USE_OPENEXR)
    message(STATUS "OpenEXR found - EXR export enabled")
else()
    message(STATUS "OpenEXR not found - EXR export disabled (will use HDR format instead)")
    message(STATUS "To enable EXR export, install OpenEXR:")
    message(STATUS "  vcpkg install openexr:x64-windows")
endif()

# ==================== 打印配置摘要 ====================
message(STATUS "")
message(STATUS "========== Configuration Summary ==========")
message(STATUS "OpenCASCADE CAD Export: ${USE_OPENCASCADE}")
if(WIN32)
    if(glfw3_FOUND)
        message(STATUS "Interactive Viewer:     ON")
    else()
        message(STATUS "Interactive Viewer:     OFF")
    endif()
endif()
if(OpenEXR_FOUND)
    message(STATUS "OpenEXR Support:        ON")
else()
    message(STATUS "OpenEXR Support:        OFF")
endif()
message(STATUS "PTX Output Directory:   ${PTX_OUTPUT_DIR}")
message(STATUS "STB Include Directory:  ${STB_INCLUDE_DIR}")
message(STATUS "============================================")
message(STATUS "")