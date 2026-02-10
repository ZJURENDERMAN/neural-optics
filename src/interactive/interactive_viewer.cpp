// interactive_viewer.cpp
#ifdef HAS_INTERACTIVE_VIEWER

// GLAD 必须在 GLFW 之前包含
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "interactive/interactive_viewer.hpp"
#include <iostream>
#include <cstring>

namespace diff_optics {

// 简单的全屏四边形着色器
static const char* vertex_shader_source = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

static const char* fragment_shader_source = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D screenTexture;
void main() {
    FragColor = texture(screenTexture, TexCoord);
}
)";

// 辅助函数：编译着色器
static GLuint compile_shader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        std::cerr << "[Shader] Compilation error: " << info_log << std::endl;
        return 0;
    }
    return shader;
}

// 辅助函数：创建着色器程序
static GLuint create_shader_program(const char* vs_source, const char* fs_source) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source);
    
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return 0;
    }
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(program, 512, nullptr, info_log);
        std::cerr << "[Shader] Link error: " << info_log << std::endl;
        glDeleteProgram(program);
        program = 0;
    }
    
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

InteractiveViewer::InteractiveViewer() {
    m_camera.set_view_preset(ViewPreset::Front, 30.0f);
    m_camera.set_perspective(45.0f);
    
    m_config.samples_per_pixel = 1;
    m_config.max_depth = 8;
    m_config.progressive = true;
}

InteractiveViewer::~InteractiveViewer() {
    close();
}

bool InteractiveViewer::create(int width, int height, const std::string& title) {
    m_width = width;
    m_height = height;
    
    if (!glfwInit()) {
        std::cerr << "[InteractiveViewer] Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // 请求 OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!m_window) {
        std::cerr << "[InteractiveViewer] Failed to create window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(m_window);
    
    // 加载 OpenGL 函数 - GLAD 2.0 的方式
    int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) {
        std::cerr << "[InteractiveViewer] Failed to load OpenGL functions with GLAD" << std::endl;
        glfwDestroyWindow(m_window);
        glfwTerminate();
        return false;
    }
    
    std::cout << "[InteractiveViewer] OpenGL " << GLAD_VERSION_MAJOR(version) << "." 
              << GLAD_VERSION_MINOR(version) << " loaded" << std::endl;
    
    glfwSetWindowUserPointer(m_window, this);
    
    // 设置回调
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(m_window, mouse_button_callback);
    glfwSetCursorPosCallback(m_window, cursor_pos_callback);
    glfwSetScrollCallback(m_window, scroll_callback);
    glfwSetKeyCallback(m_window, key_callback);
    
    // 创建着色器程序
    m_shader_program = create_shader_program(vertex_shader_source, fragment_shader_source);
    if (!m_shader_program) {
        std::cerr << "[InteractiveViewer] Failed to create shader program" << std::endl;
        glfwDestroyWindow(m_window);
        glfwTerminate();
        return false;
    }
    
    // 创建全屏四边形 VAO/VBO
    // 顶点数据：位置 (x, y) + 纹理坐标 (u, v)
    float quad_vertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    
    glGenVertexArrays(1, &m_quad_vao);
    glGenBuffers(1, &m_quad_vbo);
    
    glBindVertexArray(m_quad_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_quad_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
    
    // 位置属性
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // 纹理坐标属性
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    // 创建显示纹理
    glGenTextures(1, &m_display_texture);
    glBindTexture(GL_TEXTURE_2D, m_display_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    
    // 创建 PBO 用于 CUDA-GL 互操作
    glGenBuffers(1, &m_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(float) * 4, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&m_cuda_pbo, m_pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "[InteractiveViewer] Failed to register PBO with CUDA: " 
                  << cudaGetErrorString(err) << std::endl;
    }
    
    std::cout << "[InteractiveViewer] Window created: " << width << "x" << height << std::endl;
    return true;
}

void InteractiveViewer::set_scene(const Scene& scene, SceneRenderer* renderer, OptiXSceneManager* optix_mgr) {
    m_renderer = renderer;
    m_camera_changed = true;
    m_accumulated_frames = 0;
}

void InteractiveViewer::run() {
    if (!m_window || !m_renderer) {
        std::cerr << "[InteractiveViewer] Not properly initialized" << std::endl;
        return;
    }
    
    std::cout << "\n=== Interactive Viewer Controls ===" << std::endl;
    std::cout << "Left Mouse + Drag: Rotate camera" << std::endl;
    std::cout << "Right Mouse + Drag: Pan camera" << std::endl;
    std::cout << "Scroll: Zoom in/out" << std::endl;
    std::cout << "1-6: View presets (Front/Back/Left/Right/Top/Bottom)" << std::endl;
    std::cout << "P: Toggle perspective/orthographic" << std::endl;
    std::cout << "R: Reset camera" << std::endl;
    std::cout << "S: Save screenshot (render.png)" << std::endl;
    std::cout << "ESC: Close window" << std::endl;
    std::cout << "==================================\n" << std::endl;
    
    while (!glfwWindowShouldClose(m_window)) {
        process_input();
        
        // 渲染
        m_config.width = m_width;
        m_config.height = m_height;
        
        if (m_camera_changed) {
            m_accumulated_frames = 0;
            m_camera_changed = false;
        }
        
        m_renderer->render_progressive(m_camera, m_config, m_accumulated_frames);
        m_accumulated_frames++;
        
        update_display();
        
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

void InteractiveViewer::close() {
    if (m_cuda_pbo) {
        cudaGraphicsUnregisterResource(m_cuda_pbo);
        m_cuda_pbo = nullptr;
    }
    if (m_pbo) {
        glDeleteBuffers(1, &m_pbo);
        m_pbo = 0;
    }
    if (m_display_texture) {
        glDeleteTextures(1, &m_display_texture);
        m_display_texture = 0;
    }
    if (m_quad_vao) {
        glDeleteVertexArrays(1, &m_quad_vao);
        m_quad_vao = 0;
    }
    if (m_quad_vbo) {
        glDeleteBuffers(1, &m_quad_vbo);
        m_quad_vbo = 0;
    }
    if (m_shader_program) {
        glDeleteProgram(m_shader_program);
        m_shader_program = 0;
    }
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

void InteractiveViewer::process_input() {
    // ESC 关闭
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(m_window, true);
    }
}

void InteractiveViewer::update_display() {
    // 从渲染器获取图像数据并更新纹理
    auto data = m_renderer->get_image_data();
    if (data.empty()) return;
    
    // 转换为 RGBA
    std::vector<float> rgba_data(m_width * m_height * 4);
    for (int i = 0; i < m_width * m_height; ++i) {
        rgba_data[i * 4 + 0] = data[i * 3 + 0];
        rgba_data[i * 4 + 1] = data[i * 3 + 1];
        rgba_data[i * 4 + 2] = data[i * 3 + 2];
        rgba_data[i * 4 + 3] = 1.0f;
    }
    
    // 更新纹理
    glBindTexture(GL_TEXTURE_2D, m_display_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, rgba_data.data());
    
    // 使用着色器渲染全屏四边形
    glClear(GL_COLOR_BUFFER_BIT);
    
    glUseProgram(m_shader_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_display_texture);
    glUniform1i(glGetUniformLocation(m_shader_program, "screenTexture"), 0);
    
    glBindVertexArray(m_quad_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    
    glUseProgram(0);
    
    // 显示帧数
    std::string title = "Diff-Optics Viewer - Samples: " + std::to_string(m_accumulated_frames);
    glfwSetWindowTitle(m_window, title.c_str());
}

// 回调实现
void InteractiveViewer::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    InteractiveViewer* viewer = static_cast<InteractiveViewer*>(glfwGetWindowUserPointer(window));
    viewer->m_width = width;
    viewer->m_height = height;
    viewer->m_camera_changed = true;
    glViewport(0, 0, width, height);
    
    // 重建纹理
    if (viewer->m_display_texture) {
        glBindTexture(GL_TEXTURE_2D, viewer->m_display_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    }
    
    // 重建 PBO
    if (viewer->m_cuda_pbo) {
        cudaGraphicsUnregisterResource(viewer->m_cuda_pbo);
        viewer->m_cuda_pbo = nullptr;
    }
    if (viewer->m_pbo) {
        glDeleteBuffers(1, &viewer->m_pbo);
        glGenBuffers(1, &viewer->m_pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, viewer->m_pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(float) * 4, nullptr, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        cudaGraphicsGLRegisterBuffer(&viewer->m_cuda_pbo, viewer->m_pbo, cudaGraphicsMapFlagsWriteDiscard);
    }
}

void InteractiveViewer::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    InteractiveViewer* viewer = static_cast<InteractiveViewer*>(glfwGetWindowUserPointer(window));
    
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        viewer->m_mouse_left_pressed = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        viewer->m_mouse_right_pressed = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        viewer->m_mouse_middle_pressed = (action == GLFW_PRESS);
    }
    
    glfwGetCursorPos(window, &viewer->m_last_mouse_x, &viewer->m_last_mouse_y);
}

void InteractiveViewer::cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    InteractiveViewer* viewer = static_cast<InteractiveViewer*>(glfwGetWindowUserPointer(window));
    
    double dx = xpos - viewer->m_last_mouse_x;
    double dy = ypos - viewer->m_last_mouse_y;
    
    if (viewer->m_mouse_left_pressed) {
        // 旋转
        viewer->m_camera.rotate(static_cast<float>(dx * 0.5), static_cast<float>(dy * 0.5));
        viewer->m_camera_changed = true;
    } else if (viewer->m_mouse_right_pressed) {
        // 平移
        viewer->m_camera.pan(static_cast<float>(-dx), static_cast<float>(dy));
        viewer->m_camera_changed = true;
    }
    
    viewer->m_last_mouse_x = xpos;
    viewer->m_last_mouse_y = ypos;
}

void InteractiveViewer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    InteractiveViewer* viewer = static_cast<InteractiveViewer*>(glfwGetWindowUserPointer(window));
    viewer->m_camera.zoom(static_cast<float>(yoffset));
    viewer->m_camera_changed = true;
}

void InteractiveViewer::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;
    
    InteractiveViewer* viewer = static_cast<InteractiveViewer*>(glfwGetWindowUserPointer(window));
    
    switch (key) {
        case GLFW_KEY_1:
            viewer->m_camera.set_view_preset(ViewPreset::Front, viewer->m_camera.distance);
            viewer->m_camera_changed = true;
            break;
        case GLFW_KEY_2:
            viewer->m_camera.set_view_preset(ViewPreset::Back, viewer->m_camera.distance);
            viewer->m_camera_changed = true;
            break;
        case GLFW_KEY_3:
            viewer->m_camera.set_view_preset(ViewPreset::Left, viewer->m_camera.distance);
            viewer->m_camera_changed = true;
            break;
        case GLFW_KEY_4:
            viewer->m_camera.set_view_preset(ViewPreset::Right, viewer->m_camera.distance);
            viewer->m_camera_changed = true;
            break;
        case GLFW_KEY_5:
            viewer->m_camera.set_view_preset(ViewPreset::Top, viewer->m_camera.distance);
            viewer->m_camera_changed = true;
            break;
        case GLFW_KEY_6:
            viewer->m_camera.set_view_preset(ViewPreset::Bottom, viewer->m_camera.distance);
            viewer->m_camera_changed = true;
            break;
        case GLFW_KEY_P:
            if (viewer->m_camera.is_orthographic()) {
                viewer->m_camera.set_perspective(45.0f);
            } else {
                viewer->m_camera.set_orthographic(viewer->m_camera.distance * 2.0f);
            }
            viewer->m_camera_changed = true;
            std::cout << "Projection: " << (viewer->m_camera.is_orthographic() ? "Orthographic" : "Perspective") << std::endl;
            break;
        case GLFW_KEY_R:
            viewer->m_camera.set_view_preset(ViewPreset::Front, 30.0f);
            viewer->m_camera.set_perspective(45.0f);
            viewer->m_camera_changed = true;
            break;
        case GLFW_KEY_S:
            if (viewer->m_renderer) {
                viewer->m_renderer->save_png("render.png");
            }
            break;
    }
    
    if (viewer->m_key_callback) {
        viewer->m_key_callback(key, action);
    }
}

} // namespace diff_optics

#endif // HAS_INTERACTIVE_VIEWER