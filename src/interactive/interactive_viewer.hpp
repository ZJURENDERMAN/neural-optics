// interactive_viewer.hpp
#pragma once

#ifdef HAS_INTERACTIVE_VIEWER

// GLAD 必须在 GLFW 之前包含
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "renderer/camera.hpp"
#include "renderer/render_config.hpp"
#include "renderer/scene_renderer.hpp"
#include "scene.hpp"
#include "optix_scene.hpp"

#include <string>
#include <functional>

namespace diff_optics {

class InteractiveViewer {
public:
    InteractiveViewer();
    ~InteractiveViewer();
    
    bool create(int width, int height, const std::string& title = "Diff-Optics Viewer");
    void set_scene(const Scene& scene, SceneRenderer* renderer, OptiXSceneManager* optix_mgr);
    void run();
    void close();
    
    Camera& get_camera() { return m_camera; }
    const Camera& get_camera() const { return m_camera; }
    
    RenderConfig& get_config() { return m_config; }
    const RenderConfig& get_config() const { return m_config; }
    
    void set_key_callback(std::function<void(int key, int action)> callback) {
        m_key_callback = std::move(callback);
    }
    
private:
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    
    void process_input();
    void update_display();
    
private:
    GLFWwindow* m_window = nullptr;
    int m_width = 1024;
    int m_height = 768;
    
    Camera m_camera;
    RenderConfig m_config;
    SceneRenderer* m_renderer = nullptr;
    
    // 鼠标状态
    bool m_mouse_left_pressed = false;
    bool m_mouse_right_pressed = false;
    bool m_mouse_middle_pressed = false;
    double m_last_mouse_x = 0;
    double m_last_mouse_y = 0;
    
    // OpenGL 资源
    GLuint m_display_texture = 0;
    GLuint m_pbo = 0;
    GLuint m_shader_program = 0;
    GLuint m_quad_vao = 0;
    GLuint m_quad_vbo = 0;
    cudaGraphicsResource* m_cuda_pbo = nullptr;
    
    // 帧计数和累积
    bool m_camera_changed = true;
    unsigned int m_accumulated_frames = 0;
    
    // 回调
    std::function<void(int key, int action)> m_key_callback;
};

} // namespace diff_optics

#endif // HAS_INTERACTIVE_VIEWER