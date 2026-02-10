// camera.hpp - 相机类（透视/正交投影）
#pragma once
#include <cmath>
#include <array>

namespace diff_optics {

enum class ProjectionType {
    Perspective,
    Orthographic
};

enum class ViewPreset {
    Custom,
    Front,      // +Z 方向看
    Back,       // -Z 方向看
    Left,       // +X 方向看
    Right,      // -X 方向看
    Top,        // -Y 方向看（俯视）
    Bottom      // +Y 方向看（仰视）
};

struct Camera {
    // 位置和朝向
    float position[3] = {0.0f, 0.0f, -10.0f};
    float target[3] = {0.0f, 0.0f, 0.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};
    
    // 投影参数
    ProjectionType projection_type = ProjectionType::Perspective;
    float fov_y = 45.0f;           // 透视投影垂直FOV（度）
    float ortho_height = 20.0f;    // 正交投影高度
    float near_clip = 0.1f;
    float far_clip = 1000.0f;
    
    // 交互控制状态
    float yaw = 0.0f;              // 水平旋转角（度）
    float pitch = 0.0f;            // 垂直旋转角（度）
    float distance = 10.0f;        // 到目标点的距离
    
    Camera() = default;
    
    // 设置位置
    void set_position(float x, float y, float z) {
        position[0] = x; position[1] = y; position[2] = z;
        update_from_position();
    }
    
    // 设置目标点
    void set_target(float x, float y, float z) {
        target[0] = x; target[1] = y; target[2] = z;
        update_from_position();
    }
    
    // 设置上方向
    void set_up(float x, float y, float z) {
        up[0] = x; up[1] = y; up[2] = z;
    }
    
    // 使用预设视角
    void set_view_preset(ViewPreset preset, float dist = 10.0f) {
        distance = dist;
        target[0] = target[1] = target[2] = 0.0f;
        
        switch (preset) {
            case ViewPreset::Front:
                position[0] = 0; position[1] = 0; position[2] = -dist;
                up[0] = 0; up[1] = 1; up[2] = 0;
                yaw = 0; pitch = 0;
                break;
            case ViewPreset::Back:
                position[0] = 0; position[1] = 0; position[2] = dist;
                up[0] = 0; up[1] = 1; up[2] = 0;
                yaw = 180; pitch = 0;
                break;
            case ViewPreset::Left:
                position[0] = -dist; position[1] = 0; position[2] = 0;
                up[0] = 0; up[1] = 1; up[2] = 0;
                yaw = -90; pitch = 0;
                break;
            case ViewPreset::Right:
                position[0] = dist; position[1] = 0; position[2] = 0;
                up[0] = 0; up[1] = 1; up[2] = 0;
                yaw = 90; pitch = 0;
                break;
            case ViewPreset::Top:
                position[0] = 0; position[1] = dist; position[2] = 0;
                up[0] = 0; up[1] = 0; up[2] = 1;
                yaw = 0; pitch = 90;
                break;
            case ViewPreset::Bottom:
                position[0] = 0; position[1] = -dist; position[2] = 0;
                up[0] = 0; up[1] = 0; up[2] = -1;
                yaw = 0; pitch = -90;
                break;
            default:
                break;
        }
    }
    
    // 设置投影类型
    void set_perspective(float fov = 45.0f) {
        projection_type = ProjectionType::Perspective;
        fov_y = fov;
    }
    
    void set_orthographic(float height = 20.0f) {
        projection_type = ProjectionType::Orthographic;
        ortho_height = height;
    }
    
    // 交互控制：旋转
    void rotate(float delta_yaw, float delta_pitch) {
        yaw += delta_yaw;
        pitch += delta_pitch;
        
        // 限制俯仰角
        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;
        
        update_position_from_angles();
    }
    
    // 交互控制：缩放
    void zoom(float delta) {
        distance *= (1.0f - delta * 0.1f);
        if (distance < 0.1f) distance = 0.1f;
        if (distance > 1000.0f) distance = 1000.0f;
        
        update_position_from_angles();
        
        // 同时更新正交投影的高度
        if (projection_type == ProjectionType::Orthographic) {
            ortho_height = distance * 2.0f;
        }
    }
    
    // 交互控制：平移目标点
    void pan(float delta_x, float delta_y) {
        // 计算相机的右向量和上向量
        float forward[3], right[3], cam_up[3];
        compute_basis(forward, right, cam_up);
        
        float scale = distance * 0.002f;
        target[0] += right[0] * delta_x * scale + cam_up[0] * delta_y * scale;
        target[1] += right[1] * delta_x * scale + cam_up[1] * delta_y * scale;
        target[2] += right[2] * delta_x * scale + cam_up[2] * delta_y * scale;
        
        update_position_from_angles();
    }
    
    // 计算相机基向量
    void compute_basis(float* forward, float* right, float* cam_up) const {
        // forward = normalize(target - position)
        forward[0] = target[0] - position[0];
        forward[1] = target[1] - position[1];
        forward[2] = target[2] - position[2];
        float len = std::sqrt(forward[0]*forward[0] + forward[1]*forward[1] + forward[2]*forward[2]);
        if (len > 1e-6f) {
            forward[0] /= len; forward[1] /= len; forward[2] /= len;
        }
        
        // right = normalize(forward × up)
        right[0] = forward[1] * up[2] - forward[2] * up[1];
        right[1] = forward[2] * up[0] - forward[0] * up[2];
        right[2] = forward[0] * up[1] - forward[1] * up[0];
        len = std::sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
        if (len > 1e-6f) {
            right[0] /= len; right[1] /= len; right[2] /= len;
        }
        
        // cam_up = right × forward
        cam_up[0] = right[1] * forward[2] - right[2] * forward[1];
        cam_up[1] = right[2] * forward[0] - right[0] * forward[2];
        cam_up[2] = right[0] * forward[1] - right[1] * forward[0];
    }
    
    // 获取用于渲染的相机参数
    void get_render_params(
        float* eye, float* U, float* V, float* W,
        int width, int height
    ) const {
        eye[0] = position[0];
        eye[1] = position[1];
        eye[2] = position[2];
        
        float forward[3], right[3], cam_up[3];
        compute_basis(forward, right, cam_up);
        
        float aspect = static_cast<float>(width) / static_cast<float>(height);
        
        if (projection_type == ProjectionType::Perspective) {
            float half_height = std::tan(fov_y * 0.5f * M_PI / 180.0f);
            float half_width = half_height * aspect;
            
            // W = forward (指向场景)
            W[0] = forward[0]; W[1] = forward[1]; W[2] = forward[2];
            
            // U = right * half_width
            U[0] = right[0] * half_width;
            U[1] = right[1] * half_width;
            U[2] = right[2] * half_width;
            
            // V = cam_up * half_height
            V[0] = cam_up[0] * half_height;
            V[1] = cam_up[1] * half_height;
            V[2] = cam_up[2] * half_height;
        } else {
            // 正交投影
            float half_height = ortho_height * 0.5f;
            float half_width = half_height * aspect;
            
            // W = forward，但正交投影中不用于计算方向
            W[0] = forward[0]; W[1] = forward[1]; W[2] = forward[2];
            
            // U, V 用于计算光线原点偏移
            U[0] = right[0] * half_width;
            U[1] = right[1] * half_width;
            U[2] = right[2] * half_width;
            
            V[0] = cam_up[0] * half_height;
            V[1] = cam_up[1] * half_height;
            V[2] = cam_up[2] * half_height;
        }
    }
    
    bool is_orthographic() const {
        return projection_type == ProjectionType::Orthographic;
    }
    
private:
    void update_from_position() {
        float dx = position[0] - target[0];
        float dy = position[1] - target[1];
        float dz = position[2] - target[2];
        
        distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (distance < 1e-6f) distance = 1.0f;
        
        // 计算 yaw 和 pitch
        float horizontal_dist = std::sqrt(dx*dx + dz*dz);
        pitch = std::atan2(dy, horizontal_dist) * 180.0f / M_PI;
        yaw = std::atan2(dx, -dz) * 180.0f / M_PI;
    }
    
    void update_position_from_angles() {
        float yaw_rad = yaw * M_PI / 180.0f;
        float pitch_rad = pitch * M_PI / 180.0f;
        
        float cos_pitch = std::cos(pitch_rad);
        position[0] = target[0] + distance * std::sin(yaw_rad) * cos_pitch;
        position[1] = target[1] + distance * std::sin(pitch_rad);
        position[2] = target[2] - distance * std::cos(yaw_rad) * cos_pitch;
    }
};

} // namespace diff_optics