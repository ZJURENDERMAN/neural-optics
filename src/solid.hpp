// solid.hpp
#pragma once
#include "object.hpp"
#include "surface.hpp"
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

namespace diff_optics {

// ============= 抽象基类 Solid =============
struct Solid : public Object {
protected:
    std::vector<std::shared_ptr<Surface>> surfaces_;
    
public:
    virtual ~Solid() = default;
    
    // ============= Object 接口实现 =============
    ObjectType object_type() const override { return ObjectType::Solid; }
    
    std::vector<std::pair<std::string, std::shared_ptr<Surface>>> 
        get_surfaces_with_names(const std::string& prefix) override {
        std::vector<std::pair<std::string, std::shared_ptr<Surface>>> result;
        for (size_t i = 0; i < surfaces_.size(); ++i) {
            if (surfaces_[i]) {
                std::string name = prefix + "_face_" + std::to_string(i);
                result.emplace_back(name, surfaces_[i]);
            }
        }
        return result;
    }
    
    size_t surface_count() const override { return surfaces_.size(); }
    
    void invalidate_all_meshes() override {
        for (auto& surf : surfaces_) {
            if (surf) surf->invalidate_mesh();
        }
    }
    
    // ============= 表面访问 =============
    std::shared_ptr<Surface> get_surface(size_t index) const {
        if (index >= surfaces_.size()) {
            throw std::out_of_range("Surface index out of range");
        }
        return surfaces_[index];
    }
    
    const std::vector<std::shared_ptr<Surface>>& get_surfaces() const {
        return surfaces_;
    }
};

// ============= Cube（六个矩形平面）=============
struct Cube : public Solid {
protected:
    ScalarType size_x_, size_y_, size_z_;
    
public:
    static constexpr int FACE_FRONT = 0;
    static constexpr int FACE_BACK = 1;
    static constexpr int FACE_LEFT = 2;
    static constexpr int FACE_RIGHT = 3;
    static constexpr int FACE_TOP = 4;
    static constexpr int FACE_BOTTOM = 5;
    
    Cube() : Cube(1.0f, 1.0f, 1.0f) {}
    
    Cube(ScalarType size) : Cube(size, size, size) {}
    
    Cube(ScalarType sx, ScalarType sy, ScalarType sz)
        : size_x_(sx), size_y_(sy), size_z_(sz) {
        create_faces();
        update_face_transforms();
    }
    
    std::string type_name() const override { return "Cube"; }
    
    ScalarType get_size_x() const { return size_x_; }
    ScalarType get_size_y() const { return size_y_; }
    ScalarType get_size_z() const { return size_z_; }
    
    void set_size(ScalarType sx, ScalarType sy, ScalarType sz) {
        size_x_ = sx;
        size_y_ = sy;
        size_z_ = sz;
        recreate_faces();
    }
    
    std::shared_ptr<RectangleSurface> get_face(int face_index) const {
        return std::static_pointer_cast<RectangleSurface>(get_surface(face_index));
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Size: " << size_x_ << " x " << size_y_ << " x " << size_z_ << std::endl;
        transform_.print("  Transform");
    }
    
private:
    void create_faces() {
        surfaces_.resize(6);
        
        surfaces_[FACE_FRONT] = make_rectangle_plane(size_x_, size_y_);
        surfaces_[FACE_BACK] = make_rectangle_plane(size_x_, size_y_);
        surfaces_[FACE_LEFT] = make_rectangle_plane(size_z_, size_y_);
        surfaces_[FACE_RIGHT] = make_rectangle_plane(size_z_, size_y_);
        surfaces_[FACE_TOP] = make_rectangle_plane(size_x_, size_z_);
        surfaces_[FACE_BOTTOM] = make_rectangle_plane(size_x_, size_z_);
    }
    
    void update_face_transforms() {
        ScalarType hx = size_x_ * 0.5f;
        ScalarType hy = size_y_ * 0.5f;
        ScalarType hz = size_z_ * 0.5f;
        
        surfaces_[FACE_FRONT]->set_transform(Transform(0, 0, hz, 0, 0, 0));
        surfaces_[FACE_FRONT]->set_parent(this);
        
        surfaces_[FACE_BACK]->set_transform(Transform(0, 0, -hz, 0, 180, 0));
        surfaces_[FACE_BACK]->set_parent(this);
        
        surfaces_[FACE_LEFT]->set_transform(Transform(-hx, 0, 0, 0, -90, 0));
        surfaces_[FACE_LEFT]->set_parent(this);
        
        surfaces_[FACE_RIGHT]->set_transform(Transform(hx, 0, 0, 0, 90, 0));
        surfaces_[FACE_RIGHT]->set_parent(this);
        
        surfaces_[FACE_TOP]->set_transform(Transform(0, hy, 0, -90, 0, 0));
        surfaces_[FACE_TOP]->set_parent(this);
        
        surfaces_[FACE_BOTTOM]->set_transform(Transform(0, -hy, 0, 90, 0, 0));
        surfaces_[FACE_BOTTOM]->set_parent(this);
    }
    
    void recreate_faces() {
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_FRONT])->set_width(size_x_);
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_FRONT])->set_height(size_y_);
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_BACK])->set_width(size_x_);
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_BACK])->set_height(size_y_);
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_LEFT])->set_width(size_z_);
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_LEFT])->set_height(size_y_);
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_RIGHT])->set_width(size_z_);
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_RIGHT])->set_height(size_y_);
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_TOP])->set_width(size_x_);
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_TOP])->set_height(size_z_);
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_BOTTOM])->set_width(size_x_);
        std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_BOTTOM])->set_height(size_z_);
        
        update_face_transforms();
        invalidate_all_meshes();
    }
};

// ============= 抽象类 Lens =============
struct Lens : public Solid {
protected:
    std::shared_ptr<VolumeMaterial> lens_material_;
    ScalarType thickness_;
    
public:
    static constexpr int FACE_FRONT = 0;
    static constexpr int FACE_BACK = 1;
    
    virtual ~Lens() = default;
    
    std::shared_ptr<Surface> get_front_surface() const { return surfaces_[FACE_FRONT]; }
    std::shared_ptr<Surface> get_back_surface() const { return surfaces_[FACE_BACK]; }
    
    std::shared_ptr<VolumeMaterial> get_lens_material() const { return lens_material_; }
    void set_lens_material(std::shared_ptr<VolumeMaterial> mat) { 
        lens_material_ = mat;
        if (surfaces_[FACE_FRONT]) {
            surfaces_[FACE_FRONT]->set_inner_material(mat);
        }
        if (surfaces_[FACE_BACK]) {
            surfaces_[FACE_BACK]->set_inner_material(mat);
        }
    }
    
    ScalarType get_thickness() const { return thickness_; }
    
    virtual void set_thickness(ScalarType t) {
        thickness_ = t;
        update_face_transforms();
    }
    
protected:
    virtual void update_face_transforms() = 0;
};

// ============= 矩形透镜 =============
struct RectangleLens : public Lens {
protected:
    ScalarType width_, height_;
public:
    RectangleLens() : RectangleLens(10.0f, 10.0f, 2.0f) {}
    
    RectangleLens(ScalarType w, ScalarType h, ScalarType thickness)
        : width_(w), height_(h){
        thickness_ = thickness;
        create_surfaces();
        update_face_transforms();
    }
    
    std::string type_name() const override { return "RectangleLens"; }
    
    ScalarType get_width() const { return width_; }
    ScalarType get_height() const { return height_; }
    
    std::shared_ptr<RectangleSurface> get_front_bspline() const {
        return std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_FRONT]);
    }
    
    std::shared_ptr<RectangleSurface> get_back_bspline() const {
        return std::static_pointer_cast<RectangleSurface>(surfaces_[FACE_BACK]);
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Size: " << width_ << " x " << height_ << std::endl;
        std::cout << "  Thickness: " << thickness_ << std::endl;
        if (lens_material_) {
            std::cout << "  Material: " << lens_material_->type_name() << std::endl;
        }
        transform_.print("  Transform");
    }
    
protected:
    void create_surfaces() {
        surfaces_.resize(2);
        
        surfaces_[FACE_FRONT] = make_rectangle_plane(
            width_, height_
        );
        surfaces_[FACE_BACK] = make_rectangle_plane(
            width_, height_
        );
        
        auto refractor = std::make_shared<SpecularRefractor>();
        surfaces_[FACE_FRONT]->set_bsdf(refractor);
        surfaces_[FACE_BACK]->set_bsdf(refractor);
    }
    
    void update_face_transforms() override {
        ScalarType half_t = thickness_ * 0.5f;
        
        surfaces_[FACE_FRONT]->set_transform(Transform(0, 0, half_t, 0, 0, 0));
        surfaces_[FACE_FRONT]->set_parent(this);
        
        surfaces_[FACE_BACK]->set_transform(Transform(0, 0, -half_t, 0, 180, 0));
        surfaces_[FACE_BACK]->set_parent(this);
    }
};

// ============= 圆形透镜 =============
struct CircleLens : public Lens {
protected:
    ScalarType radius_;
public:
    CircleLens() : CircleLens(5.0f, 2.0f) {}
    
    CircleLens(ScalarType r, ScalarType thickness)
        : radius_(r) {
        thickness_ = thickness;
        create_surfaces();
        update_face_transforms();
    }
    
    std::string type_name() const override { return "CircleLens"; }
    
    ScalarType get_radius() const { return radius_; }
    
    std::shared_ptr<CircleSurface> get_front_circle() const {
        return std::static_pointer_cast<CircleSurface>(surfaces_[FACE_FRONT]);
    }
    
    std::shared_ptr<CircleSurface> get_back_circle() const {
        return std::static_pointer_cast<CircleSurface>(surfaces_[FACE_BACK]);
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Radius: " << radius_ << std::endl;
        std::cout << "  Thickness: " << thickness_ << std::endl;
        if (lens_material_) {
            std::cout << "  Material: " << lens_material_->type_name() << std::endl;
        }
        transform_.print("  Transform");
    }
    
protected:
    void create_surfaces() {
        surfaces_.resize(2);
        
        surfaces_[FACE_FRONT] = make_circle_plane(radius_);
        surfaces_[FACE_BACK] = make_circle_plane(radius_);
        
        auto refractor = std::make_shared<SpecularRefractor>();
        surfaces_[FACE_FRONT]->set_bsdf(refractor);
        surfaces_[FACE_BACK]->set_bsdf(refractor);
    }
    
    void update_face_transforms() override {
        ScalarType half_t = thickness_ * 0.5f;
        
        surfaces_[FACE_FRONT]->set_transform(Transform(0, 0, half_t, 0, 0, 0));
        surfaces_[FACE_FRONT]->set_parent(this);
        
        surfaces_[FACE_BACK]->set_transform(Transform(0, 0, -half_t, 0, 180, 0));
        surfaces_[FACE_BACK]->set_parent(this);
    }
};

} // namespace diff_optics