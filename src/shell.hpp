// shell.hpp
#pragma once
#include "object.hpp"
#include "surface.hpp"
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

namespace diff_optics {

// ============= 抽象基类 Shell =============
struct Shell : public Object {
protected:
    std::vector<std::vector<std::shared_ptr<Surface>>> surfaces_;
    int num_rows_ = 0;
    int num_cols_ = 0;
    
public:
    virtual ~Shell() = default;
    
    // ============= Object 接口实现 =============
    ObjectType object_type() const override { return ObjectType::Shell; }
    
    std::vector<std::pair<std::string, std::shared_ptr<Surface>>> 
        get_surfaces_with_names(const std::string& prefix) override {
        std::vector<std::pair<std::string, std::shared_ptr<Surface>>> result;
        for (int i = 0; i < num_rows_; ++i) {
            for (int j = 0; j < num_cols_; ++j) {
                if (surfaces_[i][j]) {
                    std::string name = prefix + "_" + std::to_string(i) + "_" + std::to_string(j);
                    result.emplace_back(name, surfaces_[i][j]);
                }
            }
        }
        return result;
    }
    
    size_t surface_count() const override {
        return static_cast<size_t>(num_rows_) * static_cast<size_t>(num_cols_);
    }
    
    void invalidate_all_meshes() override {
        for (auto& row : surfaces_) {
            for (auto& surf : row) {
                if (surf) surf->invalidate_mesh();
            }
        }
    }
    
    // ============= Shell 基本访问 =============
    int get_num_rows() const { return num_rows_; }
    int get_num_cols() const { return num_cols_; }
    
    std::shared_ptr<Surface> get_surface(int row, int col) const {
        if (row < 0 || row >= num_rows_ || col < 0 || col >= num_cols_) {
            throw std::out_of_range("Surface index out of range");
        }
        return surfaces_[row][col];
    }
    
    // ============= 抽象方法 =============
    virtual ScalarType get_width() const = 0;
    virtual ScalarType get_height() const = 0;
};

// ============= 矩形 Shell（抽象类）=============
struct RectangleShell : public Shell {
protected:
    ScalarType total_width_ = 0;
    ScalarType total_height_ = 0;
    ScalarType cell_width_ = 0;
    ScalarType cell_height_ = 0;
    
    std::vector<ScalarType> u_gaps_;
    std::vector<ScalarType> v_gaps_;
    
public:
    virtual ~RectangleShell() = default;
    
    ScalarType get_width() const override { return total_width_; }
    ScalarType get_height() const override { return total_height_; }
    ScalarType get_cell_width() const { return cell_width_; }
    ScalarType get_cell_height() const { return cell_height_; }
    
    // ============= 间隙管理 =============
    const std::vector<ScalarType>& get_u_gaps() const { return u_gaps_; }
    const std::vector<ScalarType>& get_v_gaps() const { return v_gaps_; }
    
    void set_u_gap(int index, ScalarType gap) {
        if (index >= 0 && index < static_cast<int>(u_gaps_.size())) {
            u_gaps_[index] = gap;
            update_surface_transforms();
        }
    }
    
    void set_v_gap(int index, ScalarType gap) {
        if (index >= 0 && index < static_cast<int>(v_gaps_.size())) {
            v_gaps_[index] = gap;
            update_surface_transforms();
        }
    }
    
    // ============= 计算总尺寸 =============
    void compute_total_size() {
        total_width_ = cell_width_ * num_cols_;
        for (auto g : u_gaps_) total_width_ += g;
        
        total_height_ = cell_height_ * num_rows_;
        for (auto g : v_gaps_) total_height_ += g;
    }
    
    // ============= 更新子表面变换 =============
    virtual void update_surface_transforms() {
        ScalarType half_w = total_width_ * 0.5f;
        ScalarType half_h = total_height_ * 0.5f;
        
        ScalarType y_offset = -half_h + cell_height_ * 0.5f;
        
        for (int i = 0; i < num_rows_; ++i) {
            ScalarType x_offset = -half_w + cell_width_ * 0.5f;
            
            for (int j = 0; j < num_cols_; ++j) {
                if (surfaces_[i][j]) {
                    Transform t;
                    t.set_translation(x_offset, y_offset, 0.0f);
                    surfaces_[i][j]->set_transform(t);
                    surfaces_[i][j]->set_parent(this);
                }
                
                x_offset += cell_width_;
                if (j < static_cast<int>(u_gaps_.size())) {
                    x_offset += u_gaps_[j];
                }
            }
            
            y_offset += cell_height_;
            if (i < static_cast<int>(v_gaps_.size())) {
                y_offset += v_gaps_[i];
            }
        }
    }
    
protected:
    void init_gaps() {
        u_gaps_.resize(num_cols_ > 0 ? num_cols_ - 1 : 0, 0.0f);
        v_gaps_.resize(num_rows_ > 0 ? num_rows_ - 1 : 0, 0.0f);
    }
};

//// ============= RectanglePlaneShell =============
//struct RectanglePlaneShell : public RectangleShell {
//public:
//    RectanglePlaneShell() = default;
//    
//    RectanglePlaneShell(int rows, int cols, ScalarType cell_w, ScalarType cell_h) {
//        num_rows_ = rows;
//        num_cols_ = cols;
//        cell_width_ = cell_w;
//        cell_height_ = cell_h;
//        
//        init_gaps();
//        create_surfaces();
//        compute_total_size();
//        update_surface_transforms();
//    }
//    
//    std::string type_name() const override { return "RectanglePlaneShell"; }
//    
//    void print(const std::string& name = "") const override {
//        if (!name.empty()) {
//            std::cout << name << ":" << std::endl;
//        }
//        std::cout << "  Type: " << type_name() << std::endl;
//        std::cout << "  Grid: " << num_rows_ << " x " << num_cols_ << std::endl;
//        std::cout << "  Cell Size: " << cell_width_ << " x " << cell_height_ << std::endl;
//        std::cout << "  Total Size: " << total_width_ << " x " << total_height_ << std::endl;
//        transform_.print("  Transform");
//    }
//    
//private:
//    void create_surfaces() {
//        surfaces_.resize(num_rows_);
//        for (int i = 0; i < num_rows_; ++i) {
//            surfaces_[i].resize(num_cols_);
//            for (int j = 0; j < num_cols_; ++j) {
//                surfaces_[i][j] = std::make_shared<RectangleSurface>(cell_width_, cell_height_);
//            }
//        }
//    }
//};
//
//// ============= RectangleBSplineShell =============
//struct RectangleBSplineShell : public RectangleShell {
//protected:
//    BSplineConfig bspline_config_;
//    
//    Float global_cp_z_;
//    
//    int global_u_cp_ = 0;
//    int global_v_cp_ = 0;
//    
//    std::vector<std::vector<UInt32C>> cp_index_maps_;
//    
//public:
//    RectangleBSplineShell() = default;
//    
//    RectangleBSplineShell(int rows, int cols, ScalarType cell_w, ScalarType cell_h,
//                          const BSplineConfig& config = BSplineConfig()) {
//        num_rows_ = rows;
//        num_cols_ = cols;
//        cell_width_ = cell_w;
//        cell_height_ = cell_h;
//        bspline_config_ = config;
//        
//        init_gaps();
//        compute_global_cp_size();
//        init_global_control_points();
//        create_surfaces();
//        compute_total_size();
//        update_surface_transforms();
//        update_control_point_mappings();
//    }
//    
//    std::string type_name() const override { return "RectangleBSplineShell"; }
//    
//    // ============= 全局控制点访问 =============
//    Float& get_global_cp_z() { return global_cp_z_; }
//    const Float& get_global_cp_z() const { return global_cp_z_; }
//    
//    void set_global_cp_z(const Float& z) {
//        global_cp_z_ = z;
//        invalidate_all_meshes();
//    }
//    
//    int get_global_u_cp() const { return global_u_cp_; }
//    int get_global_v_cp() const { return global_v_cp_; }
//    int get_total_control_points() const { return global_u_cp_ * global_v_cp_; }
//    
//    // ============= 间隙设置 =============
//    void set_u_gap(int index, ScalarType gap) {
//        if (index >= 0 && index < static_cast<int>(u_gaps_.size())) {
//            bool was_zero = (u_gaps_[index] == 0.0f);
//            bool will_be_zero = (gap == 0.0f);
//            
//            u_gaps_[index] = gap;
//            compute_total_size();
//            update_surface_transforms();
//            
//            if (was_zero != will_be_zero) {
//                compute_global_cp_size();
//                init_global_control_points();
//                update_control_point_mappings();
//            }
//        }
//    }
//    
//    void set_v_gap(int index, ScalarType gap) {
//        if (index >= 0 && index < static_cast<int>(v_gaps_.size())) {
//            bool was_zero = (v_gaps_[index] == 0.0f);
//            bool will_be_zero = (gap == 0.0f);
//            
//            v_gaps_[index] = gap;
//            compute_total_size();
//            update_surface_transforms();
//            
//            if (was_zero != will_be_zero) {
//                compute_global_cp_size();
//                init_global_control_points();
//                update_control_point_mappings();
//            }
//        }
//    }
//    
//    // ============= 获取单个表面 =============
//    std::shared_ptr<RectangleSurface> get_bspline_surface(int row, int col) const {
//        return std::static_pointer_cast<RectangleSurface>(get_surface(row, col));
//    }
//    
//    void print(const std::string& name = "") const override {
//        if (!name.empty()) {
//            std::cout << name << ":" << std::endl;
//        }
//        std::cout << "  Type: " << type_name() << std::endl;
//        std::cout << "  Grid: " << num_rows_ << " x " << num_cols_ << std::endl;
//        std::cout << "  Cell Size: " << cell_width_ << " x " << cell_height_ << std::endl;
//        std::cout << "  Total Size: " << total_width_ << " x " << total_height_ << std::endl;
//        std::cout << "  B-Spline: degree " << bspline_config_.u_degree << "x" << bspline_config_.v_degree << std::endl;
//        std::cout << "  Per-cell CP: " << bspline_config_.u_control_points << "x" << bspline_config_.v_control_points << std::endl;
//        std::cout << "  Global CP: " << global_u_cp_ << "x" << global_v_cp_ << " = " << get_total_control_points() << std::endl;
//        transform_.print("  Transform");
//    }
//    
//private:
//    void compute_global_cp_size() {
//        int single_u_cp = bspline_config_.u_control_points;
//        int single_v_cp = bspline_config_.v_control_points;
//        
//        global_u_cp_ = single_u_cp;
//        for (int j = 1; j < num_cols_; ++j) {
//            if (j - 1 < static_cast<int>(u_gaps_.size()) && u_gaps_[j - 1] == 0.0f) {
//                global_u_cp_ += single_u_cp - 1;
//            } else {
//                global_u_cp_ += single_u_cp;
//            }
//        }
//        
//        global_v_cp_ = single_v_cp;
//        for (int i = 1; i < num_rows_; ++i) {
//            if (i - 1 < static_cast<int>(v_gaps_.size()) && v_gaps_[i - 1] == 0.0f) {
//                global_v_cp_ += single_v_cp - 1;
//            } else {
//                global_v_cp_ += single_v_cp;
//            }
//        }
//    }
//    
//    void init_global_control_points() {
//        int total_cp = global_u_cp_ * global_v_cp_;
//        
//        std::vector<ScalarType> cp_z_init(total_cp, 0.0f);
//        
//        if (bspline_config_.amplitude > 0) {
//            for (int i = 0; i < total_cp; ++i) {
//                cp_z_init[i] = bspline_config_.amplitude * 
//                    (2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);
//            }
//        }
//        
//        global_cp_z_ = drjit::load<Float>(cp_z_init.data(), total_cp);
//    }
//    
//    void create_surfaces() {
//        surfaces_.resize(num_rows_);
//        for (int i = 0; i < num_rows_; ++i) {
//            surfaces_[i].resize(num_cols_);
//            for (int j = 0; j < num_cols_; ++j) {
//                auto surf = std::make_shared<RectangleSurface>(
//                    cell_width_, cell_height_, bspline_config_
//                );
//                surfaces_[i][j] = surf;
//            }
//        }
//    }
//    
//    void update_control_point_mappings() {
//        cp_index_maps_.resize(num_rows_);
//        
//        int single_u_cp = bspline_config_.u_control_points;
//        int single_v_cp = bspline_config_.v_control_points;
//        
//        std::vector<int> u_starts(num_cols_);
//        std::vector<int> v_starts(num_rows_);
//        
//        u_starts[0] = 0;
//        for (int j = 1; j < num_cols_; ++j) {
//            if (j - 1 < static_cast<int>(u_gaps_.size()) && u_gaps_[j - 1] == 0.0f) {
//                u_starts[j] = u_starts[j - 1] + single_u_cp - 1;
//            } else {
//                u_starts[j] = u_starts[j - 1] + single_u_cp;
//            }
//        }
//        
//        v_starts[0] = 0;
//        for (int i = 1; i < num_rows_; ++i) {
//            if (i - 1 < static_cast<int>(v_gaps_.size()) && v_gaps_[i - 1] == 0.0f) {
//                v_starts[i] = v_starts[i - 1] + single_v_cp - 1;
//            } else {
//                v_starts[i] = v_starts[i - 1] + single_v_cp;
//            }
//        }
//        
//        for (int i = 0; i < num_rows_; ++i) {
//            cp_index_maps_[i].resize(num_cols_);
//            
//            for (int j = 0; j < num_cols_; ++j) {
//                std::vector<uint32_t> indices(single_u_cp * single_v_cp);
//                
//                int v_base = v_starts[i];
//                int u_base = u_starts[j];
//                
//                for (int vi = 0; vi < single_v_cp; ++vi) {
//                    for (int ui = 0; ui < single_u_cp; ++ui) {
//                        int local_idx = vi * single_u_cp + ui;
//                        int global_idx = (v_base + vi) * global_u_cp_ + (u_base + ui);
//                        indices[local_idx] = static_cast<uint32_t>(global_idx);
//                    }
//                }
//                
//                cp_index_maps_[i][j] = drjit::load<UInt32C>(indices.data(), indices.size());
//                
//                auto surf = std::static_pointer_cast<RectangleSurface>(surfaces_[i][j]);
//                surf->set_external_control_points(&global_cp_z_, cp_index_maps_[i][j]);
//            }
//        }
//    }
//};

} // namespace diff_optics