#pragma once
#include "object.hpp"
#include "utils.hpp"
#include "shape.hpp"
#include "bsdf.hpp"
#include "material.hpp"
#include <memory>
#include <array>
namespace diff_optics {

// ============= 三角网格数据结构 =============
struct TriangleMesh {
    FloatC vertices_x;
    FloatC vertices_y;
    FloatC vertices_z;
    UInt32C indices;
    
    size_t num_vertices = 0;
    size_t num_triangles = 0;
    
    TriangleMesh() = default;
    
    void clear() {
        vertices_x = FloatC();
        vertices_y = FloatC();
        vertices_z = FloatC();
        indices = UInt32C();
        num_vertices = 0;
        num_triangles = 0;
    }
    
    bool is_valid() const {
        return num_vertices > 0 && num_triangles > 0;
    }
};

// ============= 抽象基类 Surface =============
struct Surface : public Object {
protected:
    std::shared_ptr<Shape> shape;
    
    std::shared_ptr<BSDF> bsdf;
    std::shared_ptr<VolumeMaterial> inner_material;
    std::shared_ptr<VolumeMaterial> outer_material;
    
    int tess_u = 1;
    int tess_v = 1;
    
    mutable TriangleMesh cached_mesh;
    mutable bool mesh_dirty = true;
public:
    virtual ~Surface() = default;
    
    // ============= Object 接口实现 =============
    ObjectType object_type() const override { return ObjectType::Surface; }
    
    std::vector<std::pair<std::string, std::shared_ptr<Surface>>> 
        get_surfaces_with_names(const std::string& prefix) override {
        return {};
    }
    
    size_t surface_count() const override { return 1; }
    
    void invalidate_all_meshes() override { invalidate_mesh(); }
    
    virtual std::string boundary_type() const = 0;
    virtual std::string type_name() const override = 0;
    virtual void print(const std::string& name = "") const override = 0;
    virtual std::array<ScalarType, 2> get_u_range()const=0;
    virtual std::array<ScalarType, 2> get_v_range()const=0;
    // ============= BSDF 和材质管理 =============
    std::shared_ptr<BSDF> get_bsdf() const { return bsdf; }
    void set_bsdf(std::shared_ptr<BSDF> b) { bsdf = b; }
    
    std::shared_ptr<VolumeMaterial> get_inner_material() const { return inner_material; }
    std::shared_ptr<VolumeMaterial> get_outer_material() const { return outer_material; }
    void set_inner_material(std::shared_ptr<VolumeMaterial> m) { inner_material = m; }
    void set_outer_material(std::shared_ptr<VolumeMaterial> m) { outer_material = m; }
    
    // ============= 细分参数 =============
    virtual void set_tessellation(int u, int v) {
        if (u != tess_u || v != tess_v) {
            tess_u = u;
            tess_v = v;
            mesh_dirty = true;
        }
    }
    
    int get_tess_u() const { return tess_u; }
    int get_tess_v() const { return tess_v; }
    
    // ============= 网格生成 =============
    virtual TriangleMesh generate_mesh() const = 0;
    
    const TriangleMesh& get_mesh() const {
        if (mesh_dirty || !cached_mesh.is_valid()) {
            cached_mesh = generate_mesh();
            mesh_dirty = false;
        }
        return cached_mesh;
    }
    
    void invalidate_mesh() const {
        mesh_dirty = true;
    }
    
    virtual void update_mesh_vertices() const {
        cached_mesh = generate_mesh();
        mesh_dirty = false;
    }
    
    // ============= UV 映射（派生类必须实现）=============
    virtual Vector2 mapping_world_position_to_local_uv(const Vector3& world_p) const = 0;
    
    // ============= 光线追踪相关 =============
    void compute_surface_record_from_t(const Ray& ray, const Float& t, SurfaceRecord& record) const {
        Vector3 world_p(
            ray.origin[0] + t * ray.direction[0],
            ray.origin[1] + t * ray.direction[1],
            ray.origin[2] + t * ray.direction[2]
        );
        
        Vector2 uv = mapping_world_position_to_local_uv(world_p);
        auto local_normal = shape->compute_normal(uv);
        
        record.position = world_p;
        record.normal = drjit::normalize(to_world_normal(local_normal));
    }
    
    // ============= 折射率计算 =============
    // 只计算默认方向的折射率：eta_default = n_outer / n_inner
    // BSDF 内部会根据实际入射方向决定是否取倒数
    Float compute_eta_default(const Ray& ray) const {
        size_t n = ray.size();
        
        Float n_inner = inner_material ? 
            inner_material->ior(ray.wavelength) : 
            drjit::full<Float>(1.0f, n);
        Float n_outer = outer_material ? 
            outer_material->ior(ray.wavelength) : 
            drjit::full<Float>(1.0f, n);
        
        Float eps = drjit::full<Float>(1e-10f, n);
        Float eta = n_outer / drjit::maximum(n_inner, eps);
        
        return eta;
    }
    
    // 获取内外折射率（供 BSDF 使用）
    std::pair<Float, Float> get_ior_pair(const Ray& ray) const {
        size_t n = ray.size();
        
        Float n_inner = inner_material ? 
            inner_material->ior(ray.wavelength) : 
            drjit::full<Float>(1.0f, n);
        Float n_outer = outer_material ? 
            outer_material->ior(ray.wavelength) : 
            drjit::full<Float>(1.0f, n);
        
        return {n_outer, n_inner};
    }
    
    Ray sample_ray(const Ray& ray, const SurfaceRecord& hit, const Vector2& uv) const {
        if (bsdf != nullptr) {
            auto [n_outer, n_inner] = get_ior_pair(ray);
            auto sample = bsdf->sample(ray, hit, uv, n_outer, n_inner);
            Ray ret = ray;
            ret.origin = hit.position;
            ret.direction = sample.direction;
            ret.radiance *= sample.weight;
            ret.pdf *= sample.pdf;
            return ret.masked_select(sample.valid);
        } else {
            Ray ret = ray;
            ret.origin = hit.position;
            return ret;
        }
    }
    
    std::shared_ptr<Shape> get_shape() const { return shape; }

protected:
    void print_materials() const {
        if (bsdf) {
            std::cout << "  BSDF: " << bsdf->type_name() << std::endl;
        } else {
            std::cout << "  BSDF: None" << std::endl;
        }
        
        if (inner_material) {
            std::cout << "  Inner Material: " << inner_material->type_name() << std::endl;
        } else {
            std::cout << "  Inner Material: None (vacuum)" << std::endl;
        }
        
        if (outer_material) {
            std::cout << "  Outer Material: " << outer_material->type_name() << std::endl;
        } else {
            std::cout << "  Outer Material: None (vacuum)" << std::endl;
        }
    }
};

// ============= 矩形边界表面 =============
struct RectangleSurface : public Surface {
    ScalarType width;
    ScalarType height;
    
    RectangleSurface() : width(1.0f), height(1.0f) {
        tess_u = 1;
        tess_v = 1;
        shape = std::make_shared<Plane>();
    }
    std::array<ScalarType, 2> get_u_range()const{
        return {-0.5f*width,0.5f*width};
    }
    std::array<ScalarType, 2> get_v_range()const{
        return {-0.5f*height,0.5f*height};
    }
    RectangleSurface(ScalarType w, ScalarType h, std::shared_ptr<Shape> s) 
        : width(w), height(h){
        shape = s;
    }
    
    std::string boundary_type() const override { return "Rectangle"; }
    
    ScalarType get_width() const { return width; }
    ScalarType get_height() const { return height; }
    void set_width(ScalarType val) { width = val; mesh_dirty = true; }
    void set_height(ScalarType val) { height = val; mesh_dirty = true; }
    
    /// UV 映射：世界坐标 -> 局部物理坐标 [-w/2, w/2] × [-h/2, h/2]
    Vector2 mapping_world_position_to_local_uv(const Vector3& world_p) const override {
        Vector3 local_p = from_world_point(world_p);
        return Vector2(local_p[0], local_p[1]);
    }
    
    std::string type_name() const override {
        return "RectangleSurface<" + shape->type_name() + ">";
    }
    
    /// 获取有效的细分参数（考虑 DisplacementMesh 的特殊需求）
    std::pair<int, int> get_effective_tessellation() const {
        auto disp_mesh = std::dynamic_pointer_cast<DisplacementMeshSurface>(shape);
        if (disp_mesh) {
            return disp_mesh->get_recommended_tessellation();
        }
        return {tess_u, tess_v};
    }
    
    TriangleMesh generate_mesh() const override {
        TriangleMesh mesh;
        
        auto [eff_tess_u, eff_tess_v] = get_effective_tessellation();
        
        int nu = eff_tess_u + 1;
        int nv = eff_tess_v + 1;
        
        mesh.num_vertices = static_cast<size_t>(nu) * static_cast<size_t>(nv);
        mesh.num_triangles = static_cast<size_t>(eff_tess_u) * static_cast<size_t>(eff_tess_v) * 2;
        
        size_t N = mesh.num_vertices;
        
        UInt32C indices_arr = drjit::arange<UInt32C>(static_cast<uint32_t>(N));
        UInt32C i_idx = indices_arr % UInt32C(static_cast<uint32_t>(nu));
        UInt32C j_idx = indices_arr / UInt32C(static_cast<uint32_t>(nu));
        
        float half_w = width * 0.5f;
        float half_h = height * 0.5f;
        
        FloatC u = -half_w + FloatC(i_idx) * (width / static_cast<float>(eff_tess_u));
        FloatC v = -half_h + FloatC(j_idx) * (height / static_cast<float>(eff_tess_v));

        Vector3C pos = shape->compute_position_c(Vector2C(u, v));
        
        mesh.vertices_x = pos[0];
        mesh.vertices_y = pos[1];
        mesh.vertices_z = pos[2];
        
        std::vector<uint32_t> idx(mesh.num_triangles * 3);
        size_t tri_idx = 0;
        
        for (int j = 0; j < eff_tess_v; ++j) {
            for (int i = 0; i < eff_tess_u; ++i) {
                uint32_t v00 = static_cast<uint32_t>(j * nu + i);
                uint32_t v10 = static_cast<uint32_t>(j * nu + (i + 1));
                uint32_t v01 = static_cast<uint32_t>((j + 1) * nu + i);
                uint32_t v11 = static_cast<uint32_t>((j + 1) * nu + (i + 1));
                
                idx[tri_idx * 3 + 0] = v00;
                idx[tri_idx * 3 + 1] = v10;
                idx[tri_idx * 3 + 2] = v11;
                tri_idx++;
                
                idx[tri_idx * 3 + 0] = v00;
                idx[tri_idx * 3 + 1] = v11;
                idx[tri_idx * 3 + 2] = v01;
                tri_idx++;
            }
        }
        
        mesh.indices = drjit::load<UInt32C>(idx.data(), mesh.num_triangles * 3);
        drjit::eval(mesh.vertices_x, mesh.vertices_y, mesh.vertices_z, mesh.indices);
        
        return mesh;
    }
    
    void update_mesh_vertices() const override {
        auto [eff_tess_u, eff_tess_v] = get_effective_tessellation();
        
        int nu = eff_tess_u + 1;
        int nv = eff_tess_v + 1;
        size_t N = static_cast<size_t>(nu) * static_cast<size_t>(nv);
        
        UInt32C indices_arr = drjit::arange<UInt32C>(static_cast<uint32_t>(N));
        UInt32C i_idx = indices_arr % UInt32C(static_cast<uint32_t>(nu));
        UInt32C j_idx = indices_arr / UInt32C(static_cast<uint32_t>(nu));
        
        float half_w = width * 0.5f;
        float half_h = height * 0.5f;
        
        FloatC u = -half_w + FloatC(i_idx) * (width / static_cast<float>(eff_tess_u));
        FloatC v = -half_h + FloatC(j_idx) * (height / static_cast<float>(eff_tess_v));
        
        Vector3C pos = shape->compute_position_c(Vector2C(u, v));
        
        cached_mesh.vertices_x = pos[0];
        cached_mesh.vertices_y = pos[1];
        cached_mesh.vertices_z = pos[2];

        drjit::eval(cached_mesh.vertices_x, cached_mesh.vertices_y, cached_mesh.vertices_z);
        mesh_dirty = false;
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Boundary: " << width << " x " << height << std::endl;
        std::cout << "  UV Domain: [" << -width/2 << ", " << width/2 << "] x [" 
                  << -height/2 << ", " << height/2 << "]" << std::endl;
        std::cout << "  Tessellation: " << tess_u << " x " << tess_v << std::endl;
        std::cout << "  Shape: " << shape->type_name() << std::endl;
        print_materials();
    }
    
};

// ============= 圆形边界表面 =============
struct CircleSurface : public Surface {
    ScalarType radius;
    
    CircleSurface() : radius(1.0f) {
        shape = std::make_shared<Plane>();
        tess_u = 16;
        tess_v = 20;
    }
    
    CircleSurface(ScalarType r,std::shared_ptr<Shape> s)
        : radius(std::abs(r)){
        shape = s;
    }
    
    std::array<ScalarType, 2> get_u_range()const {
        return { 0,radius };
    }
    std::array<ScalarType, 2> get_v_range()const {
        return { 0,M_PI*2.0 };
    }

    std::string boundary_type() const override { return "Circle"; }
    
    ScalarType get_radius() const { return radius; }
    void set_radius(ScalarType val) { radius = std::abs(val); mesh_dirty = true; }
    
    /// UV 映射：世界坐标 -> 局部物理坐标 (x, y)
    Vector2 mapping_world_position_to_local_uv(const Vector3& world_p) const override {
        Vector3 local_p = from_world_point(world_p);
        return Vector2(local_p[0], local_p[1]);
    }
    
    std::string type_name() const override {
        return "CircleSurface<" + shape->type_name() + ">";
    }
    
    /// 统一的网格生成：使用极坐标采样 + shape->compute_position_c()
    TriangleMesh generate_mesh() const override {
        TriangleMesh mesh;
        
        int n_radial = tess_u;
        int n_angular = tess_v;
        
        mesh.num_vertices = 1 + static_cast<size_t>(n_radial) * static_cast<size_t>(n_angular);
        mesh.num_triangles = static_cast<size_t>(n_angular) + 
                             static_cast<size_t>(n_radial - 1) * static_cast<size_t>(n_angular) * 2;
        
        std::vector<float> u_coords(mesh.num_vertices);
        std::vector<float> v_coords(mesh.num_vertices);
        
        u_coords[0] = 0.0f;
        v_coords[0] = 0.0f;
        
        size_t vidx = 1;
        for (int ri = 1; ri <= n_radial; ++ri) {
            float r = radius * static_cast<float>(ri) / static_cast<float>(n_radial);
            
            for (int ai = 0; ai < n_angular; ++ai) {
                float theta = 2.0f * static_cast<float>(M_PI) * static_cast<float>(ai) / static_cast<float>(n_angular);
                u_coords[vidx] = r * std::cos(theta);
                v_coords[vidx] = r * std::sin(theta);
                vidx++;
            }
        }
        
        FloatC u = drjit::load<FloatC>(u_coords.data(), mesh.num_vertices);
        FloatC v = drjit::load<FloatC>(v_coords.data(), mesh.num_vertices);
        
        Vector3C pos = shape->compute_position_c(Vector2C(u, v));
        
        mesh.vertices_x = pos[0];
        mesh.vertices_y = pos[1];
        mesh.vertices_z = pos[2];
        
        std::vector<uint32_t> idx(mesh.num_triangles * 3);
        size_t tri_idx = 0;
        
        for (int ai = 0; ai < n_angular; ++ai) {
            uint32_t v0 = 0;
            uint32_t v1 = 1 + static_cast<uint32_t>(ai);
            uint32_t v2 = 1 + static_cast<uint32_t>((ai + 1) % n_angular);
            
            idx[tri_idx * 3 + 0] = v0;
            idx[tri_idx * 3 + 1] = v1;
            idx[tri_idx * 3 + 2] = v2;
            tri_idx++;
        }
        
        for (int ri = 1; ri < n_radial; ++ri) {
            uint32_t ring_start_inner = 1 + static_cast<uint32_t>((ri - 1) * n_angular);
            uint32_t ring_start_outer = 1 + static_cast<uint32_t>(ri * n_angular);
            
            for (int ai = 0; ai < n_angular; ++ai) {
                uint32_t v00 = ring_start_inner + static_cast<uint32_t>(ai);
                uint32_t v10 = ring_start_inner + static_cast<uint32_t>((ai + 1) % n_angular);
                uint32_t v01 = ring_start_outer + static_cast<uint32_t>(ai);
                uint32_t v11 = ring_start_outer + static_cast<uint32_t>((ai + 1) % n_angular);
                
                idx[tri_idx * 3 + 0] = v00;
                idx[tri_idx * 3 + 1] = v10;
                idx[tri_idx * 3 + 2] = v11;
                tri_idx++;
                
                idx[tri_idx * 3 + 0] = v00;
                idx[tri_idx * 3 + 1] = v11;
                idx[tri_idx * 3 + 2] = v01;
                tri_idx++;
            }
        }
        
        mesh.indices = drjit::load<UInt32C>(idx.data(), mesh.num_triangles * 3);
        drjit::eval(mesh.vertices_x, mesh.vertices_y, mesh.vertices_z, mesh.indices);
        
        return mesh;
    }
    
    void update_mesh_vertices() const override {
        int n_radial = tess_u;
        int n_angular = tess_v;
        size_t N = 1 + static_cast<size_t>(n_radial) * static_cast<size_t>(n_angular);
        
        std::vector<float> u_coords(N);
        std::vector<float> v_coords(N);
        
        u_coords[0] = 0.0f;
        v_coords[0] = 0.0f;
        
        size_t vidx = 1;
        for (int ri = 1; ri <= n_radial; ++ri) {
            float r = radius * static_cast<float>(ri) / static_cast<float>(n_radial);
            
            for (int ai = 0; ai < n_angular; ++ai) {
                float theta = 2.0f * static_cast<float>(M_PI) * static_cast<float>(ai) / static_cast<float>(n_angular);
                u_coords[vidx] = r * std::cos(theta);
                v_coords[vidx] = r * std::sin(theta);
                vidx++;
            }
        }
        
        FloatC u = drjit::load<FloatC>(u_coords.data(), N);
        FloatC v = drjit::load<FloatC>(v_coords.data(), N);
        
        Vector3C pos = shape->compute_position_c(Vector2C(u, v));
        
        cached_mesh.vertices_x = pos[0];
        cached_mesh.vertices_y = pos[1];
        cached_mesh.vertices_z = pos[2];
        
        drjit::eval(cached_mesh.vertices_x, cached_mesh.vertices_y, cached_mesh.vertices_z);
        mesh_dirty = false;
    }
    
    void print(const std::string& name = "") const override {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << "  Type: " << type_name() << std::endl;
        std::cout << "  Radius: " << radius << std::endl;
        std::cout << "  Tessellation: " << tess_u << " (radial) x " << tess_v << " (angular)" << std::endl;
        std::cout << "  Shape: " << shape->type_name() << std::endl;
        print_materials();
    }
    
};

// ============= 工厂函数 =============
inline std::shared_ptr<RectangleSurface> make_rectangle_plane(ScalarType w, ScalarType h) {
    return std::make_shared<RectangleSurface>(w, h, std::make_shared<Plane>());
}

inline std::shared_ptr<RectangleSurface> make_rectangle_bspline(
    ScalarType w, ScalarType h, 
    const BSplineConfig& config = BSplineConfig()) {
    return std::make_shared<RectangleSurface>(w, h, std::make_shared<BSplineSurface>(w, h, config));
}

inline std::shared_ptr<RectangleSurface> make_rectangle_xypolynomial(
    ScalarType w, ScalarType h,
    const XYPolynomialConfig& config = XYPolynomialConfig()) {
    return std::make_shared<RectangleSurface>(w, h, std::make_shared<XYPolynomialSurface>(w, h, config));
}

inline std::shared_ptr<RectangleSurface> make_rectangle_displacementmesh(
    ScalarType w, ScalarType h,
    const DisplacementMeshConfig& config = DisplacementMeshConfig()) {
    return std::make_shared<RectangleSurface>(w, h, std::make_shared<DisplacementMeshSurface>(w, h, config));
}

inline std::shared_ptr<CircleSurface> make_circle_plane(ScalarType r) {
    return std::make_shared<CircleSurface>(r, std::make_shared<Plane>());
}
} // namespace diff_optics