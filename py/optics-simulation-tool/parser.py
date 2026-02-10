# parser.py
"""
JSON configuration parser for diff_optics.
Creates Scene, Simulator, and configs from JSON specification.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

# 延迟导入，避免在模块加载时就需要初始化环境
_do = None
_dr = None


def _get_do():
    global _do
    if _do is None:
        from setup import get_diff_optics
        _do = get_diff_optics()
    return _do


def _get_dr():
    global _dr
    if _dr is None:
        from setup import get_drjit
        _dr = get_drjit()
    return _dr


class ParseResult:
    """Container for parsed configuration results."""
    
    def __init__(self):
        self.scene = None  # do.Scene
        self.simulator = None  # do.ForwardSimulator
        self.sim_config = None  # do.SimulatorConfig
        self.opt_config: Optional[List[Dict[str, Any]]] = None
        self.raw_scene_config: Dict[str, Any] = {}
        self.raw_sim_config: Dict[str, Any] = {}
        self.raw_opt_config: Optional[List[Dict[str, Any]]] = None
        self.surface_params_range: Dict[str, List[float]] = {}  # surface_name -> [min, max]
    
    def __repr__(self):
        return (f"ParseResult(scene={self.scene is not None}, "
                f"simulator={self.simulator is not None}, "
                f"sim_config={self.sim_config is not None}, "
                f"opt_config={self.opt_config is not None})")


class ConfigParser:
    """Parser for diff_optics JSON configuration files."""
    
    def __init__(self, scene_path: str, sim_path: str, opt_path: Optional[str] = None):
        """
        Initialize parser with separate config files.
        
        Args:
            scene_path: Path to scene configuration JSON
            sim_path: Path to simulation configuration JSON
            opt_path: Optional path to optimization configuration JSON
        """
        self.scene_path = Path(scene_path)
        self.sim_path = Path(sim_path)
        self.opt_path = Path(opt_path) if opt_path else None
        self.base_dir = self.scene_path.parent
        self._do = None
        
    def parse(self) -> ParseResult:
        """Parse all configuration files and return components."""
        self._do = _get_do()
        
        result = ParseResult()
        
        # 1. 加载场景配置
        with open(self.scene_path, 'r', encoding='utf-8') as f:
            result.raw_scene_config = json.load(f)
        
        # 2. 加载仿真配置
        with open(self.sim_path, 'r', encoding='utf-8') as f:
            result.raw_sim_config = json.load(f)
        
        # 3. 加载优化配置（可选）
        if self.opt_path and self.opt_path.exists():
            with open(self.opt_path, 'r', encoding='utf-8') as f:
                result.raw_opt_config = json.load(f)
                result.opt_config = result.raw_opt_config
        
        # 4. 构建组件
        result.scene, result.surface_params_range = self._build_scene(result.raw_scene_config)
        result.simulator = self._build_simulator()
        result.sim_config = self._build_sim_config(result.raw_sim_config)
        
        return result
    
    # ==================== Scene Building ====================
    
    def _build_scene(self, scene_cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, List[float]]]:
        """Build the complete scene from configuration."""
        do = self._do
        scene = do.Scene()
        
        # 1. 解析 Spectrums（先解析，因为其他组件可能引用）
        for spec_cfg in scene_cfg.get("spectrums", []):
            self._create_spectrum(scene, spec_cfg)
        
        # 2. 解析 Volume Materials
        for mat_cfg in scene_cfg.get("volume_materials", []):
            self._create_volume_material(scene, mat_cfg)
        
        # 3. 解析 BSDFs
        for bsdf_cfg in scene_cfg.get("bsdfs", []):
            self._create_bsdf(scene, bsdf_cfg)
        
        # 4. 解析 Emitters
        for emitter_cfg in scene_cfg.get("emitters", []):
            self._create_emitter(scene, emitter_cfg)
        
        # 5. 解析 Surfaces，同时收集 params_range
        params_range_map = {}
        for surface_cfg in scene_cfg.get("surfaces", []):
            self._create_surface(scene, surface_cfg)
            if "params_range" in surface_cfg:
                params_range_map[surface_cfg["name"]] = surface_cfg["params_range"]
        
        # 6. 解析 Lights
        for light_cfg in scene_cfg.get("lights", []):
            self._create_light(scene, light_cfg)
        
        # 7. 解析 Sensors
        for sensor_cfg in scene_cfg.get("sensors", []):
            self._create_sensor(scene, sensor_cfg)
        
        return scene, params_range_map
    
    def _create_spectrum(self, scene, cfg: Dict[str, Any]):
        """Create and add a spectrum to the scene."""
        do = self._do
        name = cfg["name"]
        spec_type = cfg.get("type", "constant")
        
        if spec_type == "discrete":
            wavelengths = cfg["wavelengths"]
            values = cfg["values"]
            scene.create_discrete_spectrum(name, wavelengths, values)
        
        elif spec_type == "blackbody":
            temperature = cfg.get("temperature", 6500.0)
            wl_min = cfg.get("wl_min", 380.0)
            wl_max = cfg.get("wl_max", 780.0)
            scene.create_blackbody_spectrum(name, temperature, wl_min, wl_max)
        
        elif spec_type == "gaussian":
            center = cfg.get("center", 550.0)
            sigma = cfg.get("sigma", 30.0)
            amplitude = cfg.get("amplitude", 1.0)
            scene.create_gaussian_spectrum(name, center, sigma, amplitude)
        
        elif spec_type == "constant":
            value = cfg.get("value", 1.0)
            wl_min = cfg.get("wl_min", 380.0)
            wl_max = cfg.get("wl_max", 780.0)
            scene.create_constant_spectrum(name, value, wl_min, wl_max)
        
        else:
            raise ValueError(f"Unknown spectrum type: {spec_type}")
    
    def _create_volume_material(self, scene, cfg: Dict[str, Any]):
        """Create and add a volume material to the scene."""
        name = cfg["name"]
        mat_type = cfg.get("type", "constant_ior")
        
        if mat_type == "air":
            scene.create_air(name)
        
        elif mat_type == "nbk7":
            scene.create_nbk7(name)
        
        elif mat_type == "pmma":
            scene.create_pmma(name)
        
        elif mat_type == "vacuum":
            scene.create_vacuum(name)
        
        elif mat_type == "constant_ior":
            ior = cfg.get("ior", 1.5)
            transmittance = cfg.get("transmittance", 1.0)
            measurement_depth = cfg.get("measurement_depth", 10.0)
            scene.create_constant_ior_material(name, ior, transmittance, measurement_depth)
        
        else:
            raise ValueError(f"Unknown volume material type: {mat_type}")
    
    def _create_bsdf(self, scene, cfg: Dict[str, Any]):
        """Create and add a BSDF to the scene."""
        name = cfg["name"]
        bsdf_type = cfg.get("type", "specular_refractor")
        
        if bsdf_type == "specular_reflector":
            reflectance = cfg.get("reflectance", 1.0)
            scene.create_specular_reflector(name, reflectance)
        
        elif bsdf_type == "specular_refractor":
            transmittance = cfg.get("transmittance", 1.0)
            scene.create_specular_refractor(name, transmittance)
        
        elif bsdf_type == "absorber":
            scene.create_absorber(name)
        
        else:
            raise ValueError(f"Unknown BSDF type: {bsdf_type}")
    
    def _create_emitter(self, scene, cfg: Dict[str, Any]):
        """Create and add an emitter to the scene."""
        name = cfg["name"]
        emitter_type = cfg.get("type", "uniform")
        
        if emitter_type == "uniform":
            upper_angle = cfg.get("upper_angle", 30)
            scene.create_uniform_emitter(name, upper_angle)
        
        elif emitter_type == "lambert":
            scene.create_lambert_emitter(name)
        
        else:
            raise ValueError(f"Unknown emitter type: {emitter_type}")
    
    def _find_file(self, filename: str) -> Optional[Path]:
        """查找文件，支持多个搜索路径"""
        search_paths = [
            Path(filename),
            self.base_dir / filename,
            Path("experiments") / filename,
        ]
        
        for p in search_paths:
            if p.exists():
                return p
        return None

    def _load_params(self, filename: str, u_num_cp: int, v_num_cp: int):
        """
        从文件加载控制点 z 值
        
        支持格式: .npy, .txt, .csv, .bin
        """
        file_path = self._find_file(filename)
        if file_path is None:
            print(f"警告: 找不到控制点文件 '{filename}'，使用默认值 0")
            return None
        
        expected_count = u_num_cp * v_num_cp
        
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == ".npy":
                data = np.load(file_path).astype(np.float32).flatten()
            elif suffix in [".txt", ".csv"]:
                data = np.loadtxt(file_path, dtype=np.float32).flatten()
            elif suffix == ".bin":
                data = np.fromfile(file_path, dtype=np.float32)
            else:
                print(f"警告: 不支持的文件格式 '{suffix}'，使用默认值 0")
                return None
            
            if len(data) != expected_count:
                print(f"警告: 控制点数量不匹配，期望 {expected_count}，实际 {len(data)}")
                if len(data) > expected_count:
                    data = data[:expected_count]
                else:
                    data = np.pad(data, (0, expected_count - len(data)))
            
            print(f"已加载控制点: {file_path} ({len(data)} 个)")
            print(f"  z 范围: [{data.min():.4f}, {data.max():.4f}]")
            return data.tolist()
            
        except Exception as e:
            print(f"警告: 加载控制点文件失败 ({e})，使用默认值 0")
            return None

    def _create_surface(self, scene, cfg: Dict[str, Any]):
        """Create and add a surface to the scene."""
        do = self._do
        name = cfg["name"]
        surface_type = cfg.get("type", "rectangle_plane")
        
        # 获取材质引用
        inner_material = None
        outer_material = None
        bsdf = None
        
        if "inner_material" in cfg and cfg["inner_material"]:
            inner_material = scene.get_volume_material(cfg["inner_material"])
        if "outer_material" in cfg and cfg["outer_material"]:
            outer_material = scene.get_volume_material(cfg["outer_material"])
        if "bsdf" in cfg and cfg["bsdf"]:
            bsdf = scene.get_bsdf(cfg["bsdf"])
        
        # 创建表面
        if surface_type == "rectangle_plane":
            size = cfg.get("size", [1.0, 1.0])
            scene.create_rectangle_plane(
                name=name,
                size=size,
                inner_material=inner_material,
                outer_material=outer_material,
                bsdf=bsdf
            )
        
        elif surface_type == "rectangle_bspline":
            size = cfg.get("size", [1.0, 1.0])
            u_degree = cfg.get("u_degree", 3)
            v_degree = cfg.get("v_degree", 3)

            resolution = cfg.get("resolution", [8, 8])
            u_num_cp = resolution[0]
            v_num_cp = resolution[1]

            # 加载控制点 z 值
            control_points_z = None
            cpz_file = cfg.get("params")
            if cpz_file:
                control_points_z = self._load_params(cpz_file, u_num_cp, v_num_cp)

            scene.create_rectangle_bspline(
                name=name,
                size=size,
                u_degree=u_degree,
                v_degree=v_degree,
                u_num_cp=u_num_cp,
                v_num_cp=v_num_cp,
                control_points_z=control_points_z,
                inner_material=inner_material,
                outer_material=outer_material,
                bsdf=bsdf
            )
        
        elif surface_type == "rectangle_xy":
            size = cfg.get("size", [1.0, 1.0])
            order = cfg.get("order", 4)
            b = cfg.get("b", 0.0)
            coefficients = cfg.get("coefficients", None)
            
            scene.create_rectangle_xy(
                name=name,
                size=size,
                order=order,
                b=b,
                coefficients=coefficients,
                inner_material=inner_material,
                outer_material=outer_material,
                bsdf=bsdf
            )
        
        elif surface_type == "rectangle_heightmap":
            size = cfg.get("size", [1.0, 1.0])
            resolution = cfg.get("resolution", [64, 64])
            grid_width = resolution[0]
            grid_height = resolution[1]
            
            # 加载 heightmap
            heightmap = None
            heightmap_file = cfg.get("params")
            if heightmap_file:
                heightmap = self._load_params(heightmap_file, grid_width, grid_height)
            
            scene.create_rectangle_heightmap(
                name=name,
                size=size,
                grid_width=grid_width,
                grid_height=grid_height,
                heightmap=heightmap,
                inner_material=inner_material,
                outer_material=outer_material,
                bsdf=bsdf
            )
        
        elif surface_type == "circle_plane":
            radius = cfg.get("radius", 1.0)
            scene.create_circle_plane(name=name, radius=radius)
        
        else:
            raise ValueError(f"Unknown surface type: {surface_type}")
        
        # 设置变换
        surface = scene.get_surface(name)
        if "transform" in cfg:
            transform_cfg = cfg["transform"]
            pos = transform_cfg.get("position", [0, 0, 0])
            rot = transform_cfg.get("rotation", [0, 0, 0])
            transform = do.Transform(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])
            surface.transform = transform
    
    def _create_light(self, scene, cfg: Dict[str, Any]):
        """Create and add a light to the scene."""
        do = self._do
        name = cfg["name"]
        light_type = cfg.get("type", "point")
        
        # 获取 emitter 和 spectrum 引用
        emitter = None
        spectrum = None
        
        if "emitter" in cfg:
            emitter = scene.get_emitter(cfg["emitter"])
        if "spectrum" in cfg:
            spectrum = scene.get_spectrum(cfg["spectrum"])
        
        if light_type == "point":
            scene.create_point_light(name, emitter, spectrum)
            
            # 设置变换（如果有）
            if "transform" in cfg:
                light = scene.get_light(name)
                transform_cfg = cfg["transform"]
                pos = transform_cfg.get("position", [0, 0, 0])
                rot = transform_cfg.get("rotation", [0, 0, 0])
                transform = do.Transform(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])
                light.transform = transform
        
        elif light_type == "surface":
            surface_name = cfg["surface"]
            scene.create_surface_light(name, surface_name, emitter, spectrum)
        
        else:
            raise ValueError(f"Unknown light type: {light_type}")
    
    def _create_sensor(self, scene, cfg: Dict[str, Any]):
        """Create and add a sensor to the scene."""
        do = self._do
        name = cfg["name"]
        sensor_type = cfg.get("type", "irradiance")
        
        resolution = cfg.get("resolution", [256, 256])
        
        filter_type = cfg.get("filter", "bilinear")
        
        if sensor_type == "irradiance":
            surface_name = cfg["surface"]
            scene.create_irradiance_sensor(
                name, surface_name, resolution, filter_type
            )
        
        elif sensor_type == "intensity":
            ies_type = cfg.get("ies_type", "TypeA")
            surface_name = cfg.get("surface")  # 可能不存在
            u_range = cfg.get("u_range", [-5.0, 5.0])
            v_range = cfg.get("v_range", [-5.0, 5.0])
            if surface_name:
                # 近场：绑定 Surface
                scene.create_intensity_sensor(
                    name, surface_name, ies_type, resolution, u_range, v_range, filter_type
                )
            else:
                # 远场：无 Surface
                scene.create_far_field_sensor(
                    name, ies_type, resolution, u_range, v_range, filter_type
                )
        
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
    
    # ==================== Simulator Building ====================
    
    def _build_simulator(self):
        """Create a ForwardSimulator instance."""
        do = self._do
        return do.ForwardSimulator()
    
    def _build_sim_config(self, sim_cfg: Dict[str, Any]):
        """Build SimulatorConfig from configuration."""
        do = self._do
        
        config = do.SimulatorConfig()
        config.num_rays = sim_cfg.get("num_rays", 1000000)
        config.max_depth = sim_cfg.get("max_depth", 10)
        config.min_radiance = sim_cfg.get("min_radiance", 1e-6)
        config.seed = sim_cfg.get("seed", 0)
        config.use_optix = sim_cfg.get("use_optix", True)
        
        if "ptx_dir" in sim_cfg:
            config.ptx_dir = sim_cfg["ptx_dir"]
        
        # 解析序列配置
        if "sequence" in sim_cfg:
            sequence = sim_cfg["sequence"]
            if len(sequence) >= 2:
                light_name = sequence[0]
                surfaces = sequence[1:]
                config.set_sequence(light_name, surfaces)
        
        return config


def parse_config(scene_path: str, sim_path: str, opt_path: Optional[str] = None) -> ParseResult:
    """
    Convenience function to parse configuration files.
    
    Args:
        scene_path: Path to scene configuration JSON
        sim_path: Path to simulation configuration JSON
        opt_path: Optional path to optimization configuration JSON
    
    Returns:
        ParseResult containing scene, simulator, and configs
    """
    parser = ConfigParser(scene_path, sim_path, opt_path)
    return parser.parse()


def load_json(path: str) -> Dict[str, Any]:
    """Load raw JSON configuration without parsing."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)