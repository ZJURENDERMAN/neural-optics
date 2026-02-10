# scene_data_generator.py

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

# Import visualizer
from scene_visualizer import SceneVisualizer


@dataclass
class SceneGeometry:
    """Scene geometry parameters"""
    light_size: float
    light_z: float
    upper_angle: float
    element_size: float
    element_z: float
    sensor_size: float
    sensor_z: float
    lens_thickness: float = 0.0
    resolution: tuple = (64, 64)


def calculate_min_reflector_distance(element_size: float, max_angle: float = 60.0) -> float:
    """
    Calculate minimum distance from light to reflector.
    Based on point light (light_size=0) at max_angle constraint.
    """
    angle_rad = np.radians(max_angle)
    half_size = element_size / 2
    min_distance = half_size / np.tan(angle_rad)
    return min_distance


def calculate_upper_angle_reflection(light_size: float, element_size: float,
                                      reflector_z: float) -> float:
    """
    Calculate emission angle for reflection scene.
    """
    delta_x = (element_size - light_size) / 2
    distance = abs(reflector_z)
    
    if distance <= 0 or delta_x <= 0:
        return 0.0
    
    return np.degrees(np.arctan(delta_x / distance))


def calculate_min_light_distance_refraction(element_size: float, lens_thickness: float,
                                             max_angle: float = 60.0) -> float:
    """
    Calculate minimum distance from light to front surface (z=0).
    """
    angle_rad = np.radians(max_angle)
    half_size = element_size / 2
    total_distance = half_size / np.tan(angle_rad)
    min_light_distance = total_distance - lens_thickness
    return max(min_light_distance, 1.0)


def calculate_upper_angle_refraction(light_size: float, element_size: float,
                                      light_z: float, lens_thickness: float) -> float:
    """
    Calculate emission angle for refraction scene.
    """
    delta_x = (element_size - light_size) / 2
    distance = lens_thickness - light_z
    
    if distance <= 0 or delta_x <= 0:
        return 0.0
    
    return np.degrees(np.arctan(delta_x / distance))


def calculate_sensor_size_reflection(light_size: float, element_size: float,
                                      reflector_z: float, sensor_z: float) -> float:
    """
    Calculate sensor size for reflection scene.
    """
    x_light = light_size / 2
    x_reflector = element_size / 2
    z_reflector = reflector_z
    
    dx_in = x_reflector - x_light
    dz_in = z_reflector - 0
    length_in = np.sqrt(dx_in**2 + dz_in**2)
    dx_in /= length_in
    dz_in /= length_in
    
    dx_out = dx_in
    dz_out = -dz_in
    
    t = (sensor_z - z_reflector) / dz_out
    x_sensor = x_reflector + dx_out * t
    
    return 2 * abs(x_sensor)


def calculate_sensor_size_refraction(light_size: float, element_size: float,
                                      light_z: float, lens_thickness: float,
                                      sensor_z: float) -> float:
    """
    Calculate sensor size for refraction scene.
    """
    x_light = light_size / 2
    x_back = element_size / 2
    
    dx = x_back - x_light
    dz = lens_thickness - light_z
    length = np.sqrt(dx**2 + dz**2)
    dx /= length
    dz /= length
    
    t = (sensor_z - lens_thickness) / dz
    x_sensor = x_back + dx * t
    
    return 2 * abs(x_sensor)


class SceneDataGenerator:
    def __init__(self, output_dir: str = "generated_configs", generate_thumbnails: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ior = 1.5
        self.num_rays = 2000000
        self.target_file = "targets/Lenna.png"
        self.resolution_options = [(256, 256)]
        self.element_size_range = (10.0, 20.0)
        self.max_emission_angle = 60.0
        
        self.generate_thumbnails = generate_thumbnails
        self.thumbnail_figsize = (8, 6)
        
        self._visualizer = None
        self._rng = None  # 使用独立的随机数生成器
    
    def _get_rng(self, seed: int) -> np.random.Generator:
        """获取指定种子的随机数生成器"""
        return np.random.default_rng(seed)
    
    @property
    def visualizer(self) -> SceneVisualizer:
        """Lazy load visualizer"""
        if self._visualizer is None:
            self._visualizer = SceneVisualizer()
        return self._visualizer
    
    def _save_config_with_thumbnail(self, config: Dict[str, Any], filepath: Path) -> Path:
        """Save config to JSON file and generate thumbnail."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        if self.generate_thumbnails:
            thumbnail_path = filepath.with_suffix('.png')
            try:
                import matplotlib
                original_backend = matplotlib.get_backend()
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                self.visualizer.visualize(
                    str(filepath),
                    save_path=str(thumbnail_path),
                    figsize=self.thumbnail_figsize
                )
                plt.close('all')
                matplotlib.use(original_backend)
            except Exception as e:
                print(f"Warning: Failed to generate thumbnail for {filepath.name}: {e}")
        
        return filepath
    
    def _generate_reflection_geometry(self, light_type: str, 
                                    resolution: tuple,
                                    rng: np.random.Generator) -> SceneGeometry:
        """Generate reflection scene geometry using provided RNG."""
        element_size = rng.uniform(*self.element_size_range)
        
        min_distance = calculate_min_reflector_distance(element_size, self.max_emission_angle)
        max_distance = min_distance * 3
        distance = rng.uniform(min_distance, max_distance)
        reflector_z = -distance
        
        if light_type == 'point':
            light_size = 0.1
        elif light_type == 'parallel':
            light_size = element_size
        else:
            light_size = rng.uniform(0.1, element_size * 0.9)
        
        if light_type == 'parallel':
            upper_angle = 0.0
        else:
            upper_angle = calculate_upper_angle_reflection(light_size, element_size, reflector_z)
        
        min_sensor_z = distance * 0.3
        max_sensor_z = distance * 1.0
        sensor_z = rng.uniform(min_sensor_z, max_sensor_z)
        
        sensor_size = calculate_sensor_size_reflection(
            light_size, element_size, reflector_z, sensor_z
        )
        
        return SceneGeometry(
            light_size=round(light_size, 4),
            light_z=0.0,
            upper_angle=round(upper_angle, 4),
            element_size=round(element_size, 4),
            element_z=round(reflector_z, 4),
            sensor_size=round(sensor_size, 4),
            sensor_z=round(sensor_z, 4),
            resolution=resolution
        )

    def _generate_refraction_geometry(self, light_type: str,
                                    resolution: tuple,
                                    rng: np.random.Generator) -> SceneGeometry:
        """Generate refraction scene geometry using provided RNG."""
        element_size = rng.uniform(*self.element_size_range)
        
        min_thickness = element_size / 10
        max_thickness = element_size / 2
        lens_thickness = rng.uniform(min_thickness, max_thickness)
        
        min_light_distance = calculate_min_light_distance_refraction(
            element_size, lens_thickness, self.max_emission_angle
        )
        max_light_distance = min_light_distance * 3
        light_distance = rng.uniform(min_light_distance, max_light_distance)
        light_z = -light_distance
        
        if light_type == 'point':
            light_size = 0.1
        elif light_type == 'parallel':
            light_size = element_size
        else:
            light_size = rng.uniform(0.1, element_size * 0.9)
        
        if light_type == 'parallel':
            upper_angle = 0.0
        else:
            upper_angle = calculate_upper_angle_refraction(
                light_size, element_size, light_z, lens_thickness
            )
        
        min_sensor_z = lens_thickness + element_size
        max_sensor_z = lens_thickness + element_size * 2
        sensor_z = rng.uniform(min_sensor_z, max_sensor_z)
        
        sensor_size = calculate_sensor_size_refraction(
            light_size, element_size, light_z, lens_thickness, sensor_z
        )
        
        return SceneGeometry(
            light_size=round(light_size, 4),
            light_z=round(light_z, 4),
            upper_angle=round(upper_angle, 4),
            element_size=round(element_size, 4),
            element_z=round(lens_thickness, 4),
            sensor_size=round(sensor_size, 4),
            sensor_z=round(sensor_z, 4),
            lens_thickness=round(lens_thickness, 4),
            resolution=resolution
        )
    def _build_surface_config(self, surface_type: str, geom: SceneGeometry,
                    surface_name: str, bsdf: str, z_position: float,
                    rotation: List[float], scene_type: str = 'refraction') -> Dict[str, Any]:
        """Build surface configuration for either bspline or heightmap."""
        config = {
            "name": surface_name,
            "size": [geom.element_size, geom.element_size],
            "resolution": list(geom.resolution),
            "inner_material": "glass_1.5",
            "outer_material": "air",
            "bsdf": bsdf,
            "transform": {"position": [0, 0, z_position], "rotation": rotation}
        }
        
        # 参数范围：正负透镜厚度
        if scene_type == 'refraction':
            max_height = geom.lens_thickness
        else:
            max_height = abs(geom.element_z) * 0.3
        
        max_height = round(max_height, 4)
        
        if surface_type == 'bspline':
            config["type"] = "rectangle_bspline"
        else:
            config["type"] = "rectangle_heightmap"
        
        config["params_range"] = [-max_height, max_height]
        
        return config

    def _build_reflection_config(self, geom: SceneGeometry, surface_type: str = 'bspline') -> Dict[str, Any]:
        """Build reflection scene config"""
        type_prefix = "bspline" if surface_type == 'bspline' else "heightmap"
        
        optimizable_surface = self._build_surface_config(
            surface_type, geom, "S1", "reflector", geom.element_z, [0, 0, 0], 'reflection'
        )
        
        return {
            "name": f"reflection_scene_{type_prefix}",
            "description": f"Reflection ({type_prefix}) - light {geom.light_size:.1f}, reflector {geom.element_size:.1f}, res {geom.resolution[0]}",
            "spectrums": [
                {"name": "mono_550nm", "type": "discrete", "wavelengths": [550.0], "values": [1.0]}
            ],
            "volume_materials": [
                {"name": "air", "type": "air"},
                {"name": "glass_1.5", "type": "constant_ior", "ior": self.ior}
            ],
            "bsdfs": [
                {"name": "reflector", "type": "specular_reflector"}
            ],
            "emitters": [
                {"name": "uniform_emitter", "type": "uniform", "upper_angle": geom.upper_angle}
            ],
            "surfaces": [
                {
                    "name": "S0", "type": "rectangle_plane",
                    "size": [geom.light_size, geom.light_size],
                    "transform": {"position": [0, 0, geom.light_z], "rotation": [180, 0, 0]}
                },
                optimizable_surface,
                {
                    "name": "S2", "type": "rectangle_plane",
                    "size": [geom.sensor_size, geom.sensor_size],
                    "transform": {"position": [0, 0, geom.sensor_z], "rotation": [0, 0, 0]}
                }
            ],
            "lights": [
                {"name": "Light", "type": "surface", "surface": "S0", 
                "emitter": "uniform_emitter", "spectrum": "mono_550nm"}
            ],
            "sensors": [
                {"name": "IrradianceSensor", "type": "irradiance", "surface": "S2",
                "resolution": [256, 256], "filter": "bilinear"}
            ]
        }

    def _build_refraction_config(self, geom: SceneGeometry, surface_type: str = 'bspline') -> Dict[str, Any]:
        """Build refraction scene config"""
        type_prefix = "bspline" if surface_type == 'bspline' else "heightmap"
        
        optimizable_surface = self._build_surface_config(
            surface_type, geom, "S2", "refractor", geom.lens_thickness, [0, 0, 0], 'refraction'
        )
        
        return {
            "name": f"refraction_scene_{type_prefix}",
            "description": f"Refraction ({type_prefix}) - light {geom.light_size:.1f}, lens {geom.element_size:.1f}x{geom.lens_thickness:.1f}, res {geom.resolution[0]}",
            "spectrums": [
                {"name": "mono_550nm", "type": "discrete", "wavelengths": [550.0], "values": [1.0]}
            ],
            "volume_materials": [
                {"name": "air", "type": "air"},
                {"name": "glass_1.5", "type": "constant_ior", "ior": self.ior}
            ],
            "bsdfs": [
                {"name": "refractor", "type": "specular_refractor"}
            ],
            "emitters": [
                {"name": "uniform_emitter", "type": "uniform", "upper_angle": geom.upper_angle}
            ],
            "surfaces": [
                {
                    "name": "S0", "type": "rectangle_plane",
                    "size": [geom.light_size, geom.light_size],
                    "transform": {"position": [0, 0, geom.light_z], "rotation": [0, 0, 0]}
                },
                {
                    "name": "S1", "type": "rectangle_plane",
                    "size": [geom.element_size, geom.element_size],
                    "inner_material": "glass_1.5", "outer_material": "air",
                    "bsdf": "refractor",
                    "transform": {"position": [0, 0, 0], "rotation": [180, 0, 0]}
                },
                optimizable_surface,
                {
                    "name": "S3", "type": "rectangle_plane",
                    "size": [geom.sensor_size, geom.sensor_size],
                    "transform": {"position": [0, 0, geom.sensor_z], "rotation": [0, 0, 0]}
                }
            ],
            "lights": [
                {"name": "Light", "type": "surface", "surface": "S0",
                "emitter": "uniform_emitter", "spectrum": "mono_550nm"}
            ],
            "sensors": [
                {"name": "IrradianceSensor", "type": "irradiance", "surface": "S3",
                "resolution": [256, 256], "filter": "bilinear"}
            ]
        }

    def _generate_thumbnail(self, config_filepath: Path, thumbnail_path: Path, index: int) -> bool:
        """Generate a single thumbnail image."""
        if not self.generate_thumbnails:
            return False
        
        try:
            import matplotlib
            original_backend = matplotlib.get_backend()
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            self.visualizer.visualize(
                str(config_filepath),
                save_path=str(thumbnail_path),
                figsize=self.thumbnail_figsize
            )
            plt.close('all')
            matplotlib.use(original_backend)
            return True
        except Exception as e:
            print(f"Warning: Failed to generate thumbnail for index {index}: {e}")
            return False

    def generate_reflection_dataset(self, num_configs: int = 10, 
                                    base_seed: int = 42) -> List[Path]:
        """Generate reflection scene dataset with deterministic randomness."""
        # 创建主RNG，确保可重复性
        master_rng = np.random.default_rng(base_seed)
        
        # 预先生成所有子种子，确保顺序一致
        sub_seeds = master_rng.integers(0, 2**31, size=num_configs)
        
        configs = []
        
        # 确定每种光源类型的数量
        n_point = min(1, num_configs)
        n_parallel = min(1, max(0, num_configs - n_point))
        n_area = max(0, num_configs - n_point - n_parallel)
        
        idx = 0
        
        # Point light configurations
        for _ in range(n_point):
            rng = np.random.default_rng(sub_seeds[idx])
            res = self.resolution_options[idx % len(self.resolution_options)]
            geom = self._generate_reflection_geometry('point', res, rng)
            meta = {'type': 'reflection', 'light_type': 'point', 'seed': int(sub_seeds[idx]), 'geometry': geom.__dict__}
            configs.append((geom, meta))
            idx += 1
        
        # Parallel light configurations
        for _ in range(n_parallel):
            rng = np.random.default_rng(sub_seeds[idx])
            res = self.resolution_options[idx % len(self.resolution_options)]
            geom = self._generate_reflection_geometry('parallel', res, rng)
            meta = {'type': 'reflection', 'light_type': 'parallel', 'seed': int(sub_seeds[idx]), 'geometry': geom.__dict__}
            configs.append((geom, meta))
            idx += 1
        
        # Area light configurations
        for _ in range(n_area):
            rng = np.random.default_rng(sub_seeds[idx])
            res = self.resolution_options[idx % len(self.resolution_options)]
            geom = self._generate_reflection_geometry('area', res, rng)
            meta = {'type': 'reflection', 'light_type': 'area', 'seed': int(sub_seeds[idx]), 'geometry': geom.__dict__}
            configs.append((geom, meta))
            idx += 1
        
        # Create output directories
        base_dir = self.output_dir / "reflection"
        bspline_dir = base_dir / "bspline"
        heightmap_dir = base_dir / "heightmap"
        thumbnails_dir = base_dir / "thumbnails"
        bspline_dir.mkdir(parents=True, exist_ok=True)
        heightmap_dir.mkdir(parents=True, exist_ok=True)
        if self.generate_thumbnails:
            thumbnails_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for i, (geom, meta) in enumerate(configs):
            # Generate bspline config
            bspline_config = self._build_reflection_config(geom, 'bspline')
            bspline_filepath = bspline_dir / f"bspline_reflection_{i:03d}.json"
            with open(bspline_filepath, 'w', encoding='utf-8') as f:
                json.dump(bspline_config, f, indent=4, ensure_ascii=False)
            saved_files.append(bspline_filepath)
            
            # Generate heightmap config
            heightmap_config = self._build_reflection_config(geom, 'heightmap')
            heightmap_filepath = heightmap_dir / f"heightmap_reflection_{i:03d}.json"
            with open(heightmap_filepath, 'w', encoding='utf-8') as f:
                json.dump(heightmap_config, f, indent=4, ensure_ascii=False)
            saved_files.append(heightmap_filepath)
            
            # Generate thumbnail (saved to thumbnails folder)
            thumbnail_generated = False
            if self.generate_thumbnails:
                thumbnail_path = thumbnails_dir / f"reflection_{i:03d}.png"
                thumbnail_generated = self._generate_thumbnail(bspline_filepath, thumbnail_path, i)
            
            print(f"  Generated: reflection_{i:03d} (bspline + heightmap)" + 
                (" + thumbnail" if thumbnail_generated else ""))
        
        # Save summary with seed information
        summary = [{'index': i, 
                    'bspline_file': f"bspline/bspline_reflection_{i:03d}.json",
                    'heightmap_file': f"heightmap/heightmap_reflection_{i:03d}.json",
                    'thumbnail_file': f"thumbnails/reflection_{i:03d}.png" if self.generate_thumbnails else None,
                    **meta} 
                for i, (geom, meta) in enumerate(configs)]
        
        # 添加数据集元信息
        dataset_meta = {
            'base_seed': base_seed,
            'num_configs': num_configs,
            'generator_version': '1.0',
            'configs': summary
        }
        
        with open(base_dir / "_dataset_summary.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_meta, f, indent=2, ensure_ascii=False)
        
        print(f"Reflection dataset: saved {num_configs} configs (x2 types) to {base_dir}")
        return saved_files

    def generate_refraction_dataset(self, num_configs: int = 10,
                                    base_seed: int = 100) -> List[Path]:
        """Generate refraction scene dataset with deterministic randomness."""
        # 创建主RNG，确保可重复性
        master_rng = np.random.default_rng(base_seed)
        
        # 预先生成所有子种子
        sub_seeds = master_rng.integers(0, 2**31, size=num_configs)
        
        configs = []
        
        # 确定每种光源类型的数量
        n_point = min(1, num_configs)
        n_parallel = min(1, max(0, num_configs - n_point))
        n_area = max(0, num_configs - n_point - n_parallel)
        
        idx = 0
        
        # Point light configurations
        for _ in range(n_point):
            rng = np.random.default_rng(sub_seeds[idx])
            res = self.resolution_options[idx % len(self.resolution_options)]
            geom = self._generate_refraction_geometry('point', res, rng)
            meta = {'type': 'refraction', 'light_type': 'point', 'seed': int(sub_seeds[idx]), 'geometry': geom.__dict__}
            configs.append((geom, meta))
            idx += 1
        
        # Parallel light configurations
        for _ in range(n_parallel):
            rng = np.random.default_rng(sub_seeds[idx])
            res = self.resolution_options[idx % len(self.resolution_options)]
            geom = self._generate_refraction_geometry('parallel', res, rng)
            meta = {'type': 'refraction', 'light_type': 'parallel', 'seed': int(sub_seeds[idx]), 'geometry': geom.__dict__}
            configs.append((geom, meta))
            idx += 1
        
        # Area light configurations
        for _ in range(n_area):
            rng = np.random.default_rng(sub_seeds[idx])
            res = self.resolution_options[idx % len(self.resolution_options)]
            geom = self._generate_refraction_geometry('area', res, rng)
            meta = {'type': 'refraction', 'light_type': 'area', 'seed': int(sub_seeds[idx]), 'geometry': geom.__dict__}
            configs.append((geom, meta))
            idx += 1
        
        # Create output directories
        base_dir = self.output_dir / "refraction"
        bspline_dir = base_dir / "bspline"
        heightmap_dir = base_dir / "heightmap"
        thumbnails_dir = base_dir / "thumbnails"
        bspline_dir.mkdir(parents=True, exist_ok=True)
        heightmap_dir.mkdir(parents=True, exist_ok=True)
        if self.generate_thumbnails:
            thumbnails_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for i, (geom, meta) in enumerate(configs):
            # Generate bspline config
            bspline_config = self._build_refraction_config(geom, 'bspline')
            bspline_filepath = bspline_dir / f"bspline_refraction_{i:03d}.json"
            with open(bspline_filepath, 'w', encoding='utf-8') as f:
                json.dump(bspline_config, f, indent=4, ensure_ascii=False)
            saved_files.append(bspline_filepath)
            
            # Generate heightmap config
            heightmap_config = self._build_refraction_config(geom, 'heightmap')
            heightmap_filepath = heightmap_dir / f"heightmap_refraction_{i:03d}.json"
            with open(heightmap_filepath, 'w', encoding='utf-8') as f:
                json.dump(heightmap_config, f, indent=4, ensure_ascii=False)
            saved_files.append(heightmap_filepath)
            
            # Generate thumbnail (saved to thumbnails folder)
            thumbnail_generated = False
            if self.generate_thumbnails:
                thumbnail_path = thumbnails_dir / f"refraction_{i:03d}.png"
                thumbnail_generated = self._generate_thumbnail(bspline_filepath, thumbnail_path, i)
            
            print(f"  Generated: refraction_{i:03d} (bspline + heightmap)" + 
                (" + thumbnail" if thumbnail_generated else ""))
        
        # Save summary with seed information
        summary = [{'index': i, 
                    'bspline_file': f"bspline/bspline_refraction_{i:03d}.json",
                    'heightmap_file': f"heightmap/heightmap_refraction_{i:03d}.json",
                    'thumbnail_file': f"thumbnails/refraction_{i:03d}.png" if self.generate_thumbnails else None,
                    **meta}
                for i, (geom, meta) in enumerate(configs)]
        
        # 添加数据集元信息
        dataset_meta = {
            'base_seed': base_seed,
            'num_configs': num_configs,
            'generator_version': '1.0',
            'configs': summary
        }
        
        with open(base_dir / "_dataset_summary.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_meta, f, indent=2, ensure_ascii=False)
        
        print(f"Refraction dataset: saved {num_configs} configs (x2 types) to {base_dir}")
        return saved_files

    def preview_geometry(self, scene_type: str, light_type: str, 
                     resolution: tuple = (64, 64), seed: int = 42):
        """Preview geometry parameters"""
        rng = np.random.default_rng(seed)
        
        if scene_type == 'reflection':
            geom = self._generate_reflection_geometry(light_type, resolution, rng)
        else:
            geom = self._generate_refraction_geometry(light_type, resolution, rng)
        
        print(f"\n=== {scene_type.upper()} - {light_type} (seed={seed}) ===")
        print(f"light_size:     {geom.light_size}")
        print(f"light_z:        {geom.light_z}")
        print(f"upper_angle:    {geom.upper_angle}°")
        print(f"element_size:   {geom.element_size}")
        print(f"element_z:      {geom.element_z}")
        if geom.lens_thickness > 0:
            print(f"lens_thickness: {geom.lens_thickness}")
        print(f"sensor_size:    {geom.sensor_size}")
        print(f"sensor_z:       {geom.sensor_z}")
        print(f"resolution:     {geom.resolution[0]}x{geom.resolution[1]}")
        
        return geom


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Generate scene configuration datasets for optical simulations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 refraction scenes with seed 42
  python scene_data_generator.py --num 100 --output ./my_scenes --seed 42
  
  # Generate 50 scenes without thumbnails
  python scene_data_generator.py -n 50 -o ./scenes -s 123 --no-thumbnails
  
  # Generate both reflection and refraction datasets
  python scene_data_generator.py -n 200 -o ./data -s 42 --type both
        """
    )
    
    parser.add_argument(
        '-n', '--num',
        type=int,
        required=True,
        help='Number of configurations to generate'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory path'
    )
    
    parser.add_argument(
        '-s', '--seed',
        type=int,
        required=True,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '-t', '--type',
        type=str,
        choices=['reflection', 'refraction', 'both'],
        default='refraction',
        help='Type of scenes to generate (default: refraction)'
    )
    
    parser.add_argument(
        '--no-thumbnails',
        action='store_true',
        help='Disable thumbnail generation'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate parameter distribution visualizations after dataset creation'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Scene Data Generator")
    print("=" * 60)
    print(f"  Number of configs: {args.num}")
    print(f"  Output directory:  {args.output}")
    print(f"  Random seed:       {args.seed}")
    print(f"  Scene type:        {args.type}")
    print(f"  Generate thumbnails: {not args.no_thumbnails}")
    print("=" * 60)
    
    generator = SceneDataGenerator(
        output_dir=args.output, 
        generate_thumbnails=not args.no_thumbnails
    )
    
    # Generate datasets based on type
    if args.type == 'reflection' or args.type == 'both':
        print("\n--- Generating Reflection Dataset ---")
        generator.generate_reflection_dataset(
            num_configs=args.num, 
            base_seed=args.seed
        )
    
    if args.type == 'refraction' or args.type == 'both':
        print("\n--- Generating Refraction Dataset ---")
        # Use different seed offset for refraction to avoid collision
        refraction_seed = args.seed + 10000 if args.type == 'both' else args.seed
        generator.generate_refraction_dataset(
            num_configs=args.num, 
            base_seed=refraction_seed
        )
    
    # Generate visualizations if requested
    if args.visualize:
        try:
            from visualize_scene_distribution import SceneDistributionVisualizer
            
            visualizer = SceneDistributionVisualizer()
            
            if args.type == 'reflection' or args.type == 'both':
                summary_path = Path(args.output) / "reflection" / "_dataset_summary.json"
                if summary_path.exists():
                    print("\n--- Generating Reflection Visualizations ---")
                    visualizer.generate_full_report(
                        str(summary_path),
                        scene_type="Reflection"
                    )
            
            if args.type == 'refraction' or args.type == 'both':
                summary_path = Path(args.output) / "refraction" / "_dataset_summary.json"
                if summary_path.exists():
                    print("\n--- Generating Refraction Visualizations ---")
                    visualizer.generate_full_report(
                        str(summary_path),
                        scene_type="Refraction"
                    )
        except ImportError:
            print("Warning: visualize_scene_distribution module not found, skipping visualizations")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()