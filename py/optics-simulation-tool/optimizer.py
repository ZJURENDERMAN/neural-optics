# optimizer.py
"""
Optimizer module for diff_optics.
Handles optimization loops using pre-built Scene and Simulator.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json

# 延迟导入
_do = None
_dr = None
_Float = None
_optimizer_classes = None


def _init_backend():
    global _do, _dr, _Float, _optimizer_classes
    if _dr is None:
        from setup import get_diff_optics, get_drjit, get_Float
        _do = get_diff_optics()
        _dr = get_drjit()
        _Float = get_Float()
    
    if _optimizer_classes is None:
        from drjit.opt import Adam, SGD, RMSProp
        _optimizer_classes = {
            "adam": Adam,
            "sgd": SGD,
            "rmsprop": RMSProp
        }


def _create_optimizer(config: Dict[str, Any]):
    """
    根据配置创建优化器实例。
    
    支持的优化器:
    - adam: lr, beta_1, beta_2, epsilon
    - sgd: lr, momentum, nesterov
    - rmsprop: lr, alpha, epsilon
    """
    _init_backend()
    
    opt_type = config.get("type", "adam").lower()
    lr = config.get("learning_rate", 0.01)
    
    if _optimizer_classes is None:
        raise RuntimeError("Optimizer classes not initialized")
    
    if opt_type not in _optimizer_classes:
        raise ValueError(f"Unknown optimizer type: '{opt_type}'. "
                        f"Supported: {list(_optimizer_classes.keys())}")
    
    OptimizerClass = _optimizer_classes[opt_type]
    
    if opt_type == "adam":
        return OptimizerClass(
            lr=lr,
            beta_1=config.get("beta_1", config.get("beta1", 0.9)),
            beta_2=config.get("beta_2", config.get("beta2", 0.999)),
            epsilon=config.get("epsilon", 1e-8)
        )
    
    elif opt_type == "sgd":
        return OptimizerClass(
            lr=lr,
            momentum=config.get("momentum", 0.0),
            nesterov=config.get("nesterov", False)
        )
    
    elif opt_type == "rmsprop":
        return OptimizerClass(
            lr=lr,
            alpha=config.get("alpha", 0.99),
            epsilon=config.get("epsilon", 1e-8)
        )
    
    return OptimizerClass(lr=lr)


class OptimizationStage:
    """Configuration for a single optimization stage."""
    
    def __init__(self, config: Dict[str, Any]):
        self.name = config.get("name", "stage")
        self.steps = config.get("steps", 100)
        
        # 优化器配置
        self.optimizer_config = config.get("optimizer", {
            "type": "adam",
            "learning_rate": 0.01
        })
        
        # 每个 stage 独立的配置
        self.surface_name = config.get("surface", "")
        
        # sensor 相关
        self.sensor_name = config.get("sensor", "")
        self.sensor_resolution = config.get("sensor_resolution", [256, 256])
        
        self.save_interval = config.get("save_interval", 10)
        self.print_interval = config.get("print_interval", 10)
        
        # 图像优化相关
        self.target_config = config.get("target", {})
        self.loss_config = config.get("loss", {"type": "l1", "normalize": True})
        
        self.num_rays = config.get("num_rays", None)

        # 准直优化
        self.collimation = config.get("collimation", None)
    
    @property
    def learning_rate(self) -> float:
        return self.optimizer_config.get("learning_rate", 0.01)
    
    @property
    def optimizer_type(self) -> str:
        return self.optimizer_config.get("type", "adam").lower()


class OptimizationResult:
    """Container for optimization results."""
    
    def __init__(self):
        self.success: bool = False
        self.total_iterations: int = 0
        self.final_loss: float = float('inf')
        self.loss_history: List[float] = []
        self.stage_results: List[Dict[str, Any]] = []
        self.best_loss: float = float('inf')
        self.best_params: Optional[np.ndarray] = None
        self.total_time_s: float = 0.0
        self.message: str = ""
    
    def __repr__(self):
        status = "success" if self.success else "failed"
        return f"OptimizationResult({status}, iter={self.total_iterations}, loss={self.final_loss:.6e})"


class Optimizer:
    """Optimizer for optical systems."""
    
    def __init__(
        self,
        scene,
        simulator,
        sim_config,
        opt_config: List[Dict[str, Any]],
        output_dir: str = "results",
        save_details: bool = False,
        original_scene_path: Optional[str] = None,
        surface_params_range: Optional[Dict[str, List[float]]] = None
    ):
        _init_backend()
        
        self.scene = scene
        self.simulator = simulator
        self.sim_config = sim_config
        self.save_details = save_details
        self.original_scene_path = original_scene_path
        self.surface_params_range = surface_params_range or {}
        
        self.output_dir = Path(output_dir)
        self.stages = [OptimizationStage(cfg) for cfg in opt_config]
        
        self._iteration = 0
        self._current_stage = 0
        
        # 记录每个 stage 优化的 surface 名称和对应的参数文件
        self._optimized_surfaces: Dict[str, str] = {}  # surface_name -> params_file
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_target(self, target_config: Dict[str, Any], resolution: List[int]):
        """Load and preprocess target image."""
        from exr_utils import load_image
        
        target_file = target_config.get("file", "")
        if not target_file:
            raise ValueError("Target file not specified")
        
        target_path = Path(target_file)
        if not target_path.is_absolute():
            config_dir = self.output_dir.parent
            if (config_dir / target_file).exists():
                target_path = config_dir / target_file
        
        result = load_image(
            str(target_path),
            desired_channels=1,
            regularize=False,
            desired_resolution=resolution
        )
        
        target_data = result[0]
        target_width = result[1]
        target_height = result[2]
        
        print(f"Loaded target image: {target_path}")
        print(f"  Resolution: {target_width} x {target_height}")
        
        return target_data, target_width, target_height
    
    def _get_optimizable_params(self, surface_name: str):
        surface = self.scene.get_surface(surface_name)
        shape = surface.get_shape()
        return shape.get_diff_params()
    
    def _set_optimizable_params(self, surface_name: str, params):
        surface = self.scene.get_surface(surface_name)
        shape = surface.get_shape()
        shape.set_diff_params(params)
    
    def _apply_constraints(self, params, z_min: float, z_max: float):
        if z_min == -float('inf') and z_max == float('inf'):
            return params
        
        if z_min != -float('inf') and z_max != float('inf'):
            range_size = z_max - z_min
            sigmoid = 1.0 / (1.0 + _dr.exp(-params))
            return _Float(z_min) + _Float(range_size) * sigmoid
        elif z_min != -float('inf'):
            return _Float(z_min) + _dr.abs(params)
        else:
            return _Float(z_max) - _dr.abs(params)
    
    def _inverse_constraints(self, constrained_params, z_min: float, z_max: float):
        if z_min == -float('inf') and z_max == float('inf'):
            return constrained_params
        
        if z_min != -float('inf') and z_max != float('inf'):
            range_size = z_max - z_min
            normalized = (constrained_params - _Float(z_min)) / _Float(range_size)
            normalized = _dr.clamp(normalized, _Float(1e-6), _Float(1.0 - 1e-6))
            return _dr.log(normalized / (1.0 - normalized))
        elif z_min != -float('inf'):
            return constrained_params - _Float(z_min)
        else:
            return _Float(z_max) - constrained_params
    
    def _update_optix_mesh(self, surface_name: str):
        optix_manager = _do.get_optix_manager()
        if optix_manager and optix_manager.is_initialized():
            optix_manager.update_surface_mesh(surface_name, self.scene)
    
    def _create_loss_function(self, loss_config: Dict[str, Any], width: int, height: int):
        from losses import LossFactory
        return LossFactory.from_json_config(loss_config, _dr, _Float, width, height)
    
    def _compute_collimation_loss(self, sim_result, collimation_config):
        if not sim_result.has_exit_rays:
            return None
        
        exit_rays = sim_result.exit_rays
        if exit_rays.size() == 0:
            return None
        
        target_dir = collimation_config.get("target_direction", [0, 0, 1])
        target_x = _Float(target_dir[0])
        target_y = _Float(target_dir[1])
        target_z = _Float(target_dir[2])
        
        target_len = _dr.sqrt(target_x * target_x + target_y * target_y + target_z * target_z)
        target_x = target_x / target_len
        target_y = target_y / target_len
        target_z = target_z / target_len
        
        dir_x = exit_rays.direction[0]
        dir_y = exit_rays.direction[1]
        dir_z = exit_rays.direction[2]
        
        cos_theta = dir_x * target_x + dir_y * target_y + dir_z * target_z
        cos_theta = _dr.clamp(cos_theta, _Float(-1.0), _Float(1.0))
        
        target_divergence_deg = collimation_config.get("target_divergence", 0.0)
        
        weights = exit_rays.radiance / _dr.maximum(exit_rays.pdf, _Float(1e-10))
        total_weight = _dr.sum(weights)
        
        if target_divergence_deg <= 0:
            deviation = 1.0 - cos_theta
            weighted_loss = _dr.sum(deviation * weights) / total_weight
        else:
            target_divergence_rad = _Float(target_divergence_deg * np.pi / 180.0)
            theta = _dr.acos(cos_theta)
            angle_deviation = _dr.abs(theta - target_divergence_rad)
            weighted_loss = _dr.sum(angle_deviation * weights) / total_weight
        
        return weighted_loss
    
    def _print_optimizer_info(self, stage: OptimizationStage):
        opt_cfg = stage.optimizer_config
        opt_type = stage.optimizer_type
        
        print(f"  Optimizer: {opt_type}")
        print(f"  Learning rate: {stage.learning_rate}")
        
        if opt_type == "adam":
            beta_1 = opt_cfg.get('beta_1', opt_cfg.get('beta1', 0.9))
            beta_2 = opt_cfg.get('beta_2', opt_cfg.get('beta2', 0.999))
            print(f"  Beta: ({beta_1}, {beta_2})")
        elif opt_type == "sgd":
            momentum = opt_cfg.get('momentum', 0.0)
            if momentum > 0:
                print(f"  Momentum: {momentum}")
                if opt_cfg.get('nesterov', False):
                    print(f"  Nesterov: enabled")
        elif opt_type == "rmsprop":
            print(f"  Alpha: {opt_cfg.get('alpha', 0.99)}")
    
    def _save_image(self, data, width: int, height: int, filepath_without_ext: str, normalize: bool = True):
        """保存图像为 EXR 和 PNG 格式"""
        from exr_utils import save_exr, save_png
        
        # 保存 EXR
        exr_path = f"{filepath_without_ext}.exr"
        save_exr(data, width, height, exr_path, normalize=normalize)
        
        # 保存 PNG
        png_path = f"{filepath_without_ext}.png"
        save_png(data, width, height, png_path, normalize=normalize)
    
    def _save_stage_results(self, stage: OptimizationStage, stage_idx: int):
        """保存阶段结果：参数、仿真图像、CAD 文件"""
        prefix = f"stage_{stage_idx}"
        
        # 保存参数
        params = self._get_optimizable_params(stage.surface_name)
        params_np = np.array(params)
        params_filename = f"{prefix}_params.npy"
        np.save(self.output_dir / params_filename, params_np)
        print(f"  Saved parameters: {params_filename}")
        
        # 记录 surface -> params 文件映射（用于生成 optimized_scene.json）
        self._optimized_surfaces[stage.surface_name] = params_filename
        
        # 保存 CAD 文件
        try:
            surface = self.scene.get_surface(stage.surface_name)
            shape = surface.get_shape()
            cad_file = str(self.output_dir / f"{prefix}_surface.step")
            if shape.save_cad(cad_file):
                print(f"  Saved CAD: {prefix}_surface.step")
        except Exception as e:
            print(f"  CAD export failed: {e}")
        
        # 保存仿真结果图像
        sim_result = self.simulator.simulate(self.scene, self.sim_config)
        if sim_result.success:
            for sensor_name in sim_result.get_sensor_names():
                sensor_data = sim_result.get_sensor_data(sensor_name)
                filepath = str(self.output_dir / f"{prefix}_result_{sensor_name}")
                self._save_image(
                    sensor_data.data,
                    sensor_data.width,
                    sensor_data.height,
                    filepath,
                    normalize=True
                )
                print(f"  Saved result: {prefix}_result_{sensor_name}.exr/.png")
        else:
            print(f"  Warning: Final simulation for stage failed: {sim_result.error_message}")
    
    def _run_stage(self, stage: OptimizationStage, stage_idx: int) -> Dict[str, Any]:
        # 从场景配置获取约束
        if stage.surface_name in self.surface_params_range:
            pr = self.surface_params_range[stage.surface_name]
            z_min, z_max = pr[0], pr[1]
        else:
            z_min, z_max = -float('inf'), float('inf')
        
        has_constraints = (z_min != -float('inf') or z_max != float('inf'))
        
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx}: {stage.name}")
        print(f"  Surface: {stage.surface_name}")
        print(f"  Steps: {stage.steps}")
        self._print_optimizer_info(stage)
        
        if has_constraints:
            print(f"  Constraints: z_min={z_min}, z_max={z_max}")
        print(f"{'='*60}")
        
        stage_result = {
            "name": stage.name,
            "steps": stage.steps,
            "optimizer": stage.optimizer_type,
            "learning_rate": stage.learning_rate,
            "loss_history": [],
            "start_iteration": self._iteration,
            "surface": stage.surface_name
        }
        
        if not stage.surface_name:
            raise ValueError(f"Stage '{stage.name}': surface not specified")
        
        is_collimation_mode = stage.collimation is not None
        
        if is_collimation_mode:
            target_data = None
            loss_fn = None
            print(f"  Mode: Collimation")
            target_dir = stage.collimation.get("target_direction", [0, 0, 1])
            target_div = stage.collimation.get("target_divergence", 0.0)
            print(f"  Target direction: {target_dir}")
            if target_div > 0:
                print(f"  Target divergence: {target_div}° (partial collimation)")
            else:
                print(f"  Target divergence: 0° (perfect collimation)")
        else:
            if not stage.sensor_name:
                raise ValueError(f"Stage '{stage.name}': sensor not specified for image optimization")
            
            print(f"  Sensor: {stage.sensor_name}")
            print(f"  Resolution: {stage.sensor_resolution}")
            
            sensor = self.scene.get_sensor(stage.sensor_name)
            sensor.set_resolution(stage.sensor_resolution[0], stage.sensor_resolution[1])
            
            target_data, _, _ = self._load_target(stage.target_config, stage.sensor_resolution)
            loss_fn = self._create_loss_function(
                stage.loss_config,
                stage.sensor_resolution[0],
                stage.sensor_resolution[1]
            )
            print(f"  Mode: Image matching")
        
        initial_params = self._get_optimizable_params(stage.surface_name)
        
        if has_constraints:
            unconstrained_params = self._inverse_constraints(initial_params, z_min, z_max)
        else:
            unconstrained_params = _Float(initial_params)
        
        optimizer = _create_optimizer(stage.optimizer_config)
        optimizer['param'] = unconstrained_params
        _dr.enable_grad(optimizer['param'])
        
        if stage.num_rays:
            self.sim_config.num_rays = stage.num_rays
        
        for step in range(stage.steps):
            if has_constraints:
                params = self._apply_constraints(optimizer['param'], z_min, z_max)
            else:
                params = optimizer['param']
            
            self._set_optimizable_params(stage.surface_name, params)
            self._update_optix_mesh(stage.surface_name)
            
            sim_result = self.simulator.simulate(self.scene, self.sim_config)
            
            if not sim_result.success:
                print(f"  [Step {step}] Simulation failed: {sim_result.error_message}")
                continue
            
            if is_collimation_mode:
                loss = self._compute_collimation_loss(sim_result, stage.collimation)
                if loss is None:
                    print(f"  [Step {step}] No exit rays for collimation")
                    continue
            else:
                sensor_data = sim_result.get_sensor_data(stage.sensor_name)
                predicted = sensor_data.data
                loss = loss_fn(predicted, target_data)

            _dr.backward(loss)
            
            _dr.eval(loss)
            _dr.sync_thread()
            loss_value = float(loss[0]) if hasattr(loss, '__getitem__') else float(loss)
            
            stage_result["loss_history"].append(loss_value)
            
            optimizer.step()
            
            if step % stage.print_interval == 0 or step == 0:
                print(f"  [Step {step}/{stage.steps}] Loss: {loss_value:.6e}")
            
            if self.save_details and step % stage.save_interval == 0:
                self._save_checkpoint(stage, stage_idx, step, loss_value, is_collimation_mode)
            
            self._iteration += 1
        
        stage_result["final_loss"] = stage_result["loss_history"][-1] if stage_result["loss_history"] else float('inf')
        stage_result["end_iteration"] = self._iteration
        
        # 每个阶段结束后保存结果
        print(f"\n  Saving stage {stage_idx} results...")
        self._save_stage_results(stage, stage_idx)
        
        return stage_result
    
    def _save_checkpoint(self, stage: OptimizationStage, stage_idx: int, step: int, loss: float, is_collimation_mode: bool = False):
        """保存中间 checkpoint（仅在 save_details=True 时调用）"""
        params = self._get_optimizable_params(stage.surface_name)
        params_np = np.array(params)
        
        checkpoint_dir = self.output_dir / f"checkpoints_stage_{stage_idx}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(checkpoint_dir / f"params_step_{step}.npy", params_np)
        
        if not is_collimation_mode and stage.sensor_name:
            sim_result = self.simulator.simulate(self.scene, self.sim_config)
            if sim_result.success:
                sensor_data = sim_result.get_sensor_data(stage.sensor_name)
                filepath = str(checkpoint_dir / f"result_step_{step}")
                self._save_image(
                    sensor_data.data,
                    sensor_data.width,
                    sensor_data.height,
                    filepath,
                    normalize=True
                )
        
        try:
            surface = self.scene.get_surface(stage.surface_name)
            shape = surface.get_shape()
            cad_file = str(checkpoint_dir / f"surface_step_{step}.step")
            if shape.save_cad(cad_file):
                print(f"    Exported CAD: {cad_file}")
        except Exception:
            pass
    
    def _save_optimized_scene(self) -> Optional[str]:
        """
        保存优化后的场景 JSON 文件。
        
        在对应表面配置中添加 params 字段，指向 .npy 文件。
        """
        if not self.original_scene_path:
            print("  Warning: Original scene path not provided, skipping optimized scene export")
            return None
        
        try:
            # 读取原始场景 JSON
            with open(self.original_scene_path, 'r') as f:
                scene_config = json.load(f)
            
            # 更新每个被优化的表面，添加 params 字段
            surfaces_config = scene_config.get("surfaces", [])
            
            for surface_name, params_filename in self._optimized_surfaces.items():
                for surf_cfg in surfaces_config:
                    if surf_cfg.get("name") == surface_name:
                        # 使用相对于 optimized_scene.json 的路径
                        surf_cfg["params"] = params_filename
                        break
            
            # 更新描述信息
            original_desc = scene_config.get("description", "")
            scene_config["description"] = f"{original_desc} [optimized]"
            
            # 保存优化后的场景
            output_path = self.output_dir / "optimized_scene.json"
            with open(output_path, 'w') as f:
                json.dump(scene_config, f, indent=4)
            
            print(f"  Saved optimized scene: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"  Warning: Failed to save optimized scene: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self) -> OptimizationResult:
        result = OptimizationResult()
        start_time = time.time()
        
        try:
            if not self.stages:
                raise ValueError("No optimization stages specified")
            
            for stage_idx, stage in enumerate(self.stages):
                self._current_stage = stage_idx
                stage_result = self._run_stage(stage, stage_idx)
                result.stage_results.append(stage_result)
                result.loss_history.extend(stage_result["loss_history"])
                
                if stage_result["final_loss"] < result.best_loss:
                    result.best_loss = stage_result["final_loss"]
                    result.best_params = np.array(self._get_optimizable_params(stage.surface_name))
            
            result.success = True
            result.total_iterations = self._iteration
            result.final_loss = result.loss_history[-1] if result.loss_history else float('inf')
            result.message = "Optimization completed successfully"
            
        except Exception as e:
            result.success = False
            result.message = f"Optimization failed: {str(e)}"
            import traceback
            traceback.print_exc()
        
        result.total_time_s = time.time() - start_time
        
        self._save_summary(result)
        
        return result
    
    def _save_summary(self, result: OptimizationResult):
        """保存优化总结"""
        # 保存 loss 历史
        np.save(self.output_dir / "loss_history.npy", np.array(result.loss_history))
        
        # 保存优化后的场景 JSON
        optimized_scene_path = self._save_optimized_scene()
        
        # 保存总结 JSON
        summary = {
            "success": result.success,
            "total_iterations": result.total_iterations,
            "final_loss": result.final_loss,
            "best_loss": result.best_loss,
            "total_time_s": result.total_time_s,
            "message": result.message,
            "num_stages": len(self.stages),
            "optimized_surfaces": list(self._optimized_surfaces.keys()),
            "optimized_scene": optimized_scene_path,
            "stages": [
                {
                    "name": s["name"],
                    "steps": s["steps"],
                    "optimizer": s.get("optimizer", "adam"),
                    "learning_rate": s.get("learning_rate", 0.01),
                    "surface": s.get("surface", ""),
                    "final_loss": s["final_loss"]
                }
                for s in result.stage_results
            ]
        }
        
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Optimization completed!")
        print(f"  Total stages: {len(self.stages)}")
        print(f"  Total iterations: {result.total_iterations}")
        print(f"  Final loss: {result.final_loss:.6e}")
        print(f"  Total time: {result.total_time_s:.2f}s")
        print(f"  Results saved to: {self.output_dir}")
        if optimized_scene_path:
            print(f"  Optimized scene: {optimized_scene_path}")
        print(f"{'='*60}")


def run_optimization(
    scene,
    simulator,
    sim_config,
    opt_config: List[Dict[str, Any]],
    output_dir: str = "results",
    save_details: bool = False,
    original_scene_path: Optional[str] = None,
    surface_params_range: Optional[Dict[str, List[float]]] = None
) -> OptimizationResult:
    optimizer = Optimizer(
        scene, simulator, sim_config, opt_config,
        output_dir=output_dir,
        save_details=save_details,
        original_scene_path=original_scene_path,
        surface_params_range=surface_params_range
    )
    return optimizer.run()