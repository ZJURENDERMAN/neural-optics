# main.py
"""
Main entry point for diff_optics simulation and optimization.

Usage:
    # Single scene, simulation only
    python main.py scene.json --sim simulation.json
    
    # Single scene, with optimization
    python main.py scene.json --sim simulation.json --opt optimization.json
    
    # Batch mode (folder of scenes)
    python main.py scenes_folder/ --sim simulation.json --opt optimization.json
    
    # With detailed intermediate results
    python main.py scene.json --sim simulation.json --opt optimization.json --save-details
    
    # Multi-target optimization (optimize for each image in directory)
    python main.py scene.json --sim sim.json --opt opt.json --target-dir ./targets/
    
    # Single target override
    python main.py scene.json --sim sim.json --opt opt.json --target-file ./my_target.exr
    
    # Batch scenes + Multi-target (all combinations)
    python main.py ./scenes/ --sim sim.json --opt opt.json --target-dir ./targets/
    
    # Batch scenes + Random target matching (one target per scene)
    python main.py ./scenes/ --sim sim.json --opt opt.json --target-dir ./targets/ --match-targets
    
    # Specify output directory
    python main.py scene.json --sim sim.json --opt opt.json --output-dir ./results/
"""

import argparse
import sys
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple


def load_opt_config_with_target_override(opt_config_path: Path, target_file: str) -> List[Dict[str, Any]]:
    """加载优化配置并覆盖目标文件"""
    with open(opt_config_path, 'r') as f:
        config = json.load(f)
    
    for stage in config:
        stage['target'] = {'file': target_file}
    
    return config


def get_target_files(target_dir: Path) -> List[Path]:
    """获取目录中所有支持的图像文件"""
    image_extensions = {'.exr', '.jpg', '.jpeg', '.png', '.hdr', '.bmp', '.tiff', '.tif'}
    return sorted([
        f for f in target_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])


def match_scenes_to_targets(scene_files: List[Path], target_files: List[Path], 
                            seed: Optional[int] = None) -> List[Tuple[Path, Path]]:
    """
    将场景文件与目标文件进行随机匹配。
    
    以场景数量为准，从目标文件中随机采样（打乱后取前N个）。
    如果目标文件数量不足，则循环使用。
    
    Args:
        scene_files: 场景文件列表
        target_files: 目标文件列表
        seed: 随机种子，用于可重复性
    
    Returns:
        List of (scene_path, target_path) 元组
    """
    if not scene_files:
        return []
    
    if not target_files:
        raise ValueError("No target files available for matching")
    
    n_scenes = len(scene_files)
    n_targets = len(target_files)
    
    # 设置随机种子以确保可重复性
    if seed is not None:
        random.seed(seed)
    
    # 打乱目标文件列表
    shuffled_targets = target_files.copy()
    random.shuffle(shuffled_targets)
    
    # 生成匹配对
    matches = []
    for i, scene in enumerate(scene_files):
        # 循环使用目标文件（如果场景数量超过目标数量）
        target = shuffled_targets[i % n_targets]
        matches.append((scene, target))
    
    return matches


def run_simulate(result, output_prefix: str) -> int:
    """Run single simulation mode."""
    from exr_utils import save_exr
    
    print("Running simulation...")
    
    sim_result = result.simulator.simulate(result.scene, result.sim_config)
    
    if not sim_result.success:
        print(f"Simulation failed: {sim_result.error_message}")
        return 1
    
    sim_result.print()
    
    # 显示并保存传感器数据
    for sensor_name in sim_result.get_sensor_names():
        sensor_data = sim_result.get_sensor_data(sensor_name)
        
        print(f"\nSensor '{sensor_name}':")
        sensor_data.print()
        
        # 保存为 EXR 文件，使用前缀
        output_file = f"{output_prefix}_{sensor_name}.exr"
        save_exr(
            sensor_data.data,
            sensor_data.width,
            sensor_data.height,
            output_file,
            normalize=True
        )
        print(f"Saved: {output_file}")
    
    print(f"\nSimulation completed in {sim_result.total_time_ms:.2f} ms")
    return 0

def run_optimize(result, output_dir: str, save_details: bool = False, 
                 opt_config_override: Optional[List[Dict[str, Any]]] = None,
                 original_scene_path: Optional[str] = None) -> int:
    """Run optimization mode."""
    from optimizer import run_optimization
    
    print("Running optimization...")
    
    # 使用覆盖的配置或原始配置
    opt_config = opt_config_override if opt_config_override is not None else result.opt_config
    
    if opt_config is None:
        print("Error: No optimization configuration found")
        return 1
    
    opt_result = run_optimization(
        result.scene,
        result.simulator,
        result.sim_config,
        opt_config,
        output_dir=output_dir,
        save_details=save_details,
        original_scene_path=original_scene_path
    )
    
    print(f"\n{'='*60}")
    print("Optimization Result:")
    print(f"{'='*60}")
    print(f"  Status: {'Success' if opt_result.success else 'Failed'}")
    print(f"  Total iterations: {opt_result.total_iterations}")
    print(f"  Final loss: {opt_result.final_loss:.6e}")
    print(f"  Best loss: {opt_result.best_loss:.6e}")
    print(f"  Total time: {opt_result.total_time_s:.2f} s")
    print(f"  Message: {opt_result.message}")
    
    if opt_result.stage_results:
        print(f"\n  Stage Results:")
        for i, stage in enumerate(opt_result.stage_results):
            print(f"    Stage {i+1} ({stage['name']}): final_loss = {stage['final_loss']:.6e}")
    
    return 0 if opt_result.success else 1


def run_single_scene(scene_path: Path, sim_path: Path, opt_path: Optional[Path], args,
                     opt_config_override: Optional[List[Dict[str, Any]]] = None,
                     output_dir_override: Optional[str] = None,
                     output_base_dir: Optional[Path] = None) -> int:
    """Run a single scene configuration."""
    
    # 初始化环境（只在第一次调用时真正执行）
    try:
        from setup import setup_environment
        setup_environment()
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return 1
    
    # 解析配置
    print(f"Scene config: {scene_path}")
    print(f"Simulation config: {sim_path}")
    if opt_path:
        print(f"Optimization config: {opt_path}")
    
    try:
        from parser import parse_config
        result = parse_config(
            str(scene_path),
            str(sim_path),
            str(opt_path) if opt_path else None
        )
    except Exception as e:
        print(f"Error parsing config: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 应用命令行覆盖
    if args.ptx_dir:
        result.sim_config.ptx_dir = args.ptx_dir
    if args.num_rays:
        result.sim_config.num_rays = args.num_rays
    if args.seed is not None:
        result.sim_config.seed = args.seed
    
    if args.verbose:
        print(f"\nParsed: {result}")
        result.sim_config.print()
        result.scene.print()
    
    # 确定执行模式：有 opt_path 则优化，否则仅仿真
    output_prefix = get_output_prefix(scene_path, output_base_dir)
    
    if opt_path and (result.opt_config or opt_config_override):
        # 优化模式：输出到目录
        if output_dir_override:
            output_dir = output_dir_override
        else:
            output_dir = output_prefix
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"\nMode: optimize")
        print(f"Output directory: {output_dir}")
        print(f"Save details: {args.save_details}")
        return run_optimize(result, output_dir, save_details=args.save_details,
                           opt_config_override=opt_config_override,
                           original_scene_path=str(scene_path.resolve()))
    else:
        # 仿真模式：输出到文件
        # 确保输出目录存在
        output_parent = Path(output_prefix).parent
        output_parent.mkdir(parents=True, exist_ok=True)
        print(f"\nMode: simulate")
        print(f"Output prefix: {output_prefix}")
        return run_simulate(result, output_prefix)

def get_output_prefix(scene_path: Path, output_base_dir: Optional[Path] = None) -> str:
    """
    根据场景文件路径生成输出前缀。
    
    Args:
        scene_path: 场景文件路径
        output_base_dir: 可选的输出基础目录，如果指定则输出到该目录下
    
    例如: 
        scene_path=path/to/scene.json, output_base_dir=None -> path/to/scene
        scene_path=path/to/scene.json, output_base_dir=./results -> ./results/scene
    """
    if output_base_dir is not None:
        return str(output_base_dir / scene_path.stem)
    else:
        return str(scene_path.parent / scene_path.stem)


def get_combined_output_dir(scene_name: str, target_name: str, 
                            output_base_dir: Optional[Path] = None,
                            scene_parent: Optional[Path] = None) -> str:
    """
    生成扁平化的组合输出目录名。
    
    Args:
        scene_name: 场景名（不含扩展名）
        target_name: 目标名（不含扩展名）
        output_base_dir: 可选的输出基础目录
        scene_parent: 场景文件所在目录（当 output_base_dir 为 None 时使用）
    
    Returns:
        输出目录路径字符串
    
    例如:
        scene_name='lens_v1', target_name='lenna' -> 'results/lens_v1_lenna'
    """
    combined_name = f"{scene_name}_{target_name}"
    
    if output_base_dir is not None:
        return str(output_base_dir / combined_name)
    elif scene_parent is not None:
        return str(scene_parent / combined_name)
    else:
        return combined_name

def run_batch(scene_folder: Path, sim_path: Path, opt_path: Optional[Path], args,
              output_base_dir: Optional[Path] = None) -> int:
    """Process all scene JSON files in a folder."""
    
    if not scene_folder.exists():
        print(f"Error: Folder not found: {scene_folder}")
        return 1
    
    if not scene_folder.is_dir():
        print(f"Error: {scene_folder} is not a directory")
        return 1
    
    scene_files = sorted(scene_folder.glob("*.json"))
    
    if not scene_files:
        print(f"Error: No .json files found in {scene_folder}")
        return 1
    
    print(f"{'='*60}")
    print(f"Batch Processing: {len(scene_files)} scene files")
    print(f"Scene folder: {scene_folder}")
    print(f"Simulation config: {sim_path}")
    if opt_path:
        print(f"Optimization config: {opt_path}")
    if output_base_dir:
        print(f"Output base directory: {output_base_dir}")
    print(f"Save details: {args.save_details}")
    print(f"{'='*60}\n")
    
    total = len(scene_files)
    success_count = 0
    failed_count = 0
    failed_scenes = []
    
    for idx, scene_file in enumerate(scene_files):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total}] Processing: {scene_file.name}")
        print(f"{'='*60}")
        
        try:
            ret = run_single_scene(scene_file, sim_path, opt_path, args,
                                   output_base_dir=output_base_dir)
            if ret == 0:
                success_count += 1
                print(f"\n[{idx+1}/{total}] SUCCESS: {scene_file.name}")
            else:
                failed_count += 1
                failed_scenes.append(scene_file.name)
                print(f"\n[{idx+1}/{total}] FAILED: {scene_file.name}")
        except Exception as e:
            failed_count += 1
            failed_scenes.append(scene_file.name)
            print(f"\n[{idx+1}/{total}] ERROR: {scene_file.name}")
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Complete")
    print(f"{'='*60}")
    print(f"  Total:   {total}")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {failed_count}")
    
    if failed_scenes:
        print(f"\n  Failed scenes:")
        for name in failed_scenes:
            print(f"    - {name}")
    
    return 0 if failed_count == 0 else 1


def run_with_targets(scene_path: Path, sim_path: Path, opt_path: Path, 
                     target_files: List[Path], args,
                     output_base_dir: Optional[Path] = None) -> Tuple[int, int, List[str]]:
    """
    对单个场景和多个目标图像运行优化。
    
    输出目录采用扁平化命名: {output_base_dir}/{scene_name}_{target_name}/
    
    Args:
        scene_path: 场景文件路径
        sim_path: 仿真配置路径
        opt_path: 优化配置路径
        target_files: 目标图像文件列表
        args: 命令行参数
        output_base_dir: 可选的输出基础目录
    
    Returns:
        (success_count, failed_count, failed_targets)
    """
    scene_name = scene_path.stem
    
    total = len(target_files)
    success_count = 0
    failed_count = 0
    failed_targets = []
    
    for idx, target_file in enumerate(target_files):
        target_name = target_file.stem
        
        print(f"\n{'-'*50}")
        print(f"  [{idx+1}/{total}] Target: {target_name}")
        print(f"  Target file: {target_file}")
        print(f"{'-'*50}")
        
        try:
            # 加载配置并覆盖目标
            opt_config = load_opt_config_with_target_override(
                opt_path,
                target_file=str(target_file.resolve())
            )
            
            # 创建扁平化的输出目录: scene_name_target_name
            output_dir = get_combined_output_dir(
                scene_name=scene_name,
                target_name=target_name,
                output_base_dir=output_base_dir,
                scene_parent=scene_path.parent
            )
            
            # 运行优化
            ret = run_single_scene(
                scene_path, sim_path, opt_path, args,
                opt_config_override=opt_config,
                output_dir_override=output_dir
            )
            
            if ret == 0:
                success_count += 1
                print(f"\n  [{idx+1}/{total}] Target SUCCESS: {target_name}")
            else:
                failed_count += 1
                failed_targets.append(target_name)
                print(f"\n  [{idx+1}/{total}] Target FAILED: {target_name}")
                
        except Exception as e:
            failed_count += 1
            failed_targets.append(target_name)
            print(f"\n  [{idx+1}/{total}] Target ERROR: {target_name}")
            print(f"    Exception: {e}")
            import traceback
            traceback.print_exc()
    
    return (success_count, failed_count, failed_targets)


def run_multi_target(scene_path: Path, sim_path: Path, opt_path: Path, 
                     target_files: List[Path], args,
                     output_base_dir: Optional[Path] = None) -> int:
    """对单个场景文件和多个目标运行优化。"""
    
    print(f"{'='*60}")
    print(f"Multi-Target Optimization")
    print(f"{'='*60}")
    print(f"Scene: {scene_path.name}")
    print(f"Targets: {len(target_files)} images")
    if output_base_dir:
        print(f"Output base directory: {output_base_dir}")
    print(f"Output naming: {{scene_name}}_{{target_name}}/")
    print(f"{'='*60}")
    
    success, failed, failed_list = run_with_targets(
        scene_path, sim_path, opt_path, target_files, args,
        output_base_dir=output_base_dir
    )
    
    print(f"\n{'='*60}")
    print(f"Multi-Target Optimization Complete")
    print(f"{'='*60}")
    print(f"  Scene: {scene_path.name}")
    print(f"  Total targets: {len(target_files)}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    
    if failed_list:
        print(f"\n  Failed targets:")
        for name in failed_list:
            print(f"    - {name}")
    
    return 0 if failed == 0 else 1


def run_batch_with_targets(scene_folder: Path, sim_path: Path, opt_path: Path,
                           target_files: List[Path], args,
                           output_base_dir: Optional[Path] = None) -> int:
    """对多个场景文件和多个目标运行优化（笛卡尔积）。"""
    
    scene_files = sorted(scene_folder.glob("*.json"))
    
    if not scene_files:
        print(f"Error: No .json files found in {scene_folder}")
        return 1
    
    total_scenes = len(scene_files)
    total_targets = len(target_files)
    total_combinations = total_scenes * total_targets
    
    print(f"{'='*60}")
    print(f"Batch + Multi-Target Optimization")
    print(f"{'='*60}")
    print(f"Scene folder: {scene_folder}")
    print(f"Scenes: {total_scenes} files")
    print(f"Targets: {total_targets} images")
    print(f"Total combinations: {total_combinations}")
    if output_base_dir:
        print(f"Output base directory: {output_base_dir}")
    print(f"Output naming: {{scene_name}}_{{target_name}}/")
    print(f"Save details: {args.save_details}")
    print(f"\nScenes:")
    for f in scene_files:
        print(f"  - {f.name}")
    print(f"\nTargets:")
    for f in target_files:
        print(f"  - {f.name}")
    print(f"{'='*60}\n")
    
    overall_success = 0
    overall_failed = 0
    failed_combinations = []
    
    for scene_idx, scene_file in enumerate(scene_files):
        print(f"\n{'='*60}")
        print(f"[Scene {scene_idx+1}/{total_scenes}] {scene_file.name}")
        print(f"{'='*60}")
        
        success, failed, failed_targets = run_with_targets(
            scene_file, sim_path, opt_path, target_files, args,
            output_base_dir=output_base_dir
        )
        
        overall_success += success
        overall_failed += failed
        
        for target_name in failed_targets:
            failed_combinations.append(f"{scene_file.stem}_{target_name}")
        
        print(f"\n[Scene {scene_idx+1}/{total_scenes}] {scene_file.name} - "
              f"Success: {success}, Failed: {failed}")
    
    print(f"\n{'='*60}")
    print(f"Batch + Multi-Target Optimization Complete")
    print(f"{'='*60}")
    print(f"  Total scenes:       {total_scenes}")
    print(f"  Total targets:      {total_targets}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Success: {overall_success}")
    print(f"  Failed:  {overall_failed}")
    
    if failed_combinations:
        print(f"\n  Failed combinations:")
        for combo in failed_combinations:
            print(f"    - {combo}")
    
    return 0 if overall_failed == 0 else 1


def run_matched_targets(scene_folder: Path, sim_path: Path, opt_path: Path,
                        target_files: List[Path], args,
                        output_base_dir: Optional[Path] = None) -> int:
    """
    对多个场景文件和目标图像进行随机匹配优化（一对一）。
    
    以场景数量为准，从目标文件中随机采样匹配。
    
    Args:
        scene_folder: 场景文件夹路径
        sim_path: 仿真配置路径
        opt_path: 优化配置路径
        target_files: 目标图像文件列表
        args: 命令行参数
        output_base_dir: 可选的输出基础目录
    
    Returns:
        退出码（0 表示全部成功）
    """
    scene_files = sorted(scene_folder.glob("*.json"))
    
    if not scene_files:
        print(f"Error: No .json files found in {scene_folder}")
        return 1
    
    total_scenes = len(scene_files)
    total_targets = len(target_files)
    
    # 获取随机种子
    match_seed = args.match_seed if hasattr(args, 'match_seed') and args.match_seed is not None else 42
    
    # 生成场景-目标匹配对
    matches = match_scenes_to_targets(scene_files, target_files, seed=match_seed)
    
    print(f"{'='*60}")
    print(f"Matched Target Optimization (1:1 Random Matching)")
    print(f"{'='*60}")
    print(f"Scene folder: {scene_folder}")
    print(f"Scenes: {total_scenes} files")
    print(f"Available targets: {total_targets} images")
    print(f"Match seed: {match_seed}")
    if total_scenes > total_targets:
        print(f"Note: Scenes ({total_scenes}) > Targets ({total_targets}), targets will be reused")
    if output_base_dir:
        print(f"Output base directory: {output_base_dir}")
    print(f"Output naming: {{scene_name}}_{{target_name}}/")
    print(f"Save details: {args.save_details}")
    print(f"\nMatching pairs:")
    for scene, target in matches:
        print(f"  {scene.name} -> {target.name}")
    print(f"{'='*60}\n")
    
    overall_success = 0
    overall_failed = 0
    failed_pairs = []
    
    for idx, (scene_file, target_file) in enumerate(matches):
        scene_name = scene_file.stem
        target_name = target_file.stem
        
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total_scenes}] Scene: {scene_file.name}")
        print(f"                  Target: {target_file.name}")
        print(f"{'='*60}")
        
        try:
            # 加载配置并覆盖目标
            opt_config = load_opt_config_with_target_override(
                opt_path,
                target_file=str(target_file.resolve())
            )
            
            # 创建扁平化的输出目录: scene_name_target_name
            output_dir = get_combined_output_dir(
                scene_name=scene_name,
                target_name=target_name,
                output_base_dir=output_base_dir,
                scene_parent=scene_folder
            )
            
            # 运行优化
            ret = run_single_scene(
                scene_file, sim_path, opt_path, args,
                opt_config_override=opt_config,
                output_dir_override=output_dir
            )
            
            if ret == 0:
                overall_success += 1
                print(f"\n[{idx+1}/{total_scenes}] SUCCESS: {scene_name} -> {target_name}")
            else:
                overall_failed += 1
                failed_pairs.append(f"{scene_name}_{target_name}")
                print(f"\n[{idx+1}/{total_scenes}] FAILED: {scene_name} -> {target_name}")
                
        except Exception as e:
            overall_failed += 1
            failed_pairs.append(f"{scene_name}_{target_name}")
            print(f"\n[{idx+1}/{total_scenes}] ERROR: {scene_name} -> {target_name}")
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Matched Target Optimization Complete")
    print(f"{'='*60}")
    print(f"  Total pairs: {total_scenes}")
    print(f"  Success: {overall_success}")
    print(f"  Failed:  {overall_failed}")
    
    if failed_pairs:
        print(f"\n  Failed pairs:")
        for pair in failed_pairs:
            print(f"    - {pair}")
    
    return 0 if overall_failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="diff_optics - Differentiable Ray Tracing for Optical Design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single scene, simulation only
  python main.py scene.json --sim simulation.json
  
  # Single scene, with optimization
  python main.py scene.json --sim simulation.json --opt optimization.json
  
  # Batch mode (all scenes in folder)
  python main.py ./scenes/ --sim simulation.json
  
  # Batch optimization
  python main.py ./scenes/ --sim simulation.json --opt optimization.json
  
  # Multi-target optimization (single scene, multiple targets)
  # Output: ./lens_v1_lenna/, ./lens_v1_peppers/, ...
  python main.py lens_v1.json --sim sim.json --opt opt.json --target-dir ./targets/
  
  # Single target override
  python main.py scene.json --sim sim.json --opt opt.json --target-file ./my_target.exr
  
  # Batch scenes + Multi-target (all combinations) [Cartesian product]
  # Output: ./results/scene1_target1/, ./results/scene1_target2/, ./results/scene2_target1/, ...
  python main.py ./scenes/ --sim sim.json --opt opt.json --target-dir ./targets/ -d ./results/
  
  # Batch scenes + Matched targets (random 1:1 matching)
  # Output: ./results/scene1_targetA/, ./results/scene2_targetB/, ...
  python main.py ./scenes/ --sim sim.json --opt opt.json --target-dir ./targets/ --match-targets -d ./results/
  
  # With specific matching seed for reproducibility
  python main.py ./scenes/ --sim sim.json --opt opt.json --target-dir ./targets/ --match-targets --match-seed 123
  
  # Specify output directory
  python main.py scene.json --sim sim.json --opt opt.json --output-dir ./results/
  
  # With detailed intermediate results
  python main.py scene.json --sim sim.json --opt opt.json --save-details
"""
    )
    
    parser.add_argument(
        "scene",
        type=str,
        help="Path to scene JSON file or folder containing scene JSON files"
    )
    
    parser.add_argument(
        "--sim", "-s",
        type=str,
        required=True,
        help="Path to simulation configuration JSON (required)"
    )
    
    parser.add_argument(
        "--opt", "-o",
        type=str,
        default=None,
        help="Path to optimization configuration JSON (optional, enables optimization mode)"
    )
    
    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        default=None,
        help="Base directory for output files. If not specified, outputs are saved alongside scene files."
    )
    
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Directory containing target images. Will optimize for each image separately."
    )
    
    parser.add_argument(
        "--target-file",
        type=str,
        default=None,
        help="Single target image file to override the target in optimization config."
    )
    
    parser.add_argument(
        "--match-targets",
        action="store_true",
        help="Match scenes to targets 1:1 (randomly sampled) instead of Cartesian product. "
             "Number of optimizations equals number of scenes."
    )
    
    parser.add_argument(
        "--match-seed",
        type=int,
        default=42,
        help="Random seed for target matching when using --match-targets (default: 42)"
    )
    
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save detailed intermediate results during optimization"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--ptx-dir",
        type=str,
        default=None,
        help="Override PTX directory path"
    )
    
    parser.add_argument(
        "--num-rays",
        type=int,
        default=None,
        help="Override number of rays"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed"
    )
    
    args = parser.parse_args()
    
    scene_path = Path(args.scene)
    sim_path = Path(args.sim)
    opt_path = Path(args.opt) if args.opt else None
    output_base_dir = Path(args.output_dir) if args.output_dir else None
    
    # 验证路径
    if not scene_path.exists():
        print(f"Error: Scene path not found: {scene_path}")
        return 1
    
    if not sim_path.exists():
        print(f"Error: Simulation config not found: {sim_path}")
        return 1
    
    if opt_path and not opt_path.exists():
        print(f"Error: Optimization config not found: {opt_path}")
        return 1
    
    # 如果指定了输出目录，确保它存在
    if output_base_dir:
        output_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output base directory: {output_base_dir}")
    
    # 验证 target-dir 和 target-file 的互斥性
    if args.target_dir and args.target_file:
        print("Error: Cannot specify both --target-dir and --target-file")
        return 1
    
    # 如果指定了 target-dir 或 target-file，必须同时指定 --opt
    if (args.target_dir or args.target_file) and not opt_path:
        print("Error: --target-dir and --target-file require --opt to be specified")
        return 1
    
    # --match-targets 只在 batch + target-dir 模式下有效
    if args.match_targets and not args.target_dir:
        print("Error: --match-targets requires --target-dir to be specified")
        return 1
    
    if args.match_targets and not scene_path.is_dir():
        print("Error: --match-targets requires scene path to be a directory")
        return 1
    
    # 解析 target 文件列表
    target_files = None
    if args.target_dir:
        target_dir = Path(args.target_dir)
        if not target_dir.exists():
            print(f"Error: Target directory not found: {target_dir}")
            return 1
        if not target_dir.is_dir():
            print(f"Error: {target_dir} is not a directory")
            return 1
        target_files = get_target_files(target_dir)
        if not target_files:
            print(f"Error: No image files found in {target_dir}")
            return 1
    elif args.target_file:
        target_file = Path(args.target_file)
        if not target_file.exists():
            print(f"Error: Target file not found: {target_file}")
            return 1
        target_files = [target_file]
    
    print("Initializing environment...")
    
    # 根据输入类型和参数选择模式
    is_batch_scene = scene_path.is_dir()
    has_targets = target_files is not None
    
    if is_batch_scene and has_targets:
        if args.match_targets:
            # 批量场景 + 随机匹配目标模式（一对一）
            return run_matched_targets(scene_path, sim_path, opt_path, target_files, args,
                                       output_base_dir=output_base_dir)
        else:
            # 批量场景 + 多目标模式（笛卡尔积）
            return run_batch_with_targets(scene_path, sim_path, opt_path, target_files, args,
                                          output_base_dir=output_base_dir)
    
    elif is_batch_scene and not has_targets:
        # 批量场景模式（原有逻辑）
        return run_batch(scene_path, sim_path, opt_path, args,
                        output_base_dir=output_base_dir)
    
    elif not is_batch_scene and has_targets:
        # 单场景 + 多目标模式
        if not scene_path.suffix.lower() == '.json':
            print(f"Warning: Scene file does not have .json extension: {scene_path}")
        return run_multi_target(scene_path, sim_path, opt_path, target_files, args,
                               output_base_dir=output_base_dir)
    
    else:
        # 单场景模式（原有逻辑）
        if not scene_path.suffix.lower() == '.json':
            print(f"Warning: Scene file does not have .json extension: {scene_path}")
        return run_single_scene(scene_path, sim_path, opt_path, args,
                               output_base_dir=output_base_dir)


if __name__ == "__main__":
    sys.exit(main())