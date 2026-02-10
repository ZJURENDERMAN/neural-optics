# visualize_scene_distribution.py

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import seaborn as sns


class SceneDistributionVisualizer:
    """可视化场景参数分布"""
    
    def __init__(self, figsize=(16, 12)):
        self.figsize = figsize
        # 设置中文字体（如果需要）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_summary(self, summary_path: str) -> List[Dict[str, Any]]:
        """加载数据集摘要"""
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 兼容两种格式：
        # 1. 直接是列表 [{...}, {...}, ...]
        # 2. 字典格式 {"base_seed": ..., "configs": [{...}, {...}, ...]}
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'configs' in data:
            return data['configs']
        else:
            raise ValueError(f"Unexpected summary format in {summary_path}")
    
    def extract_parameters(self, summary: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """提取所有几何参数"""
        params = {
            'light_size': [],
            'light_z': [],
            'upper_angle': [],
            'element_size': [],
            'element_z': [],
            'sensor_size': [],
            'sensor_z': [],
            'lens_thickness': [],
            'light_type': [],
        }
        
        for item in summary:
            geom = item['geometry']
            params['light_size'].append(geom['light_size'])
            params['light_z'].append(geom['light_z'])
            params['upper_angle'].append(geom['upper_angle'])
            params['element_size'].append(geom['element_size'])
            params['element_z'].append(geom['element_z'])
            params['sensor_size'].append(geom['sensor_size'])
            params['sensor_z'].append(geom['sensor_z'])
            params['lens_thickness'].append(geom.get('lens_thickness', 0))
            params['light_type'].append(item['light_type'])
        
        # 转换为numpy数组
        for key in params:
            if key != 'light_type':
                params[key] = np.array(params[key])
        
        return params
    
    def plot_distributions(self, params: Dict[str, np.ndarray], 
                          scene_type: str = "Scene",
                          save_path: str = None):
        """绘制参数分布图"""
        
        fig, axes = plt.subplots(3, 3, figsize=self.figsize)
        fig.suptitle(f'{scene_type} Parameter Distributions (n={len(params["light_size"])})', 
                     fontsize=14, fontweight='bold')
        
        # 参数配置
        param_configs = [
            ('light_size', 'Light Size', 'steelblue'),
            ('light_z', 'Light Z Position', 'coral'),
            ('upper_angle', 'Upper Angle (°)', 'seagreen'),
            ('element_size', 'Element Size', 'mediumpurple'),
            ('element_z', 'Element Z / Thickness', 'goldenrod'),
            ('sensor_size', 'Sensor Size', 'crimson'),
            ('sensor_z', 'Sensor Z Position', 'teal'),
            ('lens_thickness', 'Lens Thickness', 'darkorange'),
        ]
        
        for idx, (param_name, label, color) in enumerate(param_configs):
            ax = axes[idx // 3, idx % 3]
            data = params[param_name]
            
            # 跳过全零数据
            if np.all(data == 0):
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       fontsize=12, transform=ax.transAxes)
                ax.set_title(label)
                continue
            
            # 直方图 + KDE
            sns.histplot(data, kde=True, ax=ax, color=color, alpha=0.6, bins=20)
            
            # 添加统计信息
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            
            stats_text = f'μ={mean_val:.2f}, σ={std_val:.2f}\nmin={min_val:.2f}, max={max_val:.2f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(label)
            ax.set_ylabel('Count')
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # 第9个子图：光源类型分布
        ax = axes[2, 2]
        light_types = params['light_type']
        unique_types, counts = np.unique(light_types, return_counts=True)
        colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(unique_types)]
        bars = ax.bar(unique_types, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Light Type')
        ax.set_ylabel('Count')
        ax.set_title('Light Type Distribution')
        
        # 在柱状图上显示数量
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved distribution plot to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(self, params: Dict[str, np.ndarray],
                                scene_type: str = "Scene",
                                save_path: str = None):
        """绘制参数相关性矩阵"""
        
        # 准备数据
        numeric_params = ['light_size', 'light_z', 'upper_angle', 'element_size',
                         'element_z', 'sensor_size', 'sensor_z', 'lens_thickness']
        
        # 过滤掉全零的参数
        valid_params = []
        for p in numeric_params:
            if not np.all(params[p] == 0):
                valid_params.append(p)
        
        data_matrix = np.column_stack([params[p] for p in valid_params])
        
        # 计算相关系数
        corr_matrix = np.corrcoef(data_matrix.T)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        # 设置标签
        ax.set_xticks(range(len(valid_params)))
        ax.set_yticks(range(len(valid_params)))
        ax.set_xticklabels(valid_params, rotation=45, ha='right')
        ax.set_yticklabels(valid_params)
        
        # 添加数值标注
        for i in range(len(valid_params)):
            for j in range(len(valid_params)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
        
        ax.set_title(f'{scene_type} Parameter Correlations', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved correlation plot to {save_path}")
        
        return fig
    
    def plot_scatter_matrix(self, params: Dict[str, np.ndarray],
                           scene_type: str = "Scene",
                           save_path: str = None):
        """绘制关键参数散点图矩阵"""
        
        # 选择关键参数
        key_params = ['light_size', 'element_size', 'upper_angle', 'sensor_size']
        
        # 过滤掉全零参数
        valid_params = [p for p in key_params if not np.all(params[p] == 0)]
        n_params = len(valid_params)
        
        fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
        fig.suptitle(f'{scene_type} Key Parameter Relationships', fontsize=14, fontweight='bold')
        
        # 按光源类型着色
        light_types = params['light_type']
        type_colors = {'point': '#3498db', 'parallel': '#e74c3c', 'area': '#2ecc71'}
        colors = [type_colors.get(t, 'gray') for t in light_types]
        
        for i, param_i in enumerate(valid_params):
            for j, param_j in enumerate(valid_params):
                ax = axes[i, j]
                
                if i == j:
                    # 对角线：直方图
                    for light_type in ['point', 'parallel', 'area']:
                        mask = np.array(light_types) == light_type
                        if np.any(mask):
                            ax.hist(params[param_i][mask], bins=15, alpha=0.5,
                                   color=type_colors[light_type], label=light_type)
                    if i == 0:
                        ax.legend(fontsize=6)
                else:
                    # 非对角线：散点图
                    ax.scatter(params[param_j], params[param_i], 
                              c=colors, alpha=0.5, s=10)
                
                # 标签
                if i == n_params - 1:
                    ax.set_xlabel(param_j, fontsize=8)
                if j == 0:
                    ax.set_ylabel(param_i, fontsize=8)
                
                ax.tick_params(labelsize=6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved scatter matrix to {save_path}")
        
        return fig
    
    def plot_2d_parameter_space(self, params: Dict[str, np.ndarray],
                                x_param: str = 'element_size',
                                y_param: str = 'upper_angle',
                                color_param: str = 'sensor_size',
                                scene_type: str = "Scene",
                                save_path: str = None):
        """绘制2D参数空间图"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = params[x_param]
        y = params[y_param]
        c = params[color_param]
        
        # 按光源类型使用不同marker
        light_types = params['light_type']
        markers = {'point': 'o', 'parallel': 's', 'area': '^'}
        
        scatter = None
        for light_type in ['point', 'parallel', 'area']:
            mask = np.array(light_types) == light_type
            if np.any(mask):
                scatter = ax.scatter(x[mask], y[mask], c=c[mask], 
                                    marker=markers[light_type],
                                    s=50, alpha=0.7, cmap='viridis',
                                    label=light_type, edgecolors='white', linewidth=0.5)
        
        if scatter:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_param)
        
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f'{scene_type} Parameter Space', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 2D parameter space to {save_path}")
        
        return fig
    
    def generate_full_report(self, summary_path: str, output_dir: str = None,
                            scene_type: str = "Scene"):
        """生成完整的可视化报告"""
        
        if output_dir is None:
            output_dir = Path(summary_path).parent
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        summary = self.load_summary(summary_path)
        params = self.extract_parameters(summary)
        
        print(f"\n{'='*60}")
        print(f"Generating visualization report for {scene_type}")
        print(f"Total samples: {len(summary)}")
        print(f"{'='*60}\n")
        
        # 打印统计摘要
        self.print_statistics(params)
        
        # 生成各类图表
        figs = []
        
        # 1. 参数分布
        fig1 = self.plot_distributions(
            params, scene_type,
            save_path=str(output_dir / f"{scene_type.lower()}_distributions.png")
        )
        figs.append(fig1)
        
        # 2. 相关性矩阵
        fig2 = self.plot_correlation_matrix(
            params, scene_type,
            save_path=str(output_dir / f"{scene_type.lower()}_correlations.png")
        )
        figs.append(fig2)
        
        # 3. 散点图矩阵
        fig3 = self.plot_scatter_matrix(
            params, scene_type,
            save_path=str(output_dir / f"{scene_type.lower()}_scatter_matrix.png")
        )
        figs.append(fig3)
        
        # 4. 2D参数空间
        fig4 = self.plot_2d_parameter_space(
            params, 
            x_param='element_size',
            y_param='upper_angle',
            color_param='sensor_size',
            scene_type=scene_type,
            save_path=str(output_dir / f"{scene_type.lower()}_parameter_space.png")
        )
        figs.append(fig4)
        
        print(f"\nReport generated in {output_dir}")
        
        return figs
    
    def print_statistics(self, params: Dict[str, np.ndarray]):
        """打印统计摘要"""
        
        print("\n" + "="*50)
        print("Parameter Statistics Summary")
        print("="*50)
        
        numeric_params = ['light_size', 'light_z', 'upper_angle', 'element_size',
                         'element_z', 'sensor_size', 'sensor_z', 'lens_thickness']
        
        print(f"{'Parameter':<16} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-"*56)
        
        for param in numeric_params:
            data = params[param]
            if not np.all(data == 0):
                print(f"{param:<16} {np.mean(data):>10.3f} {np.std(data):>10.3f} "
                      f"{np.min(data):>10.3f} {np.max(data):>10.3f}")
        
        # 光源类型统计
        print("\n" + "-"*56)
        print("Light Type Distribution:")
        light_types = params['light_type']
        unique_types, counts = np.unique(light_types, return_counts=True)
        for t, c in zip(unique_types, counts):
            print(f"  {t}: {c} ({100*c/len(light_types):.1f}%)")
        
        print("="*50 + "\n")


def integrate_with_generator():
    """
    示例：如何将可视化集成到 SceneDataGenerator 中
    """
    
    code = '''
# 在 SceneDataGenerator 类末尾添加以下方法：

def visualize_dataset(self, summary_path: str = None, scene_type: str = None):
    """
    生成数据集后可视化参数分布
    
    Args:
        summary_path: 数据集摘要文件路径，默认自动检测
        scene_type: 场景类型名称
    """
    from visualize_scene_distribution import SceneDistributionVisualizer
    
    visualizer = SceneDistributionVisualizer()
    
    if summary_path is None:
        # 自动检测摘要文件
        for subdir in ['reflection', 'refraction']:
            path = self.output_dir / subdir / "_dataset_summary.json"
            if path.exists():
                visualizer.generate_full_report(
                    str(path), 
                    scene_type=subdir.capitalize()
                )
    else:
        visualizer.generate_full_report(
            summary_path,
            scene_type=scene_type or "Scene"
        )


# 修改 generate_reflection_dataset 和 generate_refraction_dataset 末尾，
# 添加自动可视化：

def generate_reflection_dataset(self, num_configs: int = 10, 
                                base_seed: int = 42,
                                visualize: bool = True) -> List[Path]:
    """Generate reflection scene dataset"""
    # ... 原有代码 ...
    
    # 在 return 之前添加：
    if visualize:
        self.visualize_dataset(
            str(base_dir / "_dataset_summary.json"),
            scene_type="Reflection"
        )
    
    return saved_files
'''
    return code


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize scene parameter distributions')
    parser.add_argument('--summary', '-s', type=str, 
                       default='scene_configs/refraction/_dataset_summary.json',
                       help='Path to dataset summary JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for plots')
    parser.add_argument('--type', '-t', type=str, default='Refraction',
                       help='Scene type name')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    visualizer = SceneDistributionVisualizer()
    
    # 生成报告
    figs = visualizer.generate_full_report(
        args.summary,
        output_dir=args.output,
        scene_type=args.type
    )
    
    if args.show:
        plt.show()
    else:
        plt.close('all')


if __name__ == "__main__":
    main()