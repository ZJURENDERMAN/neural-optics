# scene_visualizer.py

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


class SceneVisualizer:
    """Scene YOZ cross-section visualizer"""
    
    def __init__(self):
        self.colors = {
            'light': '#FFD700',
            'reflector': '#4169E1',
            'lens_front': '#32CD32',
            'lens_back': '#228B22',
            'sensor': '#DC143C',
            'ray_upper': '#FF6347',
            'ray_lower': '#00CED1',
            'glass': '#87CEEB',
        }
        self.linewidths = {
            'surface': 3,
            'ray': 1.5,
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_surface_info(self, config: Dict) -> Dict[str, Dict]:
        surfaces = {}
        for surf in config['surfaces']:
            name = surf['name']
            size = surf['size'][0]
            pos = surf['transform']['position']
            rot = surf['transform']['rotation']
            
            surfaces[name] = {
                'size': size,
                'y': pos[1],
                'z': pos[2],
                'rotation': rot,
                'type': surf['type'],
                'bsdf': surf.get('bsdf', None),
            }
        return surfaces
    
    def _detect_scene_type(self, config: Dict) -> str:
        surfaces = config['surfaces']
        if len(surfaces) == 4:
            return 'refraction'
        return 'reflection'
    
    def _get_ior(self, config: Dict) -> float:
        for mat in config['volume_materials']:
            if mat['type'] == 'constant_ior':
                return mat['ior']
        return 1.5
    
    def _get_upper_angle(self, config: Dict) -> float:
        for emitter in config['emitters']:
            if 'upper_angle' in emitter:
                return emitter['upper_angle']
        return 0.0
    
    def _trace_reflection_rays(self, surfaces: Dict, upper_angle: float) -> Tuple[List, List]:
        """Trace boundary rays for reflection scene"""
        light = surfaces['S0']
        reflector = surfaces['S1']
        sensor = surfaces['S2']
        
        light_half = light['size'] / 2
        reflector_half = reflector['size'] / 2
        
        light_z = light['z']
        reflector_z = reflector['z']
        sensor_z = sensor['z']
        
        if upper_angle <= 0:
            rays_upper = [
                (light_z, light_half),
                (reflector_z, light_half),
                (sensor_z, light_half)
            ]
            rays_lower = [
                (light_z, -light_half),
                (reflector_z, -light_half),
                (sensor_z, -light_half)
            ]
        else:
            y_light = light_half
            y_reflector = reflector_half
            
            dy_in = y_reflector - y_light
            dz_in = reflector_z - light_z
            length_in = np.sqrt(dy_in**2 + dz_in**2)
            dy_in /= length_in
            dz_in /= length_in
            
            dy_out = dy_in
            dz_out = -dz_in
            
            t_out = (sensor_z - reflector_z) / dz_out
            y_sensor = y_reflector + dy_out * t_out
            
            rays_upper = [
                (light_z, y_light),
                (reflector_z, y_reflector),
                (sensor_z, y_sensor)
            ]
            
            y_light_lower = -light_half
            y_reflector_lower = -reflector_half
            
            dy_in = y_reflector_lower - y_light_lower
            dz_in = reflector_z - light_z
            length_in = np.sqrt(dy_in**2 + dz_in**2)
            dy_in /= length_in
            dz_in /= length_in
            
            dy_out = dy_in
            dz_out = -dz_in
            
            t_out = (sensor_z - reflector_z) / dz_out
            y_sensor_lower = y_reflector_lower + dy_out * t_out
            
            rays_lower = [
                (light_z, y_light_lower),
                (reflector_z, y_reflector_lower),
                (sensor_z, y_sensor_lower)
            ]
        
        return rays_upper, rays_lower
    
    def _trace_refraction_rays(self, surfaces: Dict, upper_angle: float, 
                                ior: float) -> Tuple[List, List]:
        """Trace boundary rays for refraction scene."""
        light = surfaces['S0']
        front = surfaces['S1']
        back = surfaces['S2']
        sensor = surfaces['S3']
        
        light_half = light['size'] / 2
        back_half = back['size'] / 2
        
        light_z = light['z']
        front_z = front['z']
        back_z = back['z']
        sensor_z = sensor['z']
        
        def trace_single_ray(y_light: float, y_back_target: float) -> List[Tuple[float, float]]:
            path = [(light_z, y_light)]
            
            dy_total = y_back_target - y_light
            dz_total = back_z - light_z
            length_total = np.sqrt(dy_total**2 + dz_total**2)
            
            dy_in = dy_total / length_total
            dz_in = dz_total / length_total
            
            t_front = (front_z - light_z) / dz_in
            y_front = y_light + dy_in * t_front
            path.append((front_z, y_front))
            
            sin_theta_in = abs(dy_in)
            
            if sin_theta_in < 1e-10:
                dy_glass = 0.0
                dz_glass = 1.0
            else:
                sin_theta_glass = sin_theta_in / ior
                if sin_theta_glass > 1:
                    sin_theta_glass = 1.0
                cos_theta_glass = np.sqrt(1 - sin_theta_glass**2)
                
                dy_glass = np.sign(dy_in) * sin_theta_glass
                dz_glass = cos_theta_glass
            
            t_glass = (back_z - front_z) / dz_glass
            y_back = y_front + dy_glass * t_glass
            path.append((back_z, y_back))
            
            sin_theta_glass_out = abs(dy_glass)
            
            if sin_theta_glass_out < 1e-10:
                dy_out = 0.0
                dz_out = 1.0
            else:
                sin_theta_out = sin_theta_glass_out * ior
                if sin_theta_out > 1:
                    sin_theta_out = 1.0
                cos_theta_out = np.sqrt(1 - sin_theta_out**2)
                
                dy_out = np.sign(dy_glass) * sin_theta_out
                dz_out = cos_theta_out
            
            t_out = (sensor_z - back_z) / dz_out
            y_sensor = y_back + dy_out * t_out
            path.append((sensor_z, y_sensor))
            
            return path
        
        if upper_angle <= 0:
            rays_upper = trace_single_ray(light_half, light_half)
            rays_lower = trace_single_ray(-light_half, -light_half)
        else:
            rays_upper = trace_single_ray(light_half, back_half)
            rays_lower = trace_single_ray(-light_half, -back_half)
        
        return rays_upper, rays_lower
    
    def _get_nice_ticks(self, vmin: float, vmax: float, num_ticks: int = 5) -> np.ndarray:
        """Generate nice tick values covering the data range"""
        range_val = vmax - vmin
        if range_val == 0:
            return np.array([vmin])
        
        raw_step = range_val / (num_ticks - 1)
        magnitude = 10 ** np.floor(np.log10(abs(raw_step)))
        normalized = raw_step / magnitude
        
        if normalized < 1.5:
            nice_step = 1 * magnitude
        elif normalized < 3:
            nice_step = 2 * magnitude
        elif normalized < 7:
            nice_step = 5 * magnitude
        else:
            nice_step = 10 * magnitude
        
        nice_min = np.floor(vmin / nice_step) * nice_step
        nice_max = np.ceil(vmax / nice_step) * nice_step
        
        ticks = np.arange(nice_min, nice_max + nice_step * 0.5, nice_step)
        return ticks
    
    # scene_visualizer.py 中的修改

    def visualize(self, config_path: str, save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (12, 8), show: bool = True):
        """
        Visualize scene YOZ cross-section
        
        Args:
            config_path: Path to scene config JSON file
            save_path: Optional path to save the image
            figsize: Figure size (width, height)
            show: Whether to display the figure (default True)
        """
        config = self.load_config(config_path)
        surfaces = self._get_surface_info(config)
        scene_type = self._detect_scene_type(config)
        upper_angle = self._get_upper_angle(config)
        ior = self._get_ior(config)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        all_z = []
        all_y = []
        
        if scene_type == 'reflection':
            self._draw_surface(ax, surfaces['S0'], self.colors['light'], 'S0 (Light)', all_z, all_y)
            self._draw_surface(ax, surfaces['S1'], self.colors['reflector'], 'S1 (Reflector)', all_z, all_y)
            self._draw_surface(ax, surfaces['S2'], self.colors['sensor'], 'S2 (Sensor)', all_z, all_y)
            rays_upper, rays_lower = self._trace_reflection_rays(surfaces, upper_angle)
        else:
            self._draw_surface(ax, surfaces['S0'], self.colors['light'], 'S0 (Light)', all_z, all_y)
            self._draw_surface(ax, surfaces['S1'], self.colors['lens_front'], 'S1 (Front)', all_z, all_y)
            self._draw_surface(ax, surfaces['S2'], self.colors['lens_back'], 'S2 (Back)', all_z, all_y)
            self._draw_surface(ax, surfaces['S3'], self.colors['sensor'], 'S3 (Sensor)', all_z, all_y)
            self._draw_glass_region(ax, surfaces['S1'], surfaces['S2'])
            rays_upper, rays_lower = self._trace_refraction_rays(surfaces, upper_angle, ior)
        
        self._draw_ray(ax, rays_upper, self.colors['ray_upper'], 'Upper Ray', all_z, all_y)
        self._draw_ray(ax, rays_lower, self.colors['ray_lower'], 'Lower Ray', all_z, all_y)
        
        # Calculate data range
        z_min, z_max = min(all_z), max(all_z)
        y_min, y_max = min(all_y), max(all_y)
        
        z_range = z_max - z_min
        y_range = y_max - y_min
        
        # Get nice ticks
        z_ticks = self._get_nice_ticks(z_min, z_max, 6)
        y_ticks = self._get_nice_ticks(y_min, y_max, 6)
        
        # Z axis at y=0, Y axis outside data region
        z_axis_y = 0.0
        y_axis_z = z_ticks[0] - z_range * 0.12
        
        # Set plot limits to include axes
        plot_z_min = y_axis_z - z_range * 0.08
        plot_z_max = z_ticks[-1] + z_range * 0.15
        plot_y_min = y_ticks[0] - y_range * 0.1
        plot_y_max = y_ticks[-1] + y_range * 0.15
        
        ax.set_xlim(plot_z_min, plot_z_max)
        ax.set_ylim(plot_y_min, plot_y_max)
        
        # Force equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Hide all spines and ticks
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        # Tick length in data coordinates
        tick_len = min(z_range, y_range) * 0.015
        
        # Draw Z axis with arrow at y=0
        ax.annotate('',
                    xy=(z_ticks[-1] + z_range * 0.08, z_axis_y),
                    xytext=(y_axis_z, z_axis_y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                                   shrinkA=0, shrinkB=0))
        ax.text(z_ticks[-1] + z_range * 0.1, z_axis_y, 'Z',
                fontsize=12, ha='left', va='center', fontweight='bold')
        
        # Draw Y axis with arrow
        ax.annotate('',
                    xy=(y_axis_z, y_ticks[-1] + y_range * 0.08),
                    xytext=(y_axis_z, y_ticks[0] - y_range * 0.02),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                                   shrinkA=0, shrinkB=0))
        ax.text(y_axis_z, y_ticks[-1] + y_range * 0.1, 'Y',
                fontsize=12, ha='center', va='bottom', fontweight='bold')
        
        # Draw Z axis ticks and labels at y=0
        for zt in z_ticks:
            ax.plot([zt, zt], [z_axis_y - tick_len, z_axis_y + tick_len],
                    'k-', lw=1)
            label = f'{zt:.0f}' if zt == int(zt) else f'{zt:.1f}'
            ax.text(zt, z_axis_y - tick_len * 3, label,
                    fontsize=9, ha='center', va='top')
        
        # Draw Y axis ticks and labels
        for yt in y_ticks:
            ax.plot([y_axis_z - tick_len, y_axis_z + tick_len], [yt, yt],
                    'k-', lw=1)
            label = f'{yt:.0f}' if yt == int(yt) else f'{yt:.1f}'
            ax.text(y_axis_z - tick_len * 3, yt, label,
                    fontsize=9, ha='right', va='center')
        
        # Add legend outside the plot area
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10,
                  frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Image saved to: {save_path}")
        
        # Only show if requested (and not in batch generation mode)
        if show:
            plt.show()
        
        return fig, ax
        
    def _draw_surface(self, ax, surf_info: Dict, color: str, label: str,
                      all_z: List, all_y: List):
        """Draw a surface"""
        half_size = surf_info['size'] / 2
        z = surf_info['z']
        
        z_coords = [z, z]
        y_coords = [-half_size, half_size]
        
        ax.plot(z_coords, y_coords, color=color, linewidth=self.linewidths['surface'],
                label=label, solid_capstyle='round')
        ax.scatter(z_coords, y_coords, color=color, s=30, zorder=5)
        
        all_z.extend(z_coords)
        all_y.extend(y_coords)
    
    def _draw_glass_region(self, ax, front: Dict, back: Dict):
        """Draw glass medium region"""
        half_size = max(front['size'], back['size']) / 2
        z_front = front['z']
        z_back = back['z']
        
        rect = patches.Rectangle(
            (z_front, -half_size),
            z_back - z_front,
            half_size * 2,
            linewidth=0,
            edgecolor='none',
            facecolor=self.colors['glass'],
            alpha=0.3,
            label='Glass'
        )
        ax.add_patch(rect)
    
    def _draw_ray(self, ax, ray_path: List[Tuple[float, float]], color: str,
                  label: str, all_z: List, all_y: List):
        """Draw ray path"""
        if not ray_path:
            return
        
        z_coords = [p[0] for p in ray_path]
        y_coords = [p[1] for p in ray_path]
        
        ax.plot(z_coords, y_coords, color=color, linewidth=self.linewidths['ray'],
                label=label, linestyle='-', marker='o', markersize=4)
        
        for i in range(len(ray_path) - 1):
            z1, y1 = ray_path[i]
            z2, y2 = ray_path[i + 1]
            
            mid_z = (z1 + z2) / 2
            mid_y = (y1 + y2) / 2
            dz = z2 - z1
            dy = y2 - y1
            
            length = np.sqrt(dz**2 + dy**2)
            if length > 0:
                dz /= length
                dy /= length
                arrow_size = length * 0.15
                ax.annotate('', xy=(mid_z + dz * arrow_size * 0.5, mid_y + dy * arrow_size * 0.5),
                           xytext=(mid_z - dz * arrow_size * 0.5, mid_y - dy * arrow_size * 0.5),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        
        all_z.extend(z_coords)
        all_y.extend(y_coords)


def main():
    parser = argparse.ArgumentParser(description='Scene YOZ cross-section visualizer')
    parser.add_argument('config', type=str, help='Scene config file path (JSON)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output image path')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 8],
                        help='Figure size (width height)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figure window')
    
    args = parser.parse_args()
    
    if args.output is None:
        config_path = Path(args.config)
        output_path = config_path.stem + '_viz.png'
    else:
        output_path = args.output
    
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    
    visualizer = SceneVisualizer()
    
    print(f"Loading config: {args.config}")
    print(f"Output image: {output_path}")
    
    visualizer.visualize(
        args.config,
        save_path=output_path,
        figsize=tuple(args.figsize)
    )


if __name__ == "__main__":
    main()