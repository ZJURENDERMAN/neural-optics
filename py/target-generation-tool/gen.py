# generate_datasets.py
import numpy as np
import os
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
from tqdm import tqdm
import random

def save_exr(data: np.ndarray, filepath: str):
    """Save numpy array as single-channel EXR file."""
    try:
        import OpenEXR
        import Imath
        
        # Ensure data is float32 and 2D
        data = data.astype(np.float32)
        if len(data.shape) == 3:
            data = np.mean(data, axis=2)
        
        height, width = data.shape
        
        # Create header for single channel
        header = OpenEXR.Header(width, height)
        header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        
        # Create output file
        out = OpenEXR.OutputFile(filepath, header)
        
        # Write data
        out.writePixels({'Y': data.tobytes()})
        out.close()
        
        return True
        
    except ImportError:
        print("OpenEXR not available. Saving as numpy file instead.")
        np_path = filepath.replace('.exr', '.npy')
        np.save(np_path, data)
        return False


class FontManager:
    """Manage and discover available fonts on the system"""
    
    # Characters to test for font validity
    TEST_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    
    def __init__(self, max_fonts: Optional[int] = None, random_select: bool = False,
                 verbose: bool = False):
        """
        Initialize FontManager.
        
        Args:
            max_fonts: Maximum number of fonts to use. None means use all.
            random_select: If True, randomly select fonts. If False, use first N fonts.
            verbose: If True, print detailed information about font discovery.
        """
        self.verbose = verbose
        all_fonts = self._discover_fonts()
        
        # Apply max_fonts limit
        if max_fonts is not None and max_fonts > 0 and max_fonts < len(all_fonts):
            if random_select:
                random.seed(42)  # For reproducibility
                self.available_fonts = sorted(random.sample(all_fonts, max_fonts))
            else:
                self.available_fonts = all_fonts[:max_fonts]
            print(f"Found {len(all_fonts)} valid fonts, using {len(self.available_fonts)} fonts")
        else:
            self.available_fonts = all_fonts
            print(f"Found {len(self.available_fonts)} available fonts")
    
    def _discover_fonts(self) -> List[str]:
        """Discover all available TrueType fonts on the system"""
        font_paths = []
        
        # Windows font directories
        windows_dirs = [
            "C:/Windows/Fonts",
            os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
        ]
        
        # Linux font directories
        linux_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ]
        
        # macOS font directories
        macos_dirs = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
        
        # Combine all directories
        all_dirs = windows_dirs + linux_dirs + macos_dirs
        
        # Collect all font files first
        candidate_fonts = []
        for font_dir in all_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        if file.lower().endswith(('.ttf', '.otf')):
                            font_path = os.path.join(root, file)
                            candidate_fonts.append(font_path)
        
        # Remove duplicates
        candidate_fonts = sorted(list(set(candidate_fonts)))
        
        if self.verbose:
            print(f"Found {len(candidate_fonts)} font files, validating...")
        
        # Validate fonts with progress bar
        valid_fonts = []
        rejected_fonts = []
        
        for font_path in tqdm(candidate_fonts, desc="Validating fonts", disable=not self.verbose):
            is_valid, reason = self._is_valid_font(font_path)
            if is_valid:
                valid_fonts.append(font_path)
            else:
                rejected_fonts.append((font_path, reason))
        
        if self.verbose and rejected_fonts:
            print(f"\nRejected {len(rejected_fonts)} fonts:")
            # Group by reason
            reasons = {}
            for path, reason in rejected_fonts:
                if reason not in reasons:
                    reasons[reason] = []
                reasons[reason].append(os.path.basename(path))
            
            for reason, fonts in reasons.items():
                print(f"  {reason}: {len(fonts)} fonts")
                if len(fonts) <= 5:
                    for f in fonts:
                        print(f"    - {f}")
                else:
                    for f in fonts[:3]:
                        print(f"    - {f}")
                    print(f"    ... and {len(fonts) - 3} more")
        
        return valid_fonts
    
    def _is_valid_font(self, font_path: str) -> Tuple[bool, str]:
        """
        Check if a font file can be loaded and properly renders Latin characters.
        
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        try:
            font = ImageFont.truetype(font_path, 48)
        except Exception as e:
            return False, f"Cannot load: {str(e)[:50]}"
        
        try:
            # Test rendering multiple characters to detect tofu/boxes
            test_chars = "AaBbCc0123"
            
            # Create test image
            img_size = 200
            img = Image.new('L', (img_size, img_size), 0)
            draw = ImageDraw.Draw(img)
            
            # Render test string
            draw.text((10, 10), test_chars, font=font, fill=255)
            
            # Convert to numpy for analysis
            arr = np.array(img)
            
            # Check if anything was rendered
            if arr.max() == 0:
                return False, "No pixels rendered"
            
            # Count non-zero pixels
            non_zero_pixels = np.count_nonzero(arr)
            
            # If too few pixels, font might not support these characters
            if non_zero_pixels < 100:
                return False, "Too few pixels rendered"
            
            # Additional check: render individual characters and verify variety
            # Tofu boxes tend to look identical for all characters
            char_signatures = []
            for char in "AEIOU":
                char_img = Image.new('L', (60, 60), 0)
                char_draw = ImageDraw.Draw(char_img)
                char_draw.text((5, 5), char, font=font, fill=255)
                char_arr = np.array(char_img)
                
                # Calculate a simple signature (sum of pixels in different regions)
                h, w = char_arr.shape
                signature = (
                    char_arr[:h//2, :w//2].sum(),
                    char_arr[:h//2, w//2:].sum(),
                    char_arr[h//2:, :w//2].sum(),
                    char_arr[h//2:, w//2:].sum()
                )
                char_signatures.append(signature)
            
            # If all signatures are identical, likely tofu boxes
            if len(set(char_signatures)) == 1:
                return False, "All characters render identically (likely tofu)"
            
            # Check for digit support as well
            digit_signatures = []
            for char in "012":
                char_img = Image.new('L', (60, 60), 0)
                char_draw = ImageDraw.Draw(char_img)
                char_draw.text((5, 5), char, font=font, fill=255)
                char_arr = np.array(char_img)
                
                h, w = char_arr.shape
                signature = (
                    char_arr[:h//2, :w//2].sum(),
                    char_arr[:h//2, w//2:].sum(),
                    char_arr[h//2:, :w//2].sum(),
                    char_arr[h//2:, w//2:].sum()
                )
                digit_signatures.append(signature)
            
            # If all digit signatures are identical and same as letter signatures
            if len(set(digit_signatures)) == 1 and digit_signatures[0] == char_signatures[0]:
                return False, "Digits render same as letters (likely tofu)"
            
            return True, ""
            
        except Exception as e:
            return False, f"Render error: {str(e)[:50]}"
    
    def get_font(self, index: int, font_size: int) -> ImageFont.FreeTypeFont:
        """Get a specific font at the specified size"""
        if len(self.available_fonts) == 0:
            print("Warning: No fonts found, using default")
            return ImageFont.load_default()
        
        font_path = self.available_fonts[index % len(self.available_fonts)]
        try:
            return ImageFont.truetype(font_path, font_size)
        except:
            return ImageFont.load_default()
    
    def get_font_path(self, index: int) -> str:
        """Get the font path at given index"""
        if len(self.available_fonts) == 0:
            return "default"
        return self.available_fonts[index % len(self.available_fonts)]
    
    def get_font_name(self, font_path: str) -> str:
        """Extract font name from path"""
        return os.path.splitext(os.path.basename(font_path))[0]
    
    def get_num_fonts(self) -> int:
        """Return the number of available fonts"""
        return len(self.available_fonts)


class CenteredPatternGenerator:
    """Generate centered patterns (digits, letters, shapes) with diverse fonts"""
    
    def __init__(self, resolution: Tuple[int, int] = (128, 128), 
                 max_fonts: Optional[int] = None, random_select: bool = False,
                 verbose: bool = False, fill_ratio: float = 0.9):
        """
        Initialize the pattern generator.
        
        Args:
            resolution: Output image resolution (width, height)
            max_fonts: Maximum number of fonts to use
            random_select: Whether to randomly select fonts
            verbose: Whether to print detailed information
            fill_ratio: How much of the canvas the character should fill (0.0 to 1.0)
        """
        self.resolution = resolution
        self.width, self.height = resolution
        self.fill_ratio = fill_ratio
        
        # Create coordinate grids for shape generation
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Initialize font manager with limit
        self.font_manager = FontManager(max_fonts=max_fonts, random_select=random_select,
                                        verbose=verbose)
    
    def generate_character(self, char: str, font_path: str,
                          position_offset: Tuple[float, float] = (0.0, 0.0),
                          rotation: float = 0.0,
                          fill_ratio: Optional[float] = None) -> np.ndarray:
        """
        Generate a character that fills the canvas.
        
        Args:
            char: The character to render
            font_path: Path to the font file
            position_offset: (x, y) offset as fraction of available space (-1.0 to 1.0)
            rotation: Rotation angle in degrees (-180 to 180)
            fill_ratio: Override the default fill ratio (0.0 to 1.0)
        
        Returns:
            numpy array of the rendered character, scaled to fill the canvas
        """
        if fill_ratio is None:
            fill_ratio = self.fill_ratio
        
        # Target size based on fill ratio
        target_width = int(self.width * fill_ratio)
        target_height = int(self.height * fill_ratio)
        
        # Start with a large font size and render
        # Use a large canvas initially to get the character shape
        initial_font_size = 200
        try:
            font = ImageFont.truetype(font_path, initial_font_size)
        except:
            font = ImageFont.load_default()
        
        # Create a large temporary canvas
        temp_size = 800
        temp_img = Image.new('L', (temp_size, temp_size), color=0)
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Get text bounding box
        try:
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback
            text_width = initial_font_size
            text_height = initial_font_size
            bbox = (0, 0, text_width, text_height)
        
        # Draw text centered in temp canvas
        x = (temp_size - text_width) // 2 - bbox[0]
        y = (temp_size - text_height) // 2 - bbox[1]
        
        try:
            temp_draw.text((x, y), char, fill=255, font=font)
        except:
            temp_draw.text((temp_size // 3, temp_size // 3), char, fill=255)
        
        # Apply rotation before scaling
        if rotation != 0:
            temp_img = temp_img.rotate(rotation, resample=Image.BICUBIC, 
                                       center=(temp_size // 2, temp_size // 2))
        
        # Convert to numpy array
        temp_array = np.array(temp_img, dtype=np.float32)
        
        # Find the tight bounding box of non-zero content
        non_zero = np.argwhere(temp_array > 0)
        if len(non_zero) == 0:
            # Empty image, return blank canvas
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        min_y, min_x = non_zero.min(axis=0)
        max_y, max_x = non_zero.max(axis=0)
        
        # Extract content with tight crop
        content = temp_array[min_y:max_y+1, min_x:max_x+1]
        content_height, content_width = content.shape
        
        # Calculate scale factor to fit target size while maintaining aspect ratio
        scale_x = target_width / content_width
        scale_y = target_height / content_height
        scale = min(scale_x, scale_y)  # Use the smaller scale to fit both dimensions
        
        # Calculate new size
        new_width = int(content_width * scale)
        new_height = int(content_height * scale)
        
        # Ensure minimum size
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Resize the content
        content_img = Image.fromarray(content.astype(np.uint8))
        content_img = content_img.resize((new_width, new_height), resample=Image.BICUBIC)
        content = np.array(content_img, dtype=np.float32)
        
        # Calculate center position
        center_x = (self.width - new_width) // 2
        center_y = (self.height - new_height) // 2
        
        # Apply position offset
        # Offset is fraction of available movement space
        max_offset_x = center_x  # Maximum pixels we can move
        max_offset_y = center_y
        
        offset_x = int(position_offset[0] * max_offset_x)
        offset_y = int(position_offset[1] * max_offset_y)
        
        paste_x = center_x + offset_x
        paste_y = center_y + offset_y
        
        # Clamp to valid range
        paste_x = max(0, min(paste_x, self.width - new_width))
        paste_y = max(0, min(paste_y, self.height - new_height))
        
        # Create final image
        final_img = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Paste content
        final_img[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = content
        
        return final_img / 255.0
    
    def generate_shape(self, shape_type: str, variation: int = 0) -> np.ndarray:
        """Generate centered geometric shape with variations"""
        img = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Add variation to shape parameters
        np.random.seed(variation)
        # Shapes fill most of the canvas by default
        size_var = 0.85 + np.random.uniform(-0.05, 0.05)
        
        if shape_type == 'circle':
            radius = 0.45 * size_var
            mask = (self.X**2 + self.Y**2) <= radius**2
            img[mask] = 1.0
            
        elif shape_type == 'square':
            size = 0.45 * size_var
            mask = (np.abs(self.X) <= size) & (np.abs(self.Y) <= size)
            img[mask] = 1.0
            
        elif shape_type == 'triangle':
            pil_img = Image.new('L', (self.width, self.height), color=0)
            draw = ImageDraw.Draw(pil_img)
            
            size = 0.45 * size_var
            # Rotate triangle by variation
            angle_offset = variation * 0.2
            vertices = []
            for i in range(3):
                angle = 2 * np.pi * i / 3 - np.pi / 2 + angle_offset
                vx = size * np.cos(angle)
                vy = size * np.sin(angle)
                px = int((vx + 1) * self.width / 2)
                py = int((vy + 1) * self.height / 2)
                vertices.append((px, py))
            
            draw.polygon(vertices, fill=255)
            img = np.array(pil_img, dtype=np.float32) / 255.0
            
        elif shape_type == 'pentagon':
            pil_img = Image.new('L', (self.width, self.height), color=0)
            draw = ImageDraw.Draw(pil_img)
            
            n_sides = 5
            radius = 0.45 * size_var
            angle_offset = variation * 0.15
            points = []
            for i in range(n_sides):
                angle = 2 * np.pi * i / n_sides - np.pi / 2 + angle_offset
                vx = radius * np.cos(angle)
                vy = radius * np.sin(angle)
                px = int((vx + 1) * self.width / 2)
                py = int((vy + 1) * self.height / 2)
                points.append((px, py))
            
            draw.polygon(points, fill=255)
            img = np.array(pil_img, dtype=np.float32) / 255.0
            
        elif shape_type == 'hexagon':
            pil_img = Image.new('L', (self.width, self.height), color=0)
            draw = ImageDraw.Draw(pil_img)
            
            n_sides = 6
            radius = 0.45 * size_var
            angle_offset = variation * 0.12
            points = []
            for i in range(n_sides):
                angle = 2 * np.pi * i / n_sides + angle_offset
                vx = radius * np.cos(angle)
                vy = radius * np.sin(angle)
                px = int((vx + 1) * self.width / 2)
                py = int((vy + 1) * self.height / 2)
                points.append((px, py))
            
            draw.polygon(points, fill=255)
            img = np.array(pil_img, dtype=np.float32) / 255.0
            
        elif shape_type == 'star':
            pil_img = Image.new('L', (self.width, self.height), color=0)
            draw = ImageDraw.Draw(pil_img)
            
            n_points = 5
            outer_radius = 0.45 * size_var
            inner_radius = 0.2 * size_var * (1 + variation * 0.05)
            angle_offset = variation * 0.15
            points = []
            for i in range(n_points * 2):
                angle = np.pi * i / n_points - np.pi / 2 + angle_offset
                radius = outer_radius if i % 2 == 0 else inner_radius
                vx = radius * np.cos(angle)
                vy = radius * np.sin(angle)
                px = int((vx + 1) * self.width / 2)
                py = int((vy + 1) * self.height / 2)
                points.append((px, py))
            
            draw.polygon(points, fill=255)
            img = np.array(pil_img, dtype=np.float32) / 255.0
            
        elif shape_type == 'cross':
            thickness = 0.18 * size_var
            length = 0.45 * size_var
            mask_v = (np.abs(self.X) <= thickness/2) & (np.abs(self.Y) <= length)
            mask_h = (np.abs(self.X) <= length) & (np.abs(self.Y) <= thickness/2)
            img[mask_v | mask_h] = 1.0
            
        elif shape_type == 'ring':
            outer_radius = 0.45 * size_var
            inner_ratio = 0.5 + variation * 0.03
            inner_radius = outer_radius * inner_ratio
            dist = np.sqrt(self.X**2 + self.Y**2)
            mask = (dist <= outer_radius) & (dist >= inner_radius)
            img[mask] = 1.0
            
        elif shape_type == 'diamond':
            pil_img = Image.new('L', (self.width, self.height), color=0)
            draw = ImageDraw.Draw(pil_img)
            
            size = 0.45 * size_var
            aspect = 1.0 + variation * 0.05
            cx, cy = self.width // 2, self.height // 2
            vertices = [
                (cx, cy - int(size * self.height * aspect)),
                (cx + int(size * self.width), cy),
                (cx, cy + int(size * self.height * aspect)),
                (cx - int(size * self.width), cy)
            ]
            
            draw.polygon(vertices, fill=255)
            img = np.array(pil_img, dtype=np.float32) / 255.0
            
        elif shape_type == 'ellipse':
            a = 0.45 * size_var
            b = 0.35 * size_var * (1 + variation * 0.1)
            mask = (self.X/a)**2 + (self.Y/b)**2 <= 1
            img[mask] = 1.0
            
        return img


def generate_digits_dataset(output_dir: str, resolution: Tuple[int, int] = (128, 128),
                           fill_ratio: float = 0.9,
                           position_offset: Tuple[float, float] = (0.0, 0.0),
                           rotation: float = 0.0,
                           max_fonts: Optional[int] = None,
                           random_select: bool = False,
                           verbose: bool = False):
    """Generate dataset of digits 0-9 using available fonts"""
    
    os.makedirs(output_dir, exist_ok=True)
    generator = CenteredPatternGenerator(resolution, max_fonts=max_fonts, 
                                         random_select=random_select, verbose=verbose,
                                         fill_ratio=fill_ratio)
    
    n_fonts = generator.font_manager.get_num_fonts()
    if n_fonts == 0:
        print("Error: No fonts available")
        return 0
    
    print(f"Generating digits dataset with {n_fonts} fonts per digit...")
    print(f"  Fill ratio: {fill_ratio}, Position offset: {position_offset}, Rotation: {rotation}°")
    
    metadata = []
    sample_idx = 0
    
    for digit in tqdm(range(10), desc="Digits"):
        for font_idx in range(n_fonts):
            # Get font path
            font_path = generator.font_manager.get_font_path(font_idx)
            font_name = generator.font_manager.get_font_name(font_path)
            
            # Generate image - character will be scaled to fill the canvas
            img = generator.generate_character(
                str(digit), font_path,
                position_offset=position_offset,
                rotation=rotation,
                fill_ratio=fill_ratio
            )
            
            # Save as EXR: digit_{sample_idx}.exr
            filename = f"digit_{sample_idx:04d}.exr"
            filepath = os.path.join(output_dir, filename)
            save_exr(img, filepath)
            
            # Add metadata
            metadata.append({
                'index': sample_idx,
                'filename': filename,
                'type': 'digit',
                'value': digit,
                'label': str(digit),
                'font': font_name,
                'font_index': font_idx,
                'fill_ratio': fill_ratio,
                'position_offset': list(position_offset),
                'rotation': rotation
            })
            
            sample_idx += 1
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'dataset_info': {
                'name': 'digits',
                'n_samples': sample_idx,
                'n_fonts': n_fonts,
                'classes': list(range(10)),
                'resolution': resolution,
                'format': 'exr',
                'fill_ratio': fill_ratio,
                'position_offset': list(position_offset),
                'rotation': rotation,
                'created': datetime.now().isoformat()
            },
            'samples': metadata
        }, f, indent=2)
    
    print(f"Generated {sample_idx} digit images (10 digits × {n_fonts} fonts)")
    return sample_idx


def generate_letters_dataset(output_dir: str, resolution: Tuple[int, int] = (128, 128),
                            fill_ratio: float = 0.9,
                            position_offset: Tuple[float, float] = (0.0, 0.0),
                            rotation: float = 0.0,
                            max_fonts: Optional[int] = None,
                            random_select: bool = False,
                            verbose: bool = False):
    """Generate dataset of letters a-z and A-Z using available fonts"""
    
    os.makedirs(output_dir, exist_ok=True)
    generator = CenteredPatternGenerator(resolution, max_fonts=max_fonts, 
                                         random_select=random_select, verbose=verbose,
                                         fill_ratio=fill_ratio)
    
    n_fonts = generator.font_manager.get_num_fonts()
    if n_fonts == 0:
        print("Error: No fonts available")
        return 0
    
    # Include both lowercase and uppercase
    lowercase = 'abcdefghijklmnopqrstuvwxyz'
    uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    all_letters = lowercase + uppercase  # 52 characters total
    
    print(f"Generating letters dataset with {n_fonts} fonts per letter...")
    print(f"  Characters: a-z (26) + A-Z (26) = 52 characters")
    print(f"  Fill ratio: {fill_ratio}, Position offset: {position_offset}, Rotation: {rotation}°")
    
    metadata = []
    sample_idx = 0
    
    for letter in tqdm(all_letters, desc="Letters"):
        for font_idx in range(n_fonts):
            # Get font path
            font_path = generator.font_manager.get_font_path(font_idx)
            font_name = generator.font_manager.get_font_name(font_path)
            
            # Generate image - character will be scaled to fill the canvas
            img = generator.generate_character(
                letter, font_path,
                position_offset=position_offset,
                rotation=rotation,
                fill_ratio=fill_ratio
            )
            
            # Save as EXR: letter_{sample_idx}.exr
            filename = f"letter_{sample_idx:04d}.exr"
            filepath = os.path.join(output_dir, filename)
            save_exr(img, filepath)
            
            # Add metadata
            metadata.append({
                'index': sample_idx,
                'filename': filename,
                'type': 'letter',
                'value': letter,
                'label': letter,
                'is_uppercase': letter.isupper(),
                'font': font_name,
                'font_index': font_idx,
                'fill_ratio': fill_ratio,
                'position_offset': list(position_offset),
                'rotation': rotation
            })
            
            sample_idx += 1
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'dataset_info': {
                'name': 'letters',
                'n_samples': sample_idx,
                'n_fonts': n_fonts,
                'classes': list(all_letters),
                'n_lowercase': 26,
                'n_uppercase': 26,
                'resolution': resolution,
                'format': 'exr',
                'fill_ratio': fill_ratio,
                'position_offset': list(position_offset),
                'rotation': rotation,
                'created': datetime.now().isoformat()
            },
            'samples': metadata
        }, f, indent=2)
    
    print(f"Generated {sample_idx} letter images (52 letters × {n_fonts} fonts)")
    return sample_idx


def generate_shapes_dataset(output_dir: str, n_per_shape: int = 10, resolution: Tuple[int, int] = (128, 128)):
    """Generate dataset of geometric shapes with variations"""
    
    os.makedirs(output_dir, exist_ok=True)
    generator = CenteredPatternGenerator(resolution)
    
    # Define shapes to generate
    shapes = ['circle', 'square', 'triangle', 'pentagon', 'hexagon', 
              'star', 'cross', 'ring', 'diamond', 'ellipse']
    
    metadata = []
    sample_idx = 0
    
    print("Generating shapes dataset...")
    for shape_type in tqdm(shapes, desc="Shapes"):
        for i in range(n_per_shape):
            # Generate image with variation
            img = generator.generate_shape(shape_type, variation=i)
            
            # Save as EXR
            filename = f"shape_{sample_idx:04d}.exr"
            filepath = os.path.join(output_dir, filename)
            save_exr(img, filepath)
            
            # Add metadata
            metadata.append({
                'index': sample_idx,
                'filename': filename,
                'type': 'shape',
                'shape': shape_type,
                'label': shape_type,
                'variation': i
            })
            
            sample_idx += 1
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'dataset_info': {
                'name': 'shapes',
                'n_samples': sample_idx,
                'n_per_class': n_per_shape,
                'classes': shapes,
                'resolution': resolution,
                'format': 'exr',
                'created': datetime.now().isoformat()
            },
            'samples': metadata
        }, f, indent=2)
    
    print(f"Generated {sample_idx} shape images in {output_dir}")
    return sample_idx


def main():
    """Generate all three datasets"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate centered pattern datasets with diverse fonts")
    parser.add_argument('--base_dir', type=str, default='datasets', 
                       help='Base directory for datasets')
    parser.add_argument('--resolution', type=int, nargs=2, default=[128, 128],
                       help='Image resolution (width height)')
    parser.add_argument('--n_per_shape', type=int, default=10,
                       help='Number of variations per shape')
    parser.add_argument('--fill_ratio', type=float, default=0.9,
                       help='How much of the canvas characters should fill (0.1 to 1.0, default: 0.9)')
    parser.add_argument('--position_offset', type=float, nargs=2, default=[0.0, 0.0],
                       help='Character position offset as fraction (-1.0 to 1.0 for x and y, default: 0.0 0.0)')
    parser.add_argument('--rotation', type=float, default=0.0,
                       help='Character rotation angle in degrees (-180 to 180, default: 0.0)')
    parser.add_argument('--max_fonts', type=int, default=None,
                       help='Maximum number of fonts to use (default: use all available fonts)')
    parser.add_argument('--random_fonts', action='store_true',
                       help='Randomly select fonts instead of using first N fonts')
    parser.add_argument('--digits_only', action='store_true',
                       help='Generate only digits dataset')
    parser.add_argument('--letters_only', action='store_true',
                       help='Generate only letters dataset')
    parser.add_argument('--shapes_only', action='store_true',
                       help='Generate only shapes dataset')
    parser.add_argument('--list_fonts', action='store_true',
                       help='List available fonts and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output (show font validation details)')
    
    args = parser.parse_args()
    
    # List fonts if requested
    if args.list_fonts:
        fm = FontManager(verbose=True)
        print("\nAvailable fonts:")
        for i, font_path in enumerate(fm.available_fonts):
            print(f"  {i+1}. {fm.get_font_name(font_path)}")
        print(f"\nTotal: {len(fm.available_fonts)} fonts")
        print(f"\nMaximum dataset sizes (using all fonts):")
        print(f"  Digits: 10 × {len(fm.available_fonts)} = {10 * len(fm.available_fonts)} images")
        print(f"  Letters: 52 × {len(fm.available_fonts)} = {52 * len(fm.available_fonts)} images")
        print(f"\nUse --max_fonts N to limit the number of fonts used.")
        return
    
    # Validate parameters
    fill_ratio = max(0.1, min(1.0, args.fill_ratio))
    position_offset = (
        max(-1.0, min(1.0, args.position_offset[0])),
        max(-1.0, min(1.0, args.position_offset[1]))
    )
    rotation = max(-180.0, min(180.0, args.rotation))
    
    if fill_ratio != args.fill_ratio:
        print(f"Warning: fill_ratio clamped to {fill_ratio}")
    if position_offset != tuple(args.position_offset):
        print(f"Warning: position_offset clamped to {position_offset}")
    if rotation != args.rotation:
        print(f"Warning: rotation clamped to {rotation}")
    
    resolution = tuple(args.resolution)
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    total_samples = 0
    
    # Generate datasets based on flags
    generate_all = not (args.digits_only or args.letters_only or args.shapes_only)
    
    if generate_all or args.digits_only:
        digits_dir = os.path.join(args.base_dir, 'digits')
        n_digits = generate_digits_dataset(
            digits_dir, 
            resolution=resolution,
            fill_ratio=fill_ratio,
            position_offset=position_offset,
            rotation=rotation,
            max_fonts=args.max_fonts,
            random_select=args.random_fonts,
            verbose=args.verbose
        )
        total_samples += n_digits
    
    if generate_all or args.letters_only:
        letters_dir = os.path.join(args.base_dir, 'letters')
        n_letters = generate_letters_dataset(
            letters_dir, 
            resolution=resolution,
            fill_ratio=fill_ratio,
            position_offset=position_offset,
            rotation=rotation,
            max_fonts=args.max_fonts,
            random_select=args.random_fonts,
            verbose=args.verbose
        )
        total_samples += n_letters
    
    if generate_all or args.shapes_only:
        shapes_dir = os.path.join(args.base_dir, 'shapes')
        n_shapes = generate_shapes_dataset(shapes_dir, n_per_shape=args.n_per_shape, resolution=resolution)
        total_samples += n_shapes
    
    print(f"\n" + "="*60)
    print(f"Dataset generation complete!")
    print(f"Total samples generated: {total_samples}")
    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    print(f"Character parameters:")
    print(f"  Fill ratio: {fill_ratio} (characters fill {fill_ratio*100:.0f}% of canvas)")
    print(f"  Position offset: {position_offset}")
    print(f"  Rotation: {rotation}°")
    if args.max_fonts:
        print(f"  Max fonts: {args.max_fonts}" + (" (random)" if args.random_fonts else " (first N)"))
    print(f"Output directory: {args.base_dir}")
    print("="*60)


if __name__ == "__main__":
    main()