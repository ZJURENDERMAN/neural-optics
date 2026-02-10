"""
EXR 图像保存工具模块

提供将 DrJit Float 数据保存为 EXR 文件的功能。

数据布局约定：
- SensorData 存储时，索引 0 对应 (u_min, v_min)，即左下角
- 图像文件约定左上角为原点
- 因此保存时需要上下翻转（垂直翻转）
"""

import numpy as np
from pathlib import Path

def _drjit_to_numpy(data):
    """
    将 DrJit 数组转换为 NumPy 数组
    
    Args:
        data: DrJit Float 类型的数组
        
    Returns:
        numpy.ndarray: 转换后的 NumPy 数组
    """
    # 尝试使用 numpy() 方法
    if hasattr(data, 'numpy'):
        return data.numpy()
    
    # 备选方案：使用 drjit.eval 同步后转换
    try:
        import drjit as dr
        dr.eval(data)
    except ImportError:
        pass
    
    return np.array(data)


def _is_array3f(data):
    """
    检查数据是否为 Array3f 类型
    
    Args:
        data: 待检查的数据
        
    Returns:
        bool: 如果是 Array3f 类型返回 True
    """
    type_name = type(data).__name__
    
    # 检查类型名是否包含 Array3f 或类似的三分量数组
    if 'Array3f' in type_name or 'Array3' in type_name:
        return True
    
    # 如果类型名是 Float 且不包含 Array，则为一维数组
    if 'Float' in type_name and 'Array' not in type_name:
        return False
    
    # 尝试安全地检查是否有 x 属性
    try:
        _ = data.x
        return True
    except (RuntimeError, AttributeError, TypeError):
        pass
    
    # 检查是否可以通过索引访问三个分量
    try:
        if len(data) == 3 and hasattr(data[0], '__len__'):
            return True
    except (TypeError, AttributeError):
        pass
    
    return False


def save_exr(data, width, height, filename, normalize=False):
    """
    将 DrJit 数据保存为 EXR 文件
    
    根据输入数据类型自动选择保存格式：
    - 一维 Float 张量：保存为单通道 (Y) EXR
    - Array3f：保存为 RGB 三通道 EXR
    
    注意：数据会被垂直翻转以匹配图像坐标系（左上角为原点）
    
    Args:
        data: drjit.cuda.ad.Float（一维张量）或 Array3f（三通道）
        width: 图像宽度
        height: 图像高度
        filename: 输出文件名 (如 "output.exr")
        normalize: 是否归一化到 [0, 1] 范围
        
    Returns:
        bool: 保存成功返回 True，失败返回 False
    """
    try:
        import OpenEXR
        import Imath
    except ImportError:
        print("请安装 OpenEXR: pip install OpenEXR")
        return False
    
    # 判断数据类型
    is_rgb = _is_array3f(data)
    
    if is_rgb:
        return _save_exr_rgb(data, width, height, filename, normalize)
    else:
        return _save_exr_grayscale(data, width, height, filename, normalize)


def _save_exr_grayscale(data, width, height, filename, normalize=False):
    """
    保存单通道 EXR 文件
    
    Args:
        data: drjit.cuda.ad.Float 类型的一维张量
        width: 图像宽度
        height: 图像高度
        filename: 输出文件名
        normalize: 是否归一化
        
    Returns:
        bool: 保存成功返回 True
    """
    import OpenEXR
    import Imath
    
    # 1. DrJit Float -> NumPy 数组
    np_data = _drjit_to_numpy(data)
    
    # 2. 重塑为 (height, width) 的二维数组
    np_data = np_data.reshape(height, width)
    
    # 3. 垂直翻转：将左下角原点转换为左上角原点
    np_data = np.flipud(np_data)
    
    # 4. 可选：归一化
    if normalize and np_data.max() > 0:
        np_data = np_data / np_data.max()
    
    # 5. 转换为 float32（确保连续内存布局）
    np_data = np.ascontiguousarray(np_data.astype(np.float32))
    
    # 6. 保存为单通道 EXR (Y 通道)
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    
    # 将数据转换为字节
    channel_data = np_data.tobytes()
    
    # 创建 EXR 文件
    exr_file = OpenEXR.OutputFile(filename, header)
    exr_file.writePixels({
        'Y': channel_data
    })
    exr_file.close()
    
    # print(f"已保存 EXR 文件: {filename}")
    # print(f"  尺寸: {width} x {height}")
    # print(f"  模式: 单通道 (Y)")
    # print(f"  数据范围: [{np_data.min():.6f}, {np_data.max():.6f}]")
    return True


def _save_exr_rgb(data, width, height, filename, normalize=False):
    """
    保存 RGB 三通道 EXR 文件
    
    Args:
        data: Array3f 类型的数据
        width: 图像宽度
        height: 图像高度
        filename: 输出文件名
        normalize: 是否归一化
        
    Returns:
        bool: 保存成功返回 True
    """
    import OpenEXR
    import Imath
    
    # 1. 提取三个通道
    try:
        # Array3f 使用 .x, .y, .z 访问
        r_data = _drjit_to_numpy(data.x)
        g_data = _drjit_to_numpy(data.y)
        b_data = _drjit_to_numpy(data.z)
    except (RuntimeError, AttributeError):
        # 通过索引访问
        r_data = _drjit_to_numpy(data[0])
        g_data = _drjit_to_numpy(data[1])
        b_data = _drjit_to_numpy(data[2])
    
    # 2. 重塑为 (height, width) 的二维数组
    r_data = r_data.reshape(height, width)
    g_data = g_data.reshape(height, width)
    b_data = b_data.reshape(height, width)
    
    # 3. 垂直翻转：将左下角原点转换为左上角原点
    r_data = np.flipud(r_data)
    g_data = np.flipud(g_data)
    b_data = np.flipud(b_data)
    
    # 4. 可选：归一化（使用三通道的最大值）
    if normalize:
        max_val = max(r_data.max(), g_data.max(), b_data.max())
        if max_val > 0:
            r_data = r_data / max_val
            g_data = g_data / max_val
            b_data = b_data / max_val
    
    # 5. 转换为 float32（确保连续内存布局）
    r_data = np.ascontiguousarray(r_data.astype(np.float32))
    g_data = np.ascontiguousarray(g_data.astype(np.float32))
    b_data = np.ascontiguousarray(b_data.astype(np.float32))
    
    # 6. 保存为 RGB EXR
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    
    # 将数据转换为字节
    r_bytes = r_data.tobytes()
    g_bytes = g_data.tobytes()
    b_bytes = b_data.tobytes()
    
    # 创建 EXR 文件
    exr_file = OpenEXR.OutputFile(filename, header)
    exr_file.writePixels({
        'R': r_bytes,
        'G': g_bytes,
        'B': b_bytes
    })
    exr_file.close()
    
    print(f"已保存 EXR 文件: {filename}")
    print(f"  尺寸: {width} x {height}")
    print(f"  模式: RGB")
    print(f"  R 范围: [{r_data.min():.6f}, {r_data.max():.6f}]")
    print(f"  G 范围: [{g_data.min():.6f}, {g_data.max():.6f}]")
    print(f"  B 范围: [{b_data.min():.6f}, {b_data.max():.6f}]")
    return True


def save_png(data, width, height, filename, normalize=True):
    """
    将 DrJit 数据保存为 PNG 文件
    
    根据输入数据类型自动选择保存格式：
    - 一维 Float 张量：保存为灰度 PNG
    - Array3f：保存为 RGB PNG
    
    注意：数据会被垂直翻转以匹配图像坐标系（左上角为原点）
    
    Args:
        data: drjit.cuda.ad.Float（一维张量）或 Array3f（三通道）
        width: 图像宽度
        height: 图像高度
        filename: 输出文件名 (如 "output.png")
        normalize: 是否归一化到 [0, 1] 范围后再映射到 [0, 255]
        
    Returns:
        bool: 保存成功返回 True，失败返回 False
    """
    try:
        from PIL import Image
    except ImportError:
        print("请安装 Pillow: pip install Pillow")
        return False
    
    # 判断数据类型
    is_rgb = _is_array3f(data)
    
    if is_rgb:
        return _save_png_rgb(data, width, height, filename, normalize)
    else:
        return _save_png_grayscale(data, width, height, filename, normalize)


def _save_png_grayscale(data, width, height, filename, normalize=True):
    """
    保存灰度 PNG 文件
    
    Args:
        data: drjit.cuda.ad.Float 类型的一维张量
        width: 图像宽度
        height: 图像高度
        filename: 输出文件名
        normalize: 是否归一化
        
    Returns:
        bool: 保存成功返回 True
    """
    from PIL import Image
    
    # 1. DrJit Float -> NumPy 数组
    np_data = _drjit_to_numpy(data)
    
    # 2. 重塑为 (height, width) 的二维数组
    np_data = np_data.reshape(height, width)
    
    # 3. 垂直翻转：将左下角原点转换为左上角原点
    np_data = np.flipud(np_data)
    
    # 4. 归一化并映射到 [0, 255]
    if normalize:
        max_val = np_data.max()
        if max_val > 0:
            np_data = np_data / max_val
    
    # 裁剪到 [0, 1] 范围并映射到 [0, 255]
    np_data = np.clip(np_data, 0, 1)
    np_data = (np_data * 255).astype(np.uint8)
    
    # 5. 保存为 PNG
    img = Image.fromarray(np_data, mode='L')
    img.save(filename)
    
    return True


def _save_png_rgb(data, width, height, filename, normalize=True):
    """
    保存 RGB PNG 文件
    
    Args:
        data: Array3f 类型的数据
        width: 图像宽度
        height: 图像高度
        filename: 输出文件名
        normalize: 是否归一化
        
    Returns:
        bool: 保存成功返回 True
    """
    from PIL import Image
    
    # 1. 提取三个通道
    try:
        r_data = _drjit_to_numpy(data.x)
        g_data = _drjit_to_numpy(data.y)
        b_data = _drjit_to_numpy(data.z)
    except (RuntimeError, AttributeError):
        r_data = _drjit_to_numpy(data[0])
        g_data = _drjit_to_numpy(data[1])
        b_data = _drjit_to_numpy(data[2])
    
    # 2. 重塑为 (height, width) 的二维数组
    r_data = r_data.reshape(height, width)
    g_data = g_data.reshape(height, width)
    b_data = b_data.reshape(height, width)
    
    # 3. 垂直翻转：将左下角原点转换为左上角原点
    r_data = np.flipud(r_data)
    g_data = np.flipud(g_data)
    b_data = np.flipud(b_data)
    
    # 4. 归一化（使用三通道的最大值）
    if normalize:
        max_val = max(r_data.max(), g_data.max(), b_data.max())
        if max_val > 0:
            r_data = r_data / max_val
            g_data = g_data / max_val
            b_data = b_data / max_val
    
    # 5. 裁剪到 [0, 1] 范围并映射到 [0, 255]
    r_data = np.clip(r_data, 0, 1)
    g_data = np.clip(g_data, 0, 1)
    b_data = np.clip(b_data, 0, 1)
    
    r_data = (r_data * 255).astype(np.uint8)
    g_data = (g_data * 255).astype(np.uint8)
    b_data = (b_data * 255).astype(np.uint8)
    
    # 6. 组合为 RGB 数组并保存
    rgb_data = np.stack([r_data, g_data, b_data], axis=2)
    img = Image.fromarray(rgb_data, mode='RGB')
    img.save(filename)
    
    return True


def load_exr(filename, regularize=True):
    """
    从 EXR 文件加载数据为 DrJit 数组
    
    根据文件通道自动判断返回格式：
    - 单通道 (Y)：返回 (data, width, height)
    - 多通道 (RGB)：返回 (r, g, b, width, height)
    
    注意：加载时会垂直翻转以匹配 SensorData 的存储约定（左下角为原点）
    
    Args:
        filename: EXR 文件路径
        regularize: 是否进行能量归一化。如果为 True，将数据除以总能量，
                    使得所有像素值之和为 1。对于 RGB 图像，三通道能量总和为 1。
        
    Returns:
        如果原图为单通道:
            tuple: (data, width, height)
                   data 为 drjit.cuda.ad.Float，长度为 width * height
        如果原图为多通道:
            tuple: (r, g, b, width, height)
                   r, g, b 各为 drjit.cuda.ad.Float，长度为 width * height
        
    Raises:
        ImportError: 如果 OpenEXR 未安装
        FileNotFoundError: 如果文件不存在
        RuntimeError: 如果读取失败
    """
    try:
        import OpenEXR
        import Imath
    except ImportError:
        raise ImportError("请安装 OpenEXR: pip install OpenEXR")
    
    import os
    import drjit as dr
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"EXR 文件不存在: {filename}")
    
    # 打开 EXR 文件
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    
    # 获取图像尺寸
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 获取可用通道
    channels = header['channels']
    available_channels = list(channels.keys())
    
    # 确定像素类型
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    def read_channel(channel_name):
        """读取单个通道并转换为 numpy 数组，同时垂直翻转"""
        raw_data = exr_file.channel(channel_name, pt)
        np_data = np.frombuffer(raw_data, dtype=np.float32)
        np_data = np_data.reshape(height, width)
        # 垂直翻转：将左上角原点转换为左下角原点
        np_data = np.flipud(np_data)
        return np_data
    
    # 判断通道情况
    has_rgb = all(c in available_channels for c in ['R', 'G', 'B'])
    has_y = 'Y' in available_channels
    
    # 转换为 DrJit Float
    try:
        from drjit.cuda.ad import Float
    except ImportError:
        from drjit.llvm.ad import Float
    
    if has_y and not has_rgb:
        # 单通道文件
        np_data = read_channel('Y')
        
        if regularize:
            total_energy = np_data.sum()
            if total_energy > 0:
                np_data = np_data / total_energy
                print(f"  已进行能量归一化，原始总能量: {total_energy:.6f}")
        
        data = Float(np_data.flatten())
        
        print(f"已加载 EXR 文件: {filename}")
        print(f"  尺寸: {width} x {height}")
        print(f"  模式: 单通道 (Y)")
        print(f"  数据范围: [{np_data.min():.6f}, {np_data.max():.6f}]")
        
        return data, width, height

    elif has_rgb:
        # RGB 多通道文件
        r_np = read_channel('R')
        g_np = read_channel('G')
        b_np = read_channel('B')
        
        if regularize:
            total_energy = r_np.sum() + g_np.sum() + b_np.sum()
            if total_energy > 0:
                r_np = r_np / total_energy
                g_np = g_np / total_energy
                b_np = b_np / total_energy
                print(f"  已进行能量归一化，原始总能量: {total_energy:.6f}")
        
        r = Float(r_np.flatten())
        g = Float(g_np.flatten())
        b = Float(b_np.flatten())
        
        print(f"已加载 EXR 文件: {filename}")
        print(f"  尺寸: {width} x {height}")
        print(f"  模式: RGB")
        print(f"  R 范围: [{r_np.min():.6f}, {r_np.max():.6f}]")
        print(f"  G 范围: [{g_np.min():.6f}, {g_np.max():.6f}]")
        print(f"  B 范围: [{b_np.min():.6f}, {b_np.max():.6f}]")
        
        return r, g, b, width, height
    
    else:
        raise RuntimeError(f"EXR 文件没有可用通道: {filename}")


def _resize_numpy_image(np_data, target_width, target_height):
    """
    使用 PIL 对 numpy 数组进行缩放
    
    Args:
        np_data: numpy 数组，形状为 (H, W) 或 (H, W, C)
        target_width: 目标宽度
        target_height: 目标高度
        
    Returns:
        numpy.ndarray: 缩放后的数组
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("请安装 Pillow: pip install Pillow")
    
    original_shape = np_data.shape
    is_grayscale = len(original_shape) == 2
    
    if is_grayscale:
        # 灰度图像
        img = Image.fromarray(np_data, mode='F')  # 'F' 表示 32 位浮点
        img_resized = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
        return np.array(img_resized, dtype=np.float32)
    else:
        # RGB 图像，需要分通道处理以保持浮点精度
        channels = []
        for c in range(np_data.shape[2]):
            img = Image.fromarray(np_data[:, :, c], mode='F')
            img_resized = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
            channels.append(np.array(img_resized, dtype=np.float32))
        return np.stack(channels, axis=2)


def _load_standard_image(filename, desired_channels=1, regularize=True, desired_resolution=None):
    """
    加载标准图像格式（PNG, JPG, JPEG, BMP, TIFF 等）
    
    Args:
        filename: 图像文件路径
        desired_channels: 期望的输出通道数，1 为灰度，3 为 RGB
        regularize: 是否进行能量归一化
        desired_resolution: 目标分辨率 [width, height]，None 表示保持原始尺寸
        
    Returns:
        如果 desired_channels=1:
            tuple: (data, width, height)
        如果 desired_channels=3:
            tuple: (r, g, b, width, height)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("请安装 Pillow: pip install Pillow")
    
    # 打开图像
    img = Image.open(filename)
    
    # 获取原始尺寸
    original_width, original_height = img.size
    
    # 确定目标尺寸
    if desired_resolution is not None:
        target_width, target_height = desired_resolution
    else:
        target_width, target_height = original_width, original_height
    
    # 根据期望通道数转换
    if desired_channels == 1:
        # 转换为灰度
        if img.mode != 'L':
            img = img.convert('L')
        
        # 转换为 numpy 数组
        np_data = np.array(img, dtype=np.float32)
        
        # 归一化到 [0, 1]（原始范围是 [0, 255]）
        np_data = np_data / 255.0
        
        # 缩放（在翻转之前进行，因为缩放是对图像坐标系操作）
        if target_width != original_width or target_height != original_height:
            np_data = _resize_numpy_image(np_data, target_width, target_height)
            print(f"  已缩放: {original_width}x{original_height} -> {target_width}x{target_height}")
        
        # 垂直翻转：将左上角原点转换为左下角原点
        np_data = np.flipud(np_data)
        
        # 能量归一化
        if regularize:
            total_energy = np_data.sum()
            if total_energy > 0:
                np_data = np_data / total_energy
                print(f"  已进行能量归一化，原始总能量: {total_energy:.6f}")
        
        # 转换为 DrJit Float
        try:
            from drjit.cuda.ad import Float
        except ImportError:
            from drjit.llvm.ad import Float
        
        data = Float(np_data.flatten())
        
        print(f"已加载图像: {filename}")
        print(f"  尺寸: {target_width} x {target_height}")
        print(f"  模式: 灰度")
        print(f"  数据范围: [{np_data.min():.6f}, {np_data.max():.6f}]")
        
        return data, target_width, target_height
    
    elif desired_channels == 3:
        # 转换为 RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 转换为 numpy 数组
        np_data = np.array(img, dtype=np.float32)
        
        # 归一化到 [0, 1]
        np_data = np_data / 255.0
        
        # 缩放
        if target_width != original_width or target_height != original_height:
            np_data = _resize_numpy_image(np_data, target_width, target_height)
            print(f"  已缩放: {original_width}x{original_height} -> {target_width}x{target_height}")
        
        # 垂直翻转
        np_data = np.flipud(np_data)
        
        # 分离通道
        r_np = np_data[:, :, 0]
        g_np = np_data[:, :, 1]
        b_np = np_data[:, :, 2]
        
        # 能量归一化
        if regularize:
            total_energy = r_np.sum() + g_np.sum() + b_np.sum()
            if total_energy > 0:
                r_np = r_np / total_energy
                g_np = g_np / total_energy
                b_np = b_np / total_energy
                print(f"  已进行能量归一化，原始总能量: {total_energy:.6f}")
        
        # 转换为 DrJit Float
        try:
            from drjit.cuda.ad import Float
        except ImportError:
            from drjit.llvm.ad import Float
        
        r = Float(r_np.flatten())
        g = Float(g_np.flatten())
        b = Float(b_np.flatten())
        
        print(f"已加载图像: {filename}")
        print(f"  尺寸: {target_width} x {target_height}")
        print(f"  模式: RGB")
        print(f"  R 范围: [{r_np.min():.6f}, {r_np.max():.6f}]")
        print(f"  G 范围: [{g_np.min():.6f}, {g_np.max():.6f}]")
        print(f"  B 范围: [{b_np.min():.6f}, {b_np.max():.6f}]")
        
        return r, g, b, target_width, target_height
    
    else:
        raise ValueError(f"desired_channels 必须为 1 或 3，当前值: {desired_channels}")


def load_image(file_path, desired_channels=1, regularize=True, desired_resolution=[128,128]):
    """
    通用图像加载接口，支持多种格式
    
    根据文件后缀名自动选择加载方式：
    - .exr: 使用 OpenEXR 加载
    - .png, .jpg, .jpeg, .bmp, .tiff, .tif: 使用 Pillow 加载
    - .xlsx: 使用 pandas 加载 Excel 表格
    
    Args:
        file_path: 图像文件路径
        desired_channels: 期望的输出通道数
            - 1: 返回单通道灰度数据
            - 3: 返回 RGB 三通道数据
        regularize: 是否进行能量归一化。如果为 True，将数据除以总能量，
                    使得所有像素值之和为 1。
        desired_resolution: 目标分辨率 [width, height]，默认为 None（保持原始尺寸）
                           如果指定，图像将被缩放到该分辨率
        
    Returns:
        如果 desired_channels=1:
            tuple: (data, width, height)
                   data 为 drjit.cuda.ad.Float，长度为 width * height
        如果 desired_channels=3:
            tuple: (r, g, b, width, height)
                   r, g, b 各为 drjit.cuda.ad.Float，长度为 width * height
        
    Raises:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果文件格式不支持或参数错误
        
    Examples:
        >>> # 加载灰度图像，缩放到 128x128
        >>> data, w, h = load_image("target.png", desired_channels=1, desired_resolution=[128, 128])
        
        >>> # 加载 RGB 图像，保持原始尺寸
        >>> r, g, b, w, h = load_image("target.jpg", desired_channels=3)
        
        >>> # 加载 EXR 文件，缩放到 256x256
        >>> data, w, h = load_image("target.exr", desired_channels=1, desired_resolution=[256, 256])
    """
    import os
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 获取文件后缀名（小写）
    suffix = Path(file_path).suffix.lower()
    
    # EXR 格式
    if suffix == '.exr':
        result = load_exr(file_path, regularize=regularize)
        
        # load_exr 返回格式可能是 (data, w, h) 或 (r, g, b, w, h)
        if len(result) == 3:
            # 单通道
            data, width, height = result
            np_data = _drjit_to_numpy(data).reshape(height, width)
        else:
            # 三通道
            r, g, b, width, height = result
            r_np = _drjit_to_numpy(r).reshape(height, width)
            g_np = _drjit_to_numpy(g).reshape(height, width)
            b_np = _drjit_to_numpy(b).reshape(height, width)
            np_data = np.stack([r_np, g_np, b_np], axis=2)
        
        # 确定目标尺寸
        if desired_resolution is not None:
            target_width, target_height = desired_resolution
        else:
            target_width, target_height = width, height
        
        # 缩放（如果需要）
        if target_width != width or target_height != height:
            np_data = _resize_numpy_image(np_data, target_width, target_height)
            print(f"  已缩放: {width}x{height} -> {target_width}x{target_height}")
            width, height = target_width, target_height
        
        # 转换为 DrJit Float
        try:
            from drjit.cuda.ad import Float
        except ImportError:
            from drjit.llvm.ad import Float
        
        if len(np_data.shape) == 2:
            # 单通道
            if desired_channels == 1:
                return Float(np_data.flatten()), width, height
            else:
                data = Float(np_data.flatten())
                return data, data, data, width, height
        else:
            # 三通道
            r_np = np_data[:, :, 0]
            g_np = np_data[:, :, 1]
            b_np = np_data[:, :, 2]
            
            if desired_channels == 3:
                return Float(r_np.flatten()), Float(g_np.flatten()), Float(b_np.flatten()), width, height
            else:
                # 转换为灰度
                gray_np = 0.299 * r_np + 0.587 * g_np + 0.114 * b_np
                if regularize:
                    total = gray_np.sum()
                    if total > 0:
                        gray_np = gray_np / total
                return Float(gray_np.flatten()), width, height
    
    # Excel 格式
    elif suffix == '.xlsx':
        np_data, x_coords, y_coords = xlsx_to_numpy(file_path)
        height, width = np_data.shape
        
        # 确定目标尺寸
        if desired_resolution is not None:
            target_width, target_height = desired_resolution
        else:
            target_width, target_height = width, height
        
        # 缩放（如果需要）
        if target_width != width or target_height != height:
            np_data = _resize_numpy_image(np_data, target_width, target_height)
            print(f"  已缩放: {width}x{height} -> {target_width}x{target_height}")
            width, height = target_width, target_height
        
        if regularize:
            total_energy = np_data.sum()
            if total_energy > 0:
                np_data = np_data / total_energy
                print(f"  已进行能量归一化，原始总能量: {total_energy:.6f}")
        
        try:
            from drjit.cuda.ad import Float
        except ImportError:
            from drjit.llvm.ad import Float
        
        data = Float(np_data.flatten())
        
        print(f"已加载 Excel 文件: {file_path}")
        print(f"  尺寸: {width} x {height}")
        print(f"  数据范围: [{np_data.min():.6f}, {np_data.max():.6f}]")
        
        if desired_channels == 1:
            return data, width, height
        else:
            return data, data, data, width, height
    
    # 标准图像格式
    elif suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp']:
        return _load_standard_image(file_path, desired_channels, regularize, desired_resolution)
    
    else:
        raise ValueError(f"不支持的文件格式: {suffix}\n"
                        f"支持的格式: .exr, .png, .jpg, .jpeg, .bmp, .tiff, .tif, .xlsx")


import pandas as pd

def xlsx_to_exr(xlsx_filename, exr_filename, sheet_name=0):
    """
    将 Excel 表格转换为 EXR 文件
    
    Args:
        xlsx_filename: Excel 文件路径
        exr_filename: 输出 EXR 文件路径
        sheet_name: 工作表名称或索引，默认为第一个工作表
    """
    try:
        import OpenEXR
        import Imath
    except ImportError:
        print("请安装 OpenEXR: pip install OpenEXR")
        return False
    
    # 1. 读取 Excel 文件
    df = pd.read_excel(xlsx_filename, sheet_name=sheet_name, header=0, index_col=0)
    
    # 2. 提取数据（忽略第一行和第一列的表头）
    # pandas 已经自动将第一行作为列名，第一列作为索引
    np_data = df.values.astype(np.float32)
    
    # 3. 获取尺寸
    height, width = np_data.shape
    
    print(f"数据尺寸: {width} x {height}")
    print(f"X 范围: [{df.columns.min()}, {df.columns.max()}]")
    print(f"Y 范围: [{df.index.min()}, {df.index.max()}]")
    print(f"数据范围: [{np_data.min():.6f}, {np_data.max():.6f}]")
    
    # 4. 垂直翻转
    # Excel 中 Y 从大到小排列（第一行是最大 Y）
    # EXR 图像左上角为原点，所以需要翻转使左下角对应最小 Y
    # 实际上 Excel 已经是从大 Y 到小 Y，这正好对应图像从上到下
    # 如果需要让 EXR 的左下角对应 (X_min, Y_min)，则需要翻转
    # np_data = np.flipud(np_data)
    
    # 5. 确保连续内存布局
    np_data = np.ascontiguousarray(np_data)
    
    # 6. 保存为单通道 EXR
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    
    channel_data = np_data.tobytes()
    
    exr_file = OpenEXR.OutputFile(exr_filename, header)
    exr_file.writePixels({'Y': channel_data})
    exr_file.close()
    
    print(f"\n已保存 EXR 文件: {exr_filename}")
    print(f"  尺寸: {width} x {height}")
    print(f"  模式: 单通道 (Y)")
    
    return True


def xlsx_to_numpy(xlsx_filename, sheet_name=0):
    """
    将 Excel 表格转换为 NumPy 数组，同时返回坐标信息
    
    Args:
        xlsx_filename: Excel 文件路径
        sheet_name: 工作表名称或索引
        
    Returns:
        tuple: (data, x_coords, y_coords)
            - data: (height, width) 的 numpy 数组
            - x_coords: X 坐标数组
            - y_coords: Y 坐标数组（从小到大排列）
    """
    df = pd.read_excel(xlsx_filename, sheet_name=sheet_name, header=0, index_col=0)
    
    # 提取坐标
    x_coords = df.columns.values.astype(np.float64)
    y_coords = df.index.values.astype(np.float64)
    
    # 提取数据
    np_data = df.values.astype(np.float32)
    
    # Y 坐标在 Excel 中是从大到小排列的，翻转使其从小到大
    y_coords = y_coords[::-1]
    np_data = np.flipud(np_data)
    
    return np_data, x_coords, y_coords