"""环境设置和DrJit初始化"""

import sys
import os

# 在添加任何路径之前，先导入并缓存标准库模块
import random as _builtin_random
sys.modules['_builtin_random'] = _builtin_random

_initialized = False
_dr = None
_do = None
_Float = None
_UInt32 = None
_Adam = None
_project_root = None


def setup_environment(project_root=None):
    """设置环境路径并初始化DrJit"""
    global _initialized, _dr, _do, _Float, _UInt32, _Adam, _project_root
    
    if _initialized:
        return
    
    # 自动检测项目根目录
    if project_root is None:
        project_root = os.environ.get('OPTICAL_PROJECT_ROOT')
    
    if project_root is None:
        current = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            build_dir = os.path.join(current, "build")
            if os.path.isdir(build_dir):
                for f in os.listdir(build_dir):
                    if f.startswith("diff_optics") and f.endswith(".so"):
                        project_root = current
                        break
            if project_root:
                break
            current = os.path.dirname(current)
    
    if project_root is None:
        raise RuntimeError("无法找到项目根目录，请设置 OPTICAL_PROJECT_ROOT 环境变量")
    
    _project_root = project_root
    print(f"项目根目录: {_project_root}")
    
    build_dir = os.path.join(project_root, "build")
    
    # 添加 drjit Python 模块路径（必须在 import drjit 之前）
    drjit_python_path = os.path.join(project_root, "build_cmake", "drjit")
    if drjit_python_path not in sys.path:
        sys.path.insert(0, drjit_python_path)
    
    # 添加 build 目录到 Python 路径（diff_optics 模块）
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    
    # 设置 LD_LIBRARY_PATH（动态库搜索路径）
    lib_paths = [
        os.path.join(project_root, "build_cmake", "drjit", "drjit"),
        os.path.join(project_root, "build_cmake", "drjit", "ext", "drjit-core"),
        build_dir,
    ]
    existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_paths = [p for p in lib_paths if os.path.exists(p)]
    if new_paths:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths) + ':' + existing_ld_path
    
    # 确保 random 模块不被覆盖
    sys.modules['random'] = _builtin_random
    
    # 导入 DrJit
    print("正在初始化 DrJit...")
    import drjit as dr
    dr.set_backend('cuda')
    dr.set_flag(dr.JitFlag.ForceOptiX, True)
    
    print("正在加载 diff_optics 模块...")
    import diff_optics as do
    
    from drjit.cuda.ad import Float, UInt32
    from drjit.opt import Adam
    
    _dr = dr
    _do = do
    _Float = Float
    _UInt32 = UInt32
    _Adam = Adam
    _initialized = True
    
    # 验证 PTX 文件
    ptx_file = os.path.join(build_dir, "optix_kernels.ptx")
    if not os.path.exists(ptx_file):
        ptx_file_alt = os.path.join(project_root, "build_cmake", "ptx", "optix_kernels.ptx")
        if os.path.exists(ptx_file_alt):
            print(f"PTX 文件: {ptx_file_alt}")
        else:
            raise RuntimeError(f"PTX 文件未找到: {ptx_file}")
    else:
        print(f"PTX 文件: {ptx_file}")
    
    print("环境初始化成功!")


def get_drjit():
    if not _initialized:
        setup_environment()
    return _dr


def get_diff_optics():
    if not _initialized:
        setup_environment()
    return _do


def get_Float():
    if not _initialized:
        setup_environment()
    return _Float


def get_UInt32():
    if not _initialized:
        setup_environment()
    return _UInt32


def get_Adam():
    if not _initialized:
        setup_environment()
    return _Adam


def get_project_root():
    if not _initialized:
        setup_environment()
    return _project_root


def is_initialized():
    return _initialized


if __name__ == "__main__":
    setup_environment()
    print("\n测试 DrJit:")
    dr = get_drjit()
    Float = get_Float()
    x = Float([1.0, 2.0, 3.0])
    print(f"  Float 数组: {x}")
    print(f"  求和: {dr.sum(x)}")
    
    print("\n测试 diff_optics:")
    do = get_diff_optics()
    print(f"  模块: {do}")