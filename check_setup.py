#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def check_imports():
    print("检查模块导入...")
    
    try:
        from config import Config
        print("✓ config.py")
    except Exception as e:
        print(f"✗ config.py: {e}")
        return False
    
    try:
        from models.yolov5_seg_detector import YOLOv5SegDetector
        print("✓ models.yolov5_seg_detector")
    except Exception as e:
        print(f"✗ models.yolov5_seg_detector: {e}")
        return False
    
    try:
        from preprocessing.image_preprocessor import ImagePreprocessor
        print("✓ preprocessing.image_preprocessor")
    except Exception as e:
        print(f"✗ preprocessing.image_preprocessor: {e}")
        return False
    
    try:
        from postprocessing.mask_processor import MaskProcessor
        print("✓ postprocessing.mask_processor")
    except Exception as e:
        print(f"✗ postprocessing.mask_processor: {e}")
        return False
    
    try:
        from postprocessing.filter_processor import FilterProcessor
        print("✓ postprocessing.filter_processor")
    except Exception as e:
        print(f"✗ postprocessing.filter_processor: {e}")
        return False
    
    try:
        from postprocessing.blender import Blender
        print("✓ postprocessing.blender")
    except Exception as e:
        print(f"✗ postprocessing.blender: {e}")
        return False
    
    try:
        from utils import load_image, FPSCounter, Timer
        print("✓ utils")
    except Exception as e:
        print(f"✗ utils: {e}")
        return False
    
    return True


def check_directories():
    print("\n检查目录结构...")
    
    required_dirs = [
        'models',
        'preprocessing',
        'postprocessing',
        'utils',
        'backgrounds',
        'test_images'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (不存在)")
            all_exist = False
    
    return all_exist


def check_files():
    print("\n检查文件...")
    
    required_files = [
        'config.py',
        'main.py',
        'requirements.txt',
        'README.md',
        'models/__init__.py',
        'models/yolov5_seg_detector.py',
        'preprocessing/__init__.py',
        'preprocessing/image_preprocessor.py',
        'postprocessing/__init__.py',
        'postprocessing/mask_processor.py',
        'postprocessing/filter_processor.py',
        'postprocessing/blender.py',
        'utils/__init__.py',
        'utils/image_utils.py',
        'utils/timing_utils.py'
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists() and file_path.is_file():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} (不存在)")
            all_exist = False
    
    return all_exist


def main():
    print("="*60)
    print("Portrait Segmentation with YOLOv5-Seg - 项目检查")
    print("="*60)
    
    dirs_ok = check_directories()
    files_ok = check_files()
    imports_ok = check_imports()
    
    print("\n" + "="*60)
    if dirs_ok and files_ok and imports_ok:
        print("✓ 所有检查通过！项目结构完整。")
        print("\n下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 创建背景图片: python create_backgrounds.py")
        print("3. 运行程序: python main.py --mode webcam")
    else:
        print("✗ 检查失败，请修复上述问题。")
    print("="*60)


if __name__ == '__main__':
    main()
