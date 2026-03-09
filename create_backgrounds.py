import cv2
import numpy as np
from pathlib import Path


def create_sample_backgrounds():
    backgrounds_dir = Path('backgrounds')
    backgrounds_dir.mkdir(exist_ok=True)
    
    # 创建渐变背景 - 海滩风格
    beach = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(720):
        ratio = i / 720
        color = [
            int(135 * (1 - ratio) + 255 * ratio),  # R
            int(206 * (1 - ratio) + 250 * ratio),  # G
            int(235 * (1 - ratio) + 205 * ratio)   # B
        ]
        beach[i, :] = color
    cv2.imwrite(str(backgrounds_dir / 'beach.jpg'), beach)
    print(f"Created: {backgrounds_dir / 'beach.jpg'}")
    
    # 创建渐变背景 - 日落风格
    sunset = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(720):
        ratio = i / 720
        color = [
            int(255 * (1 - ratio) + 25 * ratio),   # R
            int(140 * (1 - ratio) + 25 * ratio),  # G
            int(0 * (1 - ratio) + 112 * ratio)    # B
        ]
        sunset[i, :] = color
    cv2.imwrite(str(backgrounds_dir / 'sunset.jpg'), sunset)
    print(f"Created: {backgrounds_dir / 'sunset.jpg'}")
    
    # 创建渐变背景 - 天空风格
    sky = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(720):
        ratio = i / 720
        color = [
            int(135 * (1 - ratio) + 70 * ratio),   # R
            int(206 * (1 - ratio) + 130 * ratio),  # G
            int(250 * (1 - ratio) + 180 * ratio)  # B
        ]
        sky[i, :] = color
    cv2.imwrite(str(backgrounds_dir / 'sky.jpg'), sky)
    print(f"Created: {backgrounds_dir / 'sky.jpg'}")
    
    # 创建纯色背景 - 绿色
    green = np.full((720, 1280, 3), [34, 139, 34], dtype=np.uint8)
    cv2.imwrite(str(backgrounds_dir / 'green.jpg'), green)
    print(f"Created: {backgrounds_dir / 'green.jpg'}")
    
    # 创建纯色背景 - 蓝色
    blue = np.full((720, 1280, 3), [70, 130, 180], dtype=np.uint8)
    cv2.imwrite(str(backgrounds_dir / 'blue.jpg'), blue)
    print(f"Created: {backgrounds_dir / 'blue.jpg'}")
    
    print("\nAll sample backgrounds created successfully!")


if __name__ == '__main__':
    create_sample_backgrounds()
