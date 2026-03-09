import cv2
import numpy as np
from pathlib import Path


def load_image(image_path: str) -> np.ndarray:
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def load_background(background_path: str, target_size: tuple = None) -> np.ndarray:
    if not Path(background_path).exists():
        raise FileNotFoundError(f"Background image not found: {background_path}")
    
    bg = cv2.imread(background_path)
    if bg is None:
        raise ValueError(f"Failed to load background: {background_path}")
    
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB) / 255.0
    
    if target_size is not None:
        bg = cv2.resize(bg, target_size)
    
    return bg


def save_image(image: np.ndarray, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)


def create_solid_background(width: int, height: int, 
                          color: tuple = (0.8, 0.9, 1.0)) -> np.ndarray:
    background = np.ones((height, width, 3), dtype=np.float32)
    background[:, :] = color
    return background


def create_gradient_background(width: int, height: int, 
                           color1: tuple = (0.2, 0.4, 0.8),
                           color2: tuple = (0.8, 0.6, 0.4)) -> np.ndarray:
    background = np.zeros((height, width, 3), dtype=np.float32)
    
    for i in range(height):
        ratio = i / height
        color = [
            color1[j] * (1 - ratio) + color2[j] * ratio
            for j in range(3)
        ]
        background[i, :] = color
    
    return background


def get_video_properties(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    properties = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    
    cap.release()
    return properties
