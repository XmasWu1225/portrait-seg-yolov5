import cv2
import numpy as np
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class ImagePreprocessor:
    def __init__(self):
        self.target_size = (Config.OUTPUT_WIDTH, Config.OUTPUT_HEIGHT)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        return image
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        if target_size is None:
            target_size = self.target_size
        
        return cv2.resize(image, target_size)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0
    
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        image = self.preprocess(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def preprocess_for_display(self, image: np.ndarray) -> np.ndarray:
        image = self.preprocess(image)  # 处理通道数
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # <-- 必须添加这一行，转为 RGB
        image = self.normalize_image(image)
        image = self.resize_image(image)
        return image
