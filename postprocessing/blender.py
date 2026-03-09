import cv2
import numpy as np
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class Blender:
    def __init__(self):
        self.output_width = Config.OUTPUT_WIDTH
        self.output_height = Config.OUTPUT_HEIGHT
    
    def blend_images(self, foreground: np.ndarray, mask: np.ndarray,
                    background: np.ndarray) -> np.ndarray:
        if mask is None:
            return background
        
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        result = (foreground * mask) + (background * (1 - mask))
        return np.clip(result, 0, 1)
    
    def resize_for_output(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.output_width, self.output_height))
    
    def convert_to_uint8(self, image: np.ndarray) -> np.ndarray:
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    def convert_to_bgr(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    
    def prepare_output(self, image: np.ndarray) -> np.ndarray:
        image = self.convert_to_uint8(image)
        image = self.convert_to_bgr(image)
        return image
