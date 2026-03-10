import cv2
import numpy as np
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class MaskProcessor:
    def __init__(self):
        self.threshold = Config.MASK_THRESHOLD
        self.blur_kernel = Config.GAUSSIAN_BLUR_KERNEL
        self.blur_sigma = Config.GAUSSIAN_BLUR_SIGMA
    
    def process_mask(self, mask: np.ndarray, target_size: tuple = None) -> np.ndarray:
        if mask is None: return None
        
        # 1. 此时 mask 是 160x160 (模型原始输出分辨率)
        # 在这个极小的分辨率下做填充和扩张，速度极快！
        mask_uint8 = (mask > self.threshold).astype(np.uint8) * 255
        
        # 小图填充
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_uint8, contours, -1, 255, -1)
        
        # 小图扩张
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # 核也要变小
        mask_uint8 = cv2.dilate(mask_uint8, kernel_dilate, iterations=1)
        
        # 2. 放大到目标尺寸
        if target_size is not None:
            mask_uint8 = cv2.resize(mask_uint8, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 3. 最后在大图上做一次轻微模糊
        mask_float = mask_uint8.astype(np.float32) / 255.0
        mask_final = cv2.GaussianBlur(mask_float, (15, 15), 3.0)
        
        return mask_final
    
    def refine_mask_edges(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        if mask is None:
            return None
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            mask_grabcut, bgd_model, fgd_model = cv2.grabCut(
                image, mask_uint8, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK
            )
            
            mask_grabcut = np.where((mask_grabcut == 2) | (mask_grabcut == 0), 0, 1).astype(np.float32)
            return mask_grabcut
        except:
            return mask
    
    def remove_small_objects(self, mask: np.ndarray, min_size: int = 100) -> np.ndarray:
        if mask is None:
            return None
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                mask[labels == i] = 0
        
        return mask
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        if mask is None:
            return None
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            cv2.drawContours(mask_uint8, [cnt], 0, 255, -1)
        
        return mask_uint8.astype(np.float32) / 255.0
