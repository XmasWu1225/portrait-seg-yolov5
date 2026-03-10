import cv2
import numpy as np
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class FilterProcessor:
    def __init__(self):
        self.current_filter = Config.DEFAULT_FILTER
    
    def set_filter(self, filter_name: str):
        self.current_filter = filter_name
    
    def apply_filter(self, image: np.ndarray, mask: np.ndarray, 
                   background: np.ndarray) -> np.ndarray:
        if self.current_filter == 'color_transfer':
            return self.color_transfer(background, image)
        elif self.current_filter == 'smooth_step':
            return self.smooth_step_blend(image, mask, background)
        elif self.current_filter == 'seamless_clone':
            return self.seamless_clone(image, mask, background)
        elif self.current_filter == 'harmonize':
            return self.harmonize(image, mask, background)
        else:
            return self.alpha_blend(image, mask, background)
    
    def alpha_blend(self, foreground: np.ndarray, mask: np.ndarray, 
                   background: np.ndarray) -> np.ndarray:
        # 这种写法在 Numpy 中会创建多个中间变量，占用内存和时间
        # mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        # return (foreground * mask) + (background * (1 - mask))
        
        # 优化写法：原地运算或减少乘法
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        # 只需要一次减法和两次乘法
        return background + (foreground - background) * mask
    
    def smooth_step(self, edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
        x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return x * x * (3 - 2 * x)
    
    def smooth_step_blend(self, foreground: np.ndarray, mask: np.ndarray,
                       background: np.ndarray) -> np.ndarray:
        mask_smooth = self.smooth_step(0.3, 0.5, mask)
        return self.alpha_blend(foreground, mask_smooth, background)
    
    def color_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_lab = cv2.cvtColor((source * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype("float32")
        target_lab = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype("float32")
        
        (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = self._compute_stats(source_lab)
        (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = self._compute_stats(target_lab)
        
        (l, a, b) = cv2.split(target_lab)
        l -= lMeanTar
        a -= aMeanTar
        b -= bMeanTar
        
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
        
        l += lMeanSrc
        a += aMeanSrc
        b += bMeanSrc
        
        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        transfer = cv2.merge([l, a, b])
        transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2RGB)
        
        return transfer / 255.0
    
    def _compute_stats(self, image_lab: np.ndarray):
        (l, a, b) = cv2.split(image_lab)
        return (l.mean(), l.std(), a.mean(), a.std(), b.mean(), b.std())
    
    def seamless_clone(self, foreground: np.ndarray, mask: np.ndarray,
                     background: np.ndarray) -> np.ndarray:
        src = (foreground * 255).astype(np.uint8)
        dst = (background * 255).astype(np.uint8)
        msk = (mask * 255).astype(np.uint8)
        
        kernel = np.ones((7, 7), np.uint8)
        msk = cv2.dilate(msk, kernel, iterations=1)
        
        src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        
        contours, _ = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(largest)
            center = (x + w//2, y + h//2)
            
            clone = cv2.seamlessClone(src, dst, msk, center, cv2.NORMAL_CLONE)
            clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
            return clone / 255.0
        
        return self.alpha_blend(foreground, mask, background)
    
    def harmonize(self, foreground: np.ndarray, mask: np.ndarray,
                 background: np.ndarray) -> np.ndarray:
        return self.color_transfer(background, foreground)
