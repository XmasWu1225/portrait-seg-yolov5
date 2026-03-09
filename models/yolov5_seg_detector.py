import torch
import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class YOLOv5SegDetector:
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device if device else Config.DEVICE
        self.model_path = model_path if model_path else Config.YOLOV5_SEG_MODEL_PATH
        self.confidence_threshold = Config.YOLOV5_SEG_CONFIDENCE
        self.iou_threshold = Config.YOLOV5_SEG_IOU
        self.image_size = Config.YOLOV5_SEG_IMAGE_SIZE
        self.person_class_id = Config.PERSON_CLASS_ID
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                       path=self.model_path, 
                                       device=self.device)
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.model.eval()
            print(f"YOLOv5-seg model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading YOLOv5-seg model: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def detect_and_segment(self, image: np.ndarray) -> Dict:
        image_rgb = self.preprocess(image)
        
        with torch.no_grad():
            results = self.model(image_rgb, size=self.image_size)
        
        person_detections = []
        person_masks = []
        
        if hasattr(results, 'masks'):
            masks = results.xyxyn[0][:, -1]
            boxes = results.xyxyn[0][:, :4]
            confidences = results.xyxyn[0][:, 4]
            class_ids = results.xyxyn[0][:, 5]
            
            h, w = image.shape[:2]
            
            for i, class_id in enumerate(class_ids):
                if int(class_id) == self.person_class_id:
                    x1, y1, x2, y2 = boxes[i]
                    x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                    
                    mask = masks[i].cpu().numpy()
                    mask = cv2.resize(mask, (w, h))
                    mask = (mask > Config.MASK_THRESHOLD).astype(np.float32)
                    
                    person_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidences[i]),
                        'class_id': int(class_id)
                    })
                    person_masks.append(mask)
        
        return {
            'detections': person_detections,
            'masks': person_masks,
            'original_shape': image.shape
        }
    
    def get_person_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        result = self.detect_and_segment(image)
        
        if not result['masks']:
            return None
        
        if Config.MULTI_PERSON_STRATEGY == 'merge':
            combined_mask = np.zeros(result['original_shape'][:2], dtype=np.float32)
            for mask in result['masks']:
                combined_mask = np.maximum(combined_mask, mask)
            return combined_mask
        elif Config.MULTI_PERSON_STRATEGY == 'largest':
            largest_idx = np.argmax([np.sum(mask) for mask in result['masks']])
            return result['masks'][largest_idx]
        
        return result['masks'][0]
    
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        image_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"Person: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(image_copy, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image_copy
