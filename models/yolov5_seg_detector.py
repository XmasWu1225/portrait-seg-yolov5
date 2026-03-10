import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import Config

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not installed. ONNX models will not be available.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not installed. PyTorch models will not be available.")


class YOLOv5SegDetector:
    def __init__(self, model_path: str = None, device: str = None, use_onnx: bool = None):
        self.device = device if device else Config.DEVICE
        self.model_path = model_path if model_path else Config.YOLOV5_SEG_MODEL_PATH
        self.confidence_threshold = Config.YOLOV5_SEG_CONFIDENCE
        self.iou_threshold = Config.YOLOV5_SEG_IOU
        self.image_size = Config.YOLOV5_SEG_IMAGE_SIZE
        self.person_class_id = Config.PERSON_CLASS_ID
        
        self.use_onnx = use_onnx if use_onnx is not None else Config.USE_ONNX
        self.model = None
        self.onnx_session = None
        self._load_model()
    
    def _load_model(self):
        try:
            if self.use_onnx:
                if not ONNX_AVAILABLE:
                    raise ImportError("onnxruntime is not installed.")
                
                # 自动选择运行后端
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                print(f"Loading ONNX model: {self.model_path} with {providers[0]}")
                self.onnx_session = ort.InferenceSession(str(self.model_path), providers=providers)
            else:
                if not TORCH_AVAILABLE:
                    raise ImportError("torch is not installed.")
                print(f"Loading PyTorch model: {self.model_path}")
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, device=self.device)
            
            print(f"Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        # 标准 YOLOv5 图像预处理：按比例缩放并填充
        shape = im.shape[:2] # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def detect_and_segment_onnx(self, image: np.ndarray) -> Dict:
        im_h, im_w = image.shape[:2]
        
        # 1. 预处理 (使用 Letterbox 保持长宽比)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img, ratio, (dw, dh) = self.letterbox(img_rgb, new_shape=self.image_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        
        # 2. 推理
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: img})
        
        # 3. 解析输出
        preds = outputs[0]
        if preds.shape[1] < preds.shape[2]: # 兼容 [1, 117, 25200]
            preds = preds.transpose(0, 2, 1)
        preds = preds[0]
        protos = outputs[1][0]
        
        # 4. 过滤低置信度
        conf = preds[:, 4]
        mask = conf > self.confidence_threshold
        preds = preds[mask]
        
        if len(preds) == 0:
            return {'detections': [], 'masks': [], 'original_shape': image.shape}

        # 5. 提取类别和掩码系数
        # 假设 80 类: 4(box) + 1(obj) + 80(class) + 32(coeffs) = 117
        class_probs = preds[:, 5:85]
        class_ids = np.argmax(class_probs, axis=1)
        scores = preds[:, 4] * np.max(class_probs, axis=1)
        
        # 筛选“人”类别
        person_mask = (class_ids == self.person_class_id) & (scores > self.confidence_threshold)
        preds = preds[person_mask]
        scores = scores[person_mask]
        
        if len(preds) == 0:
            return {'detections': [], 'masks': [], 'original_shape': image.shape}

        # 6. 坐标转换
        boxes = preds[:, :4]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # 7. NMS
        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), 
                                   self.confidence_threshold, self.iou_threshold)
        
        if len(indices) == 0:
            return {'detections': [], 'masks': [], 'original_shape': image.shape}
        
        indices = indices.flatten()
        final_preds = preds[indices]
        final_boxes = boxes_xyxy[indices]
        final_scores = scores[indices]

        # 8. 生成掩码
        mask_coeffs = final_preds[:, 85:] 
        proto_h, proto_w = protos.shape[1], protos.shape[2]
        masks = self.sigmoid(mask_coeffs @ protos.reshape(32, -1))
        masks = masks.reshape((-1, proto_h, proto_w))

        person_detections = []
        person_masks = []

        for i in range(len(final_preds)):
            # 还原框
            box = final_boxes[i].copy()
            box[[0, 2]] -= dw
            box[[1, 3]] -= dh
            box /= ratio
            box[[0, 2]] = np.clip(box[[0, 2]], 0, im_w)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, im_h)

            # 还原掩码
            m = masks[i]
            m = cv2.resize(m, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            m = m[int(dh):int(self.image_size-dh), int(dw):int(self.image_size-dw)]
            m = cv2.resize(m, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            m = (m > Config.MASK_THRESHOLD).astype(np.float32)

            person_detections.append({
                'bbox': box.astype(int).tolist(),
                'confidence': float(final_scores[i]),
                'class_id': self.person_class_id
            })
            person_masks.append(m)

        return {
            'detections': person_detections,
            'masks': person_masks,
            'original_shape': image.shape
        }

    def detect_and_segment(self, image: np.ndarray) -> Dict:
        if self.use_onnx:
            return self.detect_and_segment_onnx(image)
        else:
            return self.detect_and_segment_torch(image)

    def detect_and_segment_torch(self, image: np.ndarray) -> Dict:
        # 兼容性函数
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, size=self.image_size)
        
        person_detections = []
        person_masks = []
        
        if hasattr(results, 'masks') and results.masks is not None:
            # 根据 YOLOv5 版本不同，这里逻辑可能有差异，建议优先使用 ONNX
            # 下面是 ultralytics 官方常见输出解析
            for i, det in enumerate(results.pred[0]):
                if int(det[5]) == self.person_class_id:
                    conf = float(det[4])
                    box = det[:4].cpu().numpy().astype(int).tolist()
                    mask = results.masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    
                    person_detections.append({'bbox': box, 'confidence': conf, 'class_id': 0})
                    person_masks.append(mask)
        
        return {'detections': person_detections, 'masks': person_masks, 'original_shape': image.shape}

    def get_person_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        result = self.detect_and_segment(image)
        if not result['masks']:
            return None
        
        if Config.MULTI_PERSON_STRATEGY == 'merge':
            combined_mask = np.zeros(result['original_shape'][:2], dtype=np.float32)
            for mask in result['masks']:
                combined_mask = np.maximum(combined_mask, mask)
            return combined_mask
        return result['masks'][0]