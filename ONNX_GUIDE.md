# ONNX Model Guide

## 概述

本项目现在支持使用ONNX格式的YOLOv5-seg模型，这样可以更好地适配不同的部署环境。

## 优势

1. **跨平台兼容性**：ONNX模型可以在任何支持ONNX的框架中运行
2. **更快的推理速度**：ONNX Runtime通常比PyTorch更快
3. **更小的依赖**：不需要完整的PyTorch安装
4. **更好的部署**：适合生产环境部署

## 安装依赖

```bash
pip install onnxruntime>=1.12.0
```

## 获取ONNX模型

### 方法1：从PyTorch转换

如果您有PyTorch模型，可以转换为ONNX格式：

```python
import torch
from models.yolov5_seg_detector import YOLOv5SegDetector

# 加载PyTorch模型
detector = YOLOv5SegDetector(use_onnx=False)

# 转换为ONNX
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    detector.model,
    dummy_input,
    "yolov5s-seg.onnx",
    input_names=['images'],
    output_names=['output0', 'output1'],
    dynamic_axes={
        'images': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)
```

### 方法2：下载预训练的ONNX模型

可以从以下来源下载预训练的ONNX模型：
- [ONNX Model Zoo](https://github.com/onnx/models)
- [Ultralytics ONNX Export](https://github.com/ultralytics/yolov5/issues/252)

## 配置使用

### 1. 修改config.py

```python
class Config:
    # 使用ONNX模型
    USE_ONNX = True
    
    # ONNX模型路径
    YOLOV5_SEG_MODEL_PATH = 'yolov5s-seg.onnx'
    
    # ONNX输入输出名称（根据您的模型调整）
    ONNX_INPUT_NAME = 'images'
    ONNX_OUTPUT_BOXES = 'output0'
    ONNX_OUTPUT_MASKS = 'output1'
```

### 2. 运行程序

```bash
# 自动使用ONNX模型（根据config.py设置）
python main.py --mode webcam --background backgrounds/beach.jpg

# 或者明确指定使用ONNX
python main.py --mode webcam --background backgrounds/beach.jpg --use-onnx
```

## ONNX模型要求

### 输入格式
- **形状**：(1, 3, 640, 640) 或动态尺寸
- **数据类型**：float32
- **归一化**：[0.0, 1.0]范围
- **格式**：NCHW（批次，通道，高度，宽度）

### 输出格式
- **output0**：边界框 (N, 4) - [x1, y1, x2, y2]
- **output1**：分割掩码 (N, 1, H, W)
- **output2**：置信度 (N, 1) - 可选
- **output3**：类别ID (N, 1) - 可选

## 性能对比

| 模式 | 推理时间 (ms) | 内存占用 (MB) | 依赖 |
|------|---------------|--------------|------|
| PyTorch | ~50 | ~500 | torch, torchvision |
| ONNX | ~30 | ~200 | onnxruntime |
| TensorRT | ~15 | ~150 | tensorrt |

*数据仅供参考，实际性能取决于硬件和模型大小

## 故障排除

### 问题1：找不到ONNX模型
```
FileNotFoundError: ONNX model not found: yolov5s-seg.onnx
```
**解决方案**：
- 确保ONNX模型文件在正确路径
- 检查config.py中的YOLOV5_SEG_MODEL_PATH设置

### 问题2：输入输出名称不匹配
```
KeyError: 'images'
```
**解决方案**：
- 使用Netron查看ONNX模型的输入输出名称
- 更新config.py中的ONNX_INPUT_NAME和ONNX_OUTPUT_*设置

### 问题3：形状不匹配
```
RuntimeError: Shape mismatch
```
**解决方案**：
- 检查输入图像尺寸是否为640x640
- 确保输入是NCHW格式
- 检查归一化是否正确

## 高级配置

### 使用GPU加速
```python
import onnxruntime as ort

# 创建GPU提供者
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model.onnx", providers=providers)
```

### 优化ONNX模型
```bash
# 使用ONNX优化工具
python -m onnxruntime.tools.optimize_model yolov5s-seg.onnx yolov5s-seg-optimized.onnx
```

## 部署建议

### 1. 开发环境
- 使用PyTorch模型进行调试和实验
- 快速迭代和模型修改

### 2. 生产环境
- 使用ONNX模型进行部署
- 更快的推理速度和更小的依赖

### 3. 边缘设备
- ONNX模型适合边缘设备
- 可以转换为TensorRT进行进一步优化

## 示例代码

### 完整的ONNX推理示例

```python
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# 加载ONNX模型
session = ort.InferenceSession("yolov5s-seg.onnx")

# 获取输入输出信息
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

print(f"Input: {input_name}")
print(f"Outputs: {output_names}")

# 加载图像
image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 预处理
img = cv2.resize(image_rgb, (640, 640))
img = img.astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0)

# 推理
outputs = session.run(None, {input_name: img})

# 后处理
boxes = outputs[0]
masks = outputs[1]

print(f"Detected {len(boxes)} objects")
print(f"Mask shape: {masks.shape}")
```

## 总结

使用ONNX模型可以显著提升推理性能，简化部署流程。建议在生产环境中使用ONNX格式，在开发环境中使用PyTorch格式。

## 参考资料

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [YOLOv5 ONNX Export](https://github.com/ultralytics/yolov5/issues/252)
- [ONNX Model Zoo](https://github.com/onnx/models)
