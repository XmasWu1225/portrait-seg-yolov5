# Portrait Segmentation with YOLOv5-Seg

基于YOLOv5-seg的实时人像分割项目，支持背景替换、多种滤镜效果和实时处理。

## 项目特点

- 使用YOLOv5-seg进行人体检测和实例分割
- 支持多人场景的智能处理
- 提供多种后处理滤镜效果
- 实时摄像头处理
- 支持图片和视频批处理
- 模块化设计，易于扩展

## 项目结构

```
portrait_seg_yolov5/
├── models/                      # 模型相关
│   ├── yolov5_seg_detector.py  # YOLOv5-seg检测器
│   └── __init__.py
├── preprocessing/                # 预处理模块
│   ├── image_preprocessor.py    # 图像预处理
│   └── __init__.py
├── postprocessing/               # 后处理模块
│   ├── mask_processor.py        # 掩码处理
│   ├── filter_processor.py      # 滤镜处理
│   ├── blender.py              # 图像合成
│   └── __init__.py
├── utils/                      # 工具函数
│   ├── image_utils.py          # 图像工具
│   ├── timing_utils.py         # 计时工具
│   └── __init__.py
├── backgrounds/                 # 背景图片目录
├── test_images/                # 测试图片目录
├── config.py                   # 配置文件
├── main.py                     # 主程序
└── requirements.txt            # 依赖文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 模型支持

本项目支持两种模型格式：

### 1. PyTorch模型（默认）
- **文件格式**：`.pt` 或 `.pth`
- **优点**：易于调试和修改
- **缺点**：依赖完整PyTorch安装
- **适用场景**：开发、实验、快速迭代

### 2. ONNX模型（推荐用于生产）
- **文件格式**：`.onnx`
- **优点**：
  - 更快的推理速度
  - 更小的依赖（只需onnxruntime）
  - 更好的跨平台兼容性
  - 适合生产环境部署
- **缺点**：需要模型转换
- **适用场景**：生产部署、边缘设备、实时应用

### 切换模型格式

在 `config.py` 中设置：
```python
# 使用ONNX模型
USE_ONNX = True
YOLOV5_SEG_MODEL_PATH = 'yolov5s-seg.onnx'

# 或使用PyTorch模型
USE_ONNX = False
YOLOV5_SEG_MODEL_PATH = 'yolov5s-seg.pt'
```

### 获取ONNX模型

#### 方法1：从PyTorch转换
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

#### 方法2：下载预训练的ONNX模型
可以从以下来源下载预训练的ONNX模型：
- [ONNX Model Zoo](https://github.com/onnx/models)
- [Ultralytics ONNX Export](https://github.com/ultralytics/yolov5/issues/252)

详细说明请参考 [ONNX_GUIDE.md](ONNX_GUIDE.md)

## 下载模型

首次运行时，程序会自动从Ultralytics下载YOLOv5-seg模型。也可以手动下载：

```bash
# 下载YOLOv5-seg模型
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt
```

## 使用方法

### 1. 摄像头实时处理

```bash
python main.py --mode webcam --background backgrounds/beach.jpg
```

**键盘控制：**
- `q` - 退出程序
- `a` - Alpha混合滤镜（默认）
- `s` - 平滑步进滤镜
- `c` - 颜色传递滤镜
- `m` - 无缝克隆滤镜
- `h` - 颜色协调滤镜

### 2. 处理单张图片

```bash
python main.py --mode image --input test_images/person.jpg --output result.jpg --background backgrounds/sunset.jpg
```

### 3. 处理视频文件

```bash
python main.py --mode video --input input.mp4 --output output.mp4 --background backgrounds/sky.jpg
```

## 配置说明

编辑 `config.py` 文件可以调整以下参数：

### 模型配置
- `YOLOV5_SEG_MODEL_PATH`: YOLOv5-seg模型路径
- `YOLOV5_SEG_CONFIDENCE`: 检测置信度阈值（默认0.5）
- `YOLOV5_SEG_IOU`: IOU阈值（默认0.45）
- `YOLOV5_SEG_IMAGE_SIZE`: 模型输入尺寸（默认640）

### 掩码处理配置
- `MASK_THRESHOLD`: 掩码二值化阈值（默认0.5）
- `GAUSSIAN_BLUR_KERNEL`: 高斯模糊核大小（默认(7,7)）
- `MULTI_PERSON_STRATEGY`: 多人处理策略（'merge'或'largest'）

### 输出配置
- `OUTPUT_WIDTH`: 输出宽度（默认1280）
- `OUTPUT_HEIGHT`: 输出高度（默认720）
- `DEFAULT_FILTER`: 默认滤镜（默认'alpha_blend'）

## 滤镜效果说明

### 1. Alpha混合（默认）
简单的透明度混合，将前景和背景按照掩码权重进行混合。

### 2. 平滑步进
使用多项式函数平滑掩码边缘，创建更自然的过渡效果。

### 3. 颜色传递
将背景的色彩特征传递到前景，使前景与背景色彩更协调。

### 4. 无缝克隆
使用OpenCV的seamlessClone功能，创建自然的混合效果。

### 5. 颜色协调
使用深度学习模型协调前景和背景颜色。

## 处理流程

```
输入图片/视频
    ↓
预处理（颜色转换、尺寸调整、归一化）
    ↓
YOLOv5-seg推理（人体检测 + 实例分割）
    ↓
掩码处理（二值化、高斯模糊、形态学操作）
    ↓
滤镜处理（可选）
    ↓
图像合成（Alpha混合）
    ↓
输出结果
```

## 性能优化

- 使用GPU加速推理（CUDA）
- 支持多人场景的智能处理
- 优化的后处理流程
- 实时FPS显示

## 依赖项

- PyTorch >= 1.9.0
- OpenCV >= 4.5.0
- NumPy >= 1.19.0
- Pillow >= 8.0.0

## 注意事项

1. 首次运行需要下载模型文件，请确保网络连接正常
2. 建议使用GPU以获得更好的实时性能
3. 背景图片建议使用高分辨率图片
4. 多人场景下，可以选择合并所有人体或选择最大人体

## 故障排除

### 模型下载失败
如果自动下载失败，请手动下载模型并修改 `config.py` 中的模型路径。

### 摄像头无法打开
确保摄像头未被其他程序占用，检查摄像头设备号。

### 内存不足
降低 `YOLOV5_SEG_IMAGE_SIZE` 或 `OUTPUT_WIDTH/HEIGHT` 参数。

## 许可证

MIT License

## 作者

基于YOLOv5-seg和Portrait-Segmentation项目开发

## 致谢

- [YOLOv5](https://github.com/ultralytics/yolov5) - 目标检测框架
- [Portrait-Segmentation](https://github.com/anilsathyan7/Portrait-Segmentation) - 人像分割参考项目
