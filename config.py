import torch


class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    YOLOV5_SEG_MODEL_PATH = 'yolov5s-seg.pt'
    YOLOV5_SEG_CONFIDENCE = 0.5
    YOLOV5_SEG_IOU = 0.45
    YOLOV5_SEG_IMAGE_SIZE = 640
    
    PERSON_CLASS_ID = 0
    
    MASK_THRESHOLD = 0.5
    GAUSSIAN_BLUR_KERNEL = (7, 7)
    GAUSSIAN_BLUR_SIGMA = 1.0
    
    MULTI_PERSON_STRATEGY = 'merge'
    
    OUTPUT_WIDTH = 1280
    OUTPUT_HEIGHT = 720
    
    ENABLE_FILTERS = True
    DEFAULT_FILTER = 'alpha_blend'
    
    BACKGROUND_IMAGES = [
        'backgrounds/beach.jpg',
        'backgrounds/sunset.jpg',
        'backgrounds/sky.jpg'
    ]
    
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    CAMERA_FPS = 30
    
    SAVE_OUTPUT = False
    OUTPUT_PATH = './output'
    
    DEBUG_MODE = False
