import cv2
import numpy as np
import time
import argparse
from pathlib import Path

from models.yolov5_seg_detector import YOLOv5SegDetector
from preprocessing.image_preprocessor import ImagePreprocessor
from postprocessing.mask_processor import MaskProcessor
from postprocessing.filter_processor import FilterProcessor
from postprocessing.blender import Blender
from utils import FPSCounter, load_background, create_solid_background
from config import Config


class PortraitSegmentationApp:
    def __init__(self, background_path: str = None):
        # 1. 先初始化配置参数
        self.output_width = Config.OUTPUT_WIDTH
        self.output_height = Config.OUTPUT_HEIGHT
        
        # 2. 再初始化各个组件
        self.detector = YOLOv5SegDetector()
        self.preprocessor = ImagePreprocessor()
        self.mask_processor = MaskProcessor()
        self.filter_processor = FilterProcessor()
        self.blender = Blender()
        self.fps_counter = FPSCounter()
        
        # 3. 最后加载背景（此时 output_width/height 已经存在）
        self.background = self._load_background(background_path)

    def _load_background(self, background_path: str) -> np.ndarray:
        if background_path and Path(background_path).exists():
            try:
                bg = load_background(background_path, (self.output_width, self.output_height))
                print(f"Background loaded: {background_path}")
                return bg
            except Exception as e:
                print(f"Failed to load background: {e}")
        
        print("Using default solid background")
        return create_solid_background(self.output_width, self.output_height)
    
    def process_frame(self, frame: np.ndarray, use_blur: bool = False) -> np.ndarray:
        # 1. 预处理
        img_rgb = self.preprocessor.preprocess_for_model(frame)
        img_normalized = self.preprocessor.preprocess_for_display(frame)
        
        # 2. 获取掩码
        person_mask = self.detector.get_person_mask(frame)
        
        if person_mask is None:
            # 没检测到人时，如果不开启虚化则返回原图，开启了虚化则返回全图模糊
            if use_blur:
                ksize = Config.BACKGROUND_BLUR_STRENGTH
                return self.blender.prepare_output(cv2.GaussianBlur(img_normalized, (ksize, ksize), 0))
            return frame
        
        # 3. 处理掩码
        processed_mask = self.mask_processor.process_mask(
            person_mask, 
            target_size=(self.output_width, self.output_height)
        )
        
        # 4. 确定背景：是使用静态背景图片，还是当前帧的模糊版本
        if use_blur:
            small_img = cv2.resize(img_normalized, (0, 0), fx=0.25, fy=0.25)
            small_blur = cv2.GaussianBlur(small_img, (9, 9), 0) # 缩小了，卷积核也可以相应减小
            dynamic_background = cv2.resize(small_blur, (self.output_width, self.output_height))
        else:
            dynamic_background = self.background
        
        # 5. 合成
        result = self.filter_processor.apply_filter(
            img_normalized, processed_mask, dynamic_background
        )
        
        # 6. 转换输出格式
        result = self.blender.prepare_output(result)
        
        return result

    def run_webcam(self, use_blur: bool = False): # 添加参数
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        print("\n" + "="*50)
        print("Portrait Segmentation with YOLOv5-Seg")
        print("="*50)
        print("\n控制说明:")
        print("  'q' - 退出程序")
        print("  'a' - Alpha混合滤镜 (默认)")
        print("  's' - 平滑步进滤镜")
        print("  'c' - 颜色传递滤镜")
        print("  'm' - 无缝克隆滤镜")
        print("  'h' - 颜色协调滤镜")
        print("="*50 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            result = self.process_frame(frame, use_blur=use_blur) # 传递参数
            process_time = time.time() - start_time
            
            fps = self.fps_counter.update()
            
            self._draw_info(result, fps, process_time)
            
            cv2.imshow('Portrait Segmentation', result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.filter_processor.set_filter('alpha_blend')
                print("Filter: Alpha Blend")
            elif key == ord('s'):
                self.filter_processor.set_filter('smooth_step')
                print("Filter: Smooth Step")
            elif key == ord('c'):
                self.filter_processor.set_filter('color_transfer')
                print("Filter: Color Transfer")
            elif key == ord('m'):
                self.filter_processor.set_filter('seamless_clone')
                print("Filter: Seamless Clone")
            elif key == ord('h'):
                self.filter_processor.set_filter('harmonize')
                print("Filter: Harmonize")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_image(self, image_path: str, output_path: str = None, use_blur: bool = False): # 添加参数
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image: {image_path}")
            return
         # --- 增加暖身环节 ---
        print("Warming up...")
        for _ in range(5):
            _ = self.process_frame(frame, use_blur=use_blur)

        print(f"Processing image: {image_path}")
        start_time = time.time()
        result = self.process_frame(frame, use_blur=use_blur) # 传递参数
        process_time = time.time() - start_time
        
        print(f"Actual inference + post-processing time: {process_time:.4f} seconds")
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Result saved to: {output_path}")
        
        # cv2.imshow('Portrait Segmentation', result)
        # print("Press any key to close...")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    def run_video(self, video_path: str, output_path: str = None, use_blur: bool = False): # 添加参数
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                               (self.output_width, self.output_height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.process_frame(frame, use_blur=use_blur) # 传递参数
            
            if output_path:
                out.write(result)
            
            # --- 修改这里：在服务器上不显示 ---
            # cv2.imshow('Portrait Segmentation', result)
            
            frame_idx += 1
            if frame_idx % 10 == 0: # 减少打印频率
                print(f"Processing frame {frame_idx}/{frame_count}", end='\r')

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        cap.release()
        if output_path:
            out.release()
            print(f"\nVideo saved to: {output_path}")
        
        cv2.destroyAllWindows()
    
    def _draw_info(self, image: np.ndarray, fps: float, process_time: float):
        cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(image, f"Time: {process_time*1000:.1f}ms", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(image, f"Filter: {self.filter_processor.current_filter}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(description='Portrait Segmentation')
    parser.add_argument('--mode', type=str, default='webcam', choices=['webcam', 'image', 'video'])
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--background', type=str)
    parser.add_argument('--blur', action='store_true', help='Use background blur instead of replacement') # 确保这一行存在
    
    args = parser.parse_args()
    
    try:
        app = PortraitSegmentationApp(background_path=args.background)
        
        if args.mode == 'webcam':
            app.run_webcam(use_blur=args.blur)
        elif args.mode == 'image':
            if not args.input: return
            app.run_image(args.input, args.output, use_blur=args.blur) # 这里的调用现在匹配定义了
        elif args.mode == 'video':
            if not args.input: return
            app.run_video(args.input, args.output, use_blur=args.blur)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
