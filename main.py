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
        self.detector = YOLOv5SegDetector()
        self.preprocessor = ImagePreprocessor()
        self.mask_processor = MaskProcessor()
        self.filter_processor = FilterProcessor()
        self.blender = Blender()
        self.fps_counter = FPSCounter()
        
        self.background = self._load_background(background_path)
        
        self.output_width = Config.OUTPUT_WIDTH
        self.output_height = Config.OUTPUT_HEIGHT
    
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
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        img_rgb = self.preprocessor.preprocess_for_model(frame)
        
        person_mask = self.detector.get_person_mask(frame)
        
        if person_mask is None:
            return frame
        
        processed_mask = self.mask_processor.process_mask(
            person_mask, 
            target_size=(self.output_width, self.output_height)
        )
        
        img_normalized = self.preprocessor.preprocess_for_display(frame)
        
        result = self.filter_processor.apply_filter(
            img_normalized, processed_mask, self.background
        )
        
        result = self.blender.prepare_output(result)
        
        return result
    
    def run_webcam(self):
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
            result = self.process_frame(frame)
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
    
    def run_image(self, image_path: str, output_path: str = None):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image: {image_path}")
            return
        
        print(f"Processing image: {image_path}")
        start_time = time.time()
        result = self.process_frame(frame)
        process_time = time.time() - start_time
        
        print(f"Processing time: {process_time:.4f} seconds")
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Result saved to: {output_path}")
        
        cv2.imshow('Portrait Segmentation', result)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run_video(self, video_path: str, output_path: str = None):
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
            
            result = self.process_frame(frame)
            
            if output_path:
                out.write(result)
            
            cv2.imshow('Portrait Segmentation', result)
            
            frame_idx += 1
            print(f"Processing frame {frame_idx}/{frame_count}", end='\r')
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
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
    parser = argparse.ArgumentParser(
        description='Portrait Segmentation with YOLOv5-Seg',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode webcam --background backgrounds/beach.jpg
  python main.py --mode image --input test_images/person.jpg --output result.jpg
  python main.py --mode video --input input.mp4 --output output.mp4
        """
    )
    
    parser.add_argument('--mode', type=str, default='webcam', 
                       choices=['webcam', 'image', 'video'],
                       help='运行模式: webcam(摄像头), image(图片), video(视频)')
    parser.add_argument('--input', type=str, help='输入文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--background', type=str, help='背景图片路径')
    
    args = parser.parse_args()
    
    try:
        app = PortraitSegmentationApp(background_path=args.background)
        
        if args.mode == 'webcam':
            app.run_webcam()
        elif args.mode == 'image':
            if not args.input:
                print("错误: 请指定输入图片路径 (--input)")
                return
            app.run_image(args.input, args.output)
        elif args.mode == 'video':
            if not args.input:
                print("错误: 请指定输入视频路径 (--input)")
                return
            app.run_video(args.input, args.output)
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
