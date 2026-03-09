import time
import numpy as np
from typing import Callable
import functools


class FPSCounter:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.fps = 0.0
    
    def update(self) -> float:
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            self.fps = (len(self.frame_times) - 1) / time_diff
        
        return self.fps
    
    def reset(self):
        self.frame_times = []
        self.fps = 0.0


def timing_decorator(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
