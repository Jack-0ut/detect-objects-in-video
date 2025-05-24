import cv2
import ctypes
import numpy as np

def fit_frame_to_screen(frame):
    """Resize and center the frame on a black canvas to fit the screen, preserving aspect ratio."""
    user32 = ctypes.windll.user32
    screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    y_offset = (screen_h - new_h) // 2
    x_offset = (screen_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas