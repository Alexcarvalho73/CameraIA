
import cv2
import numpy as np
import os

def detect_production_active(frame, roi_points):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Filtra tudo que não seja cinza/preto/branco puro (Saturação > 35, Value > 40)
    lower_color = np.array([0, 35, 40])
    upper_color = np.array([180, 255, 255])

    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    color_mask = cv2.bitwise_and(color_mask, mask)

    colored_area = np.sum(color_mask > 0)
    return colored_area

# ROI from main.py for camera_01
roi_pts = np.array([[280, 375], [790, 320], [810, 710], [195, 750]], np.int32)

images = [
    'input_file_0.png',
    'input_file_1.png',
    'input_file_2.png',
    'input_file_3.png'
]

for img_path in images:
    if os.path.exists(img_path):
        frame = cv2.imread(img_path)
        if frame is not None:
            # Re-scale ROI if image size is different (main.py assumes camera res, probably 1080p or 720p)
            # The images provided might be smaller.
            h, w = frame.shape[:2]
            # Let's assume ROI is for 1920x1080? No, let's check camera resolution if possible.
            # Looking at the coordinates [790, 320], it's likely 1280x720 or 1920x1080.
            # If the image is smaller, we scale the ROI.
            # But wait, input_file_0.png dimensions are... let me check.
            pass
        
        area = detect_production_active(frame, roi_pts)
        print(f"{img_path}: Area = {area}")
    else:
        print(f"{img_path} not found")
