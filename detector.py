import cv2
import numpy as np

def detect_green_stain(frame, roi_polygon):
    """
    Detects green stains within a specific ROI.
    """
    # Create a mask for the ROI
    mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_polygon], 255)
    
    # Apply ROI mask to the frame
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask_roi)
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    
    # Adjusted range to avoid yellow (gloves) and keep the green/brown bile
    # We move the lower hue to 30 to skip pure yellow/orange.
    lower_green = np.array([30, 40, 20])
    upper_green = np.array([90, 255, 255])
    
    # Create a mask for green color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up the mask (remove noise)
    kernel = np.ones((5,5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of the green stains
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 600: # Increased area to avoid reflections and small spots
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append({'rect': (x, y, w, h), 'area': area})
            
    return detections, green_mask

def find_cofre(frame):
    """
    Automatically detects the square metallic funnel (cofre) and returns its ROI points.
    """
    # Convert to grayscale and improve contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Adaptive thresholding to find edges of metallic object
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to close gaps
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_approx = None
    max_area = 0
    
    # We look for a square/rectangular shape in the upper half of the image
    height, width = frame.shape[:2]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8000 or area > (width * height * 0.2): continue
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        # If it has 4-6 points (accounting for perspective/distortion)
        if 4 <= len(approx) <= 8:
            # Check if it's roughly in the middle/top
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # The cofre in the image is centered top
                if 200 < cX < width - 200 and cY < height // 2:
                    if area > max_area:
                        max_area = area
                        best_approx = approx.reshape(-1, 2)
                        
    return best_approx

if __name__ == "__main__":
    # Test with a dummy image or prompt user to provide one
    print("Iniciando motor de detecção...")
    # ROI based on the red lines in the user's image (approximate coordinates)
    # We will need a way to let the user define this visually in the dashboard
    roi_points = np.array([
        [100, 100], [450, 50], [550, 800], [50, 900]
    ], np.int32)
    
    # Load test image
    image_path = 'test_image.jpg'
    frame = cv2.imread(image_path)
    
    if frame is not None:
        detections, mask = detect_green_stain(frame, roi_points)
        if detections:
            print(f"ALERTA: {len(detections)} rompimento(s) de fel detectado(s)!")
        else:
            print("Nenhum rompimento detectado.")
    else:
        print(f"Arquivo {image_path} não encontrado. Coloque uma imagem de teste na pasta.")
