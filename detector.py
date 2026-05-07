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
    
    # Reverted Hue to catch yellowish bile, but increased Saturation
    lower_green = np.array([25, 50, 20])
    upper_green = np.array([90, 255, 255])
    
    # Create a mask for green color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up the mask (remove noise)
    kernel = np.ones((7,7), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of the green stains
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Shape Analysis to differentiate Bile from Gloves
        # Gloves are usually small and somewhat square/round (aspect ratio near 1.0)
        # Bile spills are either massive or spread out (irregular aspect ratio)
        
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # 1. Ignore very small noise
        if area < 800:
            continue
            
        # 2. Heuristic for Gloves vs Bile
        # If the object is relatively small (like a hand/glove) AND has a compact shape
        is_compact = 0.5 < aspect_ratio < 2.0
        
        # We consider it a "Glove" if area is under 4500 and it's compact
        if area < 4500 and is_compact:
            continue
            
        # If it passed the filters, it's a valid bile spill
        detections.append({'rect': (x, y, w, h), 'area': area})
            
    return detections, green_mask

def find_cofre(frame):
    """
    Automatically detects the square metallic funnel (cofre) and returns its ROI points.
    Improved version using brightness thresholding for metallic objects.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Simple thresholding: metallic cofre is usually brighter than the floor
    # We use a lower threshold to capture more of the bin's shape
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to merge parts of the metallic object
    kernel = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_approx = None
    max_area = 0
    height, width = frame.shape[:2]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Relaxed area constraints
        if area < 3000 or area > (width * height * 0.4): continue
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        
        # We look for something roughly rectangular (4 to 8 points)
        if 4 <= len(approx) <= 8:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Check if it's in the top 70% of the image
                if cY < height * 0.7:
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
