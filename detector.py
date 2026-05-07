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
    
    # Range amplo para pegar fel (do amarelo-oliva ao verde)
    lower_full = np.array([20, 50, 20])
    upper_full = np.array([90, 255, 255])
    full_mask = cv2.inRange(hsv, lower_full, upper_full)
    
    # Inibição específica da Luva Amarela (Amarelo vibrante/fluorescente)
    # As luvas têm saturação e brilho (Value) muito altos comparados ao fel
    lower_yellow_glove = np.array([22, 160, 150])
    upper_yellow_glove = np.array([35, 255, 255])
    glove_mask = cv2.inRange(hsv, lower_yellow_glove, upper_yellow_glove)
    
    # Remove a cor da luva da detecção principal
    green_mask = cv2.subtract(full_mask, glove_mask)
    
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
        
        # O filtro de aspecto (compacto) não funcionou bem porque as luvas, dependendo do ângulo, 
        # formam retângulos longos e finos.
        # A solução mais robusta (sem usar redes neurais pesadas para detectar a pessoa)
        # é usar um limiar de área muito maior. Uma luva tem em média 2000 a 4000 pixels.
        # Um vazamento de fel real se espalha pela bandeja e é muito maior.
        
        # 1. Ignorar objetos do tamanho de luvas/mãos (Area < 7000)
        if area < 7000:
            continue
            
        # Se passou do filtro de tamanho, é um vazamento grande (Fel)
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
