import cv2
import numpy as np

# Estados para Auditoria da Câmera 02
STATE_IDLE = "IDLE"
STATE_PICKED = "PICKED"
STATE_COFRE = "COFRE"
STATE_BURST = "BURST"
STATE_WASTE = "WASTE"

def detect_hand(frame):
    """Detecta a posição das luvas amarelas do operador"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Range da luva amarela (vibrante)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([38, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hands = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500: # Mão do operador tem um tamanho mínimo
            x, y, w, h = cv2.boundingRect(cnt)
            hands.append({'rect': (x, y, w, h), 'center': (x + w//2, y + h//2), 'area': area})
    return hands

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
    
    # Voltou para a configuração de cor que o usuário preferia (mais estável)
    # Aumentado o mínimo de Saturação (60) e Brilho (40) para evitar pegar brancos/cinzas
    lower_full = np.array([25, 60, 40])
    upper_full = np.array([90, 255, 255])
    full_mask = cv2.inRange(hsv, lower_full, upper_full)
    
    # Inibição ULTRA-ESPECÍFICA apenas do amarelo "marca-texto" das luvas
    # Luvas são extremamente saturadas e brilhantes. O fel amarelo é mais "fosco".
    lower_glove = np.array([25, 180, 180])
    upper_glove = np.array([35, 255, 255])
    glove_mask = cv2.inRange(hsv, lower_glove, upper_glove)
    
    # Remove apenas o amarelo fluorescente
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
        
        # 1. Ignorar sujeira pequena e movimentos de mão rápidos
        # Voltando para um limite equilibrado que funcionava bem com as cores antigas
        if area < 7000:
            continue
            
        detections.append({'rect': (x, y, w, h), 'area': area})
            
    return detections, green_mask


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
