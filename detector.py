import cv2
import numpy as np

# Estados para Auditoria da Câmera 02
STATE_IDLE = "IDLE"
STATE_LIVER = "LIVER_READY"
STATE_LEANING = "LEANING"  # Debruçado sobre a esteira
STATE_PICKED = "PICKED"
STATE_COFRE = "COFRE"
STATE_WASTE = "WASTE"

def detect_operator(frame, roi_points=None):
    """Detecta o capacete branco do operador de forma rigorosa"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Branco (Capacete) - Filtro mais restrito
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Se houver ROI, limita a busca
    if roi_points is not None:
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [roi_points], 255)
        mask = cv2.bitwise_and(mask, roi_mask)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # O capacete tem um tamanho bem específico na câmera
        if area > 1200 and area < 7000:
            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w//2, y + h//2)
            
            # Filtro de Altura: Capacete nunca está no chão (Y > 900 é chão)
            if center[1] > 900:
                continue

            # Verificação de Circularidade (Capacete é redondo)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > 0.6: # 1.0 é um círculo perfeito
                return {'center': center, 'rect': (x, y, w, h)}
    return None

def detect_liver(frame, roi_points):
    """Detecta a presença de fígado (carne escura) na zona de coleta"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Vermelho Escuro / Roxo (Fígado)
    lower1 = np.array([0, 30, 20])
    upper1 = np.array([15, 150, 100])
    lower2 = np.array([150, 30, 20])
    upper2 = np.array([180, 150, 100])
    
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask_liver = cv2.bitwise_and(cv2.bitwise_or(m1, m2), mask)
    
    area = np.sum(mask_liver > 0)
    return area > 5000 # Retorna True se houver massa de fígado significativa

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
    
    # Espectro ampliado para capturar o líquido da biles (que varia entre verde e amarelo)
    lower_full = np.array([20, 50, 30])
    upper_full = np.array([95, 255, 255])
    full_mask = cv2.inRange(hsv, lower_full, upper_full)
    
    # Inibição de Amarelo Fluorescente (Luvas) - Mantida bem restrita
    lower_glove = np.array([22, 160, 160])
    upper_glove = np.array([35, 255, 255])
    glove_mask = cv2.inRange(hsv, lower_glove, upper_glove)
    
    # Remove a luva mas mantém o verde/amarelo do líquido
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
        
        # Reduzido para capturar jatos pequenos e rápidos de líquido
        if area < 800:
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
