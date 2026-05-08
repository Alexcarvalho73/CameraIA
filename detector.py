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

def _detect_glove_regions(hsv_full_frame):
    """
    Detecta regiões de luva amarela na imagem inteira.
    Retorna lista de bounding boxes (x, y, w, h) das luvas encontradas.
    Usado para suprimir falsos positivos em detect_green_stain.
    """
    # CAMADA 1: Range ampliado para cobrir todo o espectro do amarelo fluorescente
    # As luvas variam bastante dependendo da iluminação da câmera
    lower_glove = np.array([18, 80, 80])    # Amarelo-esverdeado até amarelo puro
    upper_glove = np.array([38, 255, 255])  # Cobre toda a faixa vibrante
    glove_mask = cv2.inRange(hsv_full_frame, lower_glove, upper_glove)

    kernel = np.ones((9, 9), np.uint8)
    glove_mask = cv2.morphologyEx(glove_mask, cv2.MORPH_CLOSE, kernel)
    glove_mask = cv2.morphologyEx(glove_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(glove_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        regions.append((x, y, w, h))
    return regions


def _stain_overlaps_glove(stain_rect, glove_regions, expand_px=60):
    """
    CAMADA 2: Rejeita manchas que se sobrepõem ou estão muito próximas de uma luva.
    A luva é expandida em `expand_px` pixels em todas as direções para cobrir o braço.
    """
    sx, sy, sw, sh = stain_rect
    for (gx, gy, gw, gh) in glove_regions:
        # Expande a região da luva para cobrir o antebraço adjacente
        ex = max(0, gx - expand_px)
        ey = max(0, gy - expand_px)
        ew = gw + 2 * expand_px
        eh = gh + 2 * expand_px
        # Checa sobreposição de retângulos
        if sx < ex + ew and sx + sw > ex and sy < ey + eh and sy + sh > ey:
            return True
    return False


def detect_green_stain(frame, roi_polygon):
    """
    Detecta manchas de fel (líquido biliar) dentro de uma ROI específica.

    Sistema anti-falso-positivo em 3 camadas:
      1. Inibição ampla de cor de luva amarela (range HSV expandido)
      2. Validação contextual: rejeita manchas próximas a regiões de luva/braço detectados
      3. Filtros de morfologia: área mínima elevada + penalização de formas compactas
         (luvas são pequenas e quadradas; fel espalhado é grande e irregular)
    """
    # Máscara da ROI
    mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_polygon], 255)

    roi_frame = cv2.bitwise_and(frame, frame, mask=mask_roi)

    # HSV do frame completo (para detectar luvas fora da ROI que invadem ela)
    hsv_full  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV apenas da ROI (para detectar mancha de fel)
    hsv_roi   = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # ── CAMADA 1: Detecção de luvas no frame inteiro ───────────────────────────
    glove_regions = _detect_glove_regions(hsv_full)

    # ── Espectro do fel (verde-amarelado a verde puro) ─────────────────────────
    # Estreitamos o limite inferior para excluir amarelos puros (luvas)
    # Fel real é mais esverdeado: H entre 30 e 95
    lower_fel = np.array([30, 40, 30])   # Começa no amarelo-esverdeado (H≥30)
    upper_fel = np.array([95, 255, 255]) # Até verde puro
    fel_mask  = cv2.inRange(hsv_roi, lower_fel, upper_fel)

    # Inibição adicional: remove pixels que são amarelo puro/fluorescente (luvas)
    lower_glove_inh = np.array([18, 80, 80])
    upper_glove_inh = np.array([38, 255, 255])
    glove_inh_mask  = cv2.inRange(hsv_roi, lower_glove_inh, upper_glove_inh)
    fel_mask = cv2.subtract(fel_mask, glove_inh_mask)

    # ── Limpeza morfológica: kernel maior para eliminar ruído pontual ──────────
    kernel = np.ones((9, 9), np.uint8)
    fel_mask = cv2.morphologyEx(fel_mask, cv2.MORPH_OPEN,  kernel)
    fel_mask = cv2.morphologyEx(fel_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # ── CAMADA 3a: Área mínima elevada ────────────────────────────────────
        # Luvas têm ~800-3000 px²; fel derramado visível tem >3500 px²
        if area < 3500:
            continue

        # ── CAMADA 3b: Filtro de compacidade (aspect ratio) ───────────────────
        # Luvas são compactas (w≈h). Fel espalhado é mais irregular e largo.
        # Rejeita blobs muito quadrados e pequenos (padrão de luva compacta)
        aspect = max(w, h) / max(min(w, h), 1)
        if area < 6000 and aspect < 1.3:
            # Blob pequeno E quadrado → provavelmente luva, rejeita
            continue

        # ── CAMADA 2: Validação contextual (proximidade com luva) ─────────────
        if _stain_overlaps_glove((x, y, w, h), glove_regions, expand_px=70):
            # A mancha está sobre ou muito perto de uma luva detectada → FP
            continue

        detections.append({'rect': (x, y, w, h), 'area': area})

    return detections, fel_mask


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
