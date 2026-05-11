import cv2
import numpy as np

# Estados para Auditoria da Câmera 02
STATE_IDLE    = "IDLE"
STATE_LIVER   = "LIVER_READY"
STATE_LEANING = "LEANING"
STATE_PICKED  = "PICKED"
STATE_COFRE   = "COFRE"
STATE_WASTE   = "WASTE"


# ─────────────────────────────────────────────────────────────────────────────
# BLOB TRACKER – discrimina fel (estático) de luvas (movimento errático)
# ─────────────────────────────────────────────────────────────────────────────
class BlobTracker:
    """
    Rastreador temporal de blobs para diferenciar fel de luvas.

    Princípio observado:
      - Luva: aparece/some rapidamente ou pula de posição frame a frame.
      - Fel:  permanece no mesmo local (ou desliza linearmente com a esteira)
              por vários frames consecutivos.

    Um blob só é "confirmado" após ser visto por `min_frames` consecutivos
    sem pular mais de `max_jump_px` pixels entre frames.
    """

    def __init__(self, min_frames=7, max_jump_px=110):
        """
        min_frames  : frames consecutivos necessários para confirmar fel (~0.35 s a 20 FPS)
        max_jump_px : deslocamento máximo entre frames para ser o mesmo blob
        """
        self.candidates  = []
        self.min_frames  = min_frames
        self.max_jump_px = max_jump_px

    def update(self, detections):
        """
        Atualiza o rastreador com as detecções do frame atual.

        Args:
            detections: lista de dicts {'rect': (x,y,w,h), 'area': int}

        Returns:
            Lista de detecções confirmadas como fel (blobs persistentes).
        """
        current = []
        for det in detections:
            x, y, w, h = det['rect']
            current.append({
                'cx': x + w // 2,
                'cy': y + h // 2,
                'rect': det['rect'],
                'area': det['area'],
            })

        new_candidates  = []
        matched_current = set()

        # Associa candidatos existentes a detecções atuais
        for cand in self.candidates:
            best_idx  = None
            best_dist = self.max_jump_px

            for i, cur in enumerate(current):
                if i in matched_current:
                    continue
                dist = ((cand['cx'] - cur['cx']) ** 2 +
                        (cand['cy'] - cur['cy']) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i

            if best_idx is not None:
                matched_current.add(best_idx)
                cur = current[best_idx]
                new_candidates.append({
                    'cx':     cur['cx'],
                    'cy':     cur['cy'],
                    'rect':   cur['rect'],
                    'area':   cur['area'],
                    'frames': cand['frames'] + 1,
                })
            # candidato sumiu → descarta (não carrega para o próximo frame)

        # Blobs novos sem correspondência anterior
        for i, cur in enumerate(current):
            if i not in matched_current:
                new_candidates.append({
                    'cx':     cur['cx'],
                    'cy':     cur['cy'],
                    'rect':   cur['rect'],
                    'area':   cur['area'],
                    'frames': 1,
                })

        self.candidates = new_candidates

        # Retorna apenas os confirmados
        confirmed = [c for c in self.candidates if c['frames'] >= self.min_frames]
        return [{'rect': c['rect'], 'area': c['area'], 'frames': c['frames']}
                for c in confirmed]

    def reset(self):
        self.candidates = []


# ─────────────────────────────────────────────────────────────────────────────
# DETECÇÃO DE OPERADOR (capacete branco)
# ─────────────────────────────────────────────────────────────────────────────
def detect_operators(frame, roi_points=None):
    """Detecta todos os capacetes brancos presentes na cena."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    if roi_points is not None:
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [roi_points], 255)
        mask = cv2.bitwise_and(mask, roi_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    operators = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 1000 < area < 8000:
            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w // 2, y + h // 2)
            if center[1] > 1000: # Ignora reflexos muito baixos se houver
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.55:
                operators.append({'center': center, 'rect': (x, y, w, h)})
    return operators


# ─────────────────────────────────────────────────────────────────────────────
# DETECÇÃO DE FÍGADO
# ─────────────────────────────────────────────────────────────────────────────
def detect_liver(frame, roi_points):
    """Detecta a presença de fígado (carne escura) na zona de coleta."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 30, 20])
    upper1 = np.array([15, 150, 100])
    lower2 = np.array([150, 30, 20])
    upper2 = np.array([180, 150, 100])

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask_liver = cv2.bitwise_and(cv2.bitwise_or(m1, m2), mask)

    area = np.sum(mask_liver > 0)
    return area > 5000


# ─────────────────────────────────────────────────────────────────────────────
# DETECÇÃO DE LUVAS (para rastreamento e auditoria Câmera 02)
# ─────────────────────────────────────────────────────────────────────────────
def detect_hand(frame):
    """Detecta a posição das luvas amarelas do operador."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([38, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hands = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            hands.append({'rect': (x, y, w, h),
                          'center': (x + w // 2, y + h // 2),
                          'area': area})
    return hands


# ─────────────────────────────────────────────────────────────────────────────
# AUXILIARES INTERNOS
# ─────────────────────────────────────────────────────────────────────────────
def _detect_glove_regions(hsv_full_frame):
    """
    Localiza regiões de luva amarela no frame inteiro.
    Usado para rejeição espacial em detect_green_stain.
    """
    lower_glove = np.array([18, 90, 90])
    upper_glove = np.array([38, 255, 255])
    glove_mask = cv2.inRange(hsv_full_frame, lower_glove, upper_glove)

    kernel = np.ones((9, 9), np.uint8)
    glove_mask = cv2.morphologyEx(glove_mask, cv2.MORPH_CLOSE, kernel)
    glove_mask = cv2.morphologyEx(glove_mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(glove_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        regions.append((x, y, w, h))
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# DETECÇÃO DE FEL – retorna CANDIDATOS (validação temporal feita pelo BlobTracker)
# ─────────────────────────────────────────────────────────────────────────────
def detect_green_stain(frame, roi_polygon):
    """
    Detecta candidatos de mancha de fel dentro da ROI com filtragem por proximidade humana.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1. Detecta Operadores e Luvas
    operators = detect_operators(frame)
    
    # Luvas: Amarelo muito específico e vibrante
    lower_glove = np.array([18, 100, 100])
    upper_glove = np.array([32, 255, 255])
    glove_candidates_mask = cv2.inRange(hsv, lower_glove, upper_glove)
    
    # 2. Lógica de Vínculo: Só anula o amarelo se estiver muito perto de uma pessoa
    confirmed_glove_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if operators:
        for op in operators:
            # Raio de 280px é suficiente para os braços sem invadir o centro da esteira
            proximity_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(proximity_mask, op['center'], 280, 255, -1)
            
            op_glove = cv2.bitwise_and(glove_candidates_mask, proximity_mask)
            confirmed_glove_mask = cv2.bitwise_or(confirmed_glove_mask, op_glove)
    
    confirmed_glove_mask = cv2.dilate(confirmed_glove_mask, np.ones((9, 9), np.uint8))

    # 3. Detecta Fel (Verde/Amarelo)
    # S=50 aumenta a sensibilidade para fel diluído
    lower_fel = np.array([30, 50, 40])
    upper_fel = np.array([90, 255, 255])
    fel_mask  = cv2.inRange(hsv, lower_fel, upper_fel)
    
    # Prepara ROI
    mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_polygon], 255)
    
    # FILTRAGEM: ROI - LUVAS CONFIRMADAS (PERTO DE PESSOAS)
    fel_mask = cv2.bitwise_and(fel_mask, mask_roi)
    fel_mask = cv2.subtract(fel_mask, confirmed_glove_mask)

    # Limpeza morfológica
    kernel   = np.ones((7, 7), np.uint8)
    fel_mask = cv2.morphologyEx(fel_mask, cv2.MORPH_OPEN,  kernel)
    fel_mask = cv2.morphologyEx(fel_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800: continue # Sensibilidade aumentada (área menor permitida)
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append({'rect': (x, y, w, h), 'area': area})

    return detections, fel_mask


if __name__ == "__main__":
    print("Iniciando motor de detecção...")
    roi_points = np.array([[100, 100], [450, 50], [550, 800], [50, 900]], np.int32)
    frame = cv2.imread('test_image.jpg')
    if frame is not None:
        detections, mask = detect_green_stain(frame, roi_points)
        print(f"Candidatos: {len(detections)} (precisam de 5 frames para confirmar)")
    else:
        print("Arquivo test_image.jpg não encontrado.")

