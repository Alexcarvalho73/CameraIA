import cv2
import numpy as np
import logging

# Suprime logs excessivos do Ultralytics
logging.getLogger("ultralytics").setLevel(logging.WARNING)
from ultralytics import YOLO

try:
    pose_model = YOLO('yolov8n-pose.pt')
except Exception as e:
    pose_model = None
    print(f"[AVISO] Falha ao carregar YOLO Pose: {e}")

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
    """Detecta pessoas (operadores) e extrai o esqueleto (pulsos e cabeça) usando YOLO Pose."""
    operators = []
    if pose_model is None:
        return operators

    # Roda o modelo apenas para a classe 0 (person)
    results = pose_model(frame, verbose=False, conf=0.45, classes=[0])
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        # Shape dos keypoints: (num_pessoas, 17, 2)
        keypoints = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []
        confs = r.keypoints.conf.cpu().numpy() if r.keypoints is not None else []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            
            # Centro default é o meio da bounding box
            center = (x1 + w // 2, y1 + h // 2)
            wrists = []
            
            if len(keypoints) > i and len(keypoints[i]) >= 11:
                kp = keypoints[i]
                c_conf = confs[i]
                
                # Se o nariz (0) tiver boa confiança, usamos ele como 'center' para saber se abaixou
                if c_conf[0] > 0.4:
                    center = (int(kp[0][0]), int(kp[0][1]))
                    
                # Pulso Esquerdo (9) - threshold baixo para pegar mãos sujas/camufladas
                if c_conf[9] > 0.15:
                    wrists.append((int(kp[9][0]), int(kp[9][1])))
                elif c_conf[7] > 0.15: # Fallback: Cotovelo Esquerdo
                    wrists.append((int(kp[7][0]), int(kp[7][1])))
                    
                # Pulso Direito (10)
                if c_conf[10] > 0.15:
                    wrists.append((int(kp[10][0]), int(kp[10][1])))
                elif c_conf[8] > 0.15: # Fallback: Cotovelo Direito
                    wrists.append((int(kp[8][0]), int(kp[8][1])))
            
            op_data = {'center': center, 'rect': (x1, y1, w, h), 'wrists': wrists}
            
            # Se houver ROI, verifica se o centro está dentro
            if roi_points is not None:
                if cv2.pointPolygonTest(roi_points, (float(center[0]), float(center[1])), False) >= 0:
                    operators.append(op_data)
            else:
                operators.append(op_data)
                
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
    Detecta candidatos de mancha de fel dentro da ROI com filtragem por proximidade humana e cor estrita.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1. Detecta Operadores e Luvas
    operators = detect_operators(frame)
    
    # Luvas: Amarelo muito específico e vibrante (Hue 15-35)
    lower_glove = np.array([15, 100, 100])
    upper_glove = np.array([35, 255, 255])
    glove_candidates_mask = cv2.inRange(hsv, lower_glove, upper_glove)
    
    # 2. Lógica de Vínculo: Usar Componentes Conectados para isolar SÓ a luva
    glove_candidates_mask = cv2.morphologyEx(glove_candidates_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours_yellow, _ = cv2.findContours(glove_candidates_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    confirmed_glove_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Máscara de "Amarelo Proibido" (independente de pose, para evitar falsos positivos críticos)
    # Se for muito amarelo e muito saturado, provavelmente é luva.
    strict_yellow_mask = cv2.inRange(hsv, np.array([18, 150, 100]), np.array([34, 255, 255]))

    if operators:
        for cnt in contours_yellow:
            if cv2.contourArea(cnt) < 150:
                continue
                
            is_glove = False
            for op in operators:
                if 'wrists' in op and op['wrists']:
                    for w in op['wrists']:
                        dist = cv2.pointPolygonTest(cnt, (float(w[0]), float(w[1])), True)
                        # Se o pulso está a até 120 pixels da bolha amarela, é a luva!
                        if dist >= -120:
                            is_glove = True
                            break
                else:
                    # Fallback: Se não achou pulso, vê se o centro da bolha amarela está na caixa da pessoa
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        x, y, w, h = op['rect']
                        if x < cx < x + w and y < cy < y + h:
                            is_glove = True
                            break
                if is_glove:
                    break
            
            if is_glove:
                # Desenha APENAS essa bolha exata na máscara de anulação
                cv2.drawContours(confirmed_glove_mask, [cnt], -1, 255, -1)
                
    confirmed_glove_mask = cv2.dilate(confirmed_glove_mask, np.ones((15, 15), np.uint8))
    strict_yellow_mask = cv2.dilate(strict_yellow_mask, np.ones((11, 11), np.uint8))

    # 3. Detecta Fel (Verde/Amarelo-Verde)
    # Aumentamos o Hue inicial para 38 para sair do range do amarelo puro (30-35)
    lower_fel = np.array([38, 50, 40])
    upper_fel = np.array([85, 255, 255])
    fel_mask  = cv2.inRange(hsv, lower_fel, upper_fel)
    
    # Prepara ROI
    mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_polygon], 255)
    
    # FILTRAGEM: ROI - LUVAS CONFIRMADAS - AMARELO ESTRITO
    fel_mask = cv2.bitwise_and(fel_mask, mask_roi)
    fel_mask = cv2.subtract(fel_mask, confirmed_glove_mask)
    fel_mask = cv2.subtract(fel_mask, strict_yellow_mask)

    # Limpeza morfológica
    kernel   = np.ones((7, 7), np.uint8)
    fel_mask = cv2.morphologyEx(fel_mask, cv2.MORPH_OPEN,  kernel)
    fel_mask = cv2.morphologyEx(fel_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000: continue # Área mínima aumentada ligeiramente para filtrar ruído
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append({'rect': (x, y, w, h), 'area': area})

    return detections, fel_mask

# ─────────────────────────────────────────────────────────────────────────────
# DETECÇÃO DE PRODUÇÃO ATIVA
# ─────────────────────────────────────────────────────────────────────────────
def detect_production_active(frame, roi_points):
    """
    Verifica se a esteira está cheia de miúdos ou vazia.
    Bandejas vazias são metálicas e cinzas (baixa saturação).
    Miúdos têm cores (vermelho, gordura, sangue, fel) com maior saturação e brilho.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Filtra tudo que não seja cinza/preto/branco puro (Saturação > 35, Value > 40)
    lower_color = np.array([0, 35, 40])
    upper_color = np.array([180, 255, 255])

    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    color_mask = cv2.bitwise_and(color_mask, mask)

    # Conta o número de pixels "coloridos" na ROI
    colored_area = np.sum(color_mask > 0)
    
    # Limiar: Elevado para 40.000 pixels para evitar que a camisa do operador dispare falso positivo
    return colored_area > 40000


if __name__ == "__main__":
    print("Iniciando motor de detecção...")
    roi_points = np.array([[100, 100], [450, 50], [550, 800], [50, 900]], np.int32)
    frame = cv2.imread('test_image.jpg')
    if frame is not None:
        detections, mask = detect_green_stain(frame, roi_points)
        print(f"Candidatos: {len(detections)} (precisam de 5 frames para confirmar)")
    else:
        print("Arquivo test_image.jpg não encontrado.")

