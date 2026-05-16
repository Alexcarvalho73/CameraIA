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
    Implementa lógica de duas áreas: Identificação e Alerta.
    
    Regra:
    1. Objeto deve ser identificado na Área de Identificação por no mínimo `min_frames_id` frames.
    2. Após identificado, deve-se aguardar no mínimo `min_delay_sec` segundos.
    3. Após esse tempo, o objeto deve ser detectado na Área de Alerta por no mínimo `min_frames_alert` frames.
    """

    def __init__(self, min_frames_id=10, min_frames_alert=10, min_delay_sec=5.0, max_jump_px=100, max_lateral_px=20):
        self.candidates = []
        self.min_frames_id = min_frames_id
        self.min_frames_alert = min_frames_alert
        self.min_delay_sec = min_delay_sec
        self.max_jump_px = max_jump_px
        self.max_lateral_px = max_lateral_px

    def update(self, detections, id_poly=None, alert_poly=None, config={}):
        """
        Atualiza o rastreador com as detecções do frame atual.
        """
        import time
        # Atualiza parâmetros se fornecidos no config
        self.min_frames_id = config.get('min_frames_id', self.min_frames_id)
        self.min_frames_alert = config.get('min_frames_alert', self.min_frames_alert)
        self.min_delay_sec = config.get('min_delay_sec', self.min_delay_sec)
        self.max_jump_px = config.get('max_jump_px', self.max_jump_px)
        self.max_lateral_px = config.get('max_lateral_px', self.max_lateral_px)

        current = []
        for det in detections:
            x, y, w, h = det['rect']
            current.append({
                'cx': x + w // 2,
                'cy': y + h // 2,
                'rect': det['rect'],
                'area': det['area'],
            })

        new_candidates = []
        matched_current = set()

        # Associa candidatos existentes a detecções atuais
        for cand in self.candidates:
            best_idx = None
            best_dist = self.max_jump_px

            for i, cur in enumerate(current):
                if i in matched_current:
                    continue
                dist = ((cand['cx'] - cur['cx']) ** 2 +
                        (cand['cy'] - cur['cy']) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None:
                matched_current.add(best_idx)
                cur = current[best_idx]
                
                # ── Filtro de Movimento Lateral ──
                # Se o deslocamento horizontal for maior que o limiar, consideramos como "lucas" (ruído)
                # e zeramos a pontuação do objeto.
                lateral_move = abs(cur['cx'] - cand['cx'])
                if lateral_move > self.max_lateral_px:
                    cand['id_hits'] = 0
                    cand['alert_hits'] = 0
                    cand['is_identified'] = False
                    cand['should_alert'] = False

                # Atualiza dados do candidato
                cand['cx'] = cur['cx']
                cand['cy'] = cur['cy']
                cand['rect'] = cur['rect']
                cand['area'] = cur['area']
                cand['last_seen'] = time.time()

                # Verifica em qual área o blob está
                in_id = False
                in_alert = False
                if id_poly is not None:
                    in_id = cv2.pointPolygonTest(id_poly, (float(cur['cx']), float(cur['cy'])), False) >= 0
                if alert_poly is not None:
                    in_alert = cv2.pointPolygonTest(alert_poly, (float(cur['cx']), float(cur['cy'])), False) >= 0

                # Lógica de Identificação
                if in_id:
                    cand['id_hits'] += 1
                    if cand['id_hits'] >= self.min_frames_id and not cand['is_identified']:
                        cand['is_identified'] = True
                        cand['id_timestamp'] = time.time()
                
                # Lógica de Alerta
                if cand['is_identified'] and in_alert:
                    time_passed = time.time() - cand['id_timestamp']
                    if time_passed >= self.min_delay_sec:
                        cand['alert_hits'] += 1
                        if cand['alert_hits'] >= self.min_frames_alert:
                            cand['should_alert'] = True
                
                new_candidates.append(cand)
            else:
                # Candidato NÃO encontrado neste frame.
                # Mantemos ele na lista para tentar casar no próximo (persitência contra flickering)
                new_candidates.append(cand)

        # Blobs novos sem correspondência anterior
        for i, cur in enumerate(current):
            if i not in matched_current:
                # Verifica se o novo blob já nasce em alguma área
                in_id = False
                if id_poly is not None:
                    in_id = cv2.pointPolygonTest(id_poly, (float(cur['cx']), float(cur['cy'])), False) >= 0
                
                new_candidates.append({
                    'cx': cur['cx'],
                    'cy': cur['cy'],
                    'rect': cur['rect'],
                    'area': cur['area'],
                    'id_hits': 1 if in_id else 0,
                    'alert_hits': 0,
                    'is_identified': False,
                    'id_timestamp': 0,
                    'should_alert': False,
                    'last_seen': time.time()
                })

        # Remove candidatos que sumiram há mais de 2 segundos (ou frames)
        # Para evitar que o rastreador fique "sujo" com objetos que saíram
        self.candidates = [c for c in new_candidates if time.time() - c['last_seen'] < 1.0]

        # Retorna apenas os blobs que estão confirmados para alerta ou sendo rastreados
        confirmed = []
        for c in self.candidates:
            # Informação visual
            status_txt = ""
            if c['should_alert']:
                status_txt = "ALERTA"
            elif c['is_identified']:
                dt = time.time() - c['id_timestamp']
                status_txt = f"ID ({dt:.1f}s)"
            else:
                status_txt = f"HI {c['id_hits']}"

            confirmed.append({
                'rect': c['rect'],
                'area': c['area'],
                'cx': c['cx'],
                'cy': c['cy'],
                'status': status_txt,
                'should_alert': c['should_alert'],
                'is_identified': c['is_identified']
            })

                
        return confirmed

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
def detect_green_stain(frame, roi_polygon, config={}):
    """
    Detecta candidatos de mancha de fel dentro da ROI com filtragem por proximidade humana e cor estrita.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1. Detecta Operadores e Luvas
    operators = detect_operators(frame)
    
    # Luvas: Amarelo Sólido, muito saturado e brilhante (Cor industrial)
    lower_glove = np.array([15, 160, 100])
    upper_glove = np.array([33, 255, 255])
    glove_candidates_mask = cv2.inRange(hsv, lower_glove, upper_glove)
    
    # 2. Lógica de Vínculo: Usar Componentes Conectados para isolar SÓ a luva
    glove_candidates_mask = cv2.morphologyEx(glove_candidates_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours_yellow, _ = cv2.findContours(glove_candidates_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    confirmed_glove_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Máscara de "Amarelo Proibido" (independente de pose, para evitar falsos positivos críticos)
    # Focada no núcleo da luva (muito amarelo e saturado)
    # Saturation aumentada para 200 para evitar pegar bile amarelada
    strict_yellow_mask = cv2.inRange(hsv, np.array([18, 200, 150]), np.array([32, 255, 255]))

    if operators:
        for cnt in contours_yellow:
            if cv2.contourArea(cnt) < 150:
                continue
                
            is_glove = False
            for op in operators:
                if 'wrists' in op and op['wrists']:
                    for w in op['wrists']:
                        dist = cv2.pointPolygonTest(cnt, (float(w[0]), float(w[1])), True)
                        if dist >= -120:
                            is_glove = True
                            break
                else:
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
                cv2.drawContours(confirmed_glove_mask, [cnt], -1, 255, -1)
                
    confirmed_glove_mask = cv2.dilate(confirmed_glove_mask, np.ones((15, 15), np.uint8))
    strict_yellow_mask = cv2.dilate(strict_yellow_mask, np.ones((11, 11), np.uint8))

    # 3. Detecta Fel (Verde/Amarelo-Verde) - DUAL RANGE
    # Range A: Fel Vibrante/Claro
    # Range B: Fel Escuro/Oliva
    
    lower_fel_a = config.get('lower_fel_a', np.array([22, 55, 40]))
    upper_fel_a = config.get('upper_fel_a', np.array([90, 255, 255]))
    lower_fel_b = config.get('lower_fel_b', np.array([35, 30, 20]))
    upper_fel_b = config.get('upper_fel_b', np.array([95, 255, 120]))
    
    mask_a = cv2.inRange(hsv, lower_fel_a, upper_fel_a)
    mask_b = cv2.inRange(hsv, lower_fel_b, upper_fel_b)
    fel_mask = cv2.bitwise_or(mask_a, mask_b)
    
    # Prepara ROI (pode ser um único polígono ou uma lista de polígonos)
    mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    if isinstance(roi_polygon, (list, tuple)):
        cv2.fillPoly(mask_roi, roi_polygon, 255)
    else:
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
        if area < 600: continue # Área reduzida para pegar inícios de vazamento
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
    # Filtra tudo que não seja cinza/preto/branco puro (Saturação > 50, Value > 50)
    # Aumentamos S de 35 para 50 e V de 40 para 50 para ignorar ruído de sensor em baixa luz
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([180, 255, 255])

    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    color_mask = cv2.bitwise_and(color_mask, mask)

    # Limpeza morfológica para remover ruído isolado (puntos de "sal e pimenta" do sensor à noite)
    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    # Conta o número de pixels "coloridos" na ROI
    colored_area = np.sum(color_mask > 0)
    
    # Limiar: Mantido em 40.000 pixels, mas agora com filtragem prévia
    return colored_area > 40000


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE BRILHO (para validação de turno)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_brightness(frame, roi_points=None):
    """
    Calcula o brilho médio da imagem ou da ROI. 
    Usado para evitar falso-positivos em ambientes escuros (limpeza).
    """
    if frame is None:
        return 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if roi_points is not None:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if isinstance(roi_points, (list, tuple)):
            cv2.fillPoly(mask, roi_points, 255)
        else:
            cv2.fillPoly(mask, [roi_points], 255)
        mean_val = cv2.mean(gray, mask=mask)[0]
    else:
        mean_val = cv2.mean(gray)[0]
        
    return mean_val


if __name__ == "__main__":
    print("Iniciando motor de detecção...")
    roi_points = np.array([[100, 100], [450, 50], [550, 800], [50, 900]], np.int32)
    frame = cv2.imread('test_image.jpg')
    if frame is not None:
        detections, mask = detect_green_stain(frame, roi_points)
        print(f"Candidatos: {len(detections)} (precisam de 5 frames para confirmar)")
    else:
        print("Arquivo test_image.jpg não encontrado.")

