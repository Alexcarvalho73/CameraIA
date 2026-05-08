import os
import cv2
import json
import time
import threading
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from detector import detect_green_stain, detect_hand, detect_operator, BlobTracker

try:
    import oracledb
    ORACLE_AVAILABLE = True
        instant_client_path = "/home/rdt/CameraIA/instantclient_21_1"
        oracle_wallet_path = "/home/rdt/CameraIA/DriveOracle"
        
        # Garante que as variáveis de ambiente estejam no processo atual
        os.environ['LD_LIBRARY_PATH'] = f"{instant_client_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        os.environ['TNS_ADMIN'] = oracle_wallet_path
        
        if os.path.exists(instant_client_path):
            oracledb.init_oracle_client(lib_dir=instant_client_path)
            print(f"[DB] Oracle Thick Mode ativado usando: {instant_client_path}")
        else:
            oracledb.init_oracle_client()
            print("[DB] Oracle Thick Mode ativado.")
    except Exception as e:
        print(f"[DB] Erro ao ativar modo Thick: {e}")
except ImportError:
    ORACLE_AVAILABLE = False
    print("[AVISO] oracledb não encontrado. Integração com banco de dados desativada.")

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DAS CÂMERAS
# ─────────────────────────────────────────────────────────────────────────────
CAMERAS = {
    "camera_01": {
        "name": "Esteira Principal - Ruptura",
        "rtsp_url": "rtsp://admin:013579ab@10.200.34.50:554/cam/realmonitor?channel=1&subtype=0",
        "roi": [[280, 375], [790, 320], [810, 710], [195, 750]],
        "type": "color_detection",
        "alerts_enabled": True     # Câmera 01 com alertas ATIVOS
    },
    "camera_02": {
        "name": "Cofre - Fluxo de Vesícula",
        "rtsp_url": "rtsp://admin:013579ab@10.200.96.81:554/cam/realmonitor?channel=1&subtype=0",
        "roi": [[700, 20], [1900, 20], [1900, 1000], [700, 1000]],
        "zones": {
            "cofre":     [[800, 20],  [1150, 20],  [1150, 300], [800, 300]],
            "descarte":  [[850, 320], [1150, 320], [1150, 600], [850, 600]],
            "pockets":   [[850, 650], [1150, 650], [1150, 900], [850, 900]],
            "work_area": [[700, 20],  [1300, 20],  [1300, 1050],[700, 1050]]
        },
        "type": "behavior_detection",
        "alerts_enabled": False    # Câmera 02 com alertas PAUSADOS
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTÊNCIA DE ROI  (roi_config.json)
# ─────────────────────────────────────────────────────────────────────────────
ROI_CONFIG_FILE = "roi_config.json"

def load_roi_config():
    """Carrega ROIs salvos em disco e sobrepõe os padrões do código."""
    if not os.path.exists(ROI_CONFIG_FILE):
        return
    try:
        with open(ROI_CONFIG_FILE, 'r') as f:
            saved = json.load(f)
        for cam_id, cfg in saved.items():
            if cam_id not in CAMERAS:
                continue
            if 'roi' in cfg:
                CAMERAS[cam_id]['roi'] = cfg['roi']
            if 'zones' in cfg and 'zones' in CAMERAS[cam_id]:
                for zone_name, pts in cfg['zones'].items():
                    CAMERAS[cam_id]['zones'][zone_name] = pts
        print(f"[ROI] Configurações carregadas de '{ROI_CONFIG_FILE}'.")
    except Exception as e:
        print(f"[ROI] Erro ao carregar '{ROI_CONFIG_FILE}': {e}")

def persist_roi_config():
    """Salva o estado atual dos ROIs em disco."""
    try:
        data = {}
        for cam_id, cfg in CAMERAS.items():
            data[cam_id] = {}
            if 'roi' in cfg:
                data[cam_id]['roi'] = cfg['roi']
            if 'zones' in cfg:
                data[cam_id]['zones'] = cfg['zones']
        with open(ROI_CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[ROI] Configurações persistidas em '{ROI_CONFIG_FILE}'.")
    except Exception as e:
        print(f"[ROI] Erro ao salvar '{ROI_CONFIG_FILE}': {e}")

# Carrega ROIs salvos ao iniciar (sobrepõe os padrões acima se existir arquivo)
load_roi_config()

# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DA AUDITORIA – Câmera 02
# ─────────────────────────────────────────────────────────────────────────────
audit_state = {
    "camera_02": {
        "process_active": False,   # True quando verde detectado no cofre
        "last_furo_time": 0,       # Timestamp do último furo detectado
        "hand_in_cofre_since": 0,  # Timestamp início mão dentro do cofre
        "last_helmet_y": 0,        # Última posição Y do capacete
    },
    "test_audit": {
        "process_active": False,
        "last_furo_time": 0,
        "hand_in_cofre_since": 0,
        "last_helmet_y": 0,
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGS DE AUDITORIA (painel ao vivo)
# ─────────────────────────────────────────────────────────────────────────────
audit_logs = []

def add_audit_log(message):
    global audit_logs
    timestamp = time.strftime("%H:%M:%S")
    audit_logs.insert(0, f"[{timestamp}] {message}")
    if len(audit_logs) > 30:
        audit_logs.pop()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES GERAIS
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "alert_cooldown": 20,      # segundos entre alertas da mesma câmera
    "last_alert_time": {}
}

alert_history    = []
latest_frames    = {}
recording_states = {}          # cam_id → cv2.VideoWriter | None
lock = threading.Lock()

test_video_rule  = None
test_video_speed = 1.0

# BlobTrackers por câmera — exigem persistência temporal para confirmar fel
blob_trackers = {
    "camera_01": BlobTracker(min_frames=10, max_jump_px=110),
    "test_feed":  BlobTracker(min_frames=10, max_jump_px=110),
}


# Cache para detecção de movimento da esteira
last_roi_frames = {} 



# ─────────────────────────────────────────────────────────────────────────────
# CARREGA ALERTAS DO DIA DO DISCO
# ─────────────────────────────────────────────────────────────────────────────
def load_existing_alerts():
    global alert_history
    if not os.path.exists("alerts"):
        os.makedirs("alerts")
        return

    print("Carregando alertas existentes do dia...")
    files = sorted(os.listdir("alerts"), reverse=True)
    today_str = time.strftime("%Y%m%d")
    loaded = 0

    for filename in files:
        if not (filename.startswith("alert_") and filename.endswith(".jpg")):
            continue
        try:
            parts = filename.replace("alert_", "").replace(".jpg", "").split("_")
            cam_id = "camera_01"
            date_part = ""
            time_part = ""

            if len(parts) >= 2 and parts[0].startswith("camera"):
                cam_id   = f"{parts[0]}_{parts[1]}"
                date_part = parts[2].split("-")[0]
                time_part = parts[2].split("-")[1]
            elif len(parts) >= 1:
                date_part = parts[0].split("-")[0]
                time_part = parts[0].split("-")[1] if "-" in parts[0] else "000000"

            if date_part != today_str:
                continue

            # Verifica se há vídeo associado
            vid_name = filename.replace("alert_", "event_").replace(".jpg", ".mp4")
            vid_url  = f"/alerts_files/{vid_name}" if os.path.exists(os.path.join("alerts", vid_name)) else None

            alert_history.append({
                "id":        len(alert_history) + 1,
                "time":      f"{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}",
                "date":      f"{date_part[6:8]}/{date_part[4:6]}/{date_part[0:4]}",
                "camera":    CAMERAS.get(cam_id, {}).get("name", "Câmera"),
                "message":   "Rompimento de fel detectado na linha!",
                "image_url": f"/alerts_files/{filename}",
                "video_url": vid_url
            })
            loaded += 1
        except Exception as e:
            print(f"Erro ao carregar alerta {filename}: {e}")

    print(f"Total de {loaded} alertas carregados hoje.")

load_existing_alerts()

# ─────────────────────────────────────────────────────────────────────────────
# BANCO DE DADOS ORACLE
# ─────────────────────────────────────────────────────────────────────────────
def insert_alert_to_db(phone, message, frame):
    """Insere o alerta e a imagem redimensionada na tabela DIZIMO.MENSAGENS."""
    if not ORACLE_AVAILABLE: return
    if not phone or frame is None:
        print(f"[DB] CANCELADO: Telefone não configurado para este alerta.")
        return
    
    ORACLE_WALLET_PATH = "/home/rdt/CameraIA/DriveOracle"

    def run_insert():
        conn = None
        try:
            # Redimensiona para economia de espaço no banco (640x360)
            resized = cv2.resize(frame, (640, 360))
            _, img_encoded = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_bytes = img_encoded.tobytes()
            
            # String de conexão direta (DSN completo) que funcionou no teste
            dsn_direto = '(description=(address=(protocol=tcps)(port=1522)(host=adb.sa-vinhedo-1.oraclecloud.com))(connect_data=(service_name=g674a77dea23c6a_imaculado_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))'
            
            conn = oracledb.connect(
                user="mensagem", password="crbsAcs@2026", dsn=dsn_direto,
                config_dir=ORACLE_WALLET_PATH, wallet_location=ORACLE_WALLET_PATH
            )
            cursor = conn.cursor()
            sql = "INSERT INTO DIZIMO.MENSAGENS (TELEFONE, TEXTO, STATUS, TIPO, IMAGEM) VALUES (:1, :2, :3, :4, :5)"
            print(f"[DB] Executando INSERT para {phone}...")
            cursor.execute(sql, [str(phone), str(message), 0, 'G', img_bytes])
            conn.commit()
            print(f"[DB] Alerta gravado com sucesso no Oracle.")
        except Exception as e:
            print(f"[DB] Erro Oracle: {e}")
        finally:
            if conn: conn.close()
            
    threading.Thread(target=run_insert, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# DISPARO DE ALERTA
# ─────────────────────────────────────────────────────────────────────────────
def trigger_alert(message, frame, cam_id):
    # Verifica se os alertas estão habilitados para essa câmera
    cam_cfg = CAMERAS.get(cam_id, {})
    if not cam_cfg.get("alerts_enabled", True):
        print(f"[{cam_id}] Alerta IGNORADO (câmera pausada): {message}")
        return

    current_time = time.time()
    last_time    = CONFIG["last_alert_time"].get(cam_id, 0)

    if current_time - last_time <= CONFIG["alert_cooldown"]:
        return  # em cooldown, ignora

    CONFIG["last_alert_time"][cam_id] = current_time

    timestamp   = time.strftime("%Y%m%d-%H%M%S")
    img_name    = f"alert_{cam_id}_{timestamp}.jpg"
    img_path    = os.path.join("alerts", img_name)
    cv2.imwrite(img_path, frame)

    vid_name = f"event_{cam_id}_{timestamp}.mp4"
    vid_path = os.path.join("alerts", vid_name)

    alert_data = {
        "id":        len(alert_history) + 1,
        "time":      time.strftime("%H:%M:%S"),
        "date":      time.strftime("%d/%m/%Y"),
        "camera":    CAMERAS.get(cam_id, {}).get("name", "Simulador de Teste"),
        "message":   message,
        "image_url": f"/alerts_files/{img_name}",
        "video_url": f"/alerts_files/{vid_name}"   # sempre disponível desde o início
    }

    alert_history.insert(0, alert_data)
    if len(alert_history) > 50:
        alert_history.pop()

    print(f"ALERTA [{cam_id}]: {message}")

    # Gravação no Banco de Dados Oracle
    phone = cam_cfg.get("phone_number")
    
    # Se for o simulador, tenta pegar o telefone da regra sendo testada
    if cam_id == "test_feed":
        sim_cfg = CAMERAS.get(test_video_rule, {})
        phone = sim_cfg.get("phone_number")
        if phone:
            print(f"[test_feed] Usando telefone da regra {test_video_rule}: {phone}")

    # Fallback se vazio
    if not phone:
        for c in CAMERAS.values():
            if c.get("phone_number"):
                phone = c["phone_number"]
                break
    
    if phone:
        insert_alert_to_db(phone, message, frame)
    else:
        print(f"[{cam_id}] Nenhuma regra de telefone encontrada para registrar no banco.")

    # Gravação automática de 20 s
    def auto_record():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(vid_path, fourcc, 20.0, (640, 360))
        end_t  = time.time() + 20
        while time.time() < end_t:
            with lock:
                if cam_id in latest_frames and latest_frames[cam_id] is not None:
                    resized = cv2.resize(latest_frames[cam_id], (640, 360))
                    out.write(resized)
            time.sleep(0.05)
        out.release()
        print(f"Auto-gravação concluída: {vid_name}")

    threading.Thread(target=auto_record, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE AUDITORIA DE ANOMALIAS – Câmera 02
# ─────────────────────────────────────────────────────────────────────────────
def run_behavior_audit(frame, cam_id, state_data, zones):
    """
    Executa as 4 heurísticas de segurança e desenha as marcações no frame.
    Retorna o frame anotado.
    """
    operator = detect_operator(frame, np.array(zones["work_area"]))
    hands    = detect_hand(frame)

    # ── Desenho: Operador
    if operator:
        cv2.circle(frame, operator['center'], 20, (255, 255, 255), 2)
        cv2.putText(frame, "OPERADOR", (operator['center'][0]-30, operator['center'][1]-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Desenho: Zonas
    zone_colors = {"cofre": (0, 255, 0), "descarte": (0, 255, 255),
                   "pockets": (0, 165, 255), "work_area": (50, 50, 50)}
    for zname, pts in zones.items():
        color = zone_colors.get(zname, (255, 255, 0))
        cv2.polylines(frame, [np.array(pts)], True, color, 2)
        if zname != "work_area":
            cv2.putText(frame, zname.upper(), (pts[0][0], pts[0][1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ── Heurística 0: Gatilho de processo (verde no cofre)
    green_det, _ = detect_green_stain(frame, np.array(zones["cofre"]))
    if green_det:
        if not state_data["process_active"]:
            state_data["process_active"] = True
            add_audit_log("PROCESSO INICIADO: Rompimento detectado no cofre.")
        state_data["last_furo_time"] = time.time()
        cv2.putText(frame, "FURO DETECTADO", (820, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ── Heurística 1: Mão no bolso / cintura
    pocket_pts = np.array(zones["pockets"])
    hand_in_pocket = any(
        cv2.pointPolygonTest(pocket_pts, (float(h['center'][0]), float(h['center'][1])), False) >= 0
        for h in hands
    )
    if hand_in_pocket:
        msg = "ANOMALIA: Mão na cintura/bolso detectada!"
        trigger_alert(f"AUDITORIA: {msg}", frame, cam_id)
        add_audit_log(msg)

    # ── Heurística 2: Mão dentro do cofre por mais de 4 s
    cofre_pts = np.array(zones["cofre"])
    hand_in_cofre = any(
        cv2.pointPolygonTest(cofre_pts, (float(h['center'][0]), float(h['center'][1])), False) >= 0
        for h in hands
    )
    if hand_in_cofre:
        if state_data["hand_in_cofre_since"] == 0:
            state_data["hand_in_cofre_since"] = time.time()
        elif time.time() - state_data["hand_in_cofre_since"] > 4:
            msg = "ANOMALIA: Manipulação excessiva dentro do cofre!"
            trigger_alert(f"AUDITORIA: {msg}", frame, cam_id)
            add_audit_log(msg)
            state_data["hand_in_cofre_since"] = time.time()
    else:
        state_data["hand_in_cofre_since"] = 0

    # ── Heurística 3: Operador se abaixou (capacete muito baixo na imagem)
    if operator and operator['center'][1] > 950:
        msg = "ANOMALIA: Operador se abaixou!"
        trigger_alert(f"AUDITORIA: {msg}", frame, cam_id)
        add_audit_log(msg)

    # ── Heurística 4: Abandono imediato pós-furo
    # APENAS dispara se o processo foi ativado E o operador ESTAVA presente
    # e SAIU nos primeiros 3 s. Sem operador detectado previamente = sem alerta.
    if state_data["process_active"]:
        time_since = time.time() - state_data["last_furo_time"]
        if time_since < 3 and state_data["last_helmet_y"] > 0:
            # Operador estava presente (last_helmet_y preenchido) e agora sumiu
            if not operator:
                msg = "ANOMALIA: Abandono imediato pós-furo!"
                trigger_alert(f"AUDITORIA: {msg}", frame, cam_id)
                add_audit_log(msg)
                state_data["process_active"] = False
        elif time_since > 8:
            state_data["process_active"] = False
            add_audit_log("Ciclo concluído (permanência OK).")

    # Atualiza posição do capacete para próximo ciclo
    state_data["last_helmet_y"] = operator['center'][1] if operator else 0

    # ── Status na tela
    status_txt = "PROCESSO ATIVO" if state_data["process_active"] else "MONITORANDO..."
    cv2.putText(frame, f"AUDITORIA: {status_txt}", (710, 1050),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    return frame

# ─────────────────────────────────────────────────────────────────────────────
# THREAD DE CAPTURA DE CÂMERA AO VIVO
# ─────────────────────────────────────────────────────────────────────────────
def video_stream_thread(cam_id):
    cam_cfg = CAMERAS[cam_id]
    cap     = cv2.VideoCapture(cam_cfg["rtsp_url"])
    print(f"Iniciando thread para {cam_id}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_id}] Falha ao capturar frame. Reconectando...")
            time.sleep(2)
            cap.open(cam_cfg["rtsp_url"])
            continue

        roi_points = np.array(cam_cfg["roi"], np.int32)

        if cam_cfg["type"] == "color_detection":
            # ── Detecção de Movimento da Esteira ──
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [roi_points], 255)
            roi_gray = cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=mask), cv2.COLOR_BGR2GRAY)
            
            is_moving = True
            if cam_id in last_roi_frames:
                diff = cv2.absdiff(last_roi_frames[cam_id], roi_gray)
                movement = np.mean(diff[mask > 0]) if np.any(mask > 0) else 0
                is_moving = movement > 1.5 # Limiar de movimento
            last_roi_frames[cam_id] = roi_gray

            candidates, _ = detect_green_stain(frame, roi_points)
            tracker  = blob_trackers.get(cam_id)
            confirmed = tracker.update(candidates) if tracker else candidates
            
            for det in confirmed:
                x, y, w, h = det['rect']
                frames_txt = det.get('frames', '')
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, f"FEL ({frames_txt}f)", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if confirmed and is_moving:
                trigger_alert(f"Rompimento detectado - {cam_cfg['name']}", frame, cam_id)
            elif confirmed and not is_moving:
                cv2.putText(frame, "ESTEIRA PARADA - ALERTA BLOQUEADO", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        elif cam_cfg["type"] == "behavior_detection":
            frame = run_behavior_audit(frame, cam_id, audit_state[cam_id], cam_cfg["zones"])

        # Desenha o ROI na tela
        cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

        # Gravação manual
        with lock:
            if recording_states.get(cam_id):
                recording_states[cam_id].write(cv2.resize(frame, (640, 360)))
            latest_frames[cam_id] = frame

        time.sleep(0.05)  # ~20 FPS max, libera CPU

# ─────────────────────────────────────────────────────────────────────────────
# ROTAS FLASK
# ─────────────────────────────────────────────────────────────────────────────
def trigger_alert_wrapper(message, frame, cam_id):
    trigger_alert(message, frame, cam_id)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/alerts_files/<filename>')
def get_alert_image(filename):
    return send_from_directory('alerts', filename)

@app.route('/alerts')
def get_alerts():
    return jsonify(alert_history)

@app.route('/audit_logs')
def get_audit_logs():
    return jsonify(audit_logs)

@app.route('/toggle_alerts/<cam_id>', methods=['POST'])
def toggle_alerts(cam_id):
    if cam_id not in CAMERAS:
        return jsonify({"status": "error", "message": "Câmera não encontrada"})
    current = CAMERAS[cam_id].get("alerts_enabled", True)
    CAMERAS[cam_id]["alerts_enabled"] = not current
    state = "ATIVO" if CAMERAS[cam_id]["alerts_enabled"] else "PAUSADO"
    print(f"[{cam_id}] Alertas: {state}")
    return jsonify({"status": "success", "cam_id": cam_id, "alerts_enabled": CAMERAS[cam_id]["alerts_enabled"]})

@app.route('/camera_status')
def camera_status():
    """Retorna o estado de alerts_enabled de todas as câmeras"""
    return jsonify({cam_id: cfg.get("alerts_enabled", True) for cam_id, cfg in CAMERAS.items()})

@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'POST':
        CONFIG.update(request.json)
        return jsonify({"status": "success", "config": CONFIG})
    return jsonify(CONFIG)

@app.route('/snapshot/<cam_id>')
def snapshot(cam_id):
    """Retorna um frame único da câmera como JPEG para o editor de ROI."""
    with lock:
        frame = latest_frames.get(cam_id)
    if frame is None:
        return "Câmera sem frame disponível", 503
    flag, enc = cv2.imencode('.jpg', frame)
    if not flag:
        return "Erro ao codificar frame", 500
    from flask import make_response
    resp = make_response(enc.tobytes())
    resp.headers['Content-Type'] = 'image/jpeg'
    resp.headers['Cache-Control'] = 'no-store'
    return resp

@app.route('/save_roi/<cam_id>', methods=['POST'])
def save_roi(cam_id):
    """Salva ROI principal ou uma zona específica da câmera."""
    if cam_id not in CAMERAS:
        return jsonify({"status": "error", "message": "Câmera não encontrada"})
    data   = request.json
    points = data.get('points', [])
    zone   = data.get('zone', None)  # ex: 'cofre', 'descarte', 'pockets', 'work_area'
    if len(points) < 3:
        return jsonify({"status": "error", "message": "Mínimo de 3 pontos necessários"})
    if zone and 'zones' in CAMERAS[cam_id]:
        CAMERAS[cam_id]['zones'][zone] = points
        persist_roi_config()
        print(f"[{cam_id}] Zona '{zone}' atualizada e salva: {points}")
        return jsonify({"status": "success", "zone": zone, "roi": points})
    else:
        CAMERAS[cam_id]['roi'] = points
        persist_roi_config()
        print(f"[{cam_id}] ROI principal salvo: {points}")
        return jsonify({"status": "success", "roi": points})

@app.route('/get_roi/<cam_id>')
def get_roi(cam_id):
    """Retorna ROI principal e todas as zonas da câmera."""
    if cam_id not in CAMERAS:
        return jsonify({"status": "error"})
    cam = CAMERAS[cam_id]
    return jsonify({
        "roi":   cam.get('roi', []),
        "zones": cam.get('zones', {})
    })

@app.route('/roi_editor')
def roi_editor():
    """Página do editor visual de ROI."""
    return send_from_directory('.', 'roi_editor.html')


@app.route('/set_test_speed', methods=['POST'])
def set_test_speed():
    global test_video_speed
    test_video_speed = float(request.json.get('speed', 1.0))
    return jsonify({"status": "success", "speed": test_video_speed})

@app.route('/list_server_videos')
def list_server_videos():
    """Lista todos os vídeos .mp4 disponíveis no servidor (alerts/ e uploads/)."""
    videos = []
    search_dirs = [
        ('alerts',  'alerts'),
        ('uploads', 'uploads'),
    ]
    for folder, label in search_dirs:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith('.mp4'):
                continue
            fpath = os.path.join(folder, fname)
            stat  = os.stat(fpath)
            videos.append({
                "filename": fname,
                "folder":   label,
                "path":     f"/{folder}/{fname}",   # URL relativa para servir
                "size_mb":  round(stat.st_size / (1024 * 1024), 1),
                "mtime":    stat.st_mtime,
            })
    # Ordena do mais recente para o mais antigo
    videos.sort(key=lambda v: v["mtime"], reverse=True)
    return jsonify(videos)

@app.route('/use_server_video', methods=['POST'])
def use_server_video():
    """Aponta o motor de teste para um vídeo já existente no servidor."""
    global test_video_rule
    data     = request.json or {}
    folder   = data.get('folder', 'alerts')
    filename = data.get('filename', '')
    rule     = data.get('rule', 'camera_01')

    if not filename:
        return jsonify({"status": "error", "message": "filename obrigatório"})

    src_path = os.path.join(folder, filename)
    if not os.path.exists(src_path):
        return jsonify({"status": "error", "message": f"Arquivo não encontrado: {src_path}"})

    # Copia para o slot de teste — mas evita copiar arquivo sobre si mesmo
    os.makedirs('uploads', exist_ok=True)
    import shutil
    dest_path = os.path.join('uploads', 'test_video.mp4')
    src_abs   = os.path.realpath(src_path)
    dest_abs  = os.path.realpath(dest_path)
    if src_abs != dest_abs:
        shutil.copy2(src_path, dest_path)
        print(f"[TESTE] Vídeo copiado: {src_path} → {dest_path}")
    else:
        print(f"[TESTE] Vídeo já no slot de teste: {src_path}")
    test_video_rule = rule
    print(f"[TESTE] Regra ativa: {rule}")
    return jsonify({"status": "success"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global test_video_rule
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file"})
    file = request.files['video']
    rule = request.form.get('rule', 'camera_01')
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"})
    os.makedirs('uploads', exist_ok=True)
    file.save(os.path.join('uploads', 'test_video.mp4'))
    test_video_rule = rule
    return jsonify({"status": "success"})

@app.route('/start_record/<cam_id>')
def start_record(cam_id):
    with lock:
        if recording_states.get(cam_id):
            return jsonify({"status": "error", "message": "Already recording"})
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename  = f"record_{cam_id}_{timestamp}.mp4"
        filepath  = os.path.join("alerts", filename)
        fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
        recording_states[cam_id] = cv2.VideoWriter(filepath, fourcc, 20.0, (640, 360))
    return jsonify({"status": "success", "filename": filename})

@app.route('/stop_record/<cam_id>')
def stop_record(cam_id):
    with lock:
        if recording_states.get(cam_id):
            recording_states[cam_id].release()
            recording_states[cam_id] = None
            return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Not recording"})

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):

    # ── Modo de Teste (vídeo enviado pelo usuário) ────────────────────────────
    def generate_test():
        filepath = os.path.join('uploads', 'test_video.mp4')
        if not os.path.exists(filepath):
            return

        cap     = cv2.VideoCapture(filepath)
        cam_cfg = CAMERAS.get(test_video_rule, CAMERAS["camera_02"])

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        time_debt = 0.0

        while cap.isOpened():
            target_delay = 1.0 / (fps * test_video_speed)
            start_time   = time.time()

            # Pula frames se estiver atrasado
            if time_debt > target_delay:
                skip = min(int(time_debt / target_delay), int(fps))
                for _ in range(skip):
                    cap.grab()
                time_debt -= skip * target_delay

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time_debt = 0.0
                continue

            roi_points = np.array(cam_cfg["roi"], np.int32)
            state_key  = "test_audit"

            if cam_cfg["type"] == "color_detection":
                # ── Detecção de Movimento (Teste) ──
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [roi_points], 255)
                roi_gray = cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=mask), cv2.COLOR_BGR2GRAY)
                is_moving = True
                if "test_feed" in last_roi_frames:
                    diff = cv2.absdiff(last_roi_frames["test_feed"], roi_gray)
                    movement = np.mean(diff[mask > 0]) if np.any(mask > 0) else 0
                    is_moving = movement > 1.5
                last_roi_frames["test_feed"] = roi_gray

                candidates, _ = detect_green_stain(frame, roi_points)
                tracker   = blob_trackers.get("test_feed")
                confirmed = tracker.update(candidates) if tracker else candidates
                for det in confirmed:
                    x, y, w, h = det['rect']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, f"FEL ({det.get('frames','')}f)", (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                if confirmed and is_moving:
                    cv2.putText(frame, "ALERTA: RUPTURA DETECTADA (TESTE)",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    trigger_alert(f"[TESTE] Rompimento - {cam_cfg['name']}", frame, "test_feed")
                elif confirmed and not is_moving:
                    cv2.putText(frame, "ESTEIRA PARADA (TESTE)", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif cam_cfg["type"] == "behavior_detection":
                frame = run_behavior_audit(frame, "test_feed",
                                           audit_state[state_key], cam_cfg["zones"])

            # ROI e label de modo teste
            cv2.polylines(frame, [roi_points], True, (255, 165, 0), 2)
            cv2.putText(frame, "MODO TESTE", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

            # Gravação de teste
            with lock:
                if recording_states.get("test_feed"):
                    recording_states["test_feed"].write(cv2.resize(frame, (640, 360)))

            flag, enc = cv2.imencode(".jpg", frame)
            if flag:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + enc.tobytes() + b'\r\n\r\n')

            elapsed    = time.time() - start_time
            time_debt += elapsed - target_delay
            if time_debt < 0:
                time.sleep(-time_debt)
                time_debt = 0.0

        cap.release()

    # ── Câmera ao Vivo ────────────────────────────────────────────────────────
    def generate_live():
        # Frame de placeholder enviado enquanto a câmera conecta
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Conectando camera...", (120, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        cv2.putText(placeholder, cam_id.replace("_", " ").upper(), (180, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1)
        _, ph_enc = cv2.imencode(".jpg", placeholder)
        ph_bytes  = ph_enc.tobytes()

        while True:
            with lock:
                frame = latest_frames.get(cam_id)

            if frame is None:
                # Envia o placeholder para o browser não girar
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + ph_bytes + b'\r\n\r\n')
                time.sleep(0.5)
                continue

            flag, enc = cv2.imencode(".jpg", frame)
            if flag:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + enc.tobytes() + b'\r\n\r\n')
            time.sleep(0.04)   # ~25 FPS para o navegador

    if cam_id == "test_feed":
        return Response(generate_test(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_live(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ─────────────────────────────────────────────────────────────────────────────
# INICIALIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('alerts', exist_ok=True)

    for cam_id in CAMERAS:
        t = threading.Thread(target=video_stream_thread, args=(cam_id,), daemon=True)
        t.start()

    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
