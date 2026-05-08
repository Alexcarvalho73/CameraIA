import cv2
import numpy as np
import time
import os
import threading
import json
import shutil
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from detector import detect_operator, run_behavior_audit, detect_green_stain, BlobTracker

try:
    import oracledb
    ORACLE_AVAILABLE = True
    # Ativa o modo Thick para suportar Auto-login Wallet (cwallet.sso) sem senha PEM
    try:
        instant_client_path = "/home/rdt/CameraIA/instantclient_21_1"
        oracle_wallet_path = "/home/rdt/CameraIA/DriveOracle"
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
# CONFIGURAÇÃO DAS CÂMERAS E MOTOR
# ─────────────────────────────────────────────────────────────────────────────
CAMERAS = {
    "camera_01": {
        "name": "Esteira Principal - Ruptura",
        "rtsp_url": "rtsp://admin:013579ab@10.200.34.50:554/cam/realmonitor?channel=1&subtype=0",
        "roi": [[280, 375], [790, 320], [810, 710], [195, 750]],
        "type": "color_detection",
        "alerts_enabled": True,
        "phone_number": ""
    },
    "camera_02": {
        "name": "Cofre - Fluxo de Vesícula",
        "rtsp_url": "rtsp://admin:013579ab@10.200.96.81:554/cam/realmonitor?channel=1&subtype=0",
        "roi": [[700, 20], [1900, 20], [1900, 1000], [700, 1000]],
        "zones": {
            "cofre":     [[800, 20],  [1150, 20],  [1150, 300], [800, 300]],
            "descarte":  [[850, 320], [1150, 320], [1150, 600], [850, 600]],
        },
        "type": "behavior_detection",
        "alerts_enabled": False,
        "phone_number": ""
    }
}

CONFIG = {
    "alert_cooldown": 15,
    "last_alert_time": {}
}

ROI_CONFIG_FILE = "roi_config.json"
ORACLE_WALLET_PATH = "/home/rdt/CameraIA/DriveOracle"

def load_roi_config():
    if not os.path.exists(ROI_CONFIG_FILE): return
    try:
        with open(ROI_CONFIG_FILE, 'r') as f:
            saved = json.load(f)
        for cam_id, cfg in saved.items():
            if cam_id not in CAMERAS: continue
            if 'roi' in cfg: CAMERAS[cam_id]['roi'] = cfg['roi']
            if 'phone_number' in cfg: CAMERAS[cam_id]['phone_number'] = cfg['phone_number']
            if 'zones' in cfg and 'zones' in CAMERAS[cam_id]:
                for zn, pts in cfg['zones'].items(): CAMERAS[cam_id]['zones'][zn] = pts
        print(f"[ROI] Configurações carregadas de '{ROI_CONFIG_FILE}'.")
    except Exception as e: print(f"[ROI] Erro ao carregar: {e}")

def persist_roi_config():
    try:
        data = {cid: {"roi": cfg.get("roi", []), "zones": cfg.get("zones", {}), "phone_number": cfg.get("phone_number", "")} 
                for cid, cfg in CAMERAS.items()}
        with open(ROI_CONFIG_FILE, 'w') as f: json.dump(data, f, indent=2)
    except Exception as e: print(f"[ROI] Erro ao salvar: {e}")

load_roi_config()

# Trackers e estados
blob_trackers = {
    "camera_01": BlobTracker(min_frames=10, max_jump_px=110),
    "test_feed":  BlobTracker(min_frames=10, max_jump_px=110),
}
audit_state = {
    "camera_02": {"process_active": False, "last_furo_time": 0},
    "test_audit": {"process_active": False, "last_furo_time": 0}
}
audit_logs = []
alert_history = []
latest_frames = {}
recording_states = {}
lock = threading.Lock()
last_roi_frames = {}
test_video_rule = "camera_01"
test_video_speed = 1.0

def add_audit_log(msg):
    log_entry = {"time": time.strftime("%H:%M:%S"), "message": msg}
    audit_logs.insert(0, log_entry)
    if len(audit_logs) > 50: audit_logs.pop()
    print(f"[AUDIT] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# BANCO DE DADOS ORACLE
# ─────────────────────────────────────────────────────────────────────────────
def insert_alert_to_db(phone, message, frame):
    if not ORACLE_AVAILABLE: return
    if not phone or frame is None:
        print(f"[DB] CANCELADO: Telefone não configurado. Alerta: {message}")
        return
    
    def run_insert():
        conn = None
        try:
            resized = cv2.resize(frame, (640, 360))
            _, img_encoded = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_bytes = img_encoded.tobytes()
            
            conn = oracledb.connect(
                user="mensagem", password="crbsAcs@2026", dsn="imaculado",
                config_dir=ORACLE_WALLET_PATH, wallet_location=ORACLE_WALLET_PATH
            )
            cursor = conn.cursor()
            sql = "INSERT INTO DIZIMO.MENSAGENS (TELEFONE, TEXTO, STATUS, TIPO, IMAGEM) VALUES (:1, :2, :3, :4, :5)"
            print(f"[DB] Executando INSERT para {phone}...")
            cursor.execute(sql, [str(phone), str(message), 0, 'G', img_bytes])
            conn.commit()
            print(f"[DB] Alerta gravado com sucesso no Oracle.")
        except Exception as e: print(f"[DB] Erro Oracle: {e}")
        finally:
            if conn: conn.close()
            
    threading.Thread(target=run_insert, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# DISPARO DE ALERTA
# ─────────────────────────────────────────────────────────────────────────────
def trigger_alert(message, frame, cam_id):
    cam_cfg = CAMERAS.get(cam_id, {})
    if not cam_cfg.get("alerts_enabled", True): return

    now = time.time()
    if now - CONFIG["last_alert_time"].get(cam_id, 0) <= CONFIG["alert_cooldown"]: return
    CONFIG["last_alert_time"][cam_id] = now

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    img_name = f"alert_{cam_id}_{timestamp}.jpg"
    cv2.imwrite(os.path.join("alerts", img_name), frame)

    alert_data = {
        "id": len(alert_history) + 1,
        "time": time.strftime("%H:%M:%S"),
        "camera": cam_cfg.get("name", cam_id),
        "message": message,
        "image_url": f"/alerts_files/{img_name}"
    }
    alert_history.insert(0, alert_data)
    if len(alert_history) > 100: alert_history.pop()

    # Gravação no Banco
    phone = cam_cfg.get("phone_number")
    if cam_id == "test_feed":
        phone = CAMERAS.get(test_video_rule, {}).get("phone_number")
    
    # Fallback se vazio
    if not phone:
        for c in CAMERAS.values():
            if c.get("phone_number"):
                phone = c["phone_number"]
                break

    insert_alert_to_db(phone, message, frame)
    print(f"ALERTA [{cam_id}]: {message}")

# ─────────────────────────────────────────────────────────────────────────────
# THREADS E ROTAS (Resumido para o essencial da 157f0b3)
# ─────────────────────────────────────────────────────────────────────────────
def video_stream_thread(cam_id):
    cam_cfg = CAMERAS[cam_id]
    cap = cv2.VideoCapture(cam_cfg["rtsp_url"])
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(2)
            cap.open(cam_cfg["rtsp_url"])
            continue
        
        with lock: latest_frames[cam_id] = frame.copy()
        roi_points = np.array(cam_cfg["roi"], np.int32)

        if cam_cfg["type"] == "color_detection":
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [roi_points], 255)
            roi_gray = cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=mask), cv2.COLOR_BGR2GRAY)
            is_moving = True
            if cam_id in last_roi_frames:
                diff = cv2.absdiff(last_roi_frames[cam_id], roi_gray)
                movement = np.mean(diff[mask > 0]) if np.any(mask > 0) else 0
                is_moving = movement > 1.5
            last_roi_frames[cam_id] = roi_gray

            candidates, _ = detect_green_stain(frame, roi_points)
            tracker = blob_trackers.get(cam_id)
            confirmed = tracker.update(candidates) if tracker else candidates
            
            for det in confirmed:
                x, y, w, h = det['rect']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            if confirmed and is_moving:
                trigger_alert(f"Rompimento detectado - {cam_cfg['name']}", frame, cam_id)

        elif cam_cfg["type"] == "behavior_detection":
            run_behavior_audit(frame, cam_id, audit_state[cam_id], cam_cfg["zones"])

@app.route('/')
def index(): return send_from_directory('.', 'index.html')

@app.route('/alerts_files/<filename>')
def get_alert_image(filename): return send_from_directory('alerts', filename)

@app.route('/alerts')
def get_alerts(): return jsonify(alert_history)

@app.route('/audit_logs')
def get_audit_logs(): return jsonify(audit_logs)

@app.route('/camera_status')
def camera_status():
    return jsonify({cid: {"alerts_enabled": cfg.get("alerts_enabled", True), "phone_number": cfg.get("phone_number", ""), "name": cfg.get("name", "")} 
                    for cid, cfg in CAMERAS.items()})

@app.route('/set_test_video', methods=['POST'])
def set_test_video():
    global test_video_rule
    data = request.json
    test_video_rule = data.get('rule', 'camera_01')
    return jsonify({"status": "success"})

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    def generate_live():
        while True:
            with lock: frame = latest_frames.get(cam_id)
            if frame is None:
                time.sleep(0.5); continue
            _, enc = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + enc.tobytes() + b'\r\n\r\n')
            time.sleep(0.04)

    def generate_test():
        global test_video_rule
        filepath = os.path.join('uploads', 'test_video.mp4')
        if not os.path.exists(filepath): return
        cap = cv2.VideoCapture(filepath)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            
            cam_cfg = CAMERAS.get(test_video_rule, CAMERAS["camera_01"])
            roi_points = np.array(cam_cfg["roi"], np.int32)
            
            # Detecção no teste (simplificada mas fiel à 157f0b3)
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
            tracker = blob_trackers.get("test_feed")
            confirmed = tracker.update(candidates) if tracker else candidates
            
            if confirmed and is_moving:
                trigger_alert(f"[TESTE] Rompimento - {cam_cfg['name']}", frame, "test_feed")
            
            for det in confirmed:
                x, y, w, h = det['rect']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            _, enc = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + enc.tobytes() + b'\r\n\r\n')
            time.sleep(0.05)
        cap.release()

    if cam_id == "test_feed":
        return Response(generate_test(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_live(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    os.makedirs('alerts', exist_ok=True)
    for cam_id in CAMERAS:
        threading.Thread(target=video_stream_thread, args=(cam_id,), daemon=True).start()
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
