import os
import cv2
import time
import requests
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import numpy as np
from detector import detect_green_stain, find_cofre

app = Flask(__name__)
CORS(app)

# Configuration for Multiple Cameras
CAMERAS = {
    "camera_01": {
        "name": "Esteira - Ruptura de Fel",
        "rtsp_url": "rtsp://admin:013579ab@10.200.34.50:554/cam/realmonitor?channel=1&subtype=0",
        "roi": [[280, 500], [720, 440], [1150, 1070], [400, 1070]],
        "type": "color_detection"
    },
    "camera_02": {
        "name": "Cofre - Fluxo de Vesícula",
        "rtsp_url": "rtsp://admin:013579ab@10.200.96.81:554/cam/realmonitor?channel=1&subtype=0",
        "roi": [[100, 100], [400, 100], [400, 400], [100, 400]], # Placeholder
        "type": "behavior_detection"
    }
}

CONFIG = {
    "alert_cooldown": 15,
    "last_alert_time": {} # Store per camera
}

# In-memory alert history
alert_history = []
latest_frames = {}
recording_states = {} # cam_id: VideoWriter or None
lock = threading.Lock()

# Test Video State
test_video_rule = None
test_video_speed = 1.0

def load_existing_alerts():
    global alert_history
    if not os.path.exists("alerts"):
        os.makedirs("alerts")
        return

    print("Carregando alertas existentes do dia...")
    files = sorted(os.listdir("alerts"), reverse=True)
    today_str = time.strftime("%Y%m%d")
    
    loaded_count = 0
    for filename in files:
        if filename.startswith("alert_") and filename.endswith(".jpg"):
            try:
                # Format expected: alert_camera_01_YYYYMMDD-HHMMSS.jpg
                # or legacy: alert_YYYYMMDD-HHMMSS.jpg
                parts = filename.replace("alert_", "").replace(".jpg", "").split("_")
                
                cam_id = "camera_01" # default
                date_part = ""
                
                if len(parts) >= 2:
                    if parts[0].startswith("camera"):
                        cam_id = f"{parts[0]}_{parts[1]}"
                        date_part = parts[2].split("-")[0]
                        time_part = parts[2].split("-")[1]
                    else:
                        date_part = parts[0].split("-")[0]
                        time_part = parts[0].split("-")[1]

                # Filtra apenas alertas de hoje
                if date_part != today_str:
                    continue
                
                formatted_date = f"{date_part[6:8]}/{date_part[4:6]}/{date_part[0:4]}"
                formatted_time = f"{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}"
                
                alert_history.append({
                    "id": len(alert_history) + 1,
                    "time": formatted_time,
                    "date": formatted_date,
                    "camera": CAMERAS.get(cam_id, {}).get("name", "Câmera Antiga"),
                    "message": "Rompimento de fel detectado na linha!",
                    "image_url": f"/alerts_files/{filename}"
                })
                loaded_count += 1
            except Exception as e:
                print(f"Erro ao carregar alerta {filename}: {e}")
                continue
    print(f"Total de {loaded_count} alertas carregados hoje.")

# Inicializa o histórico com os arquivos já existentes
load_existing_alerts()

def video_stream_thread(cam_id):
    global latest_frames
    cam_cfg = CAMERAS[cam_id]
    cap = cv2.VideoCapture(cam_cfg["rtsp_url"])
    
    print(f"Iniciando thread para {cam_id}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame for {cam_id}, retrying...")
            cap.open(cam_cfg["rtsp_url"])
            time.sleep(2)
            continue
            
        # Process detection based on camera type
        roi_points = np.array(cam_cfg["roi"], np.int32)
        
        if cam_cfg["type"] == "color_detection":
            detections, _ = detect_green_stain(frame, roi_points)
            for det in detections:
                x, y, w, h = det['rect']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                trigger_alert(f"Rompimento detectado - {cam_cfg['name']}", frame, cam_id)
        
        elif cam_cfg["type"] == "behavior_detection":
            # 1. Tentar localizar o cofre dinamicamente
            dynamic_roi = find_cofre(frame)
            if dynamic_roi is not None:
                cam_cfg["roi"] = dynamic_roi.tolist()
                roi_points = dynamic_roi
                cv2.polylines(frame, [roi_points], True, (255, 255, 0), 3)
                cv2.putText(frame, "COFRE LOCALIZADO", (roi_points[0][0], roi_points[0][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # 2. Verificar se há líquido verde caindo no cofre (dentro do ROI dinâmico)
                detections, _ = detect_green_stain(frame, roi_points)
                if detections:
                    cv2.putText(frame, "VERDE NO COFRE OK", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Se não achar o cofre, usa o último ROI conhecido mas avisa
                cv2.polylines(frame, [roi_points], True, (0, 0, 255), 1)
                cv2.putText(frame, "BUSCANDO COFRE...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Handle Recording
        with lock:
            if cam_id in recording_states and recording_states[cam_id]:
                # Redimensiona para WhatsApp antes de gravar
                resized = cv2.resize(frame, (640, 360))
                recording_states[cam_id].write(resized)

        # Draw ROI lines
        cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

        with lock:
            latest_frames[cam_id] = frame
            
        time.sleep(0.05)

# A lógica de teste agora será executada no generator do Flask para sincronia perfeita

def trigger_alert(message, frame, cam_id):
    current_time = time.time()
    last_time = CONFIG["last_alert_time"].get(cam_id, 0)
    
    if current_time - last_time > CONFIG["alert_cooldown"]:
        CONFIG["last_alert_time"][cam_id] = current_time
        
        # Save image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"alert_{cam_id}_{timestamp}.jpg"
        filepath = os.path.join("alerts", filename)
        cv2.imwrite(filepath, frame)
        
        alert_data = {
            "id": len(alert_history) + 1,
            "time": time.strftime("%H:%M:%S"),
            "date": time.strftime("%d/%m/%Y"),
            "camera": CAMERAS.get(cam_id, {"name": "Simulador de Teste"})["name"],
            "message": message,
            "image_url": f"/alerts_files/{filename}"
        }
        alert_history.insert(0, alert_data) # Newest first
        if len(alert_history) > 50: alert_history.pop() # Keep last 50
        
        print(f"ALERT SAVED [{cam_id}]: {message} -> {filepath}")
        
        # Iniciar gravação automática de 15 segundos
        def auto_stop_recording(cid, vid_name):
            time.sleep(15)
            with lock:
                if cid in recording_states and recording_states[cid]:
                    recording_states[cid].release()
                    recording_states[cid] = None
                    print(f"AUTO-RECORD STOPPED for {cid}: {vid_name}")

        with lock:
            if cam_id not in recording_states or not recording_states[cam_id]:
                vid_filename = f"event_{cam_id}_{timestamp}.mp4"
                vid_filepath = os.path.join("alerts", vid_filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # Resolução reduzida para compatibilidade com WhatsApp (640x360)
                out = cv2.VideoWriter(vid_filepath, fourcc, 20.0, (640, 360))
                recording_states[cam_id] = out
                alert_data["video_url"] = f"/alerts_files/{vid_filename}"
                threading.Thread(target=auto_stop_recording, args=(cam_id, vid_filename), daemon=True).start()
                print(f"AUTO-RECORD STARTED for {cam_id}: {vid_filename}")
        
        # WhatsApp integration placeholder
        try:
            pass
        except Exception as e:
            print(f"Failed to send alert: {e}")

@app.route('/alerts_files/<filename>')
def get_alert_image(filename):
    return send_from_directory('alerts', filename)

@app.route('/alerts')
def get_alerts():
    return jsonify(alert_history)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    def generate_live():
        while True:
            with lock:
                if cam_id not in latest_frames:
                    continue
                (flag, encodedImage) = cv2.imencode(".jpg", latest_frames[cam_id])
                if not flag:
                    continue
                frame_bytes = encodedImage.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                   frame_bytes + b'\r\n\r\n')
            time.sleep(0.04)

    def generate_test():
        filepath = os.path.join('uploads', "test_video.mp4")
        if not os.path.exists(filepath):
            return
            
        cap = cv2.VideoCapture(filepath)
        cam_cfg = CAMERAS.get(test_video_rule, CAMERAS["camera_01"])
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25.0
        
        time_debt = 0.0
        
        while cap.isOpened():
            target_delay = 1.0 / (fps * test_video_speed)
            start_time = time.time()
            
            # Se o processamento estiver lento ou a velocidade for > 1x, pulamos frames
            if time_debt > target_delay:
                frames_to_skip = int(time_debt / target_delay)
                frames_to_skip = min(frames_to_skip, int(fps)) # Pula no máximo 1s de vídeo para não travar
                for _ in range(frames_to_skip):
                    cap.grab()
                time_debt -= frames_to_skip * target_delay
                
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop infinito do teste
                time_debt = 0.0
                continue
                
            roi_points = np.array(cam_cfg["roi"], np.int32)
            
            if cam_cfg["type"] == "color_detection":
                detections, mask = detect_green_stain(frame, roi_points)
                for det in detections:
                    x, y, w, h = det['rect']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                if detections:
                    cv2.putText(frame, "ALERTA: RUPTURA DETECTADA (TESTE)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    trigger_alert(f"[TESTE] Rompimento - {cam_cfg['name']}", frame, "test_feed")
            
            elif cam_cfg["type"] == "behavior_detection":
                dynamic_roi = find_cofre(frame)
                if dynamic_roi is not None:
                    cam_cfg["roi"] = dynamic_roi.tolist()
                    roi_points = dynamic_roi
                    cv2.polylines(frame, [roi_points], True, (255, 255, 0), 3)
                    cv2.putText(frame, "COFRE LOCALIZADO", (roi_points[0][0], roi_points[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    detections, _ = detect_green_stain(frame, roi_points)
                    if detections:
                        cv2.putText(frame, "VERDE NO COFRE OK", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.polylines(frame, [roi_points], True, (0, 0, 255), 1)
                    cv2.putText(frame, "BUSCANDO COFRE...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.polylines(frame, [roi_points], True, (255, 165, 0), 2)
            cv2.putText(frame, "MODO TESTE", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if flag:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n\r\n')
            
            # Gravação automática no modo de teste
            with lock:
                if "test_feed" in recording_states and recording_states["test_feed"]:
                    resized_test = cv2.resize(frame, (640, 360))
                    recording_states["test_feed"].write(resized_test)
                
            elapsed = time.time() - start_time
            time_debt += elapsed - target_delay
            
            if time_debt < 0:
                # Processamos mais rápido, então dormimos a diferença exata
                sleep_sec = -time_debt
                time.sleep(sleep_sec)
                time_debt = 0.0

        cap.release()

    if cam_id == "test_feed":
        return Response(generate_test(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_live(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_record/<cam_id>')
def start_record(cam_id):
    global recording_states
    if cam_id not in latest_frames:
        return jsonify({"status": "error", "message": "Camera not active"})
    
    with lock:
        if cam_id in recording_states and recording_states[cam_id]:
            return jsonify({"status": "error", "message": "Already recording"})
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"record_{cam_id}_{timestamp}.mp4"
        filepath = os.path.join("alerts", filename)
        
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, 20.0, (640, 360))
        recording_states[cam_id] = out
        
    return jsonify({"status": "success", "filename": filename})

@app.route('/stop_record/<cam_id>')
def stop_record(cam_id):
    global recording_states
    with lock:
        if cam_id in recording_states and recording_states[cam_id]:
            recording_states[cam_id].release()
            recording_states[cam_id] = None
            return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Not recording"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global test_video_rule
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file part"})
    
    file = request.files['video']
    rule = request.form.get('rule', 'camera_01')
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})
        
    if file:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
            
        filepath = os.path.join('uploads', "test_video.mp4")
        file.save(filepath)
        
        test_video_rule = rule
        
        return jsonify({"status": "success"})

@app.route('/set_test_speed', methods=['POST'])
def set_test_speed():
    global test_video_speed
    data = request.json
    speed = data.get('speed', 1.0)
    test_video_speed = float(speed)
    return jsonify({"status": "success", "speed": test_video_speed})

@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'POST':
        new_config = request.json
        CONFIG.update(new_config)
        return jsonify({"status": "success", "config": CONFIG})
    return jsonify(CONFIG)

if __name__ == '__main__':
    # Ensure alerts folder exists
    if not os.path.exists('alerts'):
        os.makedirs('alerts')
    
    # Load history
    load_existing_alerts()

    # Start threads for each camera
    for cam_id in CAMERAS:
        t = threading.Thread(target=video_stream_thread, args=(cam_id,))
        t.daemon = True
        t.start()

    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
