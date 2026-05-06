import os
import cv2
import time
import requests
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import numpy as np
from detector import detect_green_stain

app = Flask(__name__)
CORS(app)

# Configuration (In a real app, these would come from a database or .env)
CONFIG = {
    "rtsp_url": "rtsp://admin:013579ab@10.200.34.50:554/cam/realmonitor?channel=1&subtype=0",
    "roi": [[360, 540], [640, 480], [1080, 1020], [520, 1020]],
    "whatsapp_number": "5511999999999",
    "alert_webhook": "http://localhost:3000/send-whatsapp", # Example endpoint from previous project
    "last_alert_time": 0,
    "alert_cooldown": 5 # Reduced for testing, adjust as needed
}

# In-memory alert history
alert_history = []

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
                # alert_YYYYMMDD-HHMMSS.jpg
                parts = filename.replace("alert_", "").replace(".jpg", "").split("-")
                file_date_str = parts[0]
                file_time_str = parts[1]
                
                # Filtra apenas alertas de hoje
                if file_date_str != today_str:
                    continue
                
                formatted_date = f"{file_date_str[6:8]}/{file_date_str[4:6]}/{file_date_str[0:4]}"
                formatted_time = f"{file_time_str[0:2]}:{file_time_str[2:4]}:{file_time_str[4:6]}"
                
                alert_history.append({
                    "id": len(alert_history) + 1,
                    "time": formatted_time,
                    "date": formatted_date,
                    "message": "Rompimento de fel detectado na linha!",
                    "image_url": f"/alerts_files/{filename}"
                })
                loaded_count += 1
            except Exception as e:
                continue
    print(f"Total de {loaded_count} alertas carregados hoje.")

# Inicializa o histórico com os arquivos já existentes
load_existing_alerts()

# Global variable for the latest frame
latest_frame = None
lock = threading.Lock()

def video_stream_thread():
    global latest_frame
    cap = cv2.VideoCapture(CONFIG["rtsp_url"])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            cap.open(CONFIG["rtsp_url"])
            time.sleep(2)
            continue
            
        # Process detection
        roi_points = np.array(CONFIG["roi"], np.int32)
        detections, _ = detect_green_stain(frame, roi_points)
        
        # Draw detections on frame for visual feedback
        for det in detections:
            x, y, w, h = det['rect']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "ROMPIMENTO FEL", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Trigger Alert
            trigger_alert(f"Rompimento de fel detectado na linha!", frame)

        # Draw ROI lines
        cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

        with lock:
            latest_frame = frame.copy()
            
        time.sleep(0.03) # ~30 FPS

def trigger_alert(message, frame):
    current_time = time.time()
    if current_time - CONFIG["last_alert_time"] > CONFIG["alert_cooldown"]:
        CONFIG["last_alert_time"] = current_time
        
        # Save image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"alert_{timestamp}.jpg"
        filepath = os.path.join("alerts", filename)
        cv2.imwrite(filepath, frame)
        
        alert_data = {
            "id": len(alert_history) + 1,
            "time": time.strftime("%H:%M:%S"),
            "date": time.strftime("%d/%m/%Y"),
            "message": message,
            "image_url": f"/alerts_files/{filename}"
        }
        alert_history.insert(0, alert_data) # Newest first
        if len(alert_history) > 50: alert_history.pop() # Keep last 50
        
        print(f"ALERT SAVED: {message} -> {filepath}")
        
        # WhatsApp integration placeholder
        try:
            # requests.post(CONFIG["alert_webhook"], json={"to": CONFIG["whatsapp_number"], "message": message})
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

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if latest_frame is None:
                    continue
                (flag, encodedImage) = cv2.imencode(".jpg", latest_frame)
                if not flag:
                    continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                   bytearray(encodedImage) + b'\r\n')
            time.sleep(0.04)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'POST':
        new_config = request.json
        CONFIG.update(new_config)
        return jsonify({"status": "success", "config": CONFIG})
    return jsonify(CONFIG)

if __name__ == '__main__':
    # Start the video thread
    t = threading.Thread(target=video_stream_thread)
    t.daemon = True
    t.start()
    
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
