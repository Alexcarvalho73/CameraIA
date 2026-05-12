import cv2
import numpy as np
from ultralytics import YOLO
import sys

def generate_visualization():
    # Carrega modelo
    try:
        model = YOLO('yolov8n-pose.pt')
    except:
        print("Erro ao carregar modelo.")
        return

    # Usando a terceira imagem que o usuário enviou como base
    img_path = r"C:\Users\admin\.gemini\antigravity\brain\097e26ad-cc76-4874-a87f-fec374f80093\media__1778608692050.jpg"
    frame = cv2.imread(img_path)
    if frame is None:
        print("Erro ao carregar a imagem:", img_path)
        # Tentar outra imagem
        img_path = r"C:\Users\admin\.gemini\antigravity\brain\097e26ad-cc76-4874-a87f-fec374f80093\media__1778608692072.jpg"
        frame = cv2.imread(img_path)
        if frame is None:
            print("Não foi possível carregar as imagens de teste.")
            return

    results = model(frame, verbose=False, conf=0.15, classes=[0])

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        keypoints = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []
        confs = r.keypoints.conf.cpu().numpy() if r.keypoints is not None else []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Plano C: Bounding Box (Vermelho com transparência)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "PLANO C", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if len(keypoints) > i and len(keypoints[i]) >= 11:
                kp = keypoints[i]
                c_conf = confs[i]
                
                # Plano B: Cotovelos (Amarelo)
                for j in [7, 8]:
                    if c_conf[j] > 0.15:
                        pt = (int(kp[j][0]), int(kp[j][1]))
                        cv2.circle(frame, pt, 150, (0, 255, 255), 3)
                        cv2.putText(frame, "PLANO B", (pt[0]-40, pt[1]-155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.circle(frame, pt, 5, (0, 255, 255), -1)
                        
                # Plano A: Pulsos (Verde)
                for j in [9, 10]:
                    if c_conf[j] > 0.15:
                        pt = (int(kp[j][0]), int(kp[j][1]))
                        cv2.circle(frame, pt, 150, (0, 255, 0), 3)
                        cv2.putText(frame, "PLANO A", (pt[0]-40, pt[1]-155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    out_path = r"C:\Users\admin\.gemini\antigravity\brain\097e26ad-cc76-4874-a87f-fec374f80093\visualization_plans.jpg"
    cv2.imwrite(out_path, frame)
    print("SUCESSO:", out_path)

if __name__ == "__main__":
    generate_visualization()
