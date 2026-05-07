import cv2
url = 'rtsp://admin:013579ab@10.200.34.50:554/cam/realmonitor?channel=1&subtype=0'
cap = cv2.VideoCapture(url)
ret, f = cap.read()
if ret:
    print(f'Resolucao: {f.shape[1]}x{f.shape[0]}')
else:
    print('Falha ao capturar frame')
cap.release()
