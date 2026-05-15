
import cv2
import os

def test_codecs():
    codecs = ['avc1', 'H264', 'X264', 'mp4v', 'XVID', 'MJPG', 'VP80']
    results = []
    
    for codec in codecs:
        filename = f"test_{codec}.mp4"
        if codec == 'VP80': filename = "test_vp80.webm"
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 360))
        
        if out.isOpened():
            results.append(f"{codec}: OK")
            out.release()
            if os.path.exists(filename): os.remove(filename)
        else:
            results.append(f"{codec}: FAIL")
            
    print("\n".join(results))

if __name__ == "__main__":
    test_codecs()
