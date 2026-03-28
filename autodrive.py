import cv2
import numpy as np
import requests

# ================= CONFIG =================

CAM_URL = "http://192.168.137.242:81/stream"
MOTOR_IP = "192.168.137.115"

SPEED = 35   # slow testing speed

# ==========================================

print("Connecting to camera...")

stream = requests.get(CAM_URL, stream=True)

bytes_data = b''

for chunk in stream.iter_content(chunk_size=4096):

    bytes_data += chunk

    start = bytes_data.find(b'\xff\xd8')
    end = bytes_data.find(b'\xff\xd9')

    if start != -1 and end != -1:

        jpg = bytes_data[start:end+2]
        bytes_data = bytes_data[end+2:]

        # ---- safety checks ----
        if len(jpg) < 100:
            continue

        img_array = np.frombuffer(jpg, dtype=np.uint8)

        if img_array.size == 0:
            continue

        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        frame = cv2.resize(frame,(640,480))

        # ===== IMAGE PROCESSING =====

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        edges = cv2.Canny(blur,50,150)

        h,w = edges.shape

        left = edges[:,0:w//2]
        right = edges[:,w//2:w]

        left_density = np.sum(left)
        right_density = np.sum(right)

        steer = 0
        direction = "FORWARD"

        if left_density > right_density:
            steer = 20
            direction = "RIGHT"

        elif right_density > left_density:
            steer = -20
            direction = "LEFT"

        # ===== SEND WIFI COMMAND =====

        try:
            requests.get(
                f"http://{MOTOR_IP}/move?speed={SPEED}&steer={steer}",
                timeout=0.05
            )
        except:
            pass

        # ===== DISPLAY =====

        cv2.putText(frame,
                    f"Dir: {direction}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("AutoRC Navigation", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cv2.destroyAllWindows()