import cv2
import numpy as np
import requests

url = "http://192.168.137.17:81/stream"

print("Connecting to ESP32-CAM...")
stream = requests.get(url, stream=True)

bytes_data = b''

for chunk in stream.iter_content(chunk_size=4096):

    bytes_data += chunk

    start = bytes_data.find(b'\xff\xd8')
    end = bytes_data.find(b'\xff\xd9')

    if start != -1 and end != -1:

        jpg = bytes_data[start:end+2]
        bytes_data = bytes_data[end+2:]

        # ---- safety check ----
        if len(jpg) == 0:
            continue

        img_array = np.frombuffer(jpg, dtype=np.uint8)

        if img_array.size == 0:
            continue

        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        frame = cv2.resize(frame,(640,480))

        # ===== SIMPLE OBSTACLE ANALYSIS =====

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        edges = cv2.Canny(blur,50,150)

        h,w = edges.shape

        left = edges[:,0:w//2]
        right = edges[:,w//2:w]

        left_density = np.sum(left)
        right_density = np.sum(right)

        if left_density > right_density:
            direction = "RIGHT"
        elif right_density > left_density:
            direction = "LEFT"
        else:
            direction = "FORWARD"

        cv2.putText(frame,
                    f"Direction: {direction}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("Navigation", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cv2.destroyAllWindows()