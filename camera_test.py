import cv2
import numpy as np
import requests

url = "http://192.168.137.242:81/stream"

print("Connecting to ESP32-CAM...")

stream = requests.get(url, stream=True)

if stream.status_code != 200:
    print("Connection failed")
    exit()

print("Connected!")

bytes_data = b''

for chunk in stream.iter_content(chunk_size=1024):

    bytes_data += chunk

    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')

    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]

        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("ESP32 CAM", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cv2.destroyAllWindows()