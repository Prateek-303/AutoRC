<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1a6b,100:2d2d9e&height=160&section=header&text=AutoRC%20—%20Autonomous%20RC%20Car&fontSize=36&fontColor=ffffff&fontAlignY=40&desc=YOLOv3%20%7C%20Occupancy%20Grids%20%7C%20Pure-Pursuit%20Navigation%20%7C%20STM32%20Black%20Pill&descSize=14&descAlignY=62&descColor=a8b5f5&animation=fadeIn" />
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=FFD43B)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![STM32](https://img.shields.io/badge/STM32_Black_Pill-03234B?style=flat-square&logo=stmicroelectronics&logoColor=white)
![ESP32](https://img.shields.io/badge/ESP32-E7352C?style=flat-square&logo=espressif&logoColor=white)
![YOLOv3](https://img.shields.io/badge/YOLOv3--Tiny-222222?style=flat-square&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

</div>

---

## 📌 Overview

**AutoRC** is a hybrid autonomous/manual RC car that switches between human-controlled and fully autonomous driving modes. It uses a real-time computer vision stack for obstacle detection, spline-based path planning, and a pure-pursuit controller for smooth steering. The heavy ML compute runs on a remote **Offboard Compute Node**, analyzing a live video stream from an onboard **ESP32-CAM**, which then wirelessly commands an **STM32 Black Pill** to handle the low-level motor control.

---

## 🏗️ System Stack

```
Perception
  ├── Floor Segmenter     → learns floor in real-time, flags obstacles
  └── YOLOv3-Tiny         → detects persons, chairs, objects
        └── Perspective geometry → distance estimate in CM

Planning
  ├── Dynamic Occupancy Grid  → top-down free/occupied map per frame
  ├── SciPy Spline Smoothing  → raw waypoints → smooth trajectory
  └── Interactive Mission Grid → mouse-click waypoints + stop markers

Control
  └── Pure-Pursuit Controller → steering angle from lookahead point
        └── Rate-limited WiFi commands → ESP32-MOT
```

---

## ⚙️ Hardware Used

| Component | Role |
|-----------|------|
| Offboard Compute Node | Main compute — YOLOv3 CV, path planning, control logic |
| ESP32-CAM | MJPEG wireless video streamer |
| STM32 Black Pill | Low-level motor controller & differential steering |
| ESP32-MOT | WiFi interface receiver |
| RC Car Chassis | Drive platform |
| Li-ion Battery | Power supply |

---

## 📁 Folder Structure

```
AutoRC/
│
├── autopilot.py           # Full autonomous stack — HUD + mission grid
├── vision_nav.py          # Edge-density based steering test
├── autodrive.py           # Simple WiFi driving controller
├── camera_test.py         # Device discovery + connectivity test
│
├── models/
│   ├── yolov3-tiny.cfg    # YOLOv3-Tiny architecture config
│   ├── yolov3-tiny.weights
│   └── coco.names         # Class labels
│
├── firmware/
│   └── esp32_motor/
│       └── esp32_motor.ino  # ESP32-MOT Arduino firmware
│
├── docs/
│   ├── system_diagram.png
│   └── images/
│
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/Prateek-303/AutoRC.git
cd AutoRC
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download YOLOv3-Tiny weights**
```bash
wget https://pjreddie.com/media/files/yolov3-tiny.weights -P models/
```

**4. Flash ESP32-MOT firmware**
- Open `firmware/esp32_motor/esp32_motor.ino` in Arduino IDE
- Select board: `ESP32 Dev Module`
- Flash and note the IP address shown on Serial Monitor

**5. Set motor IP**

In `autopilot.py`, set:
```python
MOTOR_IP = "192.168.137.50"   # replace with your ESP32-MOT IP
```

**6. Run**
```bash
# Full autonomous mode
python autopilot.py

# Simple WiFi drive test
python autodrive.py

# Device discovery
python camera_test.py
```

---

## 🔑 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MOTOR_IP` | `192.168.137.50` | ESP32 motor controller IP |
| `SAFE_MARGIN` | configurable | Obstacle buffer zone in pixels |
| `lookahead_dist` | configurable | Pure-pursuit lookahead distance |
| Command timeout | `1.5s` | WiFi motor command timeout |

---

## 📦 Requirements

```
opencv-python
numpy
scipy
```

---

## 👨💻 Author

**Prateek Baraiya** · B.Tech ECE · Dharmsinh Desai University, Nadiad

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Prateek%20Baraiya-0a66c2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/baraiya-prateek)
[![Gmail](https://img.shields.io/badge/Gmail-baraiyaprateek25%40gmail.com-ea4335?style=flat-square&logo=gmail&logoColor=white)](mailto:baraiyaprateek25@gmail.com)

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:2d2d9e,50:1a1a6b,100:0d1117&height=100&section=footer" />
</div>
