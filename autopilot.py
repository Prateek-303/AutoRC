"""
AutoRC — Tesla-Style Autonomous Navigation
==========================================
Full occupancy-grid path planning with:
  - Per-frame occupancy grid with safety margins
  - Scipy spline smoothing through waypoints
  - Pure-pursuit lookahead steering
  - Tesla-style HUD: corridor, spline path, bounding boxes, status panel

Controls (window must be focused):
  ESC / Q  - Quit
  A        - Start driving (unpause)
  S        - Stop driving (pause)
  D        - Toggle debug overlay
  1-5      - Set speed preset

Edit CAM_URL and MOTOR_IP below, then:  python autopilot.py
"""

import cv2
import numpy as np
import requests
import threading
import time
import math
import sys
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import socket
import concurrent.futures

# ─────────────────────────── CONFIG ─────────────────────────────────

# You can use raw IPs, or if mDNS/Bonjour is running on the ESP32s, just use their hostnames!
# Example: CAM_URL = "http://esp32-cam.local:81/stream", MOTOR_IP = "esp32-motor.local"
CAM_URL   = "http://192.168.137.136:81/stream" # Fallback — scanner auto-discovers the real IP
MOTOR_IP  = "192.168.137.50"                   # Fallback — scanner auto-discovers the real IP

FRAME_W   = 640
FRAME_H   = 480

# Driving speeds (0-255 PWM scale. CRITICAL: Motors stall below ~80 on battery!)
BASE_SPEED   = 85   # Increased slightly to prevent stalling
SPEED_SLOW   = 78   # Far-threat
SPEED_AVOID  = 78   # Near-threat
SPEED_REVERSE= 0    # DISABLED
SLOW_SPEED   = 78   # legacy alias
# Steer range: ±STEER_RATIO × speed  (keeps right motor always positive)
# e.g. 0.5 → at cruise 20 the max steer differential is ±10, right motor ≥ 10.
STEER_RATIO  = 0.6  # fraction of current speed to use as max steer differential
STEER_MAX    = 100  # hard cap on steer value sent to ESP32 (safety limit)

# Smooth acceleration — PWM units per motor command tick
ACCEL_RATE  = 0.5   # lowest possible ramp for 'creeping' motion
DECEL_RATE  = 12.0  # firm stop to prevent roll

# Horizon: ignore everything above this line (sky / ceiling)
HORIZON_Y   = 160

# ── Depth-based threat zones (REAL DISTANCES in cm) ─────────────────
# Threat levels trigger at these actual physical distances from the car.
# Measured using camera perspective geometry, NOT screen-fraction heuristics.
THREAT_CRITICAL_CM  = 10   # STOP + reverse: object is THIS close in front
THREAT_NEAR_CM      = 20   # AVOID:  hard steer, slow way down
THREAT_FAR_CM       = 40   # SLOW:   gentle steer, reduce speed

# ── Camera calibration (perspective geometry) ─────────────────────────
# Measure these for YOUR car's camera mount!
CAM_HEIGHT_CM   = 7    # Camera lens height from floor (cm) — measure with ruler
CAM_TILT_DEG    = 20   # Camera pointing downward from horizontal (degrees)
#  ^ 6° was too shallow → huge depth values. 20° is more realistic for a
#    low-mounted inward-facing cam. Adjust until on-screen cm labels look right.
CAM_H_FOV_DEG   = 70   # Horizontal field of view (ESP32-CAM OV2640 ≈ 70°)
CAM_V_FOV_DEG   = 50   # Vertical field of view (ESP32-CAM OV2640 ≈ 50°)

# Pre-derived constants (do not edit — edit the four lines above)
_cam_tilt_rad    = math.radians(CAM_TILT_DEG)
_rad_per_px_v    = math.radians(CAM_V_FOV_DEG) / FRAME_H
_rad_per_px_h    = math.radians(CAM_H_FOV_DEG) / FRAME_W

# Pixel Y zone lines derived from camera geometry (for HUD)
# Compute: which pixel Y corresponds to each cm threshold?
def _depth_to_pixel_y(depth_cm: float) -> int:
    """Inverse of pixel_to_world: given a real depth (cm), return pixel Y row."""
    if depth_cm <= 0:
        return FRAME_H
    try:
        import math as _m
        angle_total = _m.atan(CAM_HEIGHT_CM / depth_cm)  # angle below horizontal
        angle_from_axis = angle_total - _cam_tilt_rad    # relative to camera axis
        pixel_y = int(FRAME_H / 2 - angle_from_axis / _rad_per_px_v)
        return max(HORIZON_Y, min(FRAME_H, pixel_y))
    except Exception:
        return FRAME_H // 2

FAR_Y  = _depth_to_pixel_y(THREAT_FAR_CM)
MID_Y  = _depth_to_pixel_y(THREAT_NEAR_CM)
NEAR_Y = _depth_to_pixel_y(THREAT_CRITICAL_CM)

# ── Steering / planning ───────────────────────────────────────────────
SAFE_MARGIN      = 55   # px safety bubble dilated around obstacles
SPLINE_POINTS    = 120  # waypoints in smoothed spline
LOOKAHEAD_PX    = 200  # pure-pursuit lookahead distance (px)
PATH_REPLAN_SECS = 0.10 # planning interval (s)
STEER_SMOOTHING  = 0.72 # EMA factor — lower = faster response, less oscillation
STEER_DEADZONE   = 8.0  # degrees below which steering snaps to 0 (anti-wobble)

# ── Direction lock (anti-oscillation) ────────────────────────────────
LOCK_SECS   = 0.7

CMD_INTERVAL = 0.15   # Slowed down for ESP32 stability

_last_cmd = 0.0
path_history = []  # [(x, y, time)]
object_log = []    # [(label, global_x, global_y)]
pose_x, pose_y, pose_theta = 400.0, 700.0, 0.0 # centered on 800x800 grid
last_track_t = 0.0

# Mission Waypoints on the virtual grid
mission_pts = []  # [(gx, gy)]
mission_stops = [] # [index of stopping waypoints]

def grid_mouse_callback(event, x, y, flags, param):
    """Adds waypoints or stops to the virtual grid."""
    global mission_pts, mission_stops
    # If the user clicks, create a Manhattan (orthogonal) path
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start from the last point in mission or current pose
        start_x, start_y = (pose_x, pose_y) if not mission_pts else mission_pts[-1]
        
        # Manhattan leg 1: Move along Y (Vertical)
        mission_pts.append((int(start_x), int(y)))
        # Manhattan leg 2: Move along X (Horizontal)
        mission_pts.append((int(x), int(y)))
        print(f"  [Interactive] Manhattan Waypoints added: {mission_pts[-2:]}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Add a stop mark to the lAST added segment
        mission_stops.append(len(mission_pts) - 1)
        print(f"  [Interactive] STOP marker set at: {x, y}")


def send_motor(speed: int, steer: int):
    """
    Standard Differential-Drive mix. 
    Now continuous (no pulsing) for smooth Rover-like movement.
    """
    global _last_cmd
    now = time.time()
    
    # Rate limiting WiFi commands
    if now - _last_cmd < CMD_INTERVAL:
        return
    _last_cmd = now
    
    speed = int(speed)
    steer = int(steer)

    def _fire():
        try:
            url = f"http://{MOTOR_IP}/move?speed={speed}&steer={steer}"
            r = requests.get(url, timeout=1.5) # Increased from 0.1s to prevent ESP32 timeouts
            if r.status_code != 200:
                print(f"  [Motor] Error {r.status_code} calling {url}")
        except Exception as e:
            # Simple error print for diagnostics
            print(f"  [Motor] Command failure: {e}")
    threading.Thread(target=_fire, daemon=True).start()

# ─────────────────────────── PERCEPTION ─────────────────────────────

class FloorSegmenter:
    """
    Universal Obstacle Detection via Floor Plane Segmentation.
    Learns 'what floor looks like' by sampling the area in front of the car.
    Anything that isn't floor-colored = obstacle.
    """
    def __init__(self):
        self.floor_hsv_min = None
        self.floor_hsv_max = None
        self.frames_learned = 0
        self.learn_limit = 40  # frames to build floor model
        self._samples = []

    def learn(self, hsv_frame):
        if self.frames_learned >= self.learn_limit:
            return
        
        h, w = hsv_frame.shape[:2]
        # Sample bottom-center rectangle (area right in front of car)
        sample_roi = hsv_frame[h-60 : h-10, w//2 - 50 : w//2 + 50]
        
        self._samples.append(sample_roi)
        self.frames_learned += 1
        
        if self.frames_learned == self.learn_limit:
            all_samples = np.concatenate(self._samples)
            med = np.median(all_samples, axis=(0,1))
            std = np.std(all_samples, axis=(0,1))
            
            # Floor color range: median +/- 2.5 standard deviations
            self.floor_hsv_min = np.clip(med - (std * 2.5) - 15, 0, 255).astype(np.uint8)
            self.floor_hsv_max = np.clip(med + (std * 2.5) + 15, 0, 255).astype(np.uint8)
            print(f"  [Segmenter] Floor model build: MIN={self.floor_hsv_min} MAX={self.floor_hsv_max}")

    def get_obstacle_mask(self, hsv_frame):
        if self.frames_learned < self.learn_limit:
            return np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        
        # Everything that is NOT within the floor color range
        floor_mask = cv2.inRange(hsv_frame, self.floor_hsv_min, self.floor_hsv_max)
        
        # Invert to get obstacle mask (below horizon)
        obs_mask = cv2.bitwise_not(floor_mask)
        obs_mask[:HORIZON_Y, :] = 0 # Ignore sky
        
        # Clean up
        kernel = np.ones((5,5), np.uint8)
        obs_mask = cv2.morphologyEx(obs_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        obs_mask = cv2.dilate(obs_mask, kernel, iterations=2)
        
        return obs_mask


def pixel_to_world(px: int, py: int) -> tuple[float, float]:
    """
    Convert image pixel (px, py) to real-world (x_cm, depth_cm) using
    perspective geometry — same technique as RoboconVision.

    For AutoRC the camera is mounted at height CAM_HEIGHT_CM pointing
    downward by CAM_TILT_DEG degrees.  For any object that touches the
    FLOOR (height = 0), the pixel row that its lowest edge occupies lets
    us compute its true forward distance.

    Derivation:
      angle_below_horizontal = CAM_TILT_DEG + angle_from_image_center
      depth_cm = CAM_HEIGHT_CM / tan(angle_below_horizontal)
    """
    # Vertical: angle from camera optical axis to this pixel row
    angle_from_axis  = (py - FRAME_H / 2) * _rad_per_px_v
    # Total angle below horizontal (positive = looking downward)
    angle_total      = _cam_tilt_rad + angle_from_axis
    if angle_total <= 0:
        depth_cm = 999.0   # pixel is above horizon — treat as very far
    else:
        depth_cm = CAM_HEIGHT_CM / math.tan(angle_total)

    # Horizontal: lateral offset in cm at computed depth
    angle_h = (px - FRAME_W / 2) * _rad_per_px_h
    x_cm    = depth_cm * math.tan(angle_h)

    return x_cm, max(0.0, depth_cm)


def load_yolo():
    print("  Loading YOLOv3-Tiny...")
    net = cv2.dnn.readNet("d:/AutoRC/yolo/yolov3-tiny.weights", "d:/AutoRC/yolo/yolov3-tiny.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    try:
        out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        out_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    try:
        with open("d:/AutoRC/yolo/coco.names", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        class_names = []
    return net, out_layers, class_names


def detect_obstacles(frame: np.ndarray, net, output_layers, class_names: list,
                     segmenter: FloorSegmenter) -> tuple[np.ndarray, tuple | None, list[tuple]]:
    """
    HYBRID obstacle detection:
      Layer 1 — Floor Segmentation: Detects ANY physical item (shoes, bottles, walls)
      Layer 2 — YOLOv3-tiny: Optional labels for recognized classes
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Update floor model / Get mask
    segmenter.learn(hsv)
    obs_mask = segmenter.get_obstacle_mask(hsv)
    
    occupancy = np.zeros((h, w), dtype=np.uint8)
    largest_box = None
    all_boxes = []
    max_area    = 0

    # Process blobs from segmentation
    contours, _ = cv2.findContours(obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600:
            continue
            
        x, y, bw, bh = cv2.boundingRect(cnt)
        all_boxes.append((x, y, bw, bh))
        fby = y + bh # bottom edge
        
        # Fill occupancy
        cv2.rectangle(occupancy, (x, y), (x + bw, y + bh), 255, -1)
        
        # Distance (Proxy: k / sqrt(area) as fallback to geometry)
        dist_area = 6000.0 / math.sqrt(area)
        _, depth_geom = pixel_to_world(x + bw//2, fby)
        depth_cm = min(float(dist_area), float(depth_geom)) # Blend both for stability

        # Draw
        t = max(0.0, min(1.0, 1.0 - depth_cm / THREAT_FAR_CM))
        col = (0, int(255 * (1 - t)), int(255 * t))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 2)
        cv2.putText(frame, f"OBJ {depth_cm:.0f}cm", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, col, 1)

        if area > max_area and fby > h * 0.4:
            max_area = area
            largest_box = (x, y, bw, bh)

    # ── Layer 2: YOLO — class labels only (Optional) ───────────────
    # We run YOLO to add names (chair, person, etc) to our existing blobs
    if net is not None:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        for out in outs:
            for det in out:
                scores = det[5:]
                cid = int(np.argmax(scores))
                conf = float(scores[cid])
                if conf > 0.25:
                    bx = int(det[0] * w); by = int(det[1] * h)
                    bw2 = int(det[2] * w); bh2 = int(det[3] * h)
                    label = class_names[cid] if cid < len(class_names) else "obj"
                    # Just draw label near the center of the detection
                    cv2.putText(frame, f"{label} {conf:.0%}", (bx - bw2//2, by - bh2//2 - 15),
                                cv2.FONT_HERSHEY_PLAIN, 1.0, (100, 255, 100), 1)

    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SAFE_MARGIN, SAFE_MARGIN))
    occ = cv2.dilate(occupancy, kernel_d)
    return occ, largest_box, all_boxes

# SENSOR-SOLO MODE ENABLED (VISION DECOUPLED FROM MOTORS)


# ─────────────────────────── PLANNING ───────────────────────────────

def find_waypoints(occupancy: np.ndarray, last_steering_deg: float = 0, target_pt = None) -> list:
    """
    Scan occupancy grid to find gaps. 
    BIAS: Favors gaps towards the 'target_pt' (if set) or follows steering momentum.
    """
    h, w = occupancy.shape
    waypoints = []
    waypoints.append((w // 2, h))
    
    # If we have a target_pt, we prioritize reaching it BLINDLY.
    # We create a simple path from current bottom-center to the target.
    if target_pt is not None:
        tx, ty = target_pt
        # Limit ty to be at or above HORIZON_Y
        ty = max(HORIZON_Y + 10, ty)
        
        # Intermediate points for a smooth spline
        mid_x = (w // 2 + tx) // 2
        mid_y = (h + ty) // 2
        
        waypoints.append((mid_x, mid_y))
        waypoints.append((tx, ty))
        return waypoints

    # Otherwise, fallback to gap-based navigation
    slices = np.linspace(h - 20, HORIZON_Y + 20, 7, dtype=int)
        
    return waypoints


def smooth_waypoints(waypoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Fit a cubic spline through rough waypoints for a smooth path."""
    if len(waypoints) < 3:
        return waypoints
        
    # Extract x and y
    x = [p[0] for p in waypoints]
    y = [p[1] for p in waypoints]
    
    # Ensure strictly decreasing Y (Scipy requires monotonic for some knots, 
    # but splprep handles parametric. We just remove duplicate points)
    pts = []
    for px, py in zip(x, y):
        if not pts or py < pts[-1][1]:
            pts.append((px, py))
            
    if len(pts) < 3:
        return pts
        
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    
    try:
        # Fit parametric B-spline
        tck, u = splprep([x, y], s=300) # s is smoothness factor
        u_new = np.linspace(0, 1.0, SPLINE_POINTS)
        x_new, y_new = splev(u_new, tck)
        
        # Convert back to pixel ints
        return [(int(xn), int(yn)) for xn, yn in zip(x_new, y_new)]
    except Exception:
        # Fallback if spline fails (e.g., weird collinear points)
        return pts


def compute_steering_from_path(path_pts: list[tuple[int, int]]) -> float:
    """Standard Pure-Pursuit steering tracker."""
    if len(path_pts) < 2:
        return 0.0
        
    car_x, car_y = path_pts[0] # Bottom of screen
    
    # Find the point on path that is exactly LOOKAHEAD_PX away
    target_pt = path_pts[-1]
    for px, py in path_pts:
        dist = math.hypot(px - car_x, py - car_y)
        if dist >= LOOKAHEAD_PX:
            target_pt = (px, py)
            break
            
    # Calculate angle to target point
    dx = target_pt[0] - car_x
    dy = car_y - target_pt[1] # standard XY plane, y grows up
    
    if dy == 0: dy = 1
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)
    
    # Cap between -45 and 45
    return max(-45.0, min(45.0, angle_deg))


# ─────────────────────────── DRAWING ────────────────────────────────

# SENSOR-SOLO DRAWING ENGINE


def draw_detections(frame, all_boxes, class_names):
    """Cleanly draw all detected objects found by the Scout/Sensor."""
    for box in all_boxes:
        bx, by, bw, bh = box
        label = "Object" # Basic label if index not known here
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 100), 1)
        cv2.putText(frame, label, (bx, by - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 100), 1)

def draw_goal(frame, goal):
    """Draw a target marker at the user-selected goal."""
    if goal is None: return
    gx, gy = goal
    # Pulsing animation
    s = 1.0 + 0.2 * math.sin(time.time() * 5)
    size = int(20 * s)
    cv2.circle(frame, (gx, gy), size, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (gx, gy), 4, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "TARGET", (gx - 25, gy - size - 5),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 1, cv2.LINE_AA)

def generate_mission_report():
    """Create a matplotlib plot of the path and objects."""
    global path_history, object_log
    if not path_history: return
    
    print("  [Explorer] Generating Mission Report...")
    plt.figure(figsize=(10, 8))
    
    # Plot Path
    px = [p[0] for p in path_history]
    py = [p[1] for p in path_history]
    plt.plot(px, py, 'b-', label="Rover Path", linewidth=2)
    plt.plot(px[0], py[0], 'go', label="Start")
    plt.plot(px[-1], py[-1], 'ro', label="Destination")
    
    # Plot Objects
    if object_log:
        used_labels = set()
        for label, ox, oy in object_log:
            show_label = label if label not in used_labels else ""
            plt.scatter(ox, oy, marker='x', color='red', label=show_label)
            used_labels.add(label)
        
    plt.title("AutoRC — Mission Navigation Map")
    plt.xlabel("X (Est. CM)")
    plt.ylabel("Y (Est. CM)")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    filename = f"mission_report_{int(time.time())}.png"
    plt.savefig(filename)
    plt.close()
    print(f"  [Explorer] Mission Map saved to {filename}")
    
    # Show in a window
    img = cv2.imread(filename)
    if img is not None:
        cv2.imshow("Mission Map", img)

def draw_grid_map(pose_x, pose_y, pose_theta, objects):
    """Draws the 800x800 mission grid."""
    grid = np.zeros((800, 800, 3), dtype=np.uint8) + 30 # Dark grey
    
    # Draw Grid Lines
    for i in range(0, 800, 50):
        cv2.line(grid, (i, 0), (i, 800), (50,50,50), 1)
        cv2.line(grid, (0, i), (800, i), (50,50,50), 1)

    # Draw Mission Waypoints
    for i, pt in enumerate(mission_pts):
        color = (0, 0, 255) if i in mission_stops else (0, 255, 0)
        cv2.circle(grid, pt, 5, color, -1)
        if i > 0:
            cv2.line(grid, mission_pts[i-1], pt, (0, 150, 0), 1)

    # Draw Detected Objects
    for label, ox, oy in objects:
        gx, gy = int(ox), int(oy)
        if 0 <= gx < 800 and 0 <= gy < 800:
            cv2.drawMarker(grid, (gx, gy), (100, 100, 255), cv2.MARKER_CROSS, 10, 2)
            cv2.putText(grid, label, (gx+5, gy-5), cv2.FONT_HERSHEY_PLAIN, 0.8, (150, 150, 255), 1)

    # Draw Rover (Triangle pointing in pose_theta)
    tip = (int(pose_x + 15 * math.sin(pose_theta)), int(pose_y - 15 * math.cos(pose_theta)))
    l_wing = (int(pose_x + 10 * math.sin(pose_theta - 2.5)), int(pose_y - 10 * math.cos(pose_theta - 2.5)))
    r_wing = (int(pose_x + 10 * math.sin(pose_theta + 2.5)), int(pose_y - 10 * math.cos(pose_theta + 2.5)))
    cv2.polylines(grid, [np.array([tip, l_wing, r_wing])], True, (255, 255, 255), 2)
    cv2.circle(grid, (int(pose_x), int(pose_y)), 3, (0, 255, 255), -1)

    return grid


def draw_hud(frame: np.ndarray, mode: str, speed: int, steer_deg: float,
             occ_pct: float, paused: bool, fps: float, segmenter=None):
    """Draw sleek status panel at bottom right with threat level and direction lock."""
    # Floor Learning Status
    if segmenter and segmenter.frames_learned < segmenter.learn_limit:
        prog = int(100 * segmenter.frames_learned / segmenter.learn_limit)
        cv2.putText(frame, f"LEARNING FLOOR: {prog}%", (15, FRAME_H - 15), 
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
    panel_w, panel_h = 260, 120
    px, py = FRAME_W - panel_w - 10, FRAME_H - panel_h - 10

    # Background glass
    ov3 = frame.copy()
    cv2.rectangle(ov3, (px, py), (px + panel_w, py + panel_h), (18, 18, 22), -1)
    cv2.addWeighted(ov3, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (60, 60, 70), 1)

    # Mode / Status
    mode_col = {"PAUSED": (100,100,100), "FORWARD": (0,200,80),
                "SLOW": (0,200,200), "AVOID": (0,150,255),
                "STOP": (0,80,255), "REVERSE": (0,0,255),
                "ROTATE": (0, 255, 255), "MOVE_STRAIGHT": (0, 200, 80)}
    col = mode_col.get(mode.split("-")[0], (0, 165, 255))
    cv2.putText(frame, mode, (px + 12, py + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, col, 1, cv2.LINE_AA)

    # Speed & Occupancy
    cv2.putText(frame, f"SPD:{speed:03d}  BLK:{occ_pct*100:02.0f}%",
                (px + 12, py + 72),
                cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1, cv2.LINE_AA)

    # Steering bar
    bar_w = 220
    bar_x = px + 18
    bar_y = py + 100
    cv2.line(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y), (70, 70, 82), 4)
    cx2 = bar_x + bar_w // 2
    cv2.line(frame, (cx2, bar_y - 6), (cx2, bar_y + 6), (150, 150, 160), 2)
    steer_px = cx2 + int((steer_deg / 45.0) * (bar_w // 2))
    steer_px = max(bar_x, min(bar_x + bar_w, steer_px))
    bc = (0, 0, 255) if abs(steer_deg) > 28 else \
         (0, 200, 200) if abs(steer_deg) > 10 else (0, 230, 60)
    cv2.circle(frame, (steer_px, bar_y), 7, bc, -1, cv2.LINE_AA)
    cv2.putText(frame, "STEER", (bar_x, bar_y - 3),
                cv2.FONT_HERSHEY_PLAIN, 0.75, (120, 120, 135), 1)

    # FPS
    cv2.putText(frame, f"FPS:{fps:.0f}", (FRAME_W - 82, FRAME_H - 6),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (110, 110, 130), 1)


# LEGACY ZONE DRAWING REMOVED (SENSOR-SOLO MODE ENABLED)


# ─────────────────────────── MAIN ───────────────────────────────────

def discover_devices(subnet="192.168.137"):
    """Blast the subnet with concurrent connections to find the ESP32s."""
    print(f"  [Discovery] Hunting for AutoRC devices on {subnet}.x ...")
    found_cam   = [None]  # Use list so threads can mutate safely
    found_motor = [None]
    lock = threading.Lock()
    
    def scan(ip_suffix):
        ip = f"{subnet}.{ip_suffix}"
        
        # 1. Test Camera (Port 81)
        if found_cam[0] is None:
            try:
                # Raw sockets without HTTP headers freeze the esp32-camera httpd worker.
                # So we send a blind HTTP GET and drop the socket to free the worker instantly.
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    if s.connect_ex((ip, 81)) == 0:
                        try:
                            s.sendall(b"GET / HTTP/1.0\r\n\r\n")
                        except Exception: pass
                        with lock:
                            if found_cam[0] is None:
                                found_cam[0] = f"http://{ip}:81/stream"
                                print(f"    -> Found ESP32-CAM  at {found_cam[0]}")
            except Exception: pass
            
        # 2. Test Motor Controller (Port 80)
        if found_motor[0] is None and ip_suffix != 1:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    if s.connect_ex((ip, 80)) == 0:
                        try:
                            r = requests.get(f"http://{ip}/move?dir=0&speed=0", timeout=0.5)
                            if r.status_code == 200:
                                with lock:
                                    if found_motor[0] is None:
                                        found_motor[0] = ip
                                        print(f"    -> Found ESP32-MOT  at {found_motor[0]}")
                        except Exception: pass
            except Exception: pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        executor.map(scan, range(2, 255))
        
    return found_cam[0], found_motor[0]


    # State
class CameraScanner:
    """Runs a dedicated thread to drain the MJPEG buffer instantly, guaranteeing 0 lag."""
    def __init__(self, url):
        self.url = url
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        print(f"  [CamThread] Connecting to {self.url}...")
        while self.running:
            try:
                stream = requests.get(self.url, stream=True, timeout=3.0)
                if stream.status_code != 200:
                    time.sleep(1.0)
                    continue
                buf = b""
                for chunk in stream.iter_content(chunk_size=4096):
                    if not self.running: break
                    buf += chunk
                    s = buf.find(b"\xff\xd8")
                    e = buf.find(b"\xff\xd9")
                    
                    if s != -1 and e != -1:
                        if s < e:
                            jpg = buf[s:e + 2]
                            buf = buf[e + 2:] # Pop out the frame
                            
                            # Throw away massive queue lag if processing is falling behind!
                            if len(buf) > 300000: # Increased lag buffer limit to prevent false triggers
                                buf = b""
                            
                            if len(jpg) > 100:
                                arr = np.frombuffer(jpg, dtype=np.uint8)
                                frm = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if frm is not None:
                                    self.latest_frame = cv2.resize(frm, (FRAME_W, FRAME_H))
                        else:
                            # End marker is before start marker in this buffer chunk.
                            # Just toss the tail garbage before the fresh start marker.
                            buf = buf[s:]
            except Exception as ex:
                time.sleep(1.0)

    def read(self):
        return self.latest_frame

    def stop(self):
        self.running = False


def main():
    global CAM_URL, MOTOR_IP, mission_pts, mission_stops, object_log, pose_x, pose_y, pose_theta, last_track_t
    print("=" * 60)
    print("  AutoRC — Full Occupancy-Grid Autonomous Navigation")
    print("=" * 60)
    
    # PERMANENT IP SOLUTION
    # dt_cam, dt_mot = discover_devices("192.168.137")
    # if dt_cam: CAM_URL = dt_cam
    # if dt_mot: MOTOR_IP = dt_mot
    MOTOR_IP = "192.168.137.50" # Locked for stability

    print("=" * 60)
    print(f"  Target Camera : {CAM_URL}")
    print(f"  Target Motor  : {MOTOR_IP}")
    print("  ESC/Q=Quit  A=Start  S=Stop  D=Debug")
    print("=" * 60)

    # ── State variables ──────────────────────────────────────
    current_speed:  float = 0.0
    paused:         bool  = False
    show_debug:     bool  = True
    steering_deg:   float = 0.0
    raw_steer:      float = 0.0
    smooth_path_pts: list[tuple[int, int]] = []
    raw_waypoints:   list[tuple[int, int]] = []
    last_plan_t:    float = 0.0
    frame_idx:      int   = 0
    fps:            float = 0.0
    t_fps_start           = time.time()
    frames_fps:     int   = 0
    # State machine
    nav_state:      str   = "FORWARD"  # FORWARD/SLOW/AVOID/STOP/REVERSE
    threat:         str   = "NONE"     # NONE/FAR/NEAR/CRITICAL
    lock_dir:       str   = "C"        # direction lock: C/L/R
    lock_until:     float = 0.0        # time when lock expires
    mode:           str   = "FORWARD"
    speed:          int   = 0
    target_speed:   float = 0.0 # Added for new navigation controller

    cv2.namedWindow("AutoRC Autopilot")
    cv2.namedWindow("Mission Grid")
    cv2.setMouseCallback("Mission Grid", grid_mouse_callback)

    net, output_layers, class_names = load_yolo()

    # Universal Obstacle Detection (no background subtraction needed)
    segmenter = FloorSegmenter()
    print("  Floor Segmenter ready (sampling floor to detect obstacles).")

    print(f"  Starting lag-free camera thread at {CAM_URL} ", end="", flush=True)
    cam = CameraScanner(CAM_URL)

    # UI Loading dots until the thread gets a frame!
    dots = 0
    while True:
        if cam.latest_frame is not None:
            print(" Connected!")
            break
        print(".", end="", flush=True)
        time.sleep(0.5)
        dots += 1
        if dots > 15:
            print("\n  [!] Still waiting... camera might be rebooting.")
            dots = 0
    
    while True:
        try:
            # Prevent thread race condition and infinite re-drawing on the same image array.
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            frame = frame.copy()
            
            global pose_x, pose_y, pose_theta, last_track_t
            if frame_idx % 30 == 0:
                print(f"  [Main Loop] Processing frame {frame_idx} (FPS: {fps:.1f})")

            # ── FPS ─────────────────────────────────────────
            frames_fps += 1
            elapsed_fps = time.time() - t_fps_start
            if elapsed_fps >= 1.0:
                fps = frames_fps / elapsed_fps
                frames_fps = 0
                t_fps_start = time.time()

            # ── 2. Threat classification (depth-based) ─────────
            occupancy, obs_box, all_boxes = detect_obstacles(
                frame, net, output_layers, class_names, segmenter)
            
            # Record objects found during mission
            if all_boxes:
                # Log any YOLO detection or large occupancy block
                for box in all_boxes:
                    label = "Object" # Basic label if not classified
                    
                    # 1. Get relative CM distance from camera
                    _, depth_cm = pixel_to_world(box[0] + box[2]//2, box[1] + box[3])
                    h_offset_px = (box[0] + box[2]/2 - FRAME_W/2)
                    h_offset_cm = (h_offset_px / FRAME_W) * depth_cm * 0.8 # approx FOV scale
                    
                    # 2. Project into Global Grid Map
                    # Rotate relative (h_offset, depth) by pose_theta
                    cosa = math.cos(pose_theta)
                    sina = math.sin(pose_theta)
                    
                    # Depth is forward (global Y-ish), offset is sideways (global X-ish)
                    gx = pose_x + (h_offset_cm * cosa + depth_cm * sina)
                    gy = pose_y - (depth_cm * cosa - h_offset_cm * sina)
                    
                    # To avoid log spam, only add if unique-ish
                    if not any(math.hypot(gx-ox, gy-oy) < 15 for _, ox, oy in object_log):
                        object_log.append((label, gx, gy))

            # Object logging only (Sensor Solo)
            # No threat classification for motor control
            # ── 3. Orthogonal Navigation Controller ─────────────
            now = time.time()
            steering_deg = 0.0
            target_speed = 0.0
            nav_state = "IDLE"

            if mission_pts and not paused:
                target_x, target_y = mission_pts[0]
                dx = target_x - pose_x
                dy = target_y - pose_y
                dist = math.hypot(dx, dy)
                
                # Angle to target 
                angle_to_target = math.atan2(dx, -dy) 
                error_theta = angle_to_target - pose_theta
                error_theta = (error_theta + math.pi) % (2 * math.pi) - math.pi
                
                # NAVIGATION STATE MACHINE: ROTATE -> MOVE
                if abs(error_theta) > math.radians(12): # 12 deg alignment threshold
                    nav_state = "ROTATE"
                    steering_deg = 45.0 if error_theta > 0 else -45.0
                    target_speed = 0.0 # Stationary rotation
                else:
                    nav_state = "MOVE_STRAIGHT"
                    steering_deg = 0.0 # Force pure straight line
                    target_speed = BASE_SPEED
                
                if dist < 12:
                    print(f"  [Mission] reached: {mission_pts[0]}")
                    if 0 in mission_stops:
                        paused = True
                        print("  [Mission] STOP marker reached. Paused.")
                        mission_stops = [s-1 for s in mission_stops if s > 0]
                    mission_pts.pop(0)
                    if not mission_pts:
                        print("  [Mission] Final destination reached!")
                        send_motor(0, 0)
                        paused = True
                
                raw_steer = steering_deg 
            else:
                raw_steer = 0.0
                target_speed = 0.0

            # ── 4. State machine & Actuation ──────────────────────
            if paused:
                current_speed = max(0.0, current_speed - DECEL_RATE * 3)
                send_motor(0, 0)
                mode, speed = "PAUSED", 0
            else:
                # Orthogonal Control logic
                steer_pwm = int(np.clip(steering_deg / 45.0 * STEER_MAX, -STEER_MAX, STEER_MAX))
                desired = target_speed
                
                if current_speed < desired:
                    current_speed = min(desired, current_speed + ACCEL_RATE)
                elif current_speed > desired:
                    current_speed = max(desired, current_speed - DECEL_RATE)
                
                # Pivot boost for stationary rotation
                if nav_state == "ROTATE":
                    current_speed = 82 # Fixed pivot power
                elif nav_state == "MOVE_STRAIGHT" and current_speed < 78:
                    current_speed = 78
                    
                send_motor(int(current_speed), steer_pwm)
                mode, speed = nav_state, int(current_speed)

            # ── 5. Draw & Hud ────────────────────────────────────
            draw_detections(frame, all_boxes, class_names)
            
            # ── 6. Dead Reckoning & Navigation Tracking ─────────
            dt = now - last_track_t if last_track_t > 0 else 0.05
            last_track_t = now
            
            if not paused:
                # Scale: PWM 80 ~ 12 grid units/sec
                speed_units = (current_speed / 80.0) * 12.0
                rads = math.radians(steering_deg)
                
                # Pose estimation based on control surface
                pose_theta += (rads * 1.5) * dt 
                pose_theta = (pose_theta + math.pi) % (2 * math.pi) - math.pi
                
                pose_x += speed_units * math.sin(pose_theta) * dt
                pose_y -= speed_units * math.cos(pose_theta) * dt
            
            grid_img = draw_grid_map(pose_x, pose_y, pose_theta, object_log)
            cv2.imshow("Mission Grid", grid_img)

            draw_hud(frame, mode, speed, steering_deg,
                     float(np.mean(occupancy > 0)), paused, fps)

            cv2.imshow("AutoRC Autopilot", frame)
            if show_debug:
                cv2.imshow("Occupancy Grid", occupancy)

            frame_idx += 1

            # ── 7. Keyboard ──────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                print("Quitting.")
                send_motor(0, 0)
                cam.stop()
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord('a'):
                paused = False
            elif key == ord('s'):
                paused = True
            elif key == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow("Occupancy Grid")

        except Exception as e:
            print(f"\n  [!] Main loop error: {e}")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
