"""
warehouse_tracking.py

Requirements:
 pip install ultralytics opencv-python pandas deep-sort-realtime matplotlib

Usage:
 python warehouse_tracking.py
"""

import os
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import signal
import sys
import argparse

parser = argparse.ArgumentParser(description="CCTV Analytics Engine")
parser.add_argument("--video", type=str, required=True, help="Path to CCTV footage")
args = parser.parse_args()

# -------------------------
# CONFIG
# -------------------------
VIDEO_PATH = args.video        
OUTPUT_CSV = "warehouse_logs.csv"
EVIDENCE_DIR = "evidence_snapshots"
ANALYTICS_DIR = "analytics"

DETECTION_CONFIDENCE = 0.35          
RESCALE_WIDTH = 1280                  
MAX_INACTIVE_SECONDS = 4.0            
DEEPSORT_MAX_AGE = 30                 
DEEPSORT_N_INIT = 3                   

# Create outputs
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(ANALYTICS_DIR, exist_ok=True)

# -------------------------
# Helper functions
# -------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def sec_to_timestr(s):
    td = timedelta(seconds=int(round(s)))
    return str(td)

def autosave_logs(active_tracks, final_logs, filename=OUTPUT_CSV):
    """Save partial + finalized logs safely."""
    temp_logs = []

    for tid, info in active_tracks.items():
        temp_logs.append({
            "Person_ID": tid,
            "Entry_Time(s)": round(info["entry_time"], 2),
            "Exit_Time(s)": round(info["last_seen_time"], 2),
            "Duration(s)": round(info["last_seen_time"] - info["entry_time"], 2),
            "Evidence": ""
        })

    temp_logs.extend(final_logs)

    if temp_logs:
        df = pd.DataFrame(temp_logs)
        df = df.sort_values(by=["Person_ID", "Entry_Time(s)"])
        df.to_csv(filename, index=False)
        print(f"[{now_str()}] Autosaved logs to {filename}")

def save_analytics(final_logs, filename_prefix=None):
    """Generate and save analytics graph."""
    if not final_logs:
        return
    df = pd.DataFrame(final_logs)
    if df.empty:
        return

    df['Entry_Time(s)'] = pd.to_numeric(df['Entry_Time(s)'])
    df['Exit_Time(s)'] = pd.to_numeric(df['Exit_Time(s)'])

    max_time = int(df['Exit_Time(s)'].max()) + 1
    times = []
    counts = []

    for t in range(max_time):
        count = df[(df['Entry_Time(s)'] <= t) & (df['Exit_Time(s)'] >= t)].shape[0]
        times.append(t)
        counts.append(count)

    plt.figure(figsize=(8,4))
    plt.plot(times, counts, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Number of People")
    plt.title("People in Frame Over Time")
    plt.grid(True)
    plt.tight_layout()

    if filename_prefix is None:
        filename_prefix = os.path.join(ANALYTICS_DIR, "people_count")
    plt.savefig(f"{filename_prefix}_{int(time.time())}.png")
    plt.close()
    print(f"[{now_str()}] Analytics saved to {filename_prefix}.png")

# -------------------------
# Initialize trackers
# -------------------------
active_tracks = {}
recently_inactive = {}
final_logs = []
last_frame = None

# -------------------------
# Signal handler for Ctrl+C
# -------------------------
def finalize_active_tracks():
    """Finalize all currently active tracks and save snapshots."""
    global last_frame
    for tid, info in list(active_tracks.items()):
        exit_time = info["last_seen_time"]
        entry_time = info["entry_time"]
        duration = exit_time - entry_time

        # Crop snapshot if possible
        snapshot_path = os.path.join(EVIDENCE_DIR, f"person_{tid}_{int(entry_time)}_{int(exit_time)}.jpg")
        if last_frame is not None:
            cv2.imwrite(snapshot_path, last_frame.copy())
        else:
            cv2.imwrite(snapshot_path, frame.copy())

        final_logs.append({
            "Person_ID": tid,
            "Entry_Time(s)": round(entry_time, 2),
            "Exit_Time(s)": round(exit_time, 2),
            "Duration(s)": round(duration, 2),
            "Evidence": snapshot_path
        })
    active_tracks.clear()

def signal_handler(sig, frame):
    print(f"\n[{now_str()}] Ctrl+C detected. Finalizing tracks, saving logs & analytics...")
    finalize_active_tracks()
    autosave_logs(active_tracks, final_logs, OUTPUT_CSV)
    save_analytics(final_logs)
    cap.release()
    cv2.destroyAllWindows()
    print(f"[{now_str()}] All snapshots, logs, and analytics saved successfully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# -------------------------
# Load models
# -------------------------
print(f"[{now_str()}] Loading YOLOv8 model...")
yolo = YOLO("yolov8n.pt")  

print(f"[{now_str()}] Initializing DeepSort tracker...")
tracker = DeepSort(max_age=DEEPSORT_MAX_AGE, n_init=DEEPSORT_N_INIT)

# -------------------------
# Video initialization
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Error: cannot open video {VIDEO_PATH}")

orig_fps = cap.get(cv2.CAP_PROP_FPS)
if orig_fps is None or orig_fps <= 1:
    orig_fps = 25.0  # safe default for CCTV footage
frame_idx = 0

print(f"[{now_str()}] Starting processing...")
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"[{now_str()}] End of CCTV footage reached. Finalizing results...")
        break
    last_frame = frame.copy()  # Keep last frame for Ctrl+C snapshot
    frame_idx += 1
    current_time_s = frame_idx / orig_fps

    h, w = frame.shape[:2]
    scale = 1.0
    if w > RESCALE_WIDTH:
        scale = RESCALE_WIDTH / w
        frame_proc = cv2.resize(frame, (int(w*scale), int(h*scale)))
    else:
        frame_proc = frame.copy()

    frame_proc = cv2.bilateralFilter(frame_proc, d=5, sigmaColor=75, sigmaSpace=75)

    # YOLO inference
    yolo_results = yolo(frame_proc, conf=DETECTION_CONFIDENCE, stream=False)[0]
    detections_for_tracker = []  
    if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            if yolo.model.names[cls_id].lower() != "person":
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
            if scale != 1.0:
                x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale
            w_box = x2 - x1
            h_box = y2 - y1
            score = float(box.conf[0])
            detections_for_tracker.append(([x1, y1, w_box, h_box], score, "person"))

    # Update DeepSort
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    current_active_ids = set()
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        left, top, right, bottom = map(int, ltrb)
        current_active_ids.add(track_id)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
        cv2.putText(frame, f"Person {track_id}", (left, top-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)

        if track_id not in active_tracks:
            active_tracks[track_id] = {
                "entry_frame": frame_idx,
                "entry_time": current_time_s,
                "last_seen_frame": frame_idx,
                "last_seen_time": current_time_s,
            }
        else:
            active_tracks[track_id]["last_seen_frame"] = frame_idx
            active_tracks[track_id]["last_seen_time"] = current_time_s

    # Check for disappeared tracks
    to_finalize = []
    for tid, info in list(active_tracks.items()):
        if tid not in current_active_ids:
            delta_frames = frame_idx - info["last_seen_frame"]
            delta_seconds = delta_frames / orig_fps
            if delta_seconds >= MAX_INACTIVE_SECONDS:
                to_finalize.append(tid)

    # Finalize disappeared
    for tid in to_finalize:
        info = active_tracks.pop(tid)
        exit_time = info["last_seen_time"]
        entry_time = info["entry_time"]
        duration = exit_time - entry_time

        snapshot_path = os.path.join(EVIDENCE_DIR, f"person_{tid}_{int(entry_time)}_{int(exit_time)}.jpg")
        cv2.imwrite(snapshot_path, frame.copy())

        final_logs.append({
            "Person_ID": tid,
            "Entry_Time(s)": round(entry_time,2),
            "Exit_Time(s)": round(exit_time,2),
            "Duration(s)": round(duration,2),
            "Evidence": snapshot_path
        })

    # Show timestamp
    cv2.putText(frame, f"Time: {sec_to_timestr(current_time_s)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow("CCTV Footage - AI Analytics", frame)

    # Periodic autosave + analytics
    if frame_idx % 300 == 0:
        autosave_logs(active_tracks, final_logs, OUTPUT_CSV)
        save_analytics(final_logs)

    key = cv2.waitKey(1)
    if key == ord("q"):
        print(f"[{now_str()}] 'q' pressed. Saving logs, analytics and exiting.")
        finalize_active_tracks()
        autosave_logs(active_tracks, final_logs, OUTPUT_CSV)
        save_analytics(final_logs)
        break

# End of video processing
finalize_active_tracks()
autosave_logs(active_tracks, final_logs, OUTPUT_CSV)
save_analytics(final_logs)
cap.release()
cv2.destroyAllWindows()
print(f"[{now_str()}] Tracking stopped. Final logs available in {OUTPUT_CSV}")