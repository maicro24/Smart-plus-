# main.py
import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
from collections import deque

# ---------- CONFIG ----------
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
DEFAULT_CONF = 0.35
DEFAULT_IMGSZ = 640

OUTPUT_CSV = "traffic_timeseries_FINAL.csv"
OUTPUT_VIDEO = "traffic_out.mp4"

# ⏱️ Fixed time step
TIME_STEP = 0.25  # seconds

# 🚦 ZONE OF INTEREST (only inside this zone is counted)
ZONE_POLYGON = np.array([
    (430, 685),
    (490, 489),
    (514, 299),
    (482, 146),
    (350, 142),
    (310, 137),
    (65, 447),
    (25, 604),
    (465, 635)
], np.int32)

# Estimated capacity of this zone
ZONE_CAPACITY = 10
# ----------------------------

def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", required=True)
    parser.add_argument("--model", "-m", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_per_step = max(1, int(TIME_STEP * fps))

    # prepare output video
    output_video = OUTPUT_VIDEO
    if os.path.exists(output_video):
        try:
            os.remove(output_video)
        except:
            output_video = "traffic_out_new.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    model = YOLO(args.model)

    print("🚀 Processing video...")

    results_stream = model.track(
        source=args.video,
        conf=args.conf,
        imgsz=args.imgsz,
        persist=True,
        stream=True
    )

    rows = []
    frame_idx = 0
    step_idx = 0  # counts 0.25s steps

    prev_positions = {}
    density_history = deque(maxlen=5)

    for res in results_stream:
        frame_idx += 1
        img = res.orig_img
        boxes = res.boxes

        vehicle_count = 0
        speeds = []
        current_positions = {}

        for box in boxes:
            cls = int(box.cls.cpu().numpy()[0])
            if cls not in VEHICLE_CLASS_IDS:
                continue

            xy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xy)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # ONLY INSIDE ZONE
            if not point_in_poly((cx, cy), ZONE_POLYGON):
                continue

            vehicle_count += 1

            track_id = None
            if hasattr(box, "id") and box.id is not None:
                try:
                    track_id = int(box.id.cpu().numpy()[0])
                except:
                    track_id = None

            # speed estimation (pixel displacement)
            if track_id is not None:
                current_positions[track_id] = (cx, cy)
                if track_id in prev_positions:
                    px, py = prev_positions[track_id]
                    dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                    speeds.append(dist * fps)

            # draw detection
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        prev_positions = current_positions

        # draw zone
        overlay = img.copy()
        cv2.fillPoly(overlay, [ZONE_POLYGON], (0, 0, 0))
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        cv2.polylines(img, [ZONE_POLYGON], True, (0,255,255), 2)

        writer.write(img)

        # 🔥 SAVE ONE ROW EVERY 0.25s (NO AVERAGE, NO FLOAT COUNT)
        if frame_idx % frames_per_step == 0:
            step_idx += 1
            time_sec = step_idx * TIME_STEP  # ✅ perfect 0.25, 0.50, 0.75, ...

            avg_speed = np.mean(speeds) if speeds else 0.0
            speed_var = np.var(speeds) if speeds else 0.0

            # density = normalized occupancy
            density = vehicle_count / ZONE_CAPACITY
            density = min(1.0, density)

            density_history.append(vehicle_count)
            if len(density_history) >= 2:
                density_trend = density_history[-1] - density_history[-2]
            else:
                density_trend = 0

            flow_trend = avg_speed * vehicle_count

            rows.append({
                "time_sec": f"{time_sec:.2f}",          # fixed time grid
                "vehicle_count": int(vehicle_count),    # ✅ PURE INTEGER
                "average_speed_kmh": round(avg_speed * 0.1, 2),
                "density": round(density, 3),            # ✅ density column
                "speed_variance": round(speed_var, 2),
                "density_trend": int(density_trend),
                "flow_trend": round(flow_trend, 2),
                "data_confidence": 1.0
            })

    writer.release()

    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
        print("📊 CSV saved:", OUTPUT_CSV)

    print("✅ Done. Video saved:", output_video)

if __name__ == "__main__":
    main()
