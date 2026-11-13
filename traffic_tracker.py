"""
Live people + vehicle tracker with line-crossing counter.

Features
  - Live webcam / CCTV (RTSP/HTTP) / video file input
  - YOLO detection + ByteTrack: every person/vehicle gets a unique anonymous ID
  - Unique counts: people, cars, motorcycles, buses, trucks
  - Counts an object only after it is tracked for N frames (--min-frames)
  - Crossing-line counter with IN / OUT direction, per class
  - Smoothed FPS display
  - Optional annotated video output + CSV event log

Install:
  pip install ultralytics opencv-python

Run:
  python traffic_tracker.py --source 0                         # webcam
  python traffic_tracker.py --source "rtsp://user:pass@ip:554/stream"
  python traffic_tracker.py --source video.mp4 --save out.mp4
  python traffic_tracker.py --source 0 --line 0.1,0.6,0.9,0.6   # custom line (fractions of width/height)

Keys: q = quit, r = reset counters
"""
import argparse
import csv
import time
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

# COCO class ids -> label
CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {
    "person": (80, 200, 255),
    "car": (90, 220, 90),
    "motorcycle": (255, 180, 60),
    "bus": (60, 120, 255),
    "truck": (200, 90, 220),
}

# Tuned ByteTrack: keeps lost tracks alive longer (track_buffer) so a person who
# moves fast or is briefly hidden keeps the same ID instead of getting a new one.
TRACKER_YAML = """tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.5
track_buffer: 90
match_thresh: 0.85
fuse_score: True
"""
TRACKER_FILE = "tracker_tuned.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="0 = webcam, or RTSP/HTTP URL, or video path")
    p.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m.pt or yolo11n.pt etc.")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--line", default="0.05,0.75,0.95,0.75",
                   help="x1,y1,x2,y2 as fractions of frame size")
    p.add_argument("--min-frames", type=int, default=8,
                   help="frames an object must be tracked before it is counted")
    p.add_argument("--save", default="", help="path to save annotated video (optional)")
    p.add_argument("--log", default="crossings.csv", help="CSV log of crossing events")
    p.add_argument("--device", default=None, help="'cpu', '0' for GPU, etc.")
    p.add_argument("--no-show", action="store_true", help="headless mode")
    return p.parse_args()


def side_of_line(px, py, x1, y1, x2, y2):
    """>0 one side, <0 the other side, 0 on the line."""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def segments_intersect(a, b, c, d):
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def open_source(src):
    if src.isdigit():
        # On Windows, DirectShow is more reliable for webcams than the default (MSMF)
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
        if src.startswith(("rtsp://", "http://", "https://")):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low on CCTV
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {src}")
    for _ in range(10):  # give the camera a moment to warm up
        ok, _frame = cap.read()
        if ok:
            break
        time.sleep(0.1)
    return cap


def draw_panel(frame, unique, line_in, line_out, fps):
    rows = [f"FPS: {fps:5.1f}", ""]
    rows.append("UNIQUE      seen   IN  OUT")
    for name in CLASSES.values():
        rows.append(f"{name:<10}{len(unique[name]):>5}{line_in[name]:>5}{line_out[name]:>5}")
    veh = sum(len(unique[n]) for n in ("car", "motorcycle", "bus", "truck"))
    rows += ["", f"Total vehicles: {veh}", f"Total people:   {len(unique['person'])}"]

    w, h = 300, 22 * len(rows) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + w, 10 + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, t in enumerate(rows):
        cv2.putText(frame, t, (20, 34 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    model = YOLO(args.model)
    with open(TRACKER_FILE, "w") as f:
        f.write(TRACKER_YAML)
    cap = open_source(args.source)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("No frames from source.")
    H, W = frame.shape[:2]
    fx1, fy1, fx2, fy2 = map(float, args.line.split(","))
    L1 = (int(fx1 * W), int(fy1 * H))
    L2 = (int(fx2 * W), int(fy2 * H))

    writer = None
    if args.save:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))

    log_f = open(args.log, "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["timestamp", "track_id", "class", "direction"])

    unique = defaultdict(set)              # class -> set of track ids
    line_in = defaultdict(int)
    line_out = defaultdict(int)
    counted = set()                        # track ids already counted on the line
    last_pt = {}                           # track id -> last centre point
    trails = defaultdict(lambda: deque(maxlen=30))
    seen_frames = defaultdict(int)         # (class, id) -> frames tracked
    fps_hist = deque(maxlen=30)
    prev_t = time.time()

    while ok:
        results = model.track(frame, persist=True, tracker=TRACKER_FILE,
                              classes=list(CLASSES), conf=args.conf, imgsz=args.imgsz,
                              device=args.device, verbose=False)[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), tid, c, cf in zip(boxes, ids, clss, confs):
                name = CLASSES.get(int(c))
                if name is None:
                    continue
                key = (name, tid)
                seen_frames[key] += 1
                if seen_frames[key] >= args.min_frames:
                    unique[name].add(tid)   # count only stable tracks

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # use bottom-centre for people (feet), centre for vehicles
                pt = (cx, y2) if name == "person" else (cx, cy)
                trails[key].append(pt)

                if key in last_pt and key not in counted:
                    prev = last_pt[key]
                    if segments_intersect(prev, pt, L1, L2):
                        s = side_of_line(*pt, *L1, *L2)
                        direction = "IN" if s > 0 else "OUT"
                        (line_in if direction == "IN" else line_out)[name] += 1
                        counted.add(key)
                        log.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), tid, name, direction])
                        log_f.flush()
                last_pt[key] = pt

                col = COLORS[name]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                label = f"{name} #{tid} {cf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)
                pts = list(trails[key])
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], col, 2)

        # counting line
        cv2.line(frame, L1, L2, (0, 0, 255), 3)
        cv2.putText(frame, "COUNT LINE", (L1[0], L1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)

        now = time.time()
        fps_hist.append(1.0 / max(now - prev_t, 1e-6))
        prev_t = now
        draw_panel(frame, unique, line_in, line_out, sum(fps_hist) / len(fps_hist))

        if writer:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("Tracker", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                unique.clear(); line_in.clear(); line_out.clear()
                counted.clear(); last_pt.clear(); trails.clear(); seen_frames.clear()

        ok, frame = cap.read()

    cap.release()
    if writer:
        writer.release()
    log_f.close()
    cv2.destroyAllWindows()

    print("\nFinal unique counts:")
    for n in CLASSES.values():
        print(f"  {n:<10} seen={len(unique[n]):<5} in={line_in[n]:<5} out={line_out[n]}")


if __name__ == "__main__":
    main()