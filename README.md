# Real-Time People & Vehicle Tracking System

Real-time computer vision system that detects, tracks and counts **people, cars, motorcycles, buses and trucks** from a webcam, IP camera or CCTV stream, with a line-crossing IN/OUT counter.

Built with **Python, YOLOv8 (Ultralytics), ByteTrack and OpenCV**.

> Privacy by design: objects get anonymous numeric IDs only. No faces, names or personal data are captured or stored.

<!-- Add a demo GIF or screenshot here: ![demo](demo.gif) -->

## Features

- Live input from a webcam, IP camera / CCTV (RTSP or HTTP) or a video file
- Detection of 5 classes: person, car, motorcycle, bus, truck
- Unique tracking with persistent anonymous IDs (tuned ByteTrack)
- Duplicate-count reduction: an object is counted only after it is tracked for N frames
- Line-crossing counter with IN / OUT direction per class
- On-screen dashboard with unique counts, totals and smoothed FPS
- Optional annotated video export
- CSV log of every crossing event (`crossings.csv`)

## Installation

```bash
git clone https://github.com/<your-username>/traffic-tracker.git
cd traffic-tracker
pip install -r requirements.txt
```

The YOLOv8 model (`yolov8n.pt`) downloads automatically on first run.

## Usage

```bash
# Laptop webcam
python traffic_tracker.py --source 0

# Phone (IP Webcam app) or IP camera
python traffic_tracker.py --source "http://<camera-ip>:8080/video"

# CCTV over RTSP
python traffic_tracker.py --source "rtsp://<user>:<password>@<camera-ip>:554/<path>"

# Video file, saving the annotated output
python traffic_tracker.py --source traffic.mp4 --save output.mp4
```

### Options

| Option | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index, stream URL or video path |
| `--model` | `yolov8n.pt` | YOLO model (n/s/m for speed vs accuracy) |
| `--conf` | `0.35` | Detection confidence threshold |
| `--line` | `0.05,0.75,0.95,0.75` | Counting line `x1,y1,x2,y2` as fractions of the frame |
| `--min-frames` | `8` | Frames an object must be tracked before it is counted |
| `--save` | – | Path to save annotated video |
| `--device` | auto | `cpu` or `0` for GPU |

**Keys:** `q` quit, `r` reset counters

## How it works

1. **Detection:** each frame goes through YOLOv8, filtered to the 5 target classes.
2. **Tracking:** ByteTrack links detections across frames into tracks with stable IDs. A longer track buffer keeps IDs through brief occlusions.
3. **Counting:** a track adds to the unique count once it passes the `--min-frames` threshold, which filters out flicker and ID switches.
4. **Line crossing:** the movement between a track's last two positions is tested for intersection with the counting line, and the cross-product sign gives the IN / OUT direction. Each track is counted once.

## Output

`crossings.csv`:

```
timestamp,track_id,class,direction
2026-11-10 18:12:04,12,car,IN
2026-11-10 18:12:09,15,person,OUT
```

## Tech stack

Python · Ultralytics YOLOv8 · ByteTrack · OpenCV · NumPy

## Future work

- Per-hour traffic statistics dashboard
- Multiple counting lines / zones
- GPU / TensorRT optimisation for higher FPS

## License

MIT
