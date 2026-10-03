import os
import sys
import csv
import time
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
import easyocr

from detect_cli import get_ocr_reader, read_plate_text

def calculate_sharpness(gray_img):
    if gray_img is None or gray_img.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

def process_video(
    source_path,
    weights_path="weights/best.pt",
    conf_threshold=0.4,
    frame_skip=2,
    imgsz=640,
    max_seconds=None,
    out_video_path="results/video_output.mp4",
    csv_out_path="results/video_plates.csv"
):
    src_p = Path(source_path)
    if not src_p.exists():
        print(f"Error: Video file '{source_path}' does not exist.")
        return False

    weights_p = Path(weights_path)
    if not weights_p.exists():
        print(f"Warning: Weights '{weights_path}' not found. Falling back to yolov8n.pt")
        weights_p = Path("yolov8n.pt")

    print(f"=== Video Processing Pipeline ===")
    print(f"Source Video : {src_p.name}")
    print(f"Model Weights: {weights_p.name}")
    print(f"Frame Skip   : {frame_skip}")
    print(f"Image Size   : {imgsz}")

    cap = cv2.VideoCapture(str(src_p))
    if not cap.isOpened():
        print("Error: Failed to open video file.")
        return False

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate target dimensions (max width 960)
    target_w = min(orig_w, 960)
    scale_factor = target_w / float(orig_w)
    target_h = int(orig_h * scale_factor)

    max_frames = total_frames
    if max_seconds is not None and max_seconds > 0:
        max_frames = min(total_frames, int(max_seconds * orig_fps))
        print(f"Max Seconds  : {max_seconds}s ({max_frames} frames max)")

    # Prepare VideoWriter
    out_vid_p = Path(out_video_path)
    out_vid_p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_vid_p), fourcc, orig_fps / frame_skip, (target_w, target_h))

    model = YOLO(str(weights_p))
    reader = get_ocr_reader()

    # Track storage: track_id -> dict
    tracks_data = defaultdict(lambda: {
        "reads": [],          # list of (text, conf)
        "best_score": 0.0,
        "first_frame": None,
        "last_frame": None,
        "current_box": None
    })

    frame_count = 0
    processed_count = 0
    start_time = time.time()

    pbar = tqdm(total=max_frames, desc="Processing Video", unit="frame")

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pbar.update(1)

        if frame_count % frame_skip != 0:
            continue

        processed_count += 1

        # Resize frame to target width 960
        if (orig_w, orig_h) != (target_w, target_h):
            frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            frame_resized = frame.copy()

        annotated = frame_resized.copy()

        # Run YOLOv8 ByteTrack persistence
        results = model.track(
            frame_resized,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False
        )

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0 and boxes.is_track:
            for box in boxes:
                if box.id is None:
                    continue
                track_id = int(box.id[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = map(int, xyxy)
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(target_w, xmax), min(target_h, ymax)

                tr_info = tracks_data[track_id]
                if tr_info["first_frame"] is None:
                    tr_info["first_frame"] = frame_count
                tr_info["last_frame"] = frame_count
                tr_info["current_box"] = (xmin, ymin, xmax, ymax)

                # Check if candidate crop should undergo OCR (max 3 reads per track)
                if len(tr_info["reads"]) < 3:
                    crop = frame_resized[ymin:ymax, xmin:xmax]
                    if crop.size > 0:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        sharpness = calculate_sharpness(gray)
                        area = (xmax - xmin) * (ymax - ymin)
                        score = area * sharpness

                        if sharpness > 30.0 and score > tr_info["best_score"]:
                            txt, ocr_c = read_plate_text(reader, crop)
                            if txt and txt != "PLATE":
                                tr_info["reads"].append((txt, ocr_c))
                                tr_info["best_score"] = score

                # Determine best text for track
                best_text = "DETECTED"
                if tr_info["reads"]:
                    texts = [t for t, c in tr_info["reads"]]
                    most_common = Counter(texts).most_common(1)
                    if most_common:
                        best_text = most_common[0][0]

                # Draw bounding box & tracking label
                cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                lbl = f"#{track_id} {best_text}"
                (w_l, h_l), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (xmin, max(0, ymin - h_l - 8)), (xmin + w_l + 8, ymin), (0, 255, 0), -1)
                cv2.putText(annotated, lbl, (xmin + 4, max(14, ymin - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out_writer.write(annotated)

    cap.release()
    out_writer.release()
    pbar.close()

    elapsed = time.time() - start_time
    fps = processed_count / elapsed if elapsed > 0 else 0.0

    print(f"\nProcessing Complete!")
    print(f"Processed Frames: {processed_count} (skipped every {frame_skip} frames)")
    print(f"Elapsed Time    : {elapsed:.2f} seconds")
    print(f"Processing FPS  : {fps:.2f} frames/sec")
    print(f"Saved Video     : {out_video_path}")

    # Write CSV summary
    csv_p = Path(csv_out_path)
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "plate_text", "ocr_confidence", "first_frame", "last_frame"])
        for tid, data in sorted(tracks_data.items()):
            final_txt = "N/A"
            final_conf = 0.0
            if data["reads"]:
                # Pick most frequent text
                cnt = Counter([t for t, c in data["reads"]])
                top_txt = cnt.most_common(1)[0][0]
                confs = [c for t, c in data["reads"] if t == top_txt]
                final_txt = top_txt
                final_conf = sum(confs) / len(confs)
            writer.writerow([tid, final_txt, f"{final_conf:.4f}", data["first_frame"], data["last_frame"]])

    print(f"Saved Track CSV : {csv_out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="License Plate Tracking & OCR CLI for Videos")
    parser.add_argument("--source", type=str, default="videos/sample.mp4", help="Path to input video file")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size")
    parser.add_argument("--max-seconds", type=float, default=None, help="Process only first N seconds")
    parser.add_argument("--out", type=str, default="results/video_output.mp4", help="Output video path")
    parser.add_argument("--csv-out", type=str, default="results/video_plates.csv", help="Output CSV log path")
    args = parser.parse_args()

    process_video(
        source_path=args.source,
        weights_path=args.weights,
        conf_threshold=args.conf,
        frame_skip=args.frame_skip,
        imgsz=args.imgsz,
        max_seconds=args.max_seconds,
        out_video_path=args.out,
        csv_out_path=args.csv_out
    )

if __name__ == "__main__":
    main()
