import os
import sys
import re
import glob
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr

_EASYOCR_READER = None

def get_ocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        use_gpu = torch.cuda.is_available()
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=use_gpu)
    return _EASYOCR_READER

def clean_ocr_text(raw_text):
    if not raw_text:
        return ""
    text = raw_text.upper()
    text = re.sub(r'[^A-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_plate(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_2x = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_2x)
    return enhanced

def read_plate_text(reader, crop_bgr):
    enhanced = preprocess_plate(crop_bgr)
    if enhanced is None:
        return "", 0.0

    # Read text using allowlist for uppercase alphanumeric characters and spaces
    res = reader.readtext(enhanced, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
    if not res:
        return "", 0.0

    # Filter out small noise sub-boxes (area < 25% of max sub-box area)
    box_areas = [((b[0][1][0] - b[0][0][0]) * (b[0][2][1] - b[0][0][1]), b) for b in res]
    max_area = max(a for a, b in box_areas)
    filtered_res = [b for a, b in box_areas if a >= 0.25 * max_area and len(b[1].strip()) > 0]

    if not filtered_res:
        return "", 0.0

    # Sort text boxes top-to-bottom, left-to-right
    res_sorted = sorted(filtered_res, key=lambda b: (b[0][0][1] // 25, b[0][0][0]))
    raw_joined = ' '.join([r[1].strip() for r in res_sorted])
    cleaned_text = clean_ocr_text(raw_joined)
    
    avg_conf = sum(r[2] for r in res_sorted) / len(res_sorted)
    return cleaned_text, float(avg_conf)

def process_image(image_path, model, conf_threshold=0.4):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None, False, 0.0, "Error: Unable to load image."

    annotated_img = img_bgr.copy()
    h_img, w_img = img_bgr.shape[:2]

    results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        cv2.putText(annotated_img, "No license plate detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return annotated_img, False, 0.0, "N/A"

    reader = get_ocr_reader()
    detected_texts = []
    max_conf = 0.0

    for idx, box in enumerate(boxes):
        conf_val = float(box.conf[0].cpu().numpy())
        if conf_val > max_conf:
            max_conf = conf_val

        xyxy = box.xyxy[0].cpu().numpy()
        xmin, ymin, xmax, ymax = map(int, xyxy)

        # 5px padding around detected bounding box
        pad = 5
        xmin_p = max(0, xmin - pad)
        ymin_p = max(0, ymin - pad)
        xmax_p = min(w_img, xmax + pad)
        ymax_p = min(h_img, ymax + pad)

        crop = img_bgr[ymin_p:ymax_p, xmin_p:xmax_p]
        if crop.size == 0:
            continue

        text_str, ocr_conf = read_plate_text(reader, crop)
        if not text_str:
            text_str = "PLATE"

        detected_texts.append(text_str)

        # Draw bounding box & label
        cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        label_text = f"{text_str} ({conf_val:.2f})"
        (w_lbl, h_lbl), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_img, (xmin, max(0, ymin - h_lbl - 10)), (xmin + w_lbl + 10, ymin), (0, 255, 0), -1)
        cv2.putText(annotated_img, label_text, (xmin + 5, max(15, ymin - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated_img, True, max_conf, ", ".join(detected_texts)

def main():
    parser = argparse.ArgumentParser(description="License Plate Detection CLI")
    parser.add_argument("--source", type=str, default="test_images", help="Path to image file or directory")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--out", type=str, default="results/predictions", help="Output directory")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Weights {weights_path} not found. Falling back to yolov8n.pt")
        weights_path = "yolov8n.pt"

    model = YOLO(str(weights_path))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(args.source)
    if src_path.is_file():
        image_paths = [src_path]
    elif src_path.is_dir():
        image_paths = sorted(list(src_path.glob("*.jpg")) + list(src_path.glob("*.png")) + list(src_path.glob("*.jpeg")))
    else:
        print(f"Error: Source path '{args.source}' does not exist.")
        sys.exit(1)

    print(f"=== CLI License Plate Detection & OCR Summary ===")
    for img_p in image_paths:
        ann_img, detected, conf_val, text_res = process_image(img_p, model, conf_threshold=args.conf)
        if ann_img is not None:
            save_p = out_dir / f"pred_{img_p.name}"
            cv2.imwrite(str(save_p), ann_img)
            det_str = "Yes" if detected else "No"
            print(f"Image: {img_p.name:12s} | Plate Detected: {det_str:3s} | Confidence: {conf_val:.2f} | OCR Read: {text_res}")

if __name__ == "__main__":
    main()
