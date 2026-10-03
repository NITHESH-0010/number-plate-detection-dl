import os
import re
import glob
import time
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
import gradio as gr
from ultralytics import YOLO
import easyocr

_EASYOCR_READER = None
_YOLO_MODEL = None

def load_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        weights_path = Path("weights/best.pt")
        if not weights_path.exists():
            print("Warning: weights/best.pt not found. Falling back to pretrained yolov8n.pt")
            weights_path = "yolov8n.pt"
        else:
            print(f"Loading custom trained YOLOv8 model from {weights_path}")
        _YOLO_MODEL = YOLO(str(weights_path))
    return _YOLO_MODEL

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

    res = reader.readtext(enhanced, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
    if not res:
        return "", 0.0

    box_areas = [((b[0][1][0] - b[0][0][0]) * (b[0][2][1] - b[0][0][1]), b) for b in res]
    max_area = max(a for a, b in box_areas)
    filtered_res = [b for a, b in box_areas if a >= 0.25 * max_area and len(b[1].strip()) > 0]

    if not filtered_res:
        return "", 0.0

    res_sorted = sorted(filtered_res, key=lambda b: (b[0][0][1] // 25, b[0][0][0]))
    raw_joined = ' '.join([r[1].strip() for r in res_sorted])
    cleaned_text = clean_ocr_text(raw_joined)
    
    avg_conf = sum(r[2] for r in res_sorted) / len(res_sorted)
    return cleaned_text, float(avg_conf)

def calculate_sharpness(gray_img):
    if gray_img is None or gray_img.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

def detect_and_read_plate_img(input_img, conf_threshold=0.4):
    if input_img is None:
        return None, None, "Please upload or select an input image."

    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    annotated_bgr = img_bgr.copy()
    h_img, w_img = img_bgr.shape[:2]

    model = load_yolo_model()
    reader = get_ocr_reader()

    results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        cv2.putText(annotated_bgr, "No license plate detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb, None, "⚠️ No License Plate Detected"

    detected_texts = []
    crops_rgb = []

    for box in boxes:
        conf_val = float(box.conf[0].cpu().numpy())
        xyxy = box.xyxy[0].cpu().numpy()
        xmin, ymin, xmax, ymax = map(int, xyxy)

        pad = 5
        xmin_p = max(0, xmin - pad)
        ymin_p = max(0, ymin - pad)
        xmax_p = min(w_img, xmax + pad)
        ymax_p = min(h_img, ymax + pad)

        crop = img_bgr[ymin_p:ymax_p, xmin_p:xmax_p]
        if crop.size == 0:
            continue

        text_str, ocr_c = read_plate_text(reader, crop)
        crops_rgb.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        if not text_str:
            text_str = "PLATE"

        detected_texts.append((text_str, conf_val))

        cv2.rectangle(annotated_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        lbl = f"{text_str} ({conf_val:.2f})"
        (w_lbl, h_lbl), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated_bgr, (xmin, max(0, ymin - h_lbl - 12)), (xmin + w_lbl + 10, ymin), (0, 255, 0), -1)
        cv2.putText(annotated_bgr, lbl, (xmin + 5, max(18, ymin - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    plate_crop = crops_rgb[0] if crops_rgb else None
    result_str = "\n".join([f"🚘 Plate {i+1}: {txt} (Confidence: {conf:.2f})" for i, (txt, conf) in enumerate(detected_texts)])

    return annotated_rgb, plate_crop, result_str

def process_video_gradio(video_path, conf_threshold=0.4, frame_skip=2, max_seconds=15):
    if not video_path or not os.path.exists(video_path):
        return None, "Please upload a valid video file."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error opening video file."

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = min(orig_w, 960)
    scale_factor = target_w / float(orig_w)
    target_h = int(orig_h * scale_factor)

    max_frames = total_frames
    if max_seconds and max_seconds > 0:
        max_frames = min(total_frames, int(max_seconds * orig_fps))

    temp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(temp_out, fourcc, orig_fps / frame_skip, (target_w, target_h))

    model = load_yolo_model()
    reader = get_ocr_reader()

    tracks_data = defaultdict(lambda: {
        "reads": [],
        "best_score": 0.0,
        "first_frame": None,
        "last_frame": None
    })

    frame_count = 0
    processed_count = 0
    start_time = time.time()

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        processed_count += 1
        frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA) if (orig_w, orig_h) != (target_w, target_h) else frame.copy()
        annotated = frame_resized.copy()

        results = model.track(frame_resized, persist=True, tracker="bytetrack.yaml", conf=conf_threshold, imgsz=640, verbose=False)
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

                best_text = "DETECTED"
                if tr_info["reads"]:
                    texts = [t for t, c in tr_info["reads"]]
                    most_common = Counter(texts).most_common(1)
                    if most_common:
                        best_text = most_common[0][0]

                cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                lbl = f"#{track_id} {best_text}"
                (w_l, h_l), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (xmin, max(0, ymin - h_l - 8)), (xmin + w_l + 8, ymin), (0, 255, 0), -1)
                cv2.putText(annotated, lbl, (xmin + 4, max(14, ymin - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out_writer.write(annotated)

    cap.release()
    out_writer.release()

    elapsed = time.time() - start_time
    fps = processed_count / elapsed if elapsed > 0 else 0.0

    summary_lines = [
        f"⏱️ Video Processed in {elapsed:.2f}s ({fps:.1f} FPS)",
        f"📊 Processed {processed_count} frames | Tracked Plates:",
        "-" * 40
    ]

    for tid, data in sorted(tracks_data.items()):
        final_txt = "N/A"
        final_conf = 0.0
        if data["reads"]:
            top_txt = Counter([t for t, c in data["reads"]]).most_common(1)[0][0]
            confs = [c for t, c in data["reads"] if t == top_txt]
            final_txt = top_txt
            final_conf = sum(confs) / len(confs)
        summary_lines.append(f"🚘 Track #{tid:02d} | Plate: {final_txt:12s} | Conf: {final_conf:.2f} | Frames: {data['first_frame']}-{data['last_frame']}")

    return temp_out, "\n".join(summary_lines)

example_images = sorted(glob.glob("test_images/*.*"))

custom_css = """
.container { max-width: 1000px; margin: auto; }
.title-banner { text-align: center; margin-bottom: 1.5rem; }
.title-banner h1 { font-size: 2.2rem; font-weight: 700; color: #2b2b2b; }
.title-banner p { font-size: 1.1rem; color: #666; }
"""

with gr.Blocks(title="Number Plate Detection & Tracking System") as demo:
    gr.HTML("""
        <div class="title-banner">
            <h1>🚗 Deep Learning License Plate Recognition System</h1>
            <p>Laboratory Program 9: YOLOv8 Object Detection + ByteTrack + EasyOCR Pipeline</p>
        </div>
    """)

    with gr.Tabs():
        with gr.Tab("📷 Image Detection & OCR"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Upload Vehicle Image", type="numpy")
                    conf_slider_img = gr.Slider(minimum=0.1, maximum=0.9, value=0.4, step=0.05, label="Detection Confidence Threshold")
                    submit_btn_img = gr.Button("🔍 Detect & Recognize Plate", variant="primary")
                    
                    if example_images:
                        gr.Examples(examples=example_images, inputs=input_image, label="Sample Vehicle Test Images")

                with gr.Column(scale=1):
                    output_annotated_img = gr.Image(label="Detected License Plate Box", type="numpy")
                    output_crop_img = gr.Image(label="Cropped & Preprocessed Plate ROI", type="numpy")
                    output_text_img = gr.Textbox(label="Recognized License Plate Text", lines=3, interactive=False)

            submit_btn_img.click(
                fn=detect_and_read_plate_img,
                inputs=[input_image, conf_slider_img],
                outputs=[output_annotated_img, output_crop_img, output_text_img]
            )

        with gr.Tab("🎥 Video Tracking & OCR"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(label="Upload Vehicle Video Clip")
                    conf_slider_vid = gr.Slider(minimum=0.1, maximum=0.9, value=0.4, step=0.05, label="Detection Confidence Threshold")
                    frame_skip_slider = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Frame Skip (Speed Optimization)")
                    max_sec_slider = gr.Slider(minimum=5, maximum=60, value=15, step=5, label="Max Processing Seconds")
                    submit_btn_vid = gr.Button("🎬 Track & Recognize Video Plates", variant="primary")

                with gr.Column(scale=1):
                    output_video = gr.Video(label="Annotated Tracking Video Output")
                    output_text_vid = gr.Textbox(label="Tracked License Plates Summary Log", lines=10, interactive=False)

            submit_btn_vid.click(
                fn=process_video_gradio,
                inputs=[input_video, conf_slider_vid, frame_skip_slider, max_sec_slider],
                outputs=[output_video, output_text_vid]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
