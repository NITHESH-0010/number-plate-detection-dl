# Project Summary: Number Plate Detection using Deep Learning

## Overview
This project delivers an end-to-end Automated License Plate Recognition (ALPR) solution using deep learning. It integrates a trained **YOLOv8n** object detection model with **EasyOCR** text recognition and an interactive web interface powered by **Gradio**.

---

## 📌 Architecture & Pipeline
1. **Input Image**: Vehicle image uploaded via Gradio UI or passed to CLI.
2. **YOLOv8 Object Detection**: Detects license plate bounding boxes (`imgsz=640`).
3. **ROI Extraction & Preprocessing**: Crops plate region, converts to grayscale, upscales 2x using cubic interpolation, and applies adaptive thresholding.
4. **EasyOCR Character Recognition**: Extracts alphanumeric characters and filters non-alphanumeric noise.
5. **Annotated Output**: Returns bounding box overlay, cropped plate ROI, and extracted text string.

---

## 📊 Dataset & Cleaning Report
- **Dataset Source**: Kaggle (`andrewmvd/car-plate-detection`)
- **Initial Raw Images**: 433
- **Duplicate Images Removed**: 131 (MD5 hash check)
- **Retained Valid Samples**: 302
- **Data Splits (70 / 20 / 10)**:
  - **Train**: 211 images
  - **Validation**: 60 images
  - **Test**: 31 images
- **Annotation Format**: YOLO format (`class x_center y_center width height`, normalized `0-1`)

---

## 📈 Final Model Performance Metrics
- **Model**: YOLOv8n (Nano detector, 3.0M parameters)
- **Hardware & Epochs**: 15 epochs on CPU (1.19 hours)
- **Validation Metrics (60 images, 67 plates)**:
  - **Precision**: `0.8100`
  - **Recall**: `0.7660`
  - **mAP @ 0.50**: `0.7960`
  - **mAP @ 0.50-0.95**: `0.4430`
- **Test Metrics (31 held-out images, 33 plates)**:
  - **Precision**: `0.8470`
  - **Recall**: `0.8392`
  - **mAP @ 0.50**: `0.8908`
  - **mAP @ 0.50-0.95**: `0.5146`
  - **Inference Speed**: `111.5 ms / image` (CPU)

---

## ✅ Status & Progress Log
- [x] Project environment setup & virtual environment initialization.
- [x] Security audit & `.gitignore` configuration.
- [x] Installed dependencies (`ultralytics`, `easyocr`, `gradio`, `opencv-python`, etc.).
- [x] Dataset download, validation, cleaning & YOLO format conversion.
- [x] YOLOv8 model training for 15 epochs on CPU.
- [x] Model evaluation & metric generation (`results/metrics.txt`).
- [x] OCR pipeline refinement & text sanitization.
- [x] Interactive Gradio Web UI (`app.py`) & CLI detector (`detect_cli.py`).
- [x] Push readiness verification & documentation.
