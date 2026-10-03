# Project Summary: Number Plate Detection using Deep Learning

## Overview
This project delivers an end-to-end Automated License Plate Recognition (ALPR) solution using deep learning. It integrates a YOLOv8 object detection model with EasyOCR text recognition and presents an interactive web interface powered by Gradio.

## Architecture & Pipeline
1. **Input**: Vehicle image (upload or sample).
2. **Detection**: YOLOv8 model detects license plate bounding boxes.
3. **Cropping & Preprocessing**: Extracts ROI, converts to grayscale, upscales 2x, and applies adaptive thresholding.
4. **OCR Recognition**: EasyOCR reads and filters alphanumeric text.
5. **Output**: Annotated image displaying bounding box and formatted plate number text.

## Dataset & Training
- **Source**: Kaggle License Plate Dataset.
- **Format**: YOLO format annotations (`class x_center y_center width height`).
- **Splits**: 70% Train / 20% Validation / 10% Test.
- **Model**: YOLOv8n detector.

## Status & Progress Log
- [x] Project environment setup & git repo link.
- [x] Security audit & `.gitignore` configuration.
- [ ] Dependencies installation (`ultralytics`, `easyocr`, `gradio`, `opencv-python`, etc.).
- [ ] Kaggle dataset download, validation, cleaning & YOLO format conversion.
- [ ] YOLOv8 model training & validation.
- [ ] Model evaluation & metric generation (`metrics.txt`, confusion matrix, precision-recall curves).
- [ ] OCR pipeline refinement & edge-case handling.
- [ ] Interactive Gradio Web UI (`app.py`) & CLI detector (`detect_cli.py`).
- [ ] GitHub repository synchronization & final verification.
