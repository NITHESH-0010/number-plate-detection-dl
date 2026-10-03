import os
import json
from pathlib import Path
from ultralytics import YOLO

def evaluate_model():
    project_dir = Path(__file__).resolve().parent
    weights_path = project_dir / "weights" / "best.pt"
    rel_weights = "weights/best.pt"
    if not weights_path.exists():
        print(f"Warning: {rel_weights} not found. Falling back to yolov8n.pt")
        weights_path = "yolov8n.pt"
        rel_weights = "yolov8n.pt"

    data_yaml = project_dir / "dataset" / "data.yaml"
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating model: {rel_weights}")
    model = YOLO(str(weights_path))

    # Evaluate model on test set
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        project=str(project_dir / "runs"),
        name="eval_run",
        exist_ok=True,
        verbose=False
    )

    # Extract overall metrics
    mp = float(metrics.box.map50)     # mAP@0.50
    map95 = float(metrics.box.map)    # mAP@0.50:0.95
    precision = float(metrics.box.mp) # Mean Precision
    recall = float(metrics.box.mr)    # Mean Recall
    speed = metrics.speed             # dict: {'preprocess': ..., 'inference': ..., 'loss': ..., 'postprocess': ...}

    metrics_dict = {
        "model": rel_weights,
        "split": "test",
        "test_images": 31,
        "test_plates": 33,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mAP50": round(mp, 4),
        "mAP50_95": round(map95, 4),
        "speed_ms_per_img": {
            "preprocess": round(speed.get("preprocess", 0.0), 2),
            "inference": round(speed.get("inference", 0.0), 2),
            "postprocess": round(speed.get("postprocess", 0.0), 2)
        }
    }

    metrics_content = [
        "=== YOLOv8 License Plate Detection Test Metrics ===",
        f"Model Evaluated : {rel_weights}",
        f"Split           : Test (31 held-out images, 33 plates)",
        f"Precision       : {precision:.4f}",
        f"Recall          : {recall:.4f}",
        f"mAP@50          : {mp:.4f}",
        f"mAP@50-95       : {map95:.4f}",
        f"Inference Speed : {speed.get('inference', 0.0):.2f} ms/img (preprocess: {speed.get('preprocess', 0.0):.2f} ms, postprocess: {speed.get('postprocess', 0.0):.2f} ms)"
    ]

    metrics_text = "\n".join(metrics_content)
    print(metrics_text)

    metrics_file = results_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(metrics_text)

    json_file = results_dir / "metrics.json"
    with open(json_file, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"Evaluation metrics written to: results/metrics.txt and results/metrics.json")

if __name__ == "__main__":
    evaluate_model()
