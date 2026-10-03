import os
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

def train_model():
    project_dir = Path(__file__).resolve().parent
    data_yaml = project_dir / "dataset" / "data.yaml"
    weights_dir = project_dir / "weights"
    results_dir = project_dir / "results"
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    epochs = 15 if device == "cpu" else 30
    
    print(f"--- Training Configuration ---")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: 640")
    print(f"Data config: {data_yaml}")

    # Load YOLOv8 nano pre-trained model
    model = YOLO("yolov8n.pt")

    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=device,
        project=str(project_dir / "runs"),
        name="train_run",
        exist_ok=True,
        plots=True,
        seed=42
    )

    # Path to trained best weights
    train_run_dir = project_dir / "runs" / "train_run"
    best_weights = train_run_dir / "weights" / "best.pt"
    dest_weights = weights_dir / "best.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, dest_weights)
        size_mb = dest_weights.stat().st_size / (1024 * 1024)
        print(f"Successfully saved trained model to: {dest_weights} ({size_mb:.2f} MB)")
    else:
        print("Warning: best.pt not found in training output run.")

    # Copy plots & metrics to results folder
    plot_files = [
        "confusion_matrix.png", "F1_curve.png", "P_curve.png",
        "R_curve.png", "PR_curve.png", "results.png",
        "val_batch0_labels.jpg", "val_batch0_pred.jpg"
    ]
    for pf in plot_files:
        src_plot = train_run_dir / pf
        if src_plot.exists():
            shutil.copy2(src_plot, results_dir / pf)
            print(f"Copied training artifact to results/: {pf}")

if __name__ == "__main__":
    train_model()
