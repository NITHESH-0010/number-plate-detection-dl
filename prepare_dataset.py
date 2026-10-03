import os
import sys
import glob
import hashlib
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def get_kaggle_config_dir():
    # Priority: existing KAGGLE_CONFIG_DIR env var, then fallback checks
    if os.environ.get("KAGGLE_CONFIG_DIR") and os.path.exists(os.path.join(os.environ["KAGGLE_CONFIG_DIR"], "kaggle.json")):
        return os.environ["KAGGLE_CONFIG_DIR"]
    
    # Check parent workspace directories for kaggle.json silently
    curr = Path(__file__).resolve().parent
    for p in [curr, curr.parent, curr.parent.parent]:
        if (p / "kaggle.json").exists():
            return str(p)
            
    # Check default user home folder
    home_kaggle = Path.home() / ".kaggle"
    if (home_kaggle / "kaggle.json").exists():
        return str(home_kaggle)
        
    return None

def download_dataset(dataset_slug, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    cfg_dir = get_kaggle_config_dir()
    if cfg_dir:
        os.environ["KAGGLE_CONFIG_DIR"] = cfg_dir
    
    print(f"Downloading dataset '{dataset_slug}' using Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path=raw_dir, unzip=True)
        print("Download and extraction complete.")
    except Exception as e:
        print(f"Kaggle API download failed: {e}")
        print("Ensuring fallback dataset extraction...")
        # Check if zip file exists in raw_dir or current dir
        zips = glob.glob(os.path.join(raw_dir, "*.zip")) + glob.glob("*.zip")
        if zips:
            import zipfile
            with zipfile.ZipFile(zips[0], 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            print(f"Extracted {zips[0]} to {raw_dir}")
        else:
            raise RuntimeError(f"Could not download or find zip for {dataset_slug}. Error: {e}")

def parse_voc_xml(xml_path, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        # Clip bounding box to image boundaries
        xmin = max(0.0, min(xmin, float(img_w)))
        ymin = max(0.0, min(ymin, float(img_h)))
        xmax = max(0.0, min(xmax, float(img_w)))
        ymax = max(0.0, min(ymax, float(img_h)))
        
        w = xmax - xmin
        h = ymax - ymin
        
        # Check for non-positive dimensions
        if w <= 0 or h <= 0:
            continue
            
        # Convert to YOLO format (normalized 0-1)
        x_center = (xmin + xmax) / (2.0 * img_w)
        y_center = (ymin + ymax) / (2.0 * img_h)
        norm_w = w / img_w
        norm_h = h / img_h
        
        boxes.append((0, x_center, y_center, norm_w, norm_h))
        
    return boxes

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def prepare_data():
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "dataset"
    raw_dir = dataset_dir / "raw"
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_slug = "andrewmvd/car-plate-detection"
    download_dataset(dataset_slug, raw_dir)
    
    # Locate images and annotations directory
    img_dir = raw_dir / "images"
    ann_dir = raw_dir / "annotations"
    
    if not img_dir.exists():
        img_dir = raw_dir
    if not ann_dir.exists():
        ann_dir = raw_dir
        
    img_files = sorted(glob.glob(str(img_dir / "*.png")) + glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.jpeg")))
    
    report_lines = []
    report_lines.append("=== Data Cleaning & Preparation Report ===")
    report_lines.append(f"Selected Dataset Slug: {dataset_slug}")
    report_lines.append(f"Initial raw image count: {len(img_files)}")
    
    valid_samples = []
    seen_hashes = set()
    
    removed_corrupt = 0
    removed_no_ann = 0
    removed_invalid_boxes = 0
    removed_duplicates = 0
    
    for img_path in img_files:
        # Check corrupt image
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            removed_corrupt += 1
            continue
            
        # Duplicate check
        img_hash = calculate_md5(img_path)
        if img_hash in seen_hashes:
            removed_duplicates += 1
            continue
        seen_hashes.add(img_hash)
        
        # Annotation check
        stem = Path(img_path).stem
        xml_path = ann_dir / f"{stem}.xml"
        if not xml_path.exists():
            removed_no_ann += 1
            continue
            
        boxes = parse_voc_xml(xml_path, img_w, img_h)
        if not boxes:
            removed_invalid_boxes += 1
            continue
            
        valid_samples.append((img_path, boxes))
        
    report_lines.append(f"Removed corrupt images: {removed_corrupt}")
    report_lines.append(f"Removed duplicate images: {removed_duplicates}")
    report_lines.append(f"Removed missing annotations: {removed_no_ann}")
    report_lines.append(f"Removed invalid/empty boxes: {removed_invalid_boxes}")
    report_lines.append(f"Total valid samples retained: {len(valid_samples)}")
    
    # Random split (70 / 20 / 10)
    random.seed(42)
    random.shuffle(valid_samples)
    
    n_total = len(valid_samples)
    n_train = int(n_total * 0.70)
    n_valid = int(n_total * 0.20)
    
    train_samples = valid_samples[:n_train]
    val_samples = valid_samples[n_train:n_train + n_valid]
    test_samples = valid_samples[n_train + n_valid:]
    
    report_lines.append("\n=== Dataset Splits ===")
    report_lines.append(f"Train split (70%): {len(train_samples)}")
    report_lines.append(f"Val split   (20%): {len(val_samples)}")
    report_lines.append(f"Test split  (10%): {len(test_samples)}")
    
    splits = {
        "train": train_samples,
        "valid": val_samples,
        "test": test_samples
    }
    
    # Save formatted dataset
    for split_name, samples in splits.items():
        split_img_dir = dataset_dir / "images" / split_name
        split_lbl_dir = dataset_dir / "labels" / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, boxes in samples:
            stem = Path(img_path).stem
            dest_img = split_img_dir / f"{stem}{Path(img_path).suffix}"
            dest_lbl = split_lbl_dir / f"{stem}.txt"
            
            shutil.copy2(img_path, dest_img)
            with open(dest_lbl, "w") as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                    
    # Generate data.yaml
    data_yaml_path = dataset_dir / "data.yaml"
    yaml_content = """path: .
train: images/train
val: images/valid
test: images/test

names:
  0: license_plate
"""
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)
        
    report_lines.append("\nGenerated dataset configuration file at: dataset/data.yaml")
    
    # Write report
    report_text = "\n".join(report_lines)
    report_file = results_dir / "data_cleaning_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)
    print(report_text)
    
    # Generate 9-image visualization grid
    sample_grid = train_samples[:9]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (img_path, boxes) in enumerate(sample_grid):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img_rgb.shape
        
        for b in boxes:
            _, xc, yc, nw, nh = b
            xmin = int((xc - nw / 2) * w_img)
            ymin = int((yc - nh / 2) * h_img)
            xmax = int((xc + nw / 2) * w_img)
            ymax = int((yc + nh / 2) * h_img)
            cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img_rgb, "license_plate", (xmin, max(0, ymin - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"Sample {idx+1}: {Path(img_path).name}")
        axes[idx].axis("off")
        
    plt.tight_layout()
    grid_path = results_dir / "dataset_samples.png"
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"Sample grid saved to: {grid_path.as_posix()}")

if __name__ == "__main__":
    prepare_data()
