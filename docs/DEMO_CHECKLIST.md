# Laboratory Presentation & Demo Checklist

Use this checklist during your lab examination or presentation to showcase the project smoothly:

---

## 📋 Phase 1: Environment & Codebase Setup (1 Minute)

- [ ] Open Windows PowerShell inside the project directory.
- [ ] Activate the virtual environment:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- [ ] Confirm weights file exists under `weights/best.pt` (~5.96 MB).

---

## 📋 Phase 2: Live Web App Presentation (2 Minutes)

- [ ] Ensure the Gradio app is running (or launch via `python app.py`).
- [ ] Open your browser to `http://127.0.0.1:7860`.
- [ ] **Step 1**: Click on one of the pre-loaded sample vehicle images from `test_images/`.
- [ ] **Step 2**: Adjust the Confidence Slider (default `0.25`).
- [ ] **Step 3**: Click **🔍 Detect & Recognize Plate**.
- [ ] **Step 4**: Highlight the 3 output panels:
  1. Bounding box overlay image.
  2. Cropped & preprocessed plate ROI.
  3. Formatted alphanumeric OCR text.

---

## 📋 Phase 3: CLI Execution Demo (1 Minute)

- [ ] Open a terminal tab and run batch prediction on all test images:
  ```powershell
  python detect_cli.py --source test_images --out results/predictions
  ```
- [ ] Show output summary in the terminal listing confidence score and recognized text.
- [ ] Show saved prediction images in `results/predictions/`.

---

## 📋 Phase 4: Model Evaluation & Metrics (1 Minute)

- [ ] Display evaluation report:
  ```powershell
  python evaluate.py
  ```
- [ ] Point out key model evaluation metrics in `results/metrics.txt` and `README.md`:
  - **Validation (60 images, 67 plates)**: Precision `0.8100`, Recall `0.7660`, mAP50 `0.7960`, mAP50-95 `0.4430`
  - **Test (31 held-out images, 33 plates)**: Precision `0.8470`, Recall `0.8392`, mAP50 `0.8908`, mAP50-95 `0.5146`, Inference `111.5 ms/img` (CPU)

---

## 📋 Phase 5: Code & Data Hygiene Verification (30 Seconds)

- [ ] Show clean folder structure (`docs/`, `results/`, `test_images/`, `weights/`).
- [ ] Open `docs/PROJECT_SUMMARY.md` or `README.md` to demonstrate documentation completeness.
- [ ] Confirm no temporary or secret files are present.
