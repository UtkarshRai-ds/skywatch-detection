---
title: Skywatch Detection
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8501
---

# SkyWatch Detection

> **Satellite object detection on the NWPU VHR-10 dataset using YOLOv8 — from raw annotations to a live-inference Streamlit dashboard.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-ff4b4b.svg)](https://streamlit.io)
[![CI](https://img.shields.io/github/actions/workflow/status/utkarsh26rai/skywatch-detection/ci.yml?branch=master&label=CI)](https://github.com/utkarsh26rai/skywatch-detection/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end computer vision pipeline that detects 10 object classes in very-high-resolution satellite imagery. Covers data validation, exploratory analysis, model training on Colab T4, rigorous evaluation, and a five-page interactive dashboard with live inference.

---

![SkyWatch Demo](assets/demo.gif)

---

## Key Results

| Metric | YOLOv8n | YOLOv8s |
|---|---|---|
| mAP50 | **0.689** | **0.697** |
| mAP50-95 | 0.426 | 0.437 |
| Precision | 0.794 | 0.743 |
| Recall | 0.634 | 0.690 |
| CPU inference (avg) | 57 ms/img | 139 ms/img |
| FPS (CPU) | 17 | 7 |
| Parameters | 3.0 M | 11.1 M |
| Best epoch | 37 / 50 | 40 / 50 |
| Training time (T4) | 9.4 min | 11.4 min |

YOLOv8s wins on mAP50 (+0.8 pp) and recall (+5.6 pp) at the cost of 3.7× more parameters and 2.4× slower CPU inference. YOLOv8n is the better choice for edge or latency-constrained deployments.

---

## Project Highlights

- **Full pipeline in one repo** — raw VOC XML annotations → YOLO labels → training → evaluation → interactive dashboard, all reproducible from a single clone.
- **Rigorous data quality** — five automated checks (image counts, annotation parity, class coverage, readability, bbox coordinate validity) run before any model work.
- **Dual-model comparison** — YOLOv8n and YOLOv8s trained under identical conditions; per-class AP deltas, speed/accuracy scatter, and confusion matrices surfaced in the dashboard.
- **Bridge detection deep-dive** — the dominant class (29.5 % of annotations) achieves only 0.03 mAP50 due to elongated aspect ratios and severe background clutter; the dashboard exposes this failure mode explicitly.
- **Production-ready CI/CD** — GitHub Actions runs pytest (40 tests) and flake8 on every push; the app is Dockerised for one-command deployment.

---

## Dataset

**NWPU VHR-10** is a benchmark dataset for multi-class geospatial object detection in very-high-resolution satellite imagery.

| Property | Value |
|---|---|
| Total images | 800 |
| Train / Valid / Test | 560 / 160 / 80 |
| Total annotations | 4 811 |
| Image size | 640 × 640 px |
| Format | YOLO (normalised cx cy w h) |

### Classes

| ID | Class | Annotations |
|---|---|---|
| 0 | Airplane | 418 |
| 1 | Ship | 124 |
| 2 | Storage tank | 655 |
| 3 | Baseball diamond | 178 |
| 4 | Tennis court | 379 |
| 5 | Basketball court | 152 |
| 6 | Ground track field | 163 |
| 7 | Harbor | 364 |
| 8 | **Bridge** | **1 421** |
| 9 | Vehicle | 957 |

Bridge accounts for 29.5 % of all annotations, while ship contributes only 2.6 %, yielding an **11.46× imbalance ratio**.

---

## Roboflow

The dataset was sourced from [Roboflow Universe — NWPU VHR-10](https://universe.roboflow.com/yolo-7muwp/nwpu-vhr-10-sgj3z).

The Roboflow export handled the train/valid/test split (70/20/10) and converted the original Pascal VOC XML annotations to YOLO `.txt` label format. One notable issue with the export: the `names` field in the generated `data.yaml` contained citation text fragments instead of actual class names (a known Roboflow export bug with certain public datasets). This is corrected at load time in `src/data/loader.py`:

```python
# loader.py detects nc == 10 and replaces garbled names with the canonical list
NWPU_CLASSES: list[str] = [
    "airplane", "ship", "storage tank", "baseball diamond",
    "tennis court", "basketball court", "ground track field",
    "harbor", "bridge", "vehicle",
]
```

The same fix is applied inside `notebooks/training.ipynb` via `src.models.train.fix_data_yaml()` before training begins on Colab.

---

## EDA Findings

Exploratory analysis was performed across all 4 811 annotations using seven dedicated functions in `src/features/analyze.py`. Key findings:

1. **11.46× class imbalance** — bridge (1 421 instances) vs ship (124). Airplane, ship, baseball diamond, and vehicle fall below the `mean − 0.5σ` threshold and are flagged as undertrained.
2. **Tiny objects dominate** — median bounding-box area is **0.61 % of image area** (median width 0.062, height 0.057 in normalised units). Small-object detection is the core challenge.
3. **Strong co-occurrence** — ground track field and harbor share 919 annotations in the same images, the strongest co-occurrence pair. Bridge co-occurs with almost every other class due to sheer volume.
4. **All images are 640 × 640** — the Roboflow export applied stretch resizing uniformly. Letterbox padding is recommended for future fine-tuning to preserve aspect ratios.
5. **Zero negative images** — the original NWPU VHR-10 includes 150 background tiles with no objects. These were excluded from the Roboflow project, so all 800 images contain at least one annotation. This inflates precision at low thresholds.

---

## Model Architecture & Training

Both models use the Ultralytics YOLOv8 architecture with a CSPDarknet backbone, PANet feature-pyramid neck, and a decoupled anchor-free detection head.

| Component | YOLOv8n | YOLOv8s |
|---|---|---|
| Depth multiplier | 0.33 | 0.33 |
| Width multiplier | 0.25 | 0.50 |
| Parameters | 3.0 M | 11.1 M |
| Pretrained weights | COCO | COCO |

### Training Configuration

All hyperparameters are documented in `src/models/train.py::TRAIN_CONFIG` and reproduced here for reference:

| Hyperparameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 |
| Optimizer | AdamW |
| Initial LR (lr0) | 0.002 |
| Final LR ratio (lrf) | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| Seed | 42 |

### Augmentation Strategy

| Augmentation | Value | Rationale |
|---|---|---|
| Mosaic | 1.0 | Combines 4 images; critical for small-object generalisation |
| HSV hue jitter | 0.015 | Subtle colour shift for sensor variation |
| HSV saturation jitter | 0.7 | Handles atmospheric and seasonal variation |
| HSV value jitter | 0.4 | Compensates for illumination differences |
| Horizontal flip | 0.5 | Orientation invariance |
| Scale | 0.5 | Multi-scale robustness |
| Mixup | 0.0 | Disabled — small objects lose identity under blending |

Training was executed on Google Colab with a T4 GPU (16 GB VRAM). YOLOv8n completed in **9.4 minutes**; YOLOv8s in **11.4 minutes**.

---

## Results

### Per-Class mAP50

| Class | YOLOv8n | YOLOv8s | Delta (s − n) |
|---|---|---|---|
| Airplane | 0.561 | 0.675 | **+0.114** |
| Ship | 0.782 | 0.764 | −0.018 |
| Storage tank | 0.895 | 0.926 | +0.031 |
| Baseball diamond | 0.892 | 0.901 | +0.009 |
| Tennis court | 0.838 | 0.841 | +0.003 |
| Basketball court | 0.430 | 0.422 | −0.008 |
| Ground track field | 0.579 | 0.589 | +0.010 |
| Harbor | **0.928** | 0.889 | −0.039 |
| Bridge | 0.030 | 0.027 | −0.003 |
| Vehicle | **0.952** | 0.934 | −0.018 |

### Best & Worst Classes

- **Best** (both models): vehicle (>0.93) and harbor (>0.89) — compact, high-contrast objects with consistent aspect ratios.
- **Worst** (both models): bridge (≈0.03) and basketball court (<0.43) — consistently below 0.5 mAP50.

### Bridge Failure Analysis

Bridge is the most annotated class yet achieves near-zero detection performance. The root causes:

1. **Aspect ratio extremes** — bridge bounding boxes span the full image width at very low height. The standard 640 × 640 anchor-free head struggles with extreme aspect ratios.
2. **Background confusion** — bridges are visually continuous with roads and water. At low confidence thresholds the model detects road segments as bridges and vice versa.
3. **Label ambiguity** — in the NWPU VHR-10 annotation convention, a bridge annotation covers the entire deck span including approach roads, making the "object" boundary semantically ambiguous.

Potential remedies: dedicated bridge anchor priors, rotated bounding boxes (OBB), or a higher-resolution crop strategy for elongated structures.

---
## Limitations

### 1. Severe Class Imbalance (11.46×)
Bridge dominates with 1 421 instances vs ship's 124. Despite being the most annotated class, bridge achieves only **0.03 mAP50** — the model over-predicts common classes and fails on geometrically hard ones. Standard cross-entropy loss does not penalise easy negatives enough to learn rare class boundaries.

### 2. Tiny Object Detection
Median bounding-box area is **0.61% of image area**. At 640×640 resolution, small objects like airplanes and vehicles lose critical spatial detail during downsampling. The stride-32 feature map (P5) cannot resolve objects smaller than ~20×20 px, which describes a large fraction of this dataset.

### 3. Stretch Resize Distortion
Roboflow's default 'Stretch' preprocessing resizes all images to 640×640 by distorting aspect ratios. Elongated objects like bridges and runways are compressed horizontally, causing the model to learn distorted shape priors that do not generalise to real-world imagery.

### 4. Limited Training Data
800 images across 10 classes (~80 per class on average) is insufficient for robust generalisation. The model has seen very few examples of rare configurations (e.g. partially occluded ships, bridges at oblique angles).

### 5. No Hyperparameter Search
Both models used Ultralytics default hyperparameters. Learning rate, mosaic probability, and augmentation strength were not tuned — a proper search could yield +3–5% mAP50.

### 6. CPU-Only Deployment
The HuggingFace Space runs on CPU. Inference latency (57–139 ms/image) makes real-time video detection impractical in the current deployment.

---

## Scope of Future Work

### Short Term
- **Letterbox padding** — replace Roboflow's stretch resize with letterbox to preserve aspect ratios of elongated objects (bridges, runways). Expected improvement: +5–10% mAP50 on bridge class.
- **Focal loss (γ ≥ 1.5)** — penalise easy negatives more aggressively to force the model to learn rare class boundaries.
- **Class-balanced sampling** — oversample minority classes (ship, baseball diamond) to reduce the effective imbalance ratio below 3×.
- **Higher input resolution (1280×1280)** — improves small-object detection at the cost of 4× compute. Feasible on Colab T4 with batch size 4.

### Medium Term
- **SAHI (Sliced Inference)** — divide large images into overlapping tiles at inference time, run detection on each tile, then merge predictions. Proven to improve small-object recall by 15–20% on satellite imagery.
- **Oriented Bounding Boxes (OBB)** — YOLOv8-OBB supports rotated boxes natively. Bridges, runways, and ships at arbitrary angles would benefit significantly from angle-aware regression.
- **Hyperparameter Evolution** — use Ultralytics' built-in evolutionary search or Optuna to tune lr0, lrf, mosaic, and mixup over 100+ trials.
- **Dataset Expansion** — augment with DIOR (20 classes, 23 463 images) or xView (60 classes, 1 127 images) to improve generalisation across sensor types and geographic regions.

### Long Term
- **Transformer-Based Detection (RT-DETR / YOLOv9)** — attention mechanisms capture global context better than CNNs for geospatial scenes where objects appear at arbitrary scales and orientations.
- **MLOps Pipeline** — automate retraining with GitHub Actions when new data arrives; version models via MLflow or W&B Model Registry; serve via FastAPI + Docker for production deployment.
- **Multi-Temporal Fusion** — combine imagery from multiple timestamps to detect change (new construction, vessel movement) rather than static object presence.
- **Edge Deployment** — quantise YOLOv8n to INT8 with TensorRT or ONNX Runtime for deployment on satellite ground stations or UAV processors.

---
## Weights & Biases

Both training runs were tracked with [Weights & Biases](https://wandb.ai). W&B automatically logged:

- Training and validation loss curves (box, cls, dfl) for all 50 epochs
- Per-epoch precision, recall, mAP50, and mAP50-95
- Confusion matrices after each validation pass
- Sample detection images with bounding box overlays
- Full hyperparameter configuration

[🚀 Live Demo on HuggingFace](https://huggingface.co/spaces/Utkarsh-DS/Skywatch-Detection)

---

## Project Structure

```
skywatch-detection/
├── app/
│   └── streamlit_app.py        # 5-page interactive dashboard (live inference)
├── data/
│   ├── raw/                    # YOLO-format dataset (gitignored)
│   │   ├── train/images/
│   │   ├── train/labels/
│   │   ├── valid/images/
│   │   ├── valid/labels/
│   │   ├── test/images/
│   │   ├── test/labels/
│   │   └── data.yaml
│   ├── sample_detections/      # 30 annotated JPEGs (15 per model)
│   ├── class_stats.json        # Per-class AP, comparison, speed benchmarks
│   ├── dataset_stats.json      # Loader + quality + EDA results
│   ├── training_history.json   # 50-epoch loss/metric curves
│   ├── yolov8n_confusion_matrix.png
│   ├── yolov8s_confusion_matrix.png
│   ├── yolov8n_BoxF1_curve.png
│   └── yolov8s_BoxF1_curve.png
├── models/
│   ├── yolov8n_best.pt         # Trained checkpoint (gitignored)
│   └── yolov8s_best.pt         # Trained checkpoint (gitignored)
├── notebooks/
│   ├── eda.ipynb               # 25-cell Plotly EDA notebook
│   └── training.ipynb          # Colab-ready training notebook (T4 GPU)
├── src/
│   ├── data/
│   │   ├── loader.py           # NWPU_CLASSES, dataset stats, YAML fix
│   │   ├── quality.py          # 5 automated quality checks
│   │   └── converter.py        # Pascal VOC XML → YOLO TXT converter
│   ├── features/
│   │   └── analyze.py          # 7 EDA functions
│   └── models/
│       ├── evaluate.py         # ModelInfo, inference speed, CSV parser
│       ├── compare.py          # Side-by-side model comparison
│       ├── predict.py          # Batch inference with annotated output
│       └── train.py            # Hyperparameters + Colab training functions
├── tests/
│   ├── test_data_quality.py    # 17 tests for quality checks
│   ├── test_converter.py       # 12 tests for VOC→YOLO converter
│   └── test_model.py           # 11 tests for evaluate + compare
├── .github/
│   └── workflows/
│       └── ci.yml              # pytest + flake8 on push/PR
├── Dockerfile                  # python:3.12-slim, CPU torch, Streamlit
├── docker-compose.yml          # Single-service compose with data volume
├── requirements.txt
├── runtime.txt                 # python-3.12
├── run_loader.py               # Step 1: validate dataset, write dataset_stats.json
├── run_eda.py                  # Step 2: run EDA, append to dataset_stats.json
└── run_evaluation.py           # Step 3: evaluate models, write class_stats.json
```

---

## Quick Start

### Prerequisites

- Python 3.12
- Git

```bash
git clone https://github.com/utkarsh26rai/skywatch-detection.git
cd skywatch-detection
pip install -r requirements.txt
```

### Run the Dashboard

The dashboard reads from pre-computed JSON files. All data files are committed to the repo so you can launch immediately without a GPU:

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). Pages 1–3 and Page 5 are fully functional without model checkpoints. Page 4 (live inference) requires `models/yolov8n_best.pt` and/or `models/yolov8s_best.pt`.

### Regenerate Data Files (optional)

If you have the dataset in `data/raw/`:

```bash
# Step 1 — validate dataset and compute loader stats
python run_loader.py

# Step 2 — run all 7 EDA analyses
python run_eda.py

# Step 3 — evaluate models and generate sample detections
#           (requires models/yolov8n_best.pt and yolov8s_best.pt)
python run_evaluation.py
```

---

## Docker

```bash
# Build and start
docker-compose up --build

# Open http://localhost:8501
```

The `docker-compose.yml` mounts `./data` into the container so the dashboard reads live JSON files from the host. Model checkpoints are not included in the image; place them in `models/` before starting if you need live inference.

To build the image manually:

```bash
docker build -t skywatch-detection .
docker run -p 8501:8501 -v $(pwd)/data:/app/data skywatch-detection
```

---

## Reproducing Training

Training requires a Colab account (free T4 tier is sufficient).

1. Open `notebooks/training.ipynb` in Google Colab (`File → Upload notebook` or mount Drive).
2. Set runtime to **T4 GPU** (`Runtime → Change runtime type → T4 GPU`).
3. Run the cells in order:
   - **Install** — Ultralytics, W&B, Roboflow SDK
   - **W&B login** — `wandb.login()` (create a free account at wandb.ai)
   - **Roboflow download** — pulls the dataset using your API key
   - **Fix data.yaml** — corrects garbled class names via `fix_data_yaml()`
   - **Train YOLOv8n** — 50 epochs, ~9.4 minutes
   - **Train YOLOv8s** — 50 epochs, ~11.4 minutes
   - **Evaluate** — runs `model.val(split='test')` for both checkpoints
   - **Download** — saves `yolov8n_best.pt` and `yolov8s_best.pt` locally

All hyperparameters are defined in `src/models/train.py::TRAIN_CONFIG`. Override any value by passing kwargs to `train_model()`.

---

## Running Tests

```bash
pytest tests/ -v
```

40 tests covering data quality checks, the VOC→YOLO converter, `build_test_results`, and `compare_models`. Tests use `tmp_path` fixtures and do not require the full dataset or model checkpoints.

---

## Tech Stack

| Component | Library / Tool | Version |
|---|---|---|
| Object detection | Ultralytics YOLOv8 | ≥ 8.2 |
| Training infrastructure | Google Colab T4 GPU | — |
| Experiment tracking | Weights & Biases | ≥ 0.17 |
| Dashboard | Streamlit | 1.37 |
| Visualisation | Plotly | ≥ 5.20 |
| Image I/O | OpenCV | ≥ 4.9 |
| Numerics | NumPy | ≥ 1.26 |
| Dataset source | Roboflow Universe | — |
| Containerisation | Docker / Compose | — |
| CI | GitHub Actions | — |
| Language | Python | 3.12 |

---

## Citation

If you use this project or the NWPU VHR-10 dataset, please cite the original paper:

```bibtex
@article{cheng2016learning,
  title   = {Learning rotation-invariant convolutional neural networks for object detection in VHR optical remote sensing images},
  author  = {Cheng, Gong and Zhou, Peicheng and Han, Junwei},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {54},
  number  = {12},
  pages   = {7405--7415},
  year    = {2016},
  doi     = {10.1109/TGRS.2016.2601622}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The NWPU VHR-10 dataset is subject to its own terms. Please review the [Roboflow Universe dataset page](https://universe.roboflow.com/yolo-7muwp/nwpu-vhr-10-sgj3z) before use in commercial applications.

---

*Built by Utkarsh Rai · [utkarsh26rai@gmail.com](mailto:utkarsh26rai@gmail.com)*
