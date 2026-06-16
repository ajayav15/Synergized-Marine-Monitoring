# Synergized Marine Monitoring

Official code repository for the paper:

> **Synergizing YOLO-based Waste Detection with XGBoost Water Quality Analysis**
> Gowtham M, Ajay A V, Francesco Flammini
> JSS Science and Technology University (Mysuru, India) · IDSIA USI-SUPSI (Manno, Switzerland)


A lightweight, edge-deployable dual-module framework for real-time marine environmental monitoring. The system combines a **YOLOv11** detector for underwater debris (15 classes) with an **XGBoost** classifier for water potability screening, interpreted with **SHAP**. The vision pipeline uses CUDA-accelerated **Dark Channel Prior (DCP) + CLAHE** optical restoration to handle turbid water, and the full pipeline is deployable on an **NVIDIA Jetson Orin Nano** at ~31 FPS / 12.5 W with TensorRT FP16.

---

## Highlights

| Module        | Metric              | Value                |
|---------------|---------------------|----------------------|
| YOLOv11 detector | mAP@0.5          | **90.3 %**           |
| YOLOv11 detector | mAP@[0.5:0.95]   | 64.5 %               |
| YOLOv11 detector | F1 (microplastics) | **83.2 %** (vs 78.5 % YOLOv8 baseline) |
| XGBoost classifier | Accuracy (held-out 491-sample measured test fold) | **92.5 %** |
| XGBoost classifier | ROC-AUC          | 0.96                 |
| XGBoost classifier | FNR (non-potable) | 0.073                |
| Edge throughput  | Jetson Orin Nano, FP16 | **31 FPS end-to-end** (41 FPS model-only) |
| Edge power       | Jetson Orin Nano    | 12.5 W               |

---

## Repository layout

```
Synergized-Marine-Monitoring/
├── app.py                          # Streamlit demo: image upload, water quality form, summary report
├── yolo11.py                       # YOLOv11 training script (Colab-exported)
├── inference.py                    # YOLOv11 detection wrapper
├── dark_channel_prior.py           # DCP-based underwater haze removal
├── YOLO11.ipynb                    # YOLOv11 training notebook
├── OceanWaste.ipynb                # Combined-pipeline notebook (vision + classification)
├── Train_Water_Potabililty.ipynb   # XGBoost water-quality training notebook
├── Trained_Model_yolo11.pt         # Trained YOLOv11 weights (debris detector)
├── Ocean_waste_Model_yolo11.pt     # Alternate YOLOv11 weights used by app.py
├── yolov8n.pt                      # YOLOv8 baseline weights (for comparison)
├── xgboost_without_source_month.pkl # Trained XGBoost classifier (PyCaret)
├── test_df                         # Sample test CSV for the water-quality module
├── logs.log                        # Training logs
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT
└── README.md                       # This file
```

---

## Installation

The project targets **Python ≥ 3.10**. A clean virtual environment is recommended.

```bash
# clone the repo
git clone https://github.com/ajayav15/Synergized-Marine-Monitoring.git
cd Synergized-Marine-Monitoring

# create + activate a fresh environment
python -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate

# install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU (optional but recommended)

For training and faster inference, install PyTorch with CUDA support that matches your driver. See [pytorch.org](https://pytorch.org/get-started/locally/) for the exact pip command.

```bash
# example: CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Edge deployment (NVIDIA Jetson Orin Nano)

For TensorRT FP16 deployment on Jetson, follow NVIDIA's official Jetson stack:

1. Install JetPack 6.x via NVIDIA SDK Manager.
2. Install Ultralytics: `pip install ultralytics`.
3. Export the trained weights to TensorRT FP16:
   ```bash
   yolo export model=Ocean_waste_Model_yolo11.pt format=engine half=True imgsz=640 device=0
   ```
4. Use the generated `.engine` file in `inference.py` for ~31 FPS end-to-end throughput.

---

## Quick start

### 1. Launch the Streamlit demo

The fastest way to exercise the full pipeline is the Streamlit app, which provides three pages: **Underwater Waste Detection**, **Water Quality Assessment**, and **Generated Report**.

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, upload an underwater image, or enter physicochemical readings to see the predictions.

### 2. Train the YOLOv11 detector

The training notebook `YOLO11.ipynb` (and the exported `yolo11.py`) expects a YOLO-format dataset (`data.yaml` with `train/`, `val/`, `test/` splits). The TrashCan-Instance dataset can be obtained from [the TrashCan reference paper](https://arxiv.org/abs/2007.08097) and re-mapped to the 15 standardized categories used here.

```bash
yolo task=detect mode=train model=yolov8s.pt data=path/to/data.yaml epochs=60 imgsz=640
```

### 3. Train the XGBoost water-quality classifier

Open `Train_Water_Potabililty.ipynb` in Jupyter or Colab. The notebook:

1. Loads the public [Water Potability dataset (Kaggle)](https://www.kaggle.com/datasets/adityakadiwal/water-potability) (3,276 records).
2. Cleans + stratifies into 70 % / 15 % / 15 % train / val / test (≈ 2,294 / 491 / 491).
3. Imputes missing values with the training-fold feature-wise median.
4. Applies **SMOTE** to the training fold only (validation and test folds remain unresampled).
5. Trains XGBoost via PyCaret and exports the model pickle.
6. Generates SHAP summary and dependence plots.

### 4. Reproduce the SHAP interpretability analysis

The XGBoost notebook ends with SHAP plots that match Figs. 3 and 4 in the paper. The plots are generated with the standard `shap.summary_plot()` and `shap.dependence_plot()` APIs.

---

## Datasets

| Dataset | Source | Size | Used for |
|---------|--------|------|----------|
| TrashCan-Instance | [Hong, Fulton & Sattar, 2020](https://arxiv.org/abs/2007.08097) | 7,212 images | YOLOv11 detector training |
| Custom underwater set | Authors (GoPro HERO10 Black at 5–15 m depth, 10–50 NTU, Karnataka & Goa coastline) | 5,000 images | YOLOv11 detector training |
| Water Potability dataset | [Kaggle (Kadiwal, 2021)](https://www.kaggle.com/datasets/adityakadiwal/water-potability) | 3,276 records | XGBoost classifier training |

The custom underwater image subset (with annotations and class-mapping table) is available under the **CC BY-NC 4.0** licence from the corresponding author on reasonable request. The annotations and class-mapping table are included in this repository.

---

## Pre-trained models

| Filename | Module | Notes |
|----------|--------|-------|
| `Trained_Model_yolo11.pt` | YOLOv11 detector | Primary debris detection weights |
| `Ocean_waste_Model_yolo11.pt` | YOLOv11 detector | Used by `app.py` |
| `xgboost_without_source_month.pkl` | XGBoost classifier | PyCaret-exported, water potability |
| `yolov8n.pt` | YOLOv8 baseline | Reference for comparison only |

To run inference with the trained YOLO weights:

```python
from ultralytics import YOLO
model = YOLO("Trained_Model_yolo11.pt")
results = model("path/to/underwater_image.jpg", conf=0.25, iou=0.45)
results[0].save("output.jpg")
```

---

## Reproducing paper results

| Section / Table | Notebook                          |
|-----------------|-----------------------------------|
| Table 5 — detector mAP / F1, with bootstrap 95 % CI | `YOLO11.ipynb` |
| Table 6 — ablation (YOLOv8 → YOLOv11 → +DCP → +aug → +multi-scale) | `YOLO11.ipynb` |
| Table 7 — complexity vs. competing models | `YOLO11.ipynb` |
| Table 8 — water-quality classifier metrics | `Train_Water_Potabililty.ipynb` |
| Table 9 — confusion matrix (held-out 491-sample fold) | `Train_Water_Potabililty.ipynb` |
| Table 10 — turbidity-band robustness | `OceanWaste.ipynb` |
| Figs. 3–4 — SHAP summary / dependence plots | `Train_Water_Potabililty.ipynb` |
| Figs. 5–6 — qualitative detection examples | `OceanWaste.ipynb` / `app.py` |

All training runs were repeated with **3 random seeds**; reported values are means with standard deviation ≤ ±0.2 mAP@0.5. The 95 % confidence intervals in Table 5 are from **1,000 bootstrap iterations** over the held-out test images.

---

## Citation

If you use this code or the trained models, please cite:

```bibtex
@article{gowtham2026synergized,
  title   = {Synergizing YOLO-based Waste Detection with XGBoost Water Quality Analysis},
  author  = {Gowtham, M. and Ajay, A. V. and Flammini, Francesco},
  journal = {Discover Artificial Intelligence},
  year    = {2026},
  note    = {Submitted; citation details to be updated upon acceptance.}
}
```

---

## Acknowledgements

This work builds on:

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- [XGBoost](https://github.com/dmlc/xgboost) (Chen & Guestrin, 2016)
- [SHAP](https://github.com/shap/shap) (Lundberg & Lee, 2017)
- [TrashCan-Instance dataset](https://arxiv.org/abs/2007.08097) (Hong et al., 2020)
- [Water Potability dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability) (Kadiwal, 2021)
- WHO Drinking-Water Quality Guidelines and U.S. EPA National Primary Drinking Water Regulations (used as reference thresholds for the rule-based baseline only).

---

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE) for the full text.

The custom underwater image subset (not in this repository) is released separately under **CC BY-NC 4.0** and is available from the corresponding author on reasonable request.

---

## Contact

- **Gowtham M** — gouthamgouda@gmail.com (JSS Science and Technology University, Mysuru, India)
- **Ajay A V** — ([GitHub @ajayav15](https://github.com/ajayav15)) (JSS Science and Technology University, Mysuru, India)
- **Francesco Flammini** — francesco.flammini@supsi.ch *(corresponding author)* (IDSIA USI-SUPSI, Manno, Switzerland)
