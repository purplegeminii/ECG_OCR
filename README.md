# ECG Postpaid Meter Readings — Tesseract OCR Fine-Tuning Pipeline

A robust, end-to-end pipeline for training/retraining a Tesseract OCR model
specifically optimised for Ghana ECG (Electricity Company of Ghana) postpaid
meter reading images.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Quick Start](#quick-start)
3. [Pipeline Phases](#pipeline-phases)
4. [Configuration](#configuration)
5. [Scripts Reference](#scripts-reference)
6. [Evaluation](#evaluation)
7. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
ecg-ocr-project/
├── raw_images/          # Original meter reading photos
├── preprocessed/        # Cleaned/deskewed images ready for OCR
├── ground_truth/        # .gt.txt label files (paired with .tif)
├── augmented/           # Synthetically augmented images
├── results/             # Inference outputs, CSV reports, test_set.txt
├── models/
│   └── ecg_meter/
│       └── tessdata/    # Final .traineddata model installed here
├── tesstrain/           # git submodule — owns all training internals
│   └── data/
│       └── ecg_meter-ground-truth/  # ← 04_prepare_training_data.py writes here
├── scripts/             # All pipeline scripts
├── config/              # YAML configs
├── logs/                # Training logs, CER curves
├── notebooks/           # Jupyter analysis notebooks
├── tests/               # Unit tests
└── docs/                # Additional documentation
```

---

## Quick Start

```bash
# 1. Clone the repo with the tesstrain submodule
git clone --recurse-submodules <your-repo-url>
cd ecg-ocr-project

# 2.1 READ docs/macos_setup.md
# 2.1 Install system and Python dependencies
bash scripts/install_dependencies.sh

# 3. Configure the project
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your paths

# 4. Preprocess your raw images
python scripts/01_preprocess.py --input raw_images/ --output preprocessed/

# 5. Label your images
python scripts/02_annotate.py --manual --images preprocessed/ --output ground_truth/

# 6. Augment the dataset
python scripts/03_augment.py --input preprocessed/ --output augmented/ --factor 5

# 7. Copy pairs into tesstrain's ground-truth directory
python scripts/04_prepare_training_data.py

# 8. Run fine-tuning via tesstrain
bash scripts/05_run_training.sh

# 9. Evaluate the model
python scripts/06_evaluate.py --model models/ecg_meter/tessdata/ecg_meter.traineddata

# 10. Run inference on new images
python scripts/07_inference.py --input raw_images/new/ --output results/
```

---

## Pipeline Phases

### Phase 1 — Preprocessing
Converts raw meter photos into clean, normalised images:
- Automatic deskew & perspective correction
- Adaptive thresholding (handles glare, shadows)
- ROI (region of interest) extraction (isolates the meter display)
- Upscaling to ≥300 DPI equivalent

### Phase 2 — Annotation
Ground truth labeling with validation:
- Label Studio integration
- Automatic format conversion to `.gt.txt`
- Quality checks on labels

### Phase 3 — Augmentation
Synthetic data generation to expand small datasets:
- Brightness/contrast variation
- Slight rotation & perspective warp
- Gaussian noise & blur
- Morphological degradation

### Phase 4 — Training Data Preparation
Converts annotated images to Tesseract-ready format:
- Paired `.tif` + `.gt.txt` files
- Train/validation/test split (80/10/10)

### Phase 5 — Fine-Tuning
LSTM fine-tuning from `eng` base model via tesstrain:
- Configurable iterations and learning rate
- Checkpoint saving
- Real-time CER monitoring

### Phase 6 — Evaluation
Comprehensive metrics:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Field-level (reading, account number) accuracy
- Confusion matrix for misread characters

### Phase 7 — Inference
Production-ready OCR with:
- Character whitelist for meter domains
- Post-processing validation (regex + business rules)
- Confidence thresholding & human review flagging
- CSV/JSON output

---

## Configuration

Edit `config/config.yaml`:

```yaml
model:
  name: ecg_meter
  base_model: eng
  max_iterations: 10000
  learning_rate: 0.0001
  target_cer: 0.02

training:
  split: [0.80, 0.10, 0.10]
  augmentation_factor: 5

preprocessing:
  target_dpi: 300
  adaptive_thresh_block_size: 11
  adaptive_thresh_c: 2
  min_image_width: 800

ocr:
  psm: 6
  oem: 1
  whitelist: "0123456789.-kWhKWH/ABCDEFGHIJKLMNOPQRSTUVWXYZ "
  confidence_threshold: 60
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `01_preprocess.py` | Image cleaning, deskew, ROI extraction |
| `02_annotate.py` | Label Studio launcher + format converter |
| `03_augment.py` | Synthetic data augmentation |
| `04_prepare_training_data.py` | Build tesstrain-ready file pairs |
| `05_run_training.sh` | Execute tesstrain fine-tuning |
| `06_evaluate.py` | Full evaluation with metrics |
| `07_inference.py` | Production inference pipeline |
| `08_iterative_correction.py` | dshea89-style error correction loop |
| `install_dependencies.sh` | System + Python dependency installer |
| `plot_training_curves.py` | Visualise CER/WER over training |

---

## Troubleshooting

**Low accuracy on specific meter brands**
Add more images of that meter type. ECG uses Actaris, Landis+Gyr, and Conlog
meters — each has different digit fonts.

**High CER despite large dataset**
Check preprocessing: glare, perspective distortion, and low contrast are the
most common culprits. Run `scripts/01_preprocess.py --debug` to visualise each
preprocessing step.

**Training loss not decreasing**
Lower `learning_rate` to `0.00005` or reduce `max_iterations` if overfitting.

**Tesseract not found**
Run `bash scripts/install_dependencies.sh` and ensure `/usr/bin/tesseract` is
on your PATH.
