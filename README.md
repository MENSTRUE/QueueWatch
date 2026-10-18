# 👥 QueueWatch v5.1

<p align="center"><strong>Small-data crowd-count regression using a 5-fold MobileNetV3-Small ensemble</strong></p>
<p align="center">ImageNet-pretrained MobileNetV3-Small → log-count regression → 5-fold cross-validation → median ensemble</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img src="https://img.shields.io/badge/Backbone-MobileNetV3--Small-purple" />
  <img src="https://img.shields.io/badge/Task-Crowd%20Count%20Regression-orange" />
  <img src="https://img.shields.io/badge/App-Streamlit-ff4b4b" />
  <img src="https://img.shields.io/badge/Status-MVP-success" />
</p>

---

## Overview

**QueueWatch v5.1** is a computer-vision regression MVP for estimating the number of visible people in a single image.

The final model uses **ImageNet-pretrained MobileNetV3-Small** and predicts a transformed person count using **`log1p(count)` regression**. Five cross-validation models are trained, and inference combines their count-space outputs with a **median ensemble**.

The repository name is **QueueWatch**, but the current validated capability is **general crowd-count estimation**, not full queue analytics.

> **Current scope:** single-image crowd counting that can act as an early occupancy component for a future queue-monitoring system.

The current MVP does **not** yet estimate queue length, waiting time, queue order, trajectories, service throughput, density maps, per-person bounding boxes, or video-based queue behavior.

---

## 9% Portfolio Milestone

This repository represents the **9% learning milestone** in an ongoing portfolio growth journey.

The percentage refers to learning progress, not project count and not production readiness.

---

## Problem Framing

Crowded scenes are difficult to summarize quickly from a single image. A lightweight count estimate can help describe occupancy level without requiring a full detection or tracking pipeline.

This project explores whether a compact pretrained CNN can learn a useful crowd-count regression function from a **very small labeled dataset**.

Because the available dataset is general crowd imagery rather than a dedicated queue dataset, QueueWatch v5.1 should be interpreted as a **crowd-counting MVP**, not a validated queue-management system.

---

## Final v5.1 Architecture

```text
Input image
    ↓
RGB conversion
    ↓
Resize to 320 × 320
    ↓
ImageNet normalization
    ↓
MobileNetV3-Small
(ImageNet pretrained)
    ↓
Preserved classifier trunk
Linear(576 → 1024)
    ↓
Final regression head
Linear(1024 → 1)
    ↓
Predict log1p(count)
    ↓
expm1
    ↓
Count prediction
```

### 5-fold inference

```text
Input image
    ↓
Fold 1 model ─┐
Fold 2 model ─┤
Fold 3 model ─┼─→ expm1 per fold
Fold 4 model ─┤
Fold 5 model ─┘
                ↓
          median ensemble
                ↓
      final estimated count
```

---

## Dataset

The final experiment used the IOAI computer-vision dataset mounted in Kaggle as:

```text
/kaggle/input/datasets/wafabila/ioai-computer-vision
```

Important folders/files:

```text
labeled/
test/
ground_truth_train.json
sample_submission.csv
```

### Dataset facts

- **100 labeled images**
- **216 test images**
- count minimum: **15**
- count median: **89.5**
- count mean: **126.36**
- count maximum: **539**

This is a **small-data setting**, so 5-fold cross-validation is used instead of relying on one holdout split.

The 216 test images are used for final ensemble predictions/submission. Their ground-truth labels are not part of the local final benchmark reported here.

---

## Training Strategy

Final v5.1 setup:

```text
Backbone       : MobileNetV3-Small
Pretraining    : ImageNet
Input size     : 320 × 320
Target         : log1p(count)
CV             : 5 folds
Batch size     : 8
Stage 1        : regression head training
Stage 2        : fine-tune final feature blocks
Loss           : SmoothL1
Ensemble       : median of 5 fold predictions
```

The final v5.1 setup is the champion configuration used by the MVP.

---

## Cross-Validation Results

### Fold MAE

| Fold | Validation MAE |
|---|---:|
| Fold 1 | 49.726 |
| Fold 2 | 43.898 |
| Fold 3 | 33.129 |
| Fold 4 | 35.054 |
| Fold 5 | 47.221 |

### Final OOF metrics

| Metric | Result |
|---|---:|
| **OOF MAE** | **41.805** |
| **OOF RMSE** | **65.514** |

### Naive baseline comparison

| Metric | Naive baseline | QueueWatch v5.1 |
|---|---:|---:|
| MAE | 74.260 | **41.805** |
| RMSE | 111.624 | **65.514** |

Absolute MAE improvement: **32.455**  
Relative MAE reduction: **≈43.7%**

> This is a **regression task**. MAE/RMSE must not be presented as classification accuracy.

---

## Evaluation Visuals

### Fold Validation MAE

<p align="center"><img src="assets/previews/fold_validation_mae.png" width="760" alt="Fold validation MAE"/></p>

### OOF Prediction Scatter

<p align="center"><img src="assets/previews/oof_scatter.png" width="760" alt="OOF scatter plot"/></p>

### OOF Error Distribution

<p align="center"><img src="assets/previews/oof_error_hist.png" width="760" alt="OOF error histogram"/></p>

---

## Streamlit MVP

The local Streamlit app performs **single-image crowd-count inference**.

```text
Upload image
    ↓
Resize + ImageNet normalization
    ↓
5 trained MobileNetV3-Small models
    ↓
5 log-count predictions
    ↓
expm1
    ↓
median ensemble
    ↓
estimated visible people count
```

The app displays:

- final median estimate;
- prediction from each fold;
- minimum / maximum fold prediction;
- fold spread;
- fold standard deviation;
- OOF MAE / RMSE benchmark;
- downloadable fold prediction CSV.

> **Fold spread is not a confidence percentage.** It only describes disagreement among the five fold models.

---

## What QueueWatch Can Do Now

- single-image input;
- crowd-count regression;
- five-model median ensemble inference;
- per-fold diagnostics;
- local Streamlit demonstration;
- OOF benchmark reporting;
- CSV export.

## What QueueWatch Does Not Do Yet

The following are **not implemented or validated**:

- queue-specific training;
- queue-line segmentation;
- queue length estimation;
- waiting-time prediction;
- person detection;
- person tracking;
- entry / exit counting;
- density-map estimation;
- CCTV video inference;
- real-time queue monitoring;
- service-counter analytics;
- queue abandonment detection.

These belong under future work until they are implemented and evaluated.

---

## Why the Name “QueueWatch”? 

The original direction was queue monitoring, but the final available dataset and trained model support **general crowd counting**, not queue-specific structure modeling.

The correct current description is:

> **A crowd-count regression baseline that can become one component of a future queue-monitoring system.**

---

## Repository Artifacts

### Models

```text
models/
├── queuewatch_fold0_best.pt
├── queuewatch_fold1_best.pt
├── queuewatch_fold2_best.pt
├── queuewatch_fold3_best.pt
└── queuewatch_fold4_best.pt
```

### Reports

```text
reports/
├── cv_metrics.json
├── cv_training_history.csv
├── dataset_audit.json
├── final_summary.json
├── manifest_with_folds.csv
├── manifest.csv
├── oof_predictions.csv
├── submission.csv
└── test_ensemble_predictions.csv
```

### Visual evidence

```text
assets/previews/
├── fold_validation_mae.png
├── oof_error_hist.png
└── oof_scatter.png
```

### Notebook

```text
notebooks/
└── queuewatch.ipynb
```

---

## Reports

- [Cross-validation metrics](reports/cv_metrics.json)
- [Cross-validation training history](reports/cv_training_history.csv)
- [Dataset audit](reports/dataset_audit.json)
- [Final summary](reports/final_summary.json)
- [Manifest](reports/manifest.csv)
- [Manifest with fold assignment](reports/manifest_with_folds.csv)
- [OOF predictions](reports/oof_predictions.csv)
- [Submission](reports/submission.csv)
- [Test ensemble predictions](reports/test_ensemble_predictions.csv)

---

## Running Locally

```bash
git clone https://github.com/MENSTRUE/QueueWatch.git
cd QueueWatch
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Then open the local Streamlit URL and upload a crowd image.

---

## Repository Structure

```text
QueueWatch/
├── app/
│   └── streamlit_app.py
├── assets/
│   ├── previews/
│   ├── samples/
│   └── screenshots/
├── models/
│   ├── queuewatch_fold0_best.pt
│   ├── queuewatch_fold1_best.pt
│   ├── queuewatch_fold2_best.pt
│   ├── queuewatch_fold3_best.pt
│   └── queuewatch_fold4_best.pt
├── notebooks/
│   └── queuewatch.ipynb
├── reports/
├── src/
│   ├── __init__.py
│   ├── ensemble.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocess.py
│   └── utils.py
├── .gitignore
├── AGENTS.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Key Design Decisions

### Why MobileNetV3-Small?

The labeled dataset contains only 100 images. A compact ImageNet-pretrained backbone provides a practical transfer-learning baseline without introducing an unnecessarily large custom architecture.

### Why `log1p(count)`?

The target count range is broad. Modeling the logarithm of the count reduces numerical scale during optimization.

At inference:

```python
count = expm1(log_prediction)
```

### Why 5-fold cross-validation?

With only 100 labeled images, one split can be unstable. Five folds allow every labeled image to contribute to out-of-fold evaluation while producing five trained models for ensemble inference.

### Why median ensemble?

The median is less sensitive to one unusually high or low fold prediction than a simple mean.

---

## Previous Experiments

Later experimental branches were explored but did **not** outperform the v5.1 champion under the tracked comparison.

Explored directions included:

- density-based variants;
- embedding regressors;
- regime-aware models;
- residual correction.

They should be treated as **ablations / experiments**, not as the final MVP model.

---

## Limitations

1. **Very small labeled dataset** — only 100 labeled images.
2. **General crowd data** — not a dedicated queue dataset.
3. **No localization** — one count is produced for the full image.
4. **No density map** — spatial person distribution is not estimated.
5. **No video/tracking** — the current MVP processes still images.
6. **Large count errors remain possible** — OOF MAE is 41.805 people.
7. **Domain-shift risk** — performance may degrade on scenes unlike the training images.
8. **No final local test-set error benchmark** — the 216 test images are used for prediction/submission.
9. **Fold disagreement is not calibrated uncertainty.**
10. **Not production queue analytics.**

---

## Future Work

To turn QueueWatch into a real queue-monitoring system, future work should include:

- queue-specific image/video datasets;
- queue ROI definition;
- person localization;
- tracking;
- queue length estimation;
- waiting-time modeling;
- service-rate estimation;
- temporal occupancy trends;
- calibrated predictive uncertainty;
- external validation on real queue scenes.

Until these are implemented, the correct claim remains **crowd-count regression MVP**.

---

## Tech Stack

- Python
- PyTorch
- TorchVision
- MobileNetV3-Small
- NumPy
- Pandas
- scikit-learn
- Pillow
- Matplotlib
- Streamlit

---

## Reproducibility

The repository includes training notebook(s), five fold checkpoints, fold assignments, OOF predictions, training history, final metrics, test ensemble predictions, submission output, and evaluation plots.

This makes the result auditable rather than presenting only a headline number.

---

## Scope Statement

Use this description:

> **QueueWatch v5.1 is a small-data crowd-count regression MVP using an ImageNet-pretrained MobileNetV3-Small 5-fold ensemble. It estimates the number of visible people in a single image and can serve as an early occupancy component for future queue analytics.**

Do **not** describe the current system as:

- a validated queue monitoring system;
- a queue-length predictor;
- a wait-time predictor;
- a person detector;
- a density-map model;
- a real-time CCTV queue analytics platform;
- a model with “90% accuracy”.

---

## License

This repository is released under the [MIT License](LICENSE).

## Author

**Wafa Bila Syaefurokhman**  
GitHub: [@MENSTRUE](https://github.com/MENSTRUE)

---

<p align="center"><strong>Build → measure → document → improve.</strong></p>
