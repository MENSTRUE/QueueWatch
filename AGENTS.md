# AGENTS.md — QueueWatch v5.1

## Purpose

This file defines the engineering, reproducibility, and claim rules for AI coding agents working on **QueueWatch v5.1**.

QueueWatch is currently a **small-data crowd-count regression MVP**.

The repository name suggests queue monitoring, but the available dataset and validated final model support **general crowd counting from still images**, not queue-structure analysis.

All future work must preserve that distinction unless queue-specific capabilities are actually implemented and evaluated.

---

## Project Identity

**Project:** QueueWatch v5.1  
**Portfolio milestone:** 9%  
**Domain:** Computer Vision / Regression  
**Current task:** visible-person count estimation from one image  
**Backbone:** MobileNetV3-Small  
**Pretraining:** ImageNet  
**Training target:** `log1p(count)`  
**Validation:** 5-fold cross-validation  
**Final inference:** median ensemble across five fold models  
**Application:** Streamlit

---

## Current Scope

QueueWatch v5.1 currently does:

```text
image
→ resize / normalize
→ 5 MobileNetV3-Small regressors
→ log-count predictions
→ expm1
→ median ensemble
→ estimated visible-person count
```

It does **not** currently do:

```text
queue detection
queue ordering
queue length estimation
waiting-time prediction
person detection
person tracking
density maps
video analytics
service-counter analytics
real-time queue monitoring
```

Do not claim unsupported capability.

---

## Canonical Final Model

The final champion is **v5.1**.

```text
MobileNetV3-Small
    ↓
preserve classifier trunk
Linear(576 → 1024)
    ↓
replace final classifier with
Linear(1024 → 1)
```

Important:
- ImageNet-pretrained during training.
- Final regression output predicts `log1p(count)`.
- Inference converts back with `expm1`.
- Final application uses the five v5.1 fold models and median ensemble.

Do not replace the final model with later experiments unless a new experiment clearly beats v5.1 under a comparable evaluation protocol.

---

## Canonical v5.1 Metrics

### Fold MAE

```text
Fold 1: 49.726
Fold 2: 43.898
Fold 3: 33.129
Fold 4: 35.054
Fold 5: 47.221
```

### OOF metrics

```text
OOF MAE  = 41.8053436
OOF RMSE = 65.5135
```

### Naive baseline

```text
Naive MAE  = 74.260
Naive RMSE = 111.624
```

### Improvement

```text
Absolute MAE improvement = 32.455
Relative MAE reduction   ≈ 43.7%
```

These are **regression metrics**.

Never call them accuracy, confidence, detection precision, or classification score.

---

## Dataset Contract

Final dataset facts:

```text
100 labeled images
216 test images
count min    = 15
count median = 89.5
count mean   = 126.36
count max    = 539
```

Dataset mount:

```text
/kaggle/input/datasets/wafabila/ioai-computer-vision
```

Important source files:

```text
labeled/
test/
ground_truth_train.json
sample_submission.csv
```

The dataset is **general crowd-count data**, not a dedicated queue dataset.

Do not silently reframe it as queue-specific.

---

## Cross-Validation Rule

The final v5.1 experiment uses **5-fold cross-validation** on all 100 labeled images.

Out-of-fold predictions are the basis of the final OOF MAE / RMSE.

Do not report metrics from one arbitrary holdout split, the unlabeled 216 test images, a local Streamlit image, or later ablations as the canonical v5.1 benchmark.

---

## Inference Contract

### Image preprocessing

```text
RGB
→ resize to 320 × 320
→ tensor
→ ImageNet normalization
```

ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Do not apply training-only augmentation at inference.

### Fold inference

For each fold:

```text
raw model output
→ interpreted as log1p(count)
→ expm1
→ clamp below zero to zero
```

Final result:

```text
median(
    fold0_count,
    fold1_count,
    fold2_count,
    fold3_count,
    fold4_count
)
```

Required model files:

```text
models/queuewatch_fold0_best.pt
models/queuewatch_fold1_best.pt
models/queuewatch_fold2_best.pt
models/queuewatch_fold3_best.pt
models/queuewatch_fold4_best.pt
```

---

## Ensemble Diagnostics

Allowed diagnostics:
- fold predictions;
- minimum fold prediction;
- maximum fold prediction;
- fold spread;
- fold standard deviation;
- median;
- mean.

Do not label any of these as a confidence percentage.

Correct language:

> Fold spread measures disagreement between the five CV models.

No calibrated predictive uncertainty model has been validated.

---

## Queue-Specific Claim Rules

Allowed wording:

> Crowd-count regression MVP that can serve as an early occupancy component for future queue analytics.

Not allowed unless later implemented and evaluated:

- AI queue monitoring system;
- queue length detector;
- wait-time predictor;
- real-time queue management;
- queue congestion detector;
- queue analytics platform.

---

## Repository Responsibilities

### `app/`

Contains Streamlit UI code only.

The app should upload one image, preview it, run the five-fold ensemble, display final count and fold diagnostics, show canonical OOF metrics, and allow CSV export.

### `src/model.py`

Owns the exact MobileNetV3-Small regression architecture.

### `src/preprocess.py`

Owns inference preprocessing.

### `src/ensemble.py`

Owns checkpoint loading, per-fold prediction, `expm1`, median ensembling, and fold diagnostics.

### `src/inference.py`

Owns the high-level image prediction interface.

### `src/utils.py`

Owns small formatting helpers.

### `notebooks/`

Contains the training and evaluation pipeline.

### `reports/`

Contains reproducibility evidence.

### `assets/previews/`

Contains evaluation plots used by README / portfolio presentation.

---

## Repository Evidence

Expected reports:

```text
reports/cv_metrics.json
reports/cv_training_history.csv
reports/dataset_audit.json
reports/final_summary.json
reports/manifest_with_folds.csv
reports/manifest.csv
reports/oof_predictions.csv
reports/submission.csv
reports/test_ensemble_predictions.csv
```

Expected plots:

```text
assets/previews/fold_validation_mae.png
assets/previews/oof_error_hist.png
assets/previews/oof_scatter.png
```

Do not invent missing figures, missing CSV columns, external benchmark results, extra test labels, unmeasured latency, or production FPS.

---

## Later Experiments

Experiments after v5.1 were explored but did not beat the v5.1 champion in the tracked comparison.

Explored directions include density-based variants, embedding regression, regime-aware models, and residual correction.

Treat them as **ablations / experiments**, not the final deployed model.

---

## Engineering Rules

1. Keep preprocessing identical between training and inference.
2. Keep input size at 320 × 320 unless retraining.
3. Keep ImageNet normalization unchanged unless retraining.
4. Preserve `log1p` target / `expm1` inverse transform.
5. Use all five v5.1 fold models for final ensemble inference.
6. Use median ensemble unless a new experiment is properly evaluated.
7. Keep OOF metrics separate from demo-image output.
8. Do not create fake confidence scores.
9. Do not claim queue-specific capability from general crowd data.
10. Do not change architecture and reuse old checkpoints.
11. Do not silently compare incompatible evaluation protocols.
12. Do not report 216 test-image predictions as benchmark error without ground truth.
13. Keep README aligned with actual repository artifacts.
14. Keep large raw datasets out of Git.
15. Preserve honest limitation language.

---

## Evaluation Rules

For regression experiments, report at minimum:
- MAE;
- RMSE;
- split/CV protocol;
- baseline comparison when available.

Useful additional analysis:
- per-fold MAE;
- residual histogram;
- prediction-vs-target scatter;
- count-range breakdown.

Do not force classification metrics such as accuracy, precision, recall, or F1 onto this regression task.

---

## Streamlit Rules

The app must not say:

```text
Confidence: 90%
Accuracy: 95%
Queue length: 30 meters
Estimated wait: 12 minutes
```

unless those quantities have actually been modeled and validated.

The app may say:

```text
Estimated visible people: 87
Fold range: 79–96
Fold spread: 17
```

with a note that fold disagreement is not calibrated uncertainty.

---

## Code Change Checklist

```text
[ ] all five fold checkpoints exist
[ ] model architecture matches training
[ ] final layer is Linear(1024, 1)
[ ] image input is RGB
[ ] input resize is 320 × 320
[ ] ImageNet normalization is unchanged
[ ] fold outputs are interpreted as log1p(count)
[ ] expm1 is applied
[ ] negative counts are clamped to zero
[ ] final ensemble is median
[ ] Streamlit starts without error
[ ] sample image produces five fold predictions
[ ] README claims remain accurate
[ ] no unsupported queue-specific claim was introduced
```

---

## Git Rules

Preferred commits:

```text
feat: add five-fold crowd count inference
feat: add Streamlit crowd count demo
fix: align inference preprocessing with v5.1 training
docs: document final v5.1 benchmark
refactor: separate ensemble inference from UI
```

Avoid empty streak commits, fake progress commits, and force pushes unless explicitly approved.

---

## Scientific Honesty

Do not:
- call MAE accuracy;
- hide the 100-image labeled dataset size;
- imply the 216 test images have local ground-truth evaluation;
- call fold spread confidence;
- call crowd counting queue analytics;
- claim the model detects individual people;
- claim density-map estimation;
- fabricate external validation;
- fabricate production performance.

If evidence does not support a statement, write:

> **Not implemented / not validated yet.**

---

## Future QueueWatch Direction

A true queue-monitoring version may later add:
- queue-specific datasets;
- queue ROI;
- person localization;
- tracking;
- queue order;
- queue length;
- waiting time;
- service throughput;
- temporal video inference.

Until implemented, keep these under **Future Work** only.

---

## Final Agent Principle

**Name does not define capability. Evidence does.**

A reviewer should be able to trace:

```text
labeled images
→ preprocessing
→ 5-fold training
→ OOF predictions
→ MAE / RMSE
→ five trained models
→ median ensemble
→ Streamlit inference
```

without encountering claims that the current dataset or model does not support.
