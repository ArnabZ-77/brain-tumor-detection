# Brain Tumor Detection: Domain-Robust, Uncertainty-Aware MRI Classifier

A PyTorch implementation of the framework proposed in *Brain Tumor Detection Using Deep Learning* (literature review, problem statement and proposed solution, 2024–2026).

Most published models report 99%+ accuracy on a single curated dataset but degrade on data from other scanners and hospitals. This project does not try to beat the accuracy number. It targets three things:

1. **Robustness to unseen imaging sources.** A hybrid CNN–Transformer (MobileViT-V2) is trained with acquisition-aware augmentation that simulates scanner differences.
2. **Calibrated uncertainty and clinical triage.** Monte Carlo dropout plus temperature scaling. Low-confidence cases are referred to a radiologist, and the accuracy-vs-coverage curve is reported.
3. **Measurable explanations.** Grad-CAM++ heatmaps are scored (IoU / Dice) against ground-truth tumor masks.

The headline metric is the **generalization gap Δ = internal accuracy − external accuracy**.

> **Status:** the code is complete but has not yet been trained or evaluated. No results are reported here. Run the steps below to produce them.

## Repository layout

| File | Purpose |
|---|---|
| `data.py` | Dataset scanning, stratified 70/15/15 split, preprocessing (z-score), `MRIDataset` |
| `augment.py` | Acquisition-aware augmentation (gamma, bias field, noise, low-res, k-space motion, spike/ghosting) and geometric augmentation |
| `model.py` | `BrainTumorNet` (any timm backbone + dropout head) and MC-dropout inference (`predict_loader`) |
| `train.py` | Training with AdamW, label smoothing, cosine schedule, EMA, early stopping, then temperature and triage-threshold fitting |
| `calibration.py` | Temperature scaling, ECE, accuracy-vs-coverage, triage rule |
| `evaluate.py` | Internal and external evaluation; writes metrics, CSV and plot |
| `explain.py` | Grad-CAM++ and explanation IoU/Dice |
| `requirements.txt` | Python dependencies |

## 1. Setup

Python 3.9+ is recommended. A CUDA GPU is strongly recommended for training (CPU works but is slow).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

If you have a GPU, install the CUDA build of PyTorch first from <https://pytorch.org/get-started/locally/>, then run the `pip install` above.

The first run downloads ImageNet-pretrained weights through `timm`, so internet access is needed once.

## 2. Datasets

You need **two separate datasets**. Download them yourself and extract them anywhere.

| Role | Dataset | Notes |
|---|---|---|
| Train / validation / internal test | Kaggle "Brain Tumor MRI Dataset" (Masoud Nickparvar), ~7,023 images | Folders must be named `glioma`, `meningioma`, `notumor`, `pituitary`. The `Training/` and `Testing/` folders are merged and re-split by the code. |
| External test (never used for tuning) | Br35H (3,000 images, tumor / no tumor), or BRISC2025 | Br35H: folders `yes` / `no`. BRISC2025: 4-class folders with the same names as above. |

Example layout:

```
data/
  kaggle/Training/glioma/*.jpg ...      (and Testing/...)
  br35h/yes/*.jpg
  br35h/no/*.jpg
```

**Important:** the external dataset must never be used for training, model selection, early stopping or threshold setting. `train.py` never touches it, and `evaluate.py` should be run on it only once per final model.

## 3. Train the proposed model

```bash
python train.py --data data/kaggle --out runs/proposed
```

Useful options: `--arch` (default `mobilevitv2_100`; `swin_tiny_patch4_window7_224` also works), `--epochs`, `--bs`, `--lr`, `--coverage` (target auto-decision fraction for the triage threshold, default 0.90), `--workers`.

This writes to `runs/proposed/`:
- `model.pt`: best EMA weights, fitted temperature and triage threshold τ
- `split.json`: the exact train/val/test split, so evaluation reuses it
- `train_summary.json`

## 4. Train the baselines (for the comparison table)

```bash
python train.py --data data/kaggle --out runs/resnet50 --arch resnet50 --no-acq
python train.py --data data/kaggle --out runs/vit_b16  --arch vit_base_patch16_224 --no-acq
python train.py --data data/kaggle --out runs/mobilevit_plain --arch mobilevitv2_100 --no-acq
```

`--no-acq` disables acquisition-aware augmentation so these represent conventional training.

## 5. Evaluate: internal vs. external

```bash
# proposed model: calibrated, 25 MC-dropout passes, flip TTA
python evaluate.py --ckpt runs/proposed --external data/br35h

# baselines: single deterministic pass, no calibration
python evaluate.py --ckpt runs/resnet50 --external data/br35h --mc 1 --no-tta --no-calibration
python evaluate.py --ckpt runs/vit_b16  --external data/br35h --mc 1 --no-tta --no-calibration
python evaluate.py --ckpt runs/mobilevit_plain --external data/br35h --mc 1 --no-tta --no-calibration
```

Each run writes to its own folder:
- `eval.json`: internal and external accuracy, **Δ**, per-class recall (including glioma), ECE, triage result at τ, parameter count, ms per image
- `accuracy_coverage_external.csv` and `accuracy_coverage.png`: the triage curve

Br35H is binary, so the 4-class output is collapsed to tumor / no tumor for the external set, and Δ is computed against the internal test set collapsed the same way. ECE is only reported for 4-class external data.

Fill the comparison table from the `eval.json` files:

| Model | Internal acc. | External acc. | Δ (gap) | Glioma recall | ECE |
|---|---|---|---|---|---|
| ResNet50 | | | | | |
| ViT-B/16 | | | | | |
| MobileViT (plain) | | | | | |
| Proposed | | | | | |

It is acceptable for the proposed model to lose a little internal accuracy if external accuracy improves and Δ shrinks.

## 6. Explanations

```bash
python explain.py --ckpt runs/proposed --image path/to/scan.jpg --mask path/to/tumor_mask.png --out gradcam.png
```

Saves a Grad-CAM++ overlay and prints the IoU / Dice of the thresholded heatmap against the mask. The mask is optional. Run it over a dataset that ships tumor masks (e.g. Figshare or BraTS slices) to get an average explanation score.

## Method summary

- **Backbone:** MobileViT-V2, ImageNet-pretrained, with a small dropout MLP head.
- **Training:** AdamW (lr 3e-5, wd 1e-4), label smoothing 0.1, cosine schedule, EMA of weights, early stopping on validation loss.
- **Augmentation:** flips and affine transforms, plus random gamma, bias field, Gaussian noise, resolution degradation, k-space motion artifact and spike/ghosting.
- **Uncertainty:** 25 MC-dropout passes. The mean is the prediction and predictive entropy is the uncertainty score.
- **Calibration:** a temperature fitted on the validation set.
- **Triage:** cases with entropy ≥ τ are referred to a radiologist. τ is set on the validation set for a target coverage and never tuned on the external set.

## Known limitations

- Skull stripping is not implemented, because the public JPG datasets are already cropped. N4 bias correction is available only as the optional `preprocess_n4()` in `data.py` and needs `pip install SimpleITK`.
- Attention Rollout is not implemented. Only Grad-CAM++ is provided, and it works with backbones that output 4D feature maps (MobileViT-V2, ResNet).
- Dropout is only in the classifier head, so MC dropout reflects head uncertainty only. A deep ensemble of 3–5 models would capture more.
- Public datasets can contain near-duplicate images across splits, which inflates internal accuracy. This is one reason the external test matters.
- The code has been syntax-checked but not yet run end to end.
- **Research use only. Not a medical device and not for clinical decisions.**

## Reference

Based on the report *Brain Tumor Detection Using Deep Learning: Literature Review, Problem Statement, Proposed Solution and Scope for Improvement* (28 October 2026). All performance figures in that report are as stated by the cited authors.
