# MobileNetV2 Three-Class Defect Classifier (Mask-Guided Cropping)

## Abstract
This report documents a lightweight image classifier based on **MobileNetV2** for **three defect classes** (`oil`, `scr`, `sta`) using a paired **image + binary mask** dataset (`ground_truth_seperate`). The core idea is to **use the provided segmentation mask to crop each sample to the defect region (ROI)** before classification, reducing background clutter and improving feature focus. We provide a reproducible training/evaluation pipeline, saved artifacts (splits, logs, metrics, confusion matrix, checkpoint), and optional inference latency benchmarking.

---

## 1. Introduction
Lightweight CNN backbones are widely used when you care about **fast inference** and **small model size** (edge devices, production pipelines, quick iteration). MobileNetV2 introduced the *inverted residual + linear bottleneck* block to improve accuracy–efficiency trade-offs, and is a common baseline for real-world classification systems.

In this project, MobileNetV2 is adapted to **defect-type classification** on separated defect instances. The training pipeline emphasizes:
- **Reproducibility** (deterministic split via seed, persisted CSV manifests)
- **Better signal-to-noise** via **mask-guided cropping**
- **Balanced evaluation** (macro-F1 for model selection; per-class metrics + confusion matrix)

---

## 2. Method

### 2.1 Model: MobileNetV2 fine-tuning
We use TorchVision’s `mobilenet_v2` with optional ImageNet pretraining, replacing the final classifier layer for 3 classes:

- Load backbone with `weights=MobileNet_V2_Weights.DEFAULT`
- Replace `model.classifier[1]` with a new `Linear(in_features, 3)`
- Train end-to-end with cross-entropy loss

> **Why MobileNetV2?**  
> It is designed for efficient inference via depthwise separable convolutions and inverted residual bottlenecks, making it a strong lightweight baseline.

### 2.2 Dataset: paired separated images + masks
Expected structure:

```
data/ground_truth_seperate/
  images/<group_name>/*.png|jpg|...
  masks/<group_name>/*.png
```

**Pairing rule**
- For each mask at `masks/<relpath>`, find image at `images/<relpath>`
- If extensions differ, try common image extensions automatically

**Label rule**
- Infer class by searching tokens in the relative path (folder name, filename, etc.)
- Must match exactly one of: `oil`, `scr`, `sta` (samples with no match are skipped)

**Mask sanity checks**
- Convert mask to binary (mask>0)
- Skip samples where mask area `< min_mask_area`

### 2.3 Mask-guided ROI cropping (key idea)
If enabled (default), each sample is cropped to the **mask bounding box** with padding:

- Compute bbox from nonzero mask pixels
- Add padding proportional to bbox size (`pad_ratio`)
- Crop the RGB image to this padded bbox
- Resize to `img_size × img_size` (default 224)

This is a practical hybrid approach: segmentation provides localization; classification provides defect category.

### 2.4 Train/val/test split
A custom stratified split ensures per-class sampling:

- `eval_split` (default 0.2), `test_split` (default 0.1)
- Per-class shuffling using a seeded `torch.Generator`
- Attempts to keep at least one training example per class (when possible)

All splits are exported as CSV manifests for traceability.

### 2.5 Training details
Default training recipe:
- Optimizer: **AdamW**
- Loss: **CrossEntropyLoss**
- Epochs: 10
- Batch size: 32
- LR: 3e-4, weight decay: 1e-4
- Input normalization: ImageNet mean/std

**Model selection**
- Best checkpoint is selected by **evaluation macro-F1**.

### 2.6 Evaluation metrics
We compute:
- Accuracy
- Macro precision/recall/F1 (class-balanced)
- Weighted precision/recall/F1 (support-weighted)
- Per-class precision/recall/F1 + support
- Confusion matrix (CSV; optional PNG)

---

## 3. Experiments

### 3.1 How to run

**Train + evaluate**
```bash
python train_cnn.py \
  --data_dir data/ground_truth_seperate \
  --out_dir cnn_checkpoints \
  --epochs 10 \
  --batch_size 32 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --eval_split 0.2 \
  --test_split 0.1 \
  --img_size 224 \
  --device cuda \
  --seed 42
```

**Disable cropping**
```bash
python train_cnn.py --data_dir data/ground_truth_seperate --out_dir cnn_checkpoints --no_crop_by_mask
```

**Plot confusion matrix**
```bash
python train_cnn.py --data_dir data/ground_truth_seperate --out_dir cnn_checkpoints --plot_cm
```

**Benchmark inference latency (optional)**
```bash
python train_cnn.py --data_dir data/ground_truth_seperate --out_dir cnn_checkpoints --benchmark
```

### 3.2 Output artifacts
The script writes reproducibility artifacts to `out_dir/`:

- `splits_train.csv`, `splits_eval.csv`, `splits_test.csv`  
- `training_log.csv` (epoch, train_loss, eval_acc, eval_macro_f1)  
- `mobilenetv2_best.pt` (best checkpoint by eval macro-F1)  
- `confusion_matrix.csv` (+ `confusion_matrix.png` if `--plot_cm`)  
- `test_metrics.json`, `test_metrics.csv`  
- `summary_report.json` (run config, split stats, params, metrics, optional latency)

### 3.3 Results (fill from your run)
After running, copy your final metrics here:

- **Test accuracy**: 0.9968
- **Test macro-F1**: 0.9956
- **Per-class**:
  - oil: P=1.0, R=0.98, F1=0.99, support=56
  - scr: P=1.0, R=1.0, F1=1.0, support=140
  - sta: P=0.99, R=1.0, F1=0.99, support=115

Attach (or link) these files in your GitHub repo:
- `cnn_checkpoints/test_metrics.json`
- `cnn_checkpoints/confusion_matrix.png` (if enabled)
- `cnn_checkpoints/training_log.csv`

---

## 5. Key Implementation Points
The code contains several practical design choices that make the pipeline robust and easier to reproduce:

1. **Mask-to-image pairing by relative path + extension fallback**  
   Enables consistent dataset assembly even when image extensions vary.

2. **Label inference from any path component**  
   Class is inferred from folder names or filenames, avoiding extra annotation files.

3. **Mask-area filtering** (`min_mask_area`)  
   Removes nearly-empty masks that can destabilize ROI cropping and training.

4. **Mask-guided ROI cropping + padding**  
   Uses segmentation masks to focus the classifier on the defect region while preserving some context via `pad_ratio`.

5. **Custom stratified splitting with safeguards**  
   Prevents accidental class drop in the train split (when possible) and writes split manifests for traceability.

6. **Macro-F1–based best checkpoint**  
   Macro-F1 is better aligned with imbalanced multi-class performance than accuracy alone.

7. **Full artifact logging**  
   CSV logs + JSON summaries make it easy to generate plots, audit results, and compare runs.

---

## 7. Conclusion
This project provides a clean, reproducible baseline for three-class defect classification using MobileNetV2. The main practical contribution is **leveraging segmentation masks for ROI-focused classification**, along with robust dataset pairing, stratified splits, macro-F1 model selection, and comprehensive artifact export for GitHub reporting.

---