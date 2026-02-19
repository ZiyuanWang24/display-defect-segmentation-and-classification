# Fine-tuning SAM2 for Display Defect Segmentation (Scratch / Oil / Stain)

This repo fine-tunes **SAM2** (Segment Anything Model 2) to segment *display surface defects*—**scratch**, **oil**, and **stain**—from RGB inspection images.  
The goal is to improve segmentation quality on small / low-contrast defects where zero-shot SAM2 can struggle.

---

<<<<<<< HEAD
## Report structure (what you’ll typically see in SAM fine-tuning reports)

Across popular SAM/SAM2 fine-tuning writeups and repos, the common structure is:

1. **Problem / Motivation** (why zero-shot SAM fails on the target domain)  
2. **Dataset** (format, split, labeling, examples)  
3. **Method**  
   - model variant (SAM / SAM2 size)  
   - prompt strategy (points / boxes / masks)  
   - what is frozen vs trained (encoder / prompt encoder / decoder)  
   - loss functions and selection rules  
4. **Training setup** (hardware, hyperparameters, reproducibility)  
5. **Evaluation**  
   - metrics (IoU, Dice, precision, recall)  
   - baseline comparison (zero-shot vs fine-tuned)  
   - qualitative visuals  
6. **Results** (tables + figures + failure cases)  
7. **Deployment notes** (how inference is done; latency/throughput if measured)  
8. **Limitations & future work**  
9. **References / citation**

This README follows that template.

=======
>>>>>>> 5259cfd20509f35f733d28c628f48563ac6c4d1d
---

## 1) Problem & Motivation

Display defect segmentation is challenging because defects can be:
- **thin** (scratches),
- **semi-transparent / reflective** (oil films),
- **diffuse / low contrast** (stains),
- and often occupy a **tiny fraction** of pixels.

While SAM2 is strong zero-shot, domain shift + small-object bias can reduce mask quality. Fine-tuning adapts the mask decoder (and optionally prompt encoder) to this specific defect domain.

---

## 2) Dataset

This training script expects **image + binary ground-truth mask** pairs.

Key CLI args:
- `--data_dir` (default: `data/MSD-US/test`)
- `--dataset_format` in `{auto,csv,msd_us}`
- `--gt_dirname` (default: `ground_truth`)
- `--categories` (optional: `oil,scratch,stain`)

### Prompts derived from GT
A **single positive point** is generated per sample using the **center-of-mass of GT foreground** (deterministic).

This makes training/evaluation reproducible and matches a realistic “one-click” prompt at deployment.

---

## 3) Method

### 3.1 Model
- Base: SAM2 checkpoint (default: `sam2_hiera_tiny.pt`)
- Config inferred from checkpoint name (tiny/small/base/large)

### 3.2 What is trained vs frozen (parameter-efficient fine-tuning)
Default behavior in the script:
- ✅ Train **prompt encoder**
- ✅ Train **mask decoder**
- ❄️ Freeze **image encoder**

You can change this via flags:
- `--no_train_prompt_encoder`
- `--no_train_mask_decoder`
- `--no_freeze_image_encoder`

### 3.3 Prompting strategy
- One **positive point** per GT mask (no boxes, no negative points).
- Inference uses that point prompt and requests **3 candidate masks** from SAM2 (`multimask_output=True`).

### 3.4 Key training innovations in this repo (from `train.py`)

#### (A) **Oracle best-of-3 selection during training**
SAM2 outputs 3 candidate masks per prompt. During training, you compute IoU against GT for all 3 and select:

> **k_best = argmax IoU(GT, mask_k)**

Then you backprop **only through the best candidate**.

Why it matters for scratch/oil/stain:
- Defects are subtle; the “best” candidate may *not* be the one with the highest SAM score early in training.
- Oracle selection provides a **cleaner learning signal** (stronger supervision) for small objects.

#### (B) **Segmentation loss = BCEWithLogits + Soft Dice**
You train the selected mask with:

- `seg_loss = 0.5 * BCEWithLogits(best_logits, GT) + 0.5 * SoftDice(sigmoid(best_logits), GT)`

Why it matters:
- Dice helps with **class imbalance** (defects are small).
- BCE stabilizes pixel-wise learning.

#### (C) **Score calibration aligned with deploy-time selection**
At deployment you cannot use GT to pick the best mask. You must rely on SAM’s predicted per-mask score.

So the training also calibrates the score head:

- Compute **soft IoU** between predicted probability mask and GT (`iou_soft`)
- Add a loss term: `|pred_score(best_k) - iou_soft|`

This encourages SAM’s score to correlate with true overlap, improving *deployable* mask selection.

#### (D) Balanced evaluation split (stratified by defect type)
The code tries to infer class labels (scratch/oil/stain) from metadata or file paths and performs a **stratified train/eval split**.  
This avoids accidentally validating mostly on one defect type.

### 3.5 Validation during training
Every `--eval_every` steps:
- run deterministic subset evaluation on `eval_data`
- compute `val_mean_iou`
- save `fine_tuned_sam2_best.torch` when improved
- optional early stopping via `--early_stop_patience`

---

## 4) Training setup

### 4.1 Install
Create your environment (example):

```bash
conda create -n sam2_defects python=3.11 -y
conda activate sam2_defects
pip install -r requirements.txt
<<<<<<< HEAD
```
=======
```![Alt text for the image](path/to/your/image.png)

>>>>>>> 5259cfd20509f35f733d28c628f48563ac6c4d1d

### 4.2 Train
Example command:

```bash
python train.py \
  --data_dir data/MSD-US/test \
  --dataset_format msd_us \
  --categories scratch,oil,stain \
  --steps 10000 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --max_side 512 \
  --eval_every 200 \
  --eval_samples 200 \
  --save_every 500 \
  --out_dir checkpoints \
  --use_best_for_test
```

Notes:
- Uses AMP (`autocast`) + `GradScaler` on CUDA
- Uses gradient accumulation via `--accumulation_steps` (default 4)

---

## 5) Evaluation

Evaluation is integrated at the end of `train.py` and compares:

- **Zero-shot baseline** (original SAM2 weights)
- **Fine-tuned** (best-val checkpoint if `--use_best_for_test`)

### 5.1 Mask selection rule at inference (deployable)
During evaluation (and intended deployment), you select:

> **mask = argmax predicted_score(mask_k)**

This is critical: it matches what you can do without GT.

### 5.2 Metrics
Per class (scratch/oil/stain), computed from aggregated TP/FP/FN counts:
- IoU
- Dice
- Precision
- Recall

### 5.3 Artifacts produced
Saved to `--out_dir` (default `checkpoints/`):

- `test_metrics_per_class.csv`  
- `test_visual_grid.png` (Input | GT | Zero-shot | Fine-tuned)  
- Console prints include a latency paragraph (p50 / p95)

---

## 6) Results

> Replace the table below with the values from `checkpoints/test_metrics_per_class.csv`.

| Class | n | IoU (Zero-shot) | IoU (Fine-tuned) | Δ IoU | Dice (Zero-shot) | Dice (Fine-tuned) |
|------:|--:|----------------:|-----------------:|------:|-----------------:|------------------:|
<<<<<<< HEAD
| scratch |  |  |  |  |  |  |
| oil     |  |  |  |  |  |  |
| stain   |  |  |  |  |  |  |

### Qualitative comparison

Add the generated grid image here:

```md
![Zero-shot vs Fine-tuned](checkpoints/test_visual_grid.png)
```

### Latency / throughput

The script prints an on-machine benchmark similar to:

- p50 latency: **X ms**
- p95 latency: **Y ms**
- throughput: **~Z FPS (p50)**

Copy/paste that paragraph into this section after running.

=======
|scratch | 80 | 0.0052 | 0.0103 | 0.0052 | 0.5489 | 0.0286 | 0.0557 | 0.0286 | 0.9700 | 0.0235
|oil | 80 | 0.2734 | 0.4294 | 0.2856 | 0.8645 | 0.6532 | 0.7902 | 0.6656 | 0.9723 | 0.3798
|stain | 80 | 0.0008 | 0.0017 | 0.0008 | 0.5339 | 0.0024 | 0.0049 | 0.0024 | 0.9421 | 0.0016


### Qualitative comparison

```md
![Zero-shot vs Fine-tuned](checkpoints/test_visual_grid.png)

```

>>>>>>> 5259cfd20509f35f733d28c628f48563ac6c4d1d
---

## 7) Deployment notes

Target deployment: an automated display-inspection pipeline on a tester/inspection workstation.

Recommended inference flow:
1. Resize input (longest side capped, same as training).
2. Provide one prompt point (human click or heuristic center point).
3. Run SAM2, request 3 candidates, select by **highest predicted score**.
4. Threshold at 0.5 to get a binary mask.

If you need higher recall, lower the threshold; if you need fewer false positives, increase threshold.

---

## 8) Limitations & Future work

- **Only positive points** are used. Adding **negative points** or **box prompts** may reduce false positives.
- **Single-instance assumption** per image (one prompt point). Multi-defect images may require multi-point prompting.
- Consider lightweight **data augmentation** (contrast/blur/noise) to improve robustness across inspection conditions.
- Consider unfreezing (or LoRA/adapter on) parts of the **image encoder** if domain shift is large.

<<<<<<< HEAD
---

## 9) References

- SAM2: Segment Anything in Images and Videos (arXiv:2408.00714)  
- Popular SAM fine-tuning templates and repos:
  - mazurowski-lab/finetune-SAM (comprehensive empirical study + code)
  - WholeNow/finestSAM (configuration-driven fine-tuning framework)
  - MathieuNlp/Sam_LoRA (report-style README with baseline vs fine-tuned + visuals)
  - Roboflow & LearnOpenCV tutorials for SAM2 fine-tuning workflows

---

## 10) Citation

If you use SAM/SAM2 in academic work, cite the original Segment Anything / SAM2 papers (per their official BibTeX).
=======

>>>>>>> 5259cfd20509f35f733d28c628f48563ac6c4d1d
