# SAM2 → MobileNetV2 Pipeline for Display Defect Segmentation & Classification (MSD-US)

## Abstract
This project builds an end-to-end defect understanding pipeline for display images using a two-stage approach: **(1) fine-tuned SAM2** for defect segmentation and **(2) MobileNetV2** for defect type classification (**oil / scratch / stain**). We evaluate both segmentation quality and classification performance, including the full end-to-end pipeline (image → SAM2 mask → crop → classifier). Results show that the classifier achieves perfect accuracy on clean cropped patches, while end-to-end performance (**85.83%** accuracy) is limited primarily by segmentation and crop quality.

---

## Method

### Task Definition
Given an input image from the MSD-US dataset, the goal is to:
1. **Segment** the defect region (binary mask).
2. **Classify** the defect type into one of three classes: **oil**, **scratch**, **stain**.

### Dataset Layout
Expected dataset structure:
```
data/MSD-US/test/
  ground_truth/   # binary masks
  oil/            # images
  scratch/        # images
  stain/          # images
```

### Pipeline Overview
1. **Fine-tuned SAM2 Segmentation**
   - SAM2 is fine-tuned on MSD-US.
   - During inference, the model is prompted using automatically sampled points (gradient-based heuristic).
   - The predicted mask is chosen using SAM’s internal confidence score (SAM-score).

2. **Patch Generation**
   - The predicted defect mask is converted into a bounding box.
   - A defect patch is cropped from the original image (with padding).

3. **MobileNetV2 Classification**
   - A MobileNetV2 classifier is trained on the cropped patches.
   - Final inference is **end-to-end**: SAM2 generates patch → MobileNetV2 predicts defect type.

### Evaluation Protocol
We report:
- **Segmentation metrics:** IoU, Dice, and pixel Precision/Recall/F1.
- **Classification metrics (patch-only):** accuracy + confusion matrix.
- **End-to-end classification metrics:** accuracy + confusion matrix + per-class Precision/Recall/F1.

---

## Results

### 1) Fine-tuned SAM2 Segmentation Performance
**Overall (SAM-score selected mask):**
| Metric | Value |
|---|---:|
| IoU | **0.1339** |
| Dice | **0.1974** |
| Pixel Precision | **0.1738** |
| Pixel Recall | **0.5934** |
| Pixel F1 | **0.1974** |

**Per-class (SAM-score selected mask):**
| Class | IoU | Dice | Pixel Precision | Pixel Recall | Pixel F1 |
|---|---:|---:|---:|---:|---:|
| oil | **0.2178** | **0.2970** | **0.3016** | 0.4655 | **0.2970** |
| scratch | 0.1154 | 0.1793 | 0.1357 | **0.8193** | 0.1793 |
| stain | 0.0684 | 0.1160 | 0.0843 | 0.4955 | 0.1160 |

**Interpretation:**  
- SAM2 shows **high recall** (especially for *scratch*) but **low precision**, indicating it often over-segments or includes non-defect regions.
- This matters because the next stage depends on the crop quality derived from the predicted mask.

---

### 2) MobileNetV2 Classification on Cropped Patches (Upper Bound)
When MobileNetV2 is trained and evaluated on **pre-cropped defect patches**, it achieves:

- **Accuracy: 1.0000**
- Confusion matrix (rows=true, cols=pred):
```
[[400   0   0]
 [  0 400   0]
 [  0   0 400]]
```

Per-class:
- oil: P=1.0000, R=1.0000, F1=1.0000  
- scratch: P=1.0000, R=1.0000, F1=1.0000  
- stain: P=1.0000, R=1.0000, F1=1.0000  

**Interpretation:**  
The classifier can separate the three defect types perfectly **when the input patch is clean and defect-focused**. This is effectively the **upper bound** of the classification stage.

---

### 3) End-to-End Performance (SAM2 → Crop → MobileNetV2)
This is the **real deployment scenario**: segmentation affects the crop, which affects classification.

- **End-to-end Accuracy: 0.8583**
- Confusion matrix (rows=true, cols=pred):
```
[[389  11   0]
 [  0 400   0]
 [ 42 117 241]]
```

Per-class:
| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| oil | 0.9026 | **0.9725** | **0.9362** |
| scratch | 0.7576 | **1.0000** | 0.8621 |
| stain | **1.0000** | 0.6025 | 0.7520 |

Macro averages (equal class sizes):
- Macro Precision ≈ **0.8867**
- Macro Recall ≈ **0.8583**
- Macro F1 ≈ **0.8501**

**Interpretation:**  
- **Scratch recall is perfect (1.0)**, but stain is frequently misclassified as scratch (117 stain → scratch).
- The primary failure mode is likely that SAM2 crops for stain often include patterns that look scratch-like, or crops miss stain regions (low stain recall).
- Since patch-only accuracy is 1.0, the **performance bottleneck is segmentation/cropping**, not the CNN classifier.

---

## Key Takeaways
- ✅ **MobileNetV2 is strong** when given defect-focused crops (100% patch accuracy).
- ⚠️ **End-to-end performance drops** to **85.83%** due to segmentation/cropping errors.
- 🔧 Improving SAM2 segmentation quality (especially precision + stain localization) should yield the biggest end-to-end gains.

---

## Next Steps (Recommended Improvements)
To improve segmentation and end-to-end accuracy:
1. **Improve prompt strategy**
   - Add **negative points** (background prompts) to reduce over-segmentation.
   - Use **box prompts** derived from coarse heuristics if available.
2. **Better mask selection**
   - Instead of choosing the top SAM-score mask, select by **mask stability**, size priors, or ensemble multiple masks.
3. **Post-processing**
   - Remove small components, apply morphological cleanup, enforce plausible defect size/shape constraints.
4. **Training improvements**
   - More training steps / stronger augmentations
   - Loss tweaks (Dice/Focal) to reduce false positives
5. **Joint optimization**
   - Train the crop/classification step to be robust to imperfect masks (augment crops, random jitter, mask noise).
