#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_sam2_mnv2.py

End-to-end defect pipeline:
  (1) Fine-tuned SAM2 segmentation
  (2) MobileNetV2 classification on the segmented/cropped instance
  (3) Report averaged metrics (NOT per-image spam), save CSVs + example grids
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from PIL import Image

# ----------------------------
# constants / helpers
# ----------------------------
CANONICAL_CLASSES = ("oil", "scratch", "stain")
# include more extensions (mask can be jpg)
IMG_EXTS = tuple(sorted(set([e.lower() for e in IMG_EXTENSIONS] + [".bmp", ".tif", ".tiff"])))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_class_name(s: str) -> Optional[str]:
    """
    Normalize a token to canonical class names: oil/scratch/stain
    Accepts abbreviations: scr->scratch, sta->stain
    """
    t = str(s).strip().lower()
    if t in ("oil",):
        return "oil"
    if t in ("scratch", "scr"):
        return "scratch"
    if t in ("stain", "sta"):
        return "stain"
    return None


def infer_class_from_path(path: Path) -> Optional[str]:
    """
    Infer canonical class from any part of a relative path.
    Handles:
      oil_0001.png, Oil_0001__inst000.jpg
      scr_0363.png, Scr_0363__inst000.jpg
      sta_0067.png, Sta_0067__inst000.jpg
    """
    s = str(path).lower()

    # Prefer explicit long tokens if present
    if "scratch" in s:
        return "scratch"
    if "stain" in s:
        return "stain"
    if "oil" in s:
        return "oil"

    # Support abbreviations as "path tokens"
    # e.g. "/scr_0363.png/" or "scr_0363"
    # Use regex to avoid "descr" false matches
    if re.search(r"(^|[/_\\-])scr([/_\\-]|[0-9])", s):
        return "scratch"
    if re.search(r"(^|[/_\\-])sta([/_\\-]|[0-9])", s):
        return "stain"

    return None


def find_matching_image(images_root: Path, rel_mask_path: Path) -> Optional[Path]:
    """
    Given rel path under masks/, find corresponding image under images/.
    If exact path doesn't exist (ext mismatch), try common image extensions.
    """
    cand = images_root / rel_mask_path
    if cand.exists():
        return cand

    # try different extensions
    stem = cand.with_suffix("")  # remove suffix
    for ext in IMG_EXTS:
        c = Path(str(stem) + ext)
        if c.exists():
            return c
    return None


def read_rgb(path: Path) -> Optional[np.ndarray]:
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return bgr[..., ::-1]  # RGB


def read_mask01(path: Path) -> Optional[np.ndarray]:
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)


def mask_bbox(mask01: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def crop_with_pad(img_rgb: np.ndarray, bbox: Tuple[int, int, int, int], pad_ratio: float = 0.25) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    pad = int(round(pad_ratio * max(bw, bh)))

    x0p = max(0, x0 - pad)
    y0p = max(0, y0 - pad)
    x1p = min(w - 1, x1 + pad)
    y1p = min(h - 1, y1 + pad)
    return img_rgb[y0p : y1p + 1, x0p : x1p + 1]


def choose_prompt_point_oracle_dt(mask01: np.ndarray) -> Optional[np.ndarray]:
    """
    Robust oracle prompt: pick the pixel with max distance-to-boundary (distance transform).
    Works well for thin/disconnected shapes (always inside foreground if any).
    Returns point_coords shape (1,2) in (x,y) float32.
    """
    import cv2

    if mask01 is None or mask01.ndim != 2:
        return None
    if int(mask01.sum()) == 0:
        return None

    # distanceTransform expects 8-bit single channel with non-zero as foreground
    m = (mask01 > 0).astype(np.uint8)
    dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=3)
    # argmax is inside foreground; for extremely thin regions dist may be small but valid
    y, x = np.unravel_index(int(dist.argmax()), dist.shape)
    return np.array([[float(x), float(y)]], dtype=np.float32)


def counts_from_masks(gt01: np.ndarray, pred01: np.ndarray) -> Tuple[int, int, int, int]:
    gt = gt01.astype(bool)
    pr = pred01.astype(bool)
    tp = int(np.logical_and(pr, gt).sum())
    fp = int(np.logical_and(pr, ~gt).sum())
    fn = int(np.logical_and(~pr, gt).sum())
    tn = int(np.logical_and(~pr, ~gt).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    iou = safe_div(tp, tp + fp + fn)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    return {"iou": iou, "dice": dice, "precision": prec, "recall": rec}


# ----------------------------
# dataset indexing
# ----------------------------
@dataclass
class Sample:
    image_path: str
    mask_path: str
    cls: str            # canonical: oil/scratch/stain
    rel: str            # relative path under masks/
    group: str          # first folder under masks/


def index_separated_pairs(root: Path, images_dirname: str = "images", masks_dirname: str = "masks",
                          min_mask_area: int = 5) -> List[Sample]:
    """
    Scan masks/** and pair with images/** using relpath.

    Your structure has a "group folder" that can include '.png' in the folder name, e.g.:
      masks/oil_0001.png/Oil_0001__inst000.png
    That's fine: we treat it just as a folder.
    """
    images_root = root / images_dirname
    masks_root = root / masks_dirname

    if not images_root.exists():
        raise FileNotFoundError(f"Missing images folder: {images_root}")
    if not masks_root.exists():
        raise FileNotFoundError(f"Missing masks folder: {masks_root}")

    mask_files = [p for p in masks_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    mask_files.sort()

    samples: List[Sample] = []
    for mp in mask_files:
        rel = mp.relative_to(masks_root)  # e.g. oil_0001.png/Oil_0001__inst000.png
        group = rel.parts[0] if len(rel.parts) > 0 else ""

        imgp = find_matching_image(images_root, rel)
        if imgp is None:
            continue

        cls = infer_class_from_path(rel)
        if cls is None:
            cls = infer_class_from_path(imgp)
        if cls is None:
            continue

        m01 = read_mask01(mp)
        if m01 is None:
            continue
        area = int(m01.sum())
        if area < min_mask_area:
            continue

        samples.append(Sample(
            image_path=str(imgp),
            mask_path=str(mp),
            cls=cls,
            rel=str(rel),
            group=str(group),
        ))

    return samples


# ----------------------------
# MobileNetV2
# ----------------------------
def build_mobilenetv2(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_mnv2_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """
    Expects a checkpoint like:
      {"state_dict": ..., "classes": [...]}
    If "classes" missing, fallback to canonical.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    classes = ckpt.get("classes", list(CANONICAL_CLASSES))
    # normalize class names if abbreviated
    classes = [normalize_class_name(c) or str(c).lower() for c in classes]

    model = build_mobilenetv2(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    return model, classes


def preprocess_crop_for_mnv2(img_rgb: np.ndarray, img_size: int) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    pil = Image.fromarray(img_rgb)
    return tfm(pil)


# ----------------------------
# SAM2 loading (robust sys.path)
# ----------------------------
def add_sam2_to_syspath(sam2_repo: Optional[str]) -> None:
    """
    If you cloned facebookresearch/sam2 locally and didn't pip-install it,
    pass --sam2_repo /path/to/sam2 so imports work.
    """
    if sam2_repo is None or str(sam2_repo).strip() == "":
        return
    p = Path(sam2_repo).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"--sam2_repo not found: {p}")
    if str(p) not in os.sys.path:
        os.sys.path.insert(0, str(p))


def build_sam2_predictor(model_cfg: str, base_ckpt: str, ft_ckpt: str,
                         device: str, sam2_repo: Optional[str] = None):
    """
    Returns a SAM2ImagePredictor with fine-tuned weights loaded.
    """
    add_sam2_to_syspath(sam2_repo)

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as e:
        raise RuntimeError(
            "Could not import SAM2. Fix one of these:\n"
            "  (A) pip install the sam2 package (if you have a wheel), OR\n"
            "  (B) pass --sam2_repo /path/to/facebookresearch/sam2 (repo root containing sam2/)\n"
            f"Original import error: {e}"
        )

    # Try the common build API:
    #   sam2_model = build_sam2(model_cfg, checkpoint, device=..., apply_postprocessing=False)
    try:
        sam2_model = build_sam2(model_cfg, base_ckpt, device=device, apply_postprocessing=False)
    except Exception:
        # Fallback: if build_sam2 expects a Hydra/OmegaConf cfg object
        try:
            from hydra import initialize_config_module, compose  # type: ignore
            with initialize_config_module(config_module="sam2_configs", version_base=None):
                cfg = compose(config_name=model_cfg)
            sam2_model = build_sam2(cfg, base_ckpt, device=device, apply_postprocessing=False)
        except Exception as e2:
            raise RuntimeError(
                "Failed to build SAM2 model. Common causes:\n"
                "  - model_cfg not found in Hydra config path (sam2_configs)\n"
                "  - you passed a wrong model_cfg name\n"
                "Try:\n"
                "  --model_cfg sam2_hiera_t.yaml   (or _s.yaml/_b.yaml/_l.yaml)\n"
                "  --sam2_repo /path/to/sam2\n"
                f"Build error: {e2}"
            )

    # load fine-tuned weights
    state = torch.load(ft_ckpt, map_location=device)
    sam2_model.load_state_dict(state, strict=False)

    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


@torch.no_grad()
def sam2_predict_selected_mask(predictor, image_rgb: np.ndarray, point_xy: np.ndarray,
                               device_type: str = "cuda") -> Tuple[np.ndarray, float, int]:
    """
    Use SAM2 predictor.predict and select the best-of-3 by score.
    Returns:
      pred01: (H,W) uint8
      best_score: float
      k: int
    """
    # SAM2 expects point_coords shape (N,2) and point_labels shape (N,)
    point_coords = point_xy.astype(np.float32)              # (1,2)
    point_labels = np.array([1], dtype=np.int32)            # (1,)

    # predictor handles transforms internally
    with torch.amp.autocast(device_type, enabled=(device_type == "cuda")):
        predictor.set_image(image_rgb)
        masks, scores, _logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=True,
        )

    # normalize shapes
    masks = np.asarray(masks)
    scores = np.asarray(scores)

    # masks could be (3,H,W) or (1,3,H,W)
    if masks.ndim == 4:
        masks = masks[0]
    if scores.ndim == 2:
        scores = scores[0]

    k = int(np.argmax(scores))
    best_score = float(scores[k])
    pred = masks[k]

    # masks from SAM2 are typically boolean; convert to uint8 {0,1}
    pred01 = (pred > 0).astype(np.uint8)
    return pred01, best_score, k


# ----------------------------
# visualization
# ----------------------------
def overlay_gt_pred(img_rgb: np.ndarray, gt01: np.ndarray, pr01: np.ndarray) -> np.ndarray:
    """
    Overlay:
      GT in green, Pred in red, overlap becomes yellow-ish.
    Assumes all inputs already have matching HxW.
    """
    if img_rgb.shape[:2] != gt01.shape[:2] or img_rgb.shape[:2] != pr01.shape[:2]:
        raise ValueError(
            f"overlay_gt_pred shape mismatch: img={img_rgb.shape[:2]} gt={gt01.shape[:2]} pr={pr01.shape[:2]}"
        )

    out = img_rgb.astype(np.float32).copy()
    gt = gt01.astype(bool)
    pr = pr01.astype(bool)

    # Green for GT
    out[gt, 1] = np.clip(out[gt, 1] + 120.0, 0, 255)
    # Red for Pred
    out[pr, 0] = np.clip(out[pr, 0] + 120.0, 0, 255)

    return out.astype(np.uint8)


def _resize_rgb_to_hw(img_rgb: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resize RGB image to (H,W)."""
    import cv2
    H, W = hw
    if img_rgb.shape[:2] == (H, W):
        return img_rgb
    return cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)


def _resize_mask_to_hw(mask01: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resize binary mask to (H,W) with nearest-neighbor."""
    import cv2
    H, W = hw
    if mask01.shape[:2] == (H, W):
        return mask01
    m = cv2.resize(mask01.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)



def save_examples_per_class(rows: List[Dict], out_dir: Path, cls: str,
                            n_success: int = 3, n_failure: int = 3,
                            prefer_end2end: bool = True,
                            iou_thr_success: float = 0.5,
                            pad_ratio: float = 0.25) -> None:
    """
    rows entries must include:
      cls, image_path, mask_path, seg_iou, cls_correct, end2end
    Saves examples_<cls>.png with 6 rows: 3 success then 3 failure.
    """
    import matplotlib.pyplot as plt

    cand = [r for r in rows if r["cls"] == cls]
    if len(cand) == 0:
        return

    # define success / failure pools
    if prefer_end2end:
        success_pool = [r for r in cand if r["end2end"]]
        failure_pool = [r for r in cand if not r["end2end"]]
        # if too few, fallback to iou-based
        if len(success_pool) < n_success:
            success_pool = [r for r in cand if r["seg_iou"] >= iou_thr_success]
        if len(failure_pool) < n_failure:
            failure_pool = [r for r in cand if r["seg_iou"] < iou_thr_success]
    else:
        success_pool = [r for r in cand if r["seg_iou"] >= iou_thr_success]
        failure_pool = [r for r in cand if r["seg_iou"] < iou_thr_success]

    # rank
    success_pool = sorted(success_pool, key=lambda x: x["seg_iou"], reverse=True)
    failure_pool = sorted(failure_pool, key=lambda x: x["seg_iou"], reverse=False)

    success = success_pool[:n_success]
    failure = failure_pool[:n_failure]

    # if still short, pad by best/worst overall
    if len(success) < n_success:
        more = [r for r in cand if r not in success]
        more = sorted(more, key=lambda x: x["seg_iou"], reverse=True)
        success += more[: (n_success - len(success))]
    if len(failure) < n_failure:
        more = [r for r in cand if r not in failure]
        more = sorted(more, key=lambda x: x["seg_iou"], reverse=False)
        failure += more[: (n_failure - len(failure))]

    picks = success[:n_success] + failure[:n_failure]
    if len(picks) == 0:
        return

    # plot: columns = Input | GT | Pred | Overlay
    nrows = len(picks)
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(16, 4 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, r in enumerate(picks):
        img = read_rgb(Path(r["image_path"]))
        gt = read_mask01(Path(r["mask_path"]))
        pr = read_mask01(Path(r["pred_mask_path"])) if r.get("pred_mask_path") else None

        # If we didn't save pred masks, recompute overlay using stored pred01 (if present)
        if pr is None and "pred01" in r:
            pr = r["pred01"]

        if img is None or gt is None or pr is None:
            continue

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Input\n{cls}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("GT")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pr, cmap="gray")
        tag = "SUCCESS" if r in success else "FAILURE"
        axes[i, 2].set_title(f"Pred ({tag})\nIoU={r['seg_iou']:.3f}")
        axes[i, 2].axis("off")

        # ---- ALIGN SHAPES (critical) ----
        # pred01 was generated after resizing to max_side, while disk image/mask are full-res.
        # Resize img + gt to pred01's HxW so overlay indexing matches.
        target_hw = pr.shape[:2]
        img = _resize_rgb_to_hw(img, target_hw)
        gt = _resize_mask_to_hw(gt, target_hw)
        pr = _resize_mask_to_hw(pr, target_hw)

        ov = overlay_gt_pred(img, gt, pr)

        pred_name = r.get("pred_cls", "?")
        axes[i, 3].imshow(ov)
        axes[i, 3].set_title(f"Overlay\npred={pred_name} | gt={cls}")
        axes[i, 3].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"examples_{cls}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data_dir", type=str, default="data/ground_truth_seperate")
    ap.add_argument("--images_dirname", type=str, default="images")
    ap.add_argument("--masks_dirname", type=str, default="masks")
    ap.add_argument("--min_mask_area", type=int, default=5)

    # SAM2
    ap.add_argument("--sam2_repo", type=str, default="", help="Path to facebookresearch/sam2 repo if not installed")
    ap.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml")
    ap.add_argument("--sam2_checkpoint", type=str, default="sam_trained_models/sam2_hiera_tiny.pt",
                    help="Base SAM2 weights (.pt)")
    ap.add_argument("--ft_ckpt", type=str, default="checkpoints/fine_tuned_sam2_final.torch",
                    help="Fine-tuned SAM2 .torch state_dict")
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--prompt_mode", type=str, default="oracle_dt",
                    choices=["oracle_dt", "center"], help="How to generate the SAM prompt point")

    # MobileNetV2
    ap.add_argument("--cnn_ckpt", type=str, default="cnn_checkpoints/mobilenetv2_best.pt",
                    help="MobileNetV2 checkpoint produced by your training script")
    ap.add_argument("--img_size", type=int, default=224)

    # pipeline evaluation
    ap.add_argument("--out_dir", type=str, default="pipeline_results")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_class", type=int, default=0, help="0 = use all samples per class")
    ap.add_argument("--iou_thr_e2e", type=float, default=0.5, help="IoU threshold for end-to-end success")

    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # 1) index samples (paired separated instances)
    samples = index_separated_pairs(
        root=root,
        images_dirname=args.images_dirname,
        masks_dirname=args.masks_dirname,
        min_mask_area=args.min_mask_area,
    )
    if len(samples) == 0:
        raise RuntimeError(
            "No samples found. Double-check your structure.\n"
            f"Expected like:\n"
            f"  {root}/{args.images_dirname}/oil_0001.png/Oil_0001__inst000.jpg\n"
            f"  {root}/{args.masks_dirname}/oil_0001.png/Oil_0001__inst000.png\n"
        )

    # Basic counts
    counts = {c: 0 for c in CANONICAL_CLASSES}
    for s in samples:
        counts[s.cls] += 1

    print("Found samples:", len(samples))
    print("Counts by class:", counts)

    # Optional: limit per class (for speed)
    if args.max_per_class and args.max_per_class > 0:
        byc: Dict[str, List[Sample]] = {c: [] for c in CANONICAL_CLASSES}
        for s in samples:
            byc[s.cls].append(s)
        # deterministic shuffle
        rng = np.random.RandomState(args.seed)
        new_samples: List[Sample] = []
        for c in CANONICAL_CLASSES:
            arr = byc[c]
            rng.shuffle(arr)
            new_samples.extend(arr[: args.max_per_class])
        samples = new_samples
        print("After --max_per_class limit:", len(samples))

    # 2) load models
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"

    print("\nLoading SAM2 predictor...")
    predictor = build_sam2_predictor(
        model_cfg=args.model_cfg,
        base_ckpt=args.sam2_checkpoint,
        ft_ckpt=args.ft_ckpt,
        device=str(device),
        sam2_repo=(args.sam2_repo if args.sam2_repo.strip() else None),
    )

    print("Loading MobileNetV2 checkpoint...")
    mnv2, mn_classes = load_mnv2_checkpoint(Path(args.cnn_ckpt), device=device)
    # normalize ckpt classes
    mn_classes = [normalize_class_name(c) or c for c in mn_classes]
    mn_class_to_idx = {c: i for i, c in enumerate(mn_classes)}

    # sanity: make sure it can output canonical set, otherwise we still evaluate but mapping is imperfect
    # (best is to train mnv2 with canonical classes)
    print("MobileNet classes:", mn_classes)

    # 3) evaluate
    # segmentation aggregated tp/fp/fn per class
    seg_counts = {c: {"tp": 0, "fp": 0, "fn": 0, "n": 0} for c in CANONICAL_CLASSES}
    # classification
    cls_counts = {c: {"correct": 0, "n": 0} for c in CANONICAL_CLASSES}
    # end-to-end
    e2e_counts = {c: {"correct": 0, "n": 0} for c in CANONICAL_CLASSES}

    # confusion matrix for classification over canonical classes (rows=true, cols=pred)
    C = len(CANONICAL_CLASSES)
    cls_cm = [[0 for _ in range(C)] for _ in range(C)]
    canon_to_idx = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

    # store per-sample records for example selection (no large arrays)
    rows: List[Dict] = []

    def resize_longest_side(img_rgb: np.ndarray, mask01: np.ndarray, max_side: int) -> Tuple[np.ndarray, np.ndarray]:
        if max_side is None or max_side <= 0:
            return img_rgb, mask01
        h, w = img_rgb.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        if abs(scale - 1.0) < 1e-9:
            return img_rgb, mask01
        import cv2
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img2 = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        m2 = cv2.resize(mask01, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return img2, m2

    t0 = time.perf_counter()
    for s in samples:
        gt_cls = s.cls
        img = read_rgb(Path(s.image_path))
        gt = read_mask01(Path(s.mask_path))
        if img is None or gt is None:
            continue

        img, gt = resize_longest_side(img, gt, args.max_side)

        # prompt point
        if args.prompt_mode == "oracle_dt":
            pt = choose_prompt_point_oracle_dt(gt)
        else:
            # center prompt (NOT robust; only for quick debugging)
            ys, xs = np.where(gt > 0)
            if len(xs) == 0:
                pt = None
            else:
                pt = np.array([[float(xs.mean()), float(ys.mean())]], dtype=np.float32)

        if pt is None:
            continue

        # SAM2 predict
        pr, _score, _k = sam2_predict_selected_mask(predictor, img, pt, device_type=device_type)

        # seg metrics (aggregate by counts)
        tp, fp, fn, _tn = counts_from_masks(gt, pr)
        seg_counts[gt_cls]["tp"] += tp
        seg_counts[gt_cls]["fp"] += fp
        seg_counts[gt_cls]["fn"] += fn
        seg_counts[gt_cls]["n"] += 1
        seg_m = metrics_from_counts(tp, fp, fn)

        # crop by predicted mask bbox for classification
        crop = img
        bb = mask_bbox(pr)
        if bb is not None:
            crop = crop_with_pad(img, bb, pad_ratio=0.25)

        x = preprocess_crop_for_mnv2(crop, args.img_size).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = mnv2(x)
            pred_idx = int(logits.argmax(dim=1).item())
        pred_name_raw = mn_classes[pred_idx] if pred_idx < len(mn_classes) else "unknown"
        pred_cls = normalize_class_name(pred_name_raw) or str(pred_name_raw)

        # classification counts (canonical confusion matrix)
        cls_counts[gt_cls]["n"] += 1
        cls_ok = (pred_cls == gt_cls)
        if cls_ok:
            cls_counts[gt_cls]["correct"] += 1

        # update confusion matrix if prediction is canonical
        if gt_cls in canon_to_idx and pred_cls in canon_to_idx:
            cls_cm[canon_to_idx[gt_cls]][canon_to_idx[pred_cls]] += 1

        # end-to-end: correct class AND segmentation IoU >= threshold
        e2e_counts[gt_cls]["n"] += 1
        e2e_ok = cls_ok and (seg_m["iou"] >= float(args.iou_thr_e2e))
        if e2e_ok:
            e2e_counts[gt_cls]["correct"] += 1

        rows.append({
            "cls": gt_cls,
            "image_path": s.image_path,
            "mask_path": s.mask_path,
            "seg_iou": float(seg_m["iou"]),
            "seg_dice": float(seg_m["dice"]),
            "seg_precision": float(seg_m["precision"]),
            "seg_recall": float(seg_m["recall"]),
            "pred_cls": pred_cls,
            "cls_correct": bool(cls_ok),
            "end2end": bool(e2e_ok),
            # keep pred mask for example plots without recomputing:
            "pred01": pr,
            "pred_mask_path": "",  # not used (we keep pred01 in-memory)
        })

    t1 = time.perf_counter()
    print(f"\nEvaluated {len(rows)} samples in {(t1 - t0):.1f}s")

    if len(rows) == 0:
        raise RuntimeError("No valid evaluated samples (after reading / prompting). Check your data and paths.")

    # 4) summarize + save CSVs
    per_class_out = out_dir / "pipeline_metrics_per_class.csv"
    overall_out = out_dir / "pipeline_metrics_overall.csv"
    cm_out = out_dir / "pipeline_confusion_matrix_cls.csv"

    def summarize_seg(c: str) -> Dict[str, float]:
        d = seg_counts[c]
        m = metrics_from_counts(d["tp"], d["fp"], d["fn"])
        return {"n": float(d["n"]), **m}

    def summarize_cls(c: str) -> Dict[str, float]:
        d = cls_counts[c]
        acc = safe_div(d["correct"], d["n"])
        return {"n": float(d["n"]), "acc": acc}

    def summarize_e2e(c: str) -> Dict[str, float]:
        d = e2e_counts[c]
        acc = safe_div(d["correct"], d["n"])
        return {"n": float(d["n"]), "acc": acc}

    # per-class rows
    per_rows: List[Dict[str, float]] = []
    for c in CANONICAL_CLASSES:
        sseg = summarize_seg(c)
        scls = summarize_cls(c)
        se2e = summarize_e2e(c)
        per_rows.append({
            "class": c,
            "n": int(sseg["n"]),
            "seg_iou": sseg["iou"],
            "seg_dice": sseg["dice"],
            "seg_precision": sseg["precision"],
            "seg_recall": sseg["recall"],
            "cls_acc": scls["acc"],
            "end2end_acc": se2e["acc"],
        })

    # overall (micro-aggregated over all classes)
    tp_all = sum(seg_counts[c]["tp"] for c in CANONICAL_CLASSES)
    fp_all = sum(seg_counts[c]["fp"] for c in CANONICAL_CLASSES)
    fn_all = sum(seg_counts[c]["fn"] for c in CANONICAL_CLASSES)
    seg_all = metrics_from_counts(tp_all, fp_all, fn_all)

    cls_correct_all = sum(cls_counts[c]["correct"] for c in CANONICAL_CLASSES)
    cls_n_all = sum(cls_counts[c]["n"] for c in CANONICAL_CLASSES)
    cls_acc_all = safe_div(cls_correct_all, cls_n_all)

    e2e_correct_all = sum(e2e_counts[c]["correct"] for c in CANONICAL_CLASSES)
    e2e_n_all = sum(e2e_counts[c]["n"] for c in CANONICAL_CLASSES)
    e2e_acc_all = safe_div(e2e_correct_all, e2e_n_all)

    # write per-class CSV
    with per_class_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        for r in per_rows:
            w.writerow(r)

    # write overall CSV
    with overall_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_samples", len(rows)])
        w.writerow(["seg_iou", seg_all["iou"]])
        w.writerow(["seg_dice", seg_all["dice"]])
        w.writerow(["seg_precision", seg_all["precision"]])
        w.writerow(["seg_recall", seg_all["recall"]])
        w.writerow(["cls_acc", cls_acc_all])
        w.writerow(["end2end_acc", e2e_acc_all])
        w.writerow(["iou_thr_e2e", args.iou_thr_e2e])
        w.writerow(["prompt_mode", args.prompt_mode])

    # write classification confusion matrix
    with cm_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + list(CANONICAL_CLASSES))
        for i, row in enumerate(cls_cm):
            w.writerow([CANONICAL_CLASSES[i]] + row)

    # print concise summary (averages only)
    print("\n=== Pipeline summary (averaged) ===")
    print(f"Segmentation (SAM2 fine-tuned): IoU={seg_all['iou']:.4f} | Dice={seg_all['dice']:.4f}")
    print(f"Classification (MobileNetV2):  Acc={cls_acc_all:.4f}")
    print(f"End-to-end (IoU>={args.iou_thr_e2e} & class correct): Acc={e2e_acc_all:.4f}")
    print(f"\nSaved per-class CSV : {per_class_out}")
    print(f"Saved overall CSV   : {overall_out}")
    print(f"Saved cls CM CSV    : {cm_out}")

    # also save a json for convenience
    summary_json = out_dir / "pipeline_summary.json"
    with summary_json.open("w") as f:
        json.dump({
            "n_samples": len(rows),
            "prompt_mode": args.prompt_mode,
            "iou_thr_e2e": args.iou_thr_e2e,
            "overall": {
                "seg": seg_all,
                "cls_acc": cls_acc_all,
                "end2end_acc": e2e_acc_all,
            },
            "per_class": per_rows,
        }, f, indent=2)
    print(f"Saved summary JSON  : {summary_json}")

    # 5) save example grids: 3 success + 3 failure per class
    for c in CANONICAL_CLASSES:
        save_examples_per_class(
            rows=rows,
            out_dir=out_dir,
            cls=c,
            n_success=3,
            n_failure=3,
            prefer_end2end=True,
            iou_thr_success=float(args.iou_thr_e2e),
        )
    print("\nSaved example grids:", ", ".join([f"examples_{c}.png" for c in CANONICAL_CLASSES]))


if __name__ == "__main__":
    main()
