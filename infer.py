# infer.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from data import load_train_test_data, read_image, resize_to_max_side, get_points
from sam2_utils import build_predictor, load_finetuned_weights


def infer_model_cfg_from_ckpt(ckpt_path: str) -> str:
    name = Path(ckpt_path).name.lower()
    if "tiny" in name or "hiera_t" in name:
        return "sam2_hiera_t.yaml"
    if "small" in name or "hiera_s" in name:
        return "sam2_hiera_s.yaml"
    if "base_plus" in name or "bplus" in name or "b+" in name:
        return "sam2_hiera_b+.yaml"
    if "base" in name or "hiera_b" in name:
        return "sam2_hiera_b.yaml"
    if "large" in name or "hiera_l" in name:
        return "sam2_hiera_l.yaml"
    raise ValueError(f"Cannot infer model_cfg from checkpoint name: {ckpt_path}")


def best_mask_from_multimask(masks: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    masks: (K,H,W) or (K,1,H,W)
    scores: (K,) or (K,1)
    Returns: best_mask(H,W) uint8 {0,1}, best_score
    """
    m = masks
    if m.ndim == 4:
        m = m[:, 0]
    s = scores
    if s.ndim == 2:
        s = s[:, 0]
    idx = int(np.argmax(s))
    best = (m[idx] > 0).astype(np.uint8)
    return best, float(s[idx])


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = (gt > 0).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / (union + 1e-6))


def main():
    ap = argparse.ArgumentParser()

    # Either provide (data_dir) to sample a random example, OR provide explicit image/mask paths
    ap.add_argument("--data_dir", type=str, default="data/MSD-US/test")
    ap.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "csv", "msd_us"])
    ap.add_argument("--gt_dirname", type=str, default="ground_truth")
    ap.add_argument("--categories", type=str, default="", help="Comma-separated list, e.g. oil,scratch,stain (default: all)")
    ap.add_argument("--test_size", type=float, default=0.2)

    ap.add_argument("--image_path", type=str, default="", help="Optional: run inference on a specific image file")
    ap.add_argument("--mask_path", type=str, default="", help="Optional: GT mask path for prompts/IoU (recommended for eval)")

    # Model
    ap.add_argument("--sam2_checkpoint", type=str, default="sam_trained_models/sam2_hiera_tiny.pt")
    ap.add_argument("--model_cfg", type=str, default="", help="Empty -> infer from checkpoint name")
    ap.add_argument("--weights", type=str, required=True, help="Path to fine-tuned .torch weights")
    ap.add_argument("--device", type=str, default="cuda")

    # Inference
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--num_samples_per_instance", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_pred", type=str, default="", help="Optional: output path to save predicted binary mask (.png)")
    ap.add_argument("--show", action="store_true", help="Visualize the result")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Pick sample
    if args.image_path:
        if not args.mask_path:
            raise ValueError("If you pass --image_path, also pass --mask_path (for prompt points).")
        sample_image = args.image_path
        sample_mask = args.mask_path
    else:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None
        train_data, test_data = load_train_test_data(
            args.data_dir,
            test_size=args.test_size,
            random_state=args.seed,
            dataset_format=args.dataset_format,
            gt_dirname=args.gt_dirname,
            categories=categories,
            verbose=True,
        )
        pool = test_data if len(test_data) else train_data
        sample = random.choice(pool)
        sample_image, sample_mask = sample["image"], sample["annotation"]

    image, mask = read_image(sample_image, sample_mask)
    image, mask, _ = resize_to_max_side(image, mask, max_side=args.max_side)

    # Prompt points from GT
    input_points = get_points(mask, num_samples_per_instance=args.num_samples_per_instance, seed=args.seed)
    if input_points.shape[0] == 0:
        raise RuntimeError("No foreground pixels found in GT mask; cannot sample prompt points.")

    if not args.model_cfg:
        args.model_cfg = infer_model_cfg_from_ckpt(args.sam2_checkpoint)

    predictor = build_predictor(args.model_cfg, args.sam2_checkpoint, device=args.device)
    load_finetuned_weights(predictor, args.weights, device=args.device, strict=False)
    predictor.model.eval()

    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1], dtype=np.int64),
        )

    best_pred, best_score = best_mask_from_multimask(np.asarray(masks), np.asarray(scores))
    iou = iou_score(best_pred, mask)

    print(f"Sample: image={sample_image}")
    print(f"Best SAM score={best_score:.4f} | IoU={iou:.4f}")

    if args.save_pred:
        out = Path(args.save_pred)
        out.parent.mkdir(parents=True, exist_ok=True)
        import cv2 as _cv2
        _cv2.imwrite(str(out), (best_pred * 255).astype(np.uint8))
        print(f"Saved predicted mask: {out}")

    if args.show:
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("GT mask")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title(f"Prediction (IoU={iou:.3f})")
        plt.imshow(best_pred, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
