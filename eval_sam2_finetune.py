# eval_sam2_finetune.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data import load_train_test_data, read_batch
from sam2_utils import build_predictor


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


def _binarize_gt(mask_u8_or_01: np.ndarray) -> np.ndarray:
    """Ensure GT is 0/1 with shape (H,W)."""
    gt = np.squeeze(mask_u8_or_01)
    if gt.ndim != 2:
        raise ValueError(f"Expected GT to be 2D after squeeze, got {gt.shape}")
    # works for 0/255 or already 0/1
    gt01 = (gt > 0).astype(np.float32)
    return gt01


def _metrics_from_probs(prd_prob: torch.Tensor, gt01: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    prd_prob: (N,H,W) float in [0,1]
    gt01:     (N,H,W) float {0,1}
    returns: iou, dice, prec, rec, f1 each shape (N,)
    """
    prd_bin = (prd_prob > 0.5).to(gt01.dtype)

    inter = (gt01 * prd_bin).sum(dim=(1, 2))
    union = gt01.sum(dim=(1, 2)) + prd_bin.sum(dim=(1, 2)) - inter
    iou = inter / (union + 1e-6)

    dice = (2 * inter) / (gt01.sum(dim=(1, 2)) + prd_bin.sum(dim=(1, 2)) + 1e-6)

    tp = inter
    fp = (prd_bin * (1 - gt01)).sum(dim=(1, 2))
    fn = ((1 - prd_bin) * gt01).sum(dim=(1, 2))
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)

    return iou, dice, prec, rec, f1


@torch.no_grad()
def run_one_batch(predictor, image_rgb: np.ndarray, mask_u8: np.ndarray, input_point: np.ndarray) -> Dict[str, float]:
    """
    Matches your training forward pass and returns:
    - mask0 metrics (the one you trained)
    - score-selected metrics
    - oracle best-of-3 metrics
    """
    device = next(predictor.model.parameters()).device
    is_cuda = device.type == "cuda"

    predictor.set_image(image_rgb)

    # Training uses one positive label per prompt point, shape (N,1)
    num_masks = input_point.shape[0]
    input_label = np.ones((num_masks, 1), dtype=np.int64)

    mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
        input_point,
        input_label,
        box=None,
        mask_logits=None,
        normalize_coords=True,
    )
    if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0:
        return {}

    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=(unnorm_coords, labels),
        boxes=None,
        masks=None,
    )

    batched_mode = unnorm_coords.shape[0] > 1
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

    # Important: match train (multimask_output=True)
    with torch.amp.autocast("cuda", enabled=is_cuda):
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # (N,3,H,W)
        prd_prob_all = torch.sigmoid(prd_masks)  # (N,3,H,W)

    # GT expand to (N,H,W)
    gt01_np = _binarize_gt(mask_u8)  # (H,W)
    gt = torch.as_tensor(gt01_np, device=device)  # (H,W)
    gt = gt.unsqueeze(0).expand(prd_prob_all.shape[0], -1, -1)  # (N,H,W)

    # ---- metrics for mask0 (match training)
    prd_prob0 = prd_prob_all[:, 0]  # (N,H,W)
    iou0, dice0, prec0, rec0, f10 = _metrics_from_probs(prd_prob0, gt)

    # ---- score-selected among 3 masks (per prompt instance)
    # prd_scores is typically (N,3); choose best score for each prompt row
    if prd_scores.ndim == 2 and prd_scores.shape[1] == 3:
        sel = torch.argmax(prd_scores, dim=1)  # (N,)
        prd_prob_sel = prd_prob_all[torch.arange(prd_prob_all.size(0), device=device), sel]
    else:
        # fallback: just use mask0
        prd_prob_sel = prd_prob0

    ious, dices, precs, recs, f1s = _metrics_from_probs(prd_prob_sel, gt)

    # ---- oracle best-of-3 IoU (upper bound diagnostic)
    # compute iou for each of 3 masks per prompt, then max over masks
    iou_all = []
    for k in range(3):
        iouk, _, _, _, _ = _metrics_from_probs(prd_prob_all[:, k], gt)
        iou_all.append(iouk.unsqueeze(1))
    iou_all = torch.cat(iou_all, dim=1)  # (N,3)
    oracle_iou = iou_all.max(dim=1).values  # (N,)

    out = {
        "iou_mask0": float(iou0.mean().cpu()),
        "dice_mask0": float(dice0.mean().cpu()),
        "px_precision_mask0": float(prec0.mean().cpu()),
        "px_recall_mask0": float(rec0.mean().cpu()),
        "px_f1_mask0": float(f10.mean().cpu()),

        "iou_score_selected": float(ious.mean().cpu()),
        "dice_score_selected": float(dices.mean().cpu()),
        "px_precision_score_selected": float(precs.mean().cpu()),
        "px_recall_score_selected": float(recs.mean().cpu()),
        "px_f1_score_selected": float(f1s.mean().cpu()),

        "oracle_best_iou": float(oracle_iou.mean().cpu()),
        "mean_prd_score0": float(prd_scores[:, 0].mean().detach().cpu()) if prd_scores.ndim == 2 else float(np.mean(prd_scores)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()

    # Data (same as training)
    ap.add_argument("--data_dir", type=str, default="data/MSD-US/test")
    ap.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "csv", "msd_us"])
    ap.add_argument("--gt_dirname", type=str, default="ground_truth")
    ap.add_argument("--categories", type=str, default="", help="Comma-separated list, e.g. oil,scratch,stain (default: all)")
    ap.add_argument("--test_size", type=float, default=0.2)

    # Model
    ap.add_argument("--sam2_checkpoint", type=str, required=True)
    ap.add_argument("--model_cfg", type=str, default="")
    ap.add_argument("--sam_weights", type=str, required=True, help="Your fine-tuned .torch state_dict")

    # Eval
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_batches", type=int, default=0, help="0 = eval all test samples (recommended)")
    ap.add_argument("--out_csv", type=str, default="sam2_finetune_eval.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    print(f"Loaded train={len(train_data)} test={len(test_data)}")

    if not args.model_cfg:
        args.model_cfg = infer_model_cfg_from_ckpt(args.sam2_checkpoint)
    print(f"Using model_cfg={args.model_cfg} ckpt={args.sam2_checkpoint}")
    print(f"Loading fine-tuned weights: {args.sam_weights}")

    predictor = build_predictor(args.model_cfg, args.sam2_checkpoint, device=args.device)
    sd = torch.load(args.sam_weights, map_location=args.device)
    predictor.model.load_state_dict(sd, strict=True)
    predictor.model.eval()

    rows: List[Dict[str, float]] = []
    n = len(test_data) if args.num_batches == 0 else min(len(test_data), args.num_batches)

    for i in tqdm(range(n), desc="Evaluating on test batches"):
        image, mask, input_point, num_masks = read_batch(
            test_data,
            visualize_data=False,
            max_side=args.max_side,
            seed=args.seed + i,
        )
        if image is None or mask is None or num_masks == 0 or input_point.size == 0:
            continue

        m = run_one_batch(predictor, image, mask, input_point)
        if not m:
            continue
        rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved -> {args.out_csv}")

    if len(df) == 0:
        print("No valid samples evaluated (check read_batch / GT).")
        return

    print("\n=== Overall (prompted like training) ===")
    cols0 = ["iou_mask0", "dice_mask0", "px_precision_mask0", "px_recall_mask0", "px_f1_mask0"]
    print(df[cols0].mean())

    print("\n=== Score-selected (diagnostic) ===")
    colsS = ["iou_score_selected", "dice_score_selected", "px_precision_score_selected", "px_recall_score_selected", "px_f1_score_selected"]
    print(df[colsS].mean())

    print("\n=== Oracle best-of-3 IoU (upper bound diagnostic) ===")
    print(df[["oracle_best_iou"]].mean())


if __name__ == "__main__":
    main()
