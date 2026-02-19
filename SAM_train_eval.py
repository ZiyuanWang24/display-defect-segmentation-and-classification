# train.py
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F

# Keep your existing utilities
from sam2_utils import build_predictor, set_training_mode

# Optional: keep compatibility with your old dataset loaders
try:
    from data import load_train_test_data as load_train_test_data_orig
    from data import read_batch as read_batch_orig
except Exception:
    load_train_test_data_orig = None
    read_batch_orig = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def infer_model_cfg_from_ckpt(ckpt_path: str) -> str:
    """
    Infer SAM2 config filename from checkpoint filename.
    Adjust if your checkpoint naming differs.
    """
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

    raise ValueError(
        f"Cannot infer model_cfg from checkpoint name: {ckpt_path}\n"
        "Please pass the matching --model_cfg (e.g., sam2_hiera_t.yaml / _s.yaml / _b.yaml / _l.yaml / _b+.yaml)."
    )


def soft_dice_loss(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    prob, gt: (N, H, W) float tensors in [0,1]
    Returns mean(1 - dice) over batch.
    """
    prob_f = prob.flatten(1)
    gt_f = gt.flatten(1)
    inter = (prob_f * gt_f).sum(dim=1)
    den = prob_f.sum(dim=1) + gt_f.sum(dim=1)
    dice = (2.0 * inter + eps) / (den + eps)
    return (1.0 - dice).mean()


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _resize_longest_side(img_rgb: np.ndarray, mask: np.ndarray, max_side: int):
    h, w = img_rgb.shape[:2]
    if max_side is None or max_side <= 0:
        return img_rgb, mask
    scale = min(max_side / max(h, w), 1.0)
    if abs(scale - 1.0) < 1e-9:
        return img_rgb, mask
    import cv2

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img2 = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    m2 = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img2, m2


def pick_prompt_point_inside_mask(gt01: np.ndarray) -> Optional[np.ndarray]:
    """
    Robust prompt point: choose the pixel farthest from boundary (distance transform).
    Returns point array shape (1,1,2) float32 (x,y) or None if empty.
    """
    if gt01 is None or gt01.ndim != 2 or gt01.sum() == 0:
        return None
    import cv2

    dist = cv2.distanceTransform((gt01.astype(np.uint8) * 255), cv2.DIST_L2, 3)
    cy, cx = np.unravel_index(int(dist.argmax()), dist.shape)
    return np.array([[[cx, cy]]], dtype=np.float32)


def load_instances_csv_dataset(data_dir: Path, categories: Optional[List[str]] = None) -> List[Dict]:
    """
    Load instance dataset produced by make_instance_dataset.py.
    Requires: data_dir/instances.csv
    CSV columns: class, image, mask (relative to data_dir)
    Returns entries with keys: image, annotation, class
    """
    csv_path = data_dir / "instances.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"instances.csv not found at: {csv_path}")

    entries: List[Dict] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"class", "image", "mask"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}; found: {reader.fieldnames}")

        for row in reader:
            cls = (row.get("class") or "").strip().lower()
            if categories and cls not in categories:
                continue

            img_rel = (row.get("image") or "").strip()
            msk_rel = (row.get("mask") or "").strip()
            if not img_rel or not msk_rel:
                continue

            img_path = (data_dir / img_rel).resolve()
            msk_path = (data_dir / msk_rel).resolve()

            if not img_path.exists() or not msk_path.exists():
                # skip silently; you can make this a warning if you want
                continue

            entries.append({"class": cls, "image": img_path, "annotation": msk_path})

    if len(entries) == 0:
        raise RuntimeError(f"No valid entries loaded from {csv_path}. Check paths / categories.")
    return entries


from pathlib import Path

KNOWN_CLASSES = ("oil", "scr", "sta")

def infer_category(ent: dict) -> str:
    """
    Infer class label without relying on folder structure.

    Priority:
    1) ent["category"]/["label"]/["defect_type"] if it contains an actual class
    2) parse from annotation filename (recommended for instance dataset)
    3) parse from image filename (e.g. oil_0013.png -> oil)
    """
    # ---------- 1) explicit label fields ----------
    for k in ("category", "label", "defect_type", "class"):
        v = ent.get(k, None)
        if not v:
            continue
        s = str(v).strip().lower()

        # exact match
        if s in KNOWN_CLASSES:
            return s

        # if it's a filename like "oil_0013.png", extract prefix token
        base = Path(s).name           # oil_0013.png
        stem = Path(base).stem        # oil_0013
        token = stem.split("_")[0].split("-")[0].split("__")[0]
        if token in KNOWN_CLASSES:
            return token

        # last resort substring match
        for c in KNOWN_CLASSES:
            if c in stem:
                return c

    # ---------- 2) parse from annotation filename ----------
    ann = str(ent.get("annotation", "")).lower()
    if ann:
        stem = Path(ann).stem.lower()
        # if you name instance masks like: "scratch__oil_0013__inst000.png"
        token = stem.split("__")[0].split("_")[0].split("-")[0]
        if token in KNOWN_CLASSES:
            return token
        for c in KNOWN_CLASSES:
            if c in stem:
                return c

    # ---------- 3) parse from image filename ----------
    img = str(ent.get("image", "")).lower()
    if img:
        stem = Path(img).stem.lower()
        token = stem.split("_")[0].split("-")[0].split("__")[0]
        if token in KNOWN_CLASSES:
            return token
        for c in KNOWN_CLASSES:
            if c in stem:
                return c

    return "unknown"

# keep compatibility with older code that calls _infer_category
_infer_category = infer_category



def stratified_split(entries: List[Dict], holdout_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.RandomState(seed)
    if len(entries) == 0:
        return [], []

    labels = [infer_category(e) for e in entries]
    uniq = sorted(set(labels))

    # If all same label, just shuffle split
    if len(uniq) <= 1:
        idx = rng.permutation(len(entries))
        n_hold = int(round(len(entries) * holdout_ratio))
        hold_idx = idx[:n_hold]
        keep_idx = idx[n_hold:]
        return [entries[i] for i in keep_idx], [entries[i] for i in hold_idx]

    keep_idx, hold_idx = [], []
    for lab in uniq:
        inds = [i for i, y in enumerate(labels) if y == lab]
        inds = list(rng.permutation(inds))
        n_hold = int(round(len(inds) * holdout_ratio))
        if len(inds) >= 2 and n_hold >= len(inds):
            n_hold = len(inds) - 1
        hold_idx.extend(inds[:n_hold])
        keep_idx.extend(inds[n_hold:])

    rng.shuffle(keep_idx)
    rng.shuffle(hold_idx)
    return [entries[i] for i in keep_idx], [entries[i] for i in hold_idx]


def read_batch_instances(
    entries: List[Dict],
    visualize_data: bool = False,
    max_side: int = 512,
    seed: Optional[int] = None,
):
    """
    For instance dataset: one sample = one instance mask.
    Returns:
      image_rgb: (H,W,3) uint8
      mask:      (1,H,W) uint8 {0,1}
      point:     (1,1,2) float32 (x,y)
      num_masks: int (always 1)
    """
    import cv2

    if len(entries) == 0:
        return None, None, None, 0

    rng = np.random.RandomState(seed) if seed is not None else np.random
    ent = entries[int(rng.randint(len(entries)))]

    img_path = ent.get("image", None)
    ann_path = ent.get("annotation", None)
    if img_path is None or ann_path is None:
        return None, None, None, 0

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, None, 0
    img_rgb = bgr[..., ::-1]

    m = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None, None, None, 0
    gt = (m > 0).astype(np.uint8)

    img_rgb, gt = _resize_longest_side(img_rgb, gt, max_side=max_side)
    pt = pick_prompt_point_inside_mask(gt)
    if pt is None:
        return None, None, None, 0

    mask_out = gt[None, ...].astype(np.uint8)

    if visualize_data:
        import matplotlib.pyplot as plt

        cx, cy = int(pt[0, 0, 0]), int(pt[0, 0, 1])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(img_rgb)
        ax[0].scatter([cx], [cy], s=40)
        ax[0].set_title("Image + prompt point")
        ax[0].axis("off")
        ax[1].imshow(gt, cmap="gray")
        ax[1].set_title("Instance GT mask")
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()

    return img_rgb, mask_out, pt, 1


def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--data_dir", type=str, default="data/ground_truth_seperate")
    ap.add_argument(
        "--dataset_format",
        type=str,
        default="auto",
        choices=["auto", "csv", "msd_us", "instances"],
        help="Use 'instances' for the split-instance dataset (expects instances.csv).",
    )
    ap.add_argument("--gt_dirname", type=str, default="ground_truth")
    ap.add_argument("--categories", type=str, default="", help="Comma-separated list, e.g. oil,scratch,stain (default: all)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1, help="Fraction of (train+val) used as eval/val")

    # Model
    ap.add_argument("--sam2_checkpoint", type=str, default="sam_trained_models/sam2_hiera_tiny.pt")
    ap.add_argument("--model_cfg", type=str, default="")  # empty -> infer from ckpt
    ap.add_argument("--weights_init", type=str, default="", help="Optional: start from a fine-tuned .torch state_dict")

    # Train
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--scheduler_step_size", type=int, default=500)
    ap.add_argument("--scheduler_gamma", type=float, default=0.2)
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--visualize_first_batch", action="store_true")

    # Fine-tuning knobs
    ap.add_argument("--train_prompt_encoder", action="store_true", help="Train prompt encoder (default: True)")
    ap.add_argument("--no_train_prompt_encoder", action="store_true", help="Disable training prompt encoder")
    ap.add_argument("--train_mask_decoder", action="store_true", help="Train mask decoder (default: True)")
    ap.add_argument("--no_train_mask_decoder", action="store_true", help="Disable training mask decoder")
    ap.add_argument("--freeze_image_encoder", action="store_true", help="Freeze image encoder (default: True)")
    ap.add_argument("--no_freeze_image_encoder", action="store_true", help="Do not freeze image encoder")

    # Eval during training
    ap.add_argument("--eval_every", type=int, default=200, help="Run eval every N steps")
    ap.add_argument("--eval_samples", type=int, default=200, help="Max eval samples per eval run (speed)")
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="0 disables; otherwise stop if no val IoU improvement for this many evals",
    )
    ap.add_argument("--use_best_for_test", action="store_true", help="Evaluate test using best-val checkpoint (recommended)")

    args = ap.parse_args()

    # Defaults for flags: True unless explicitly disabled
    train_prompt_encoder = True if not args.no_train_prompt_encoder else False
    train_mask_decoder = True if not args.no_train_mask_decoder else False
    freeze_image_encoder = True if not args.no_freeze_image_encoder else False

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    categories = [c.strip().lower() for c in args.categories.split(",") if c.strip()] or None

    # Decide which dataset loader to use
    instances_csv = data_dir / "instances.csv"
    use_instances = (args.dataset_format == "instances") or (args.dataset_format == "auto" and instances_csv.exists())

    if use_instances:
        print(f"[Data] Using instance dataset (instances.csv): {instances_csv}")
        all_entries = load_instances_csv_dataset(data_dir, categories=categories)
        trainval_data, test_data = stratified_split(all_entries, holdout_ratio=args.test_size, seed=args.seed)
        read_batch_fn = read_batch_instances
    else:
        if load_train_test_data_orig is None or read_batch_orig is None:
            raise RuntimeError(
                "Could not import data.py loaders, and dataset_format is not 'instances'. "
                "Install/restore your original data.py or use the instance dataset."
            )
        trainval_data, test_data = load_train_test_data_orig(
            args.data_dir,
            test_size=args.test_size,
            random_state=args.seed,
            dataset_format=args.dataset_format,
            gt_dirname=args.gt_dirname,
            categories=categories,
            verbose=True,
        )
        read_batch_fn = read_batch_orig

    # Train/Eval split from trainval_data
    train_data, eval_data = stratified_split(trainval_data, holdout_ratio=args.val_size, seed=args.seed)

    print(
        f"[Data] Loaded train={len(train_data)} eval={len(eval_data)} test={len(test_data)} "
        f"(test_size={args.test_size}, val_size={args.val_size}, use_instances={use_instances})"
    )

    # Model cfg
    if not args.model_cfg:
        args.model_cfg = infer_model_cfg_from_ckpt(args.sam2_checkpoint)
    print(f"Using model_cfg={args.model_cfg} ckpt={args.sam2_checkpoint}")

    predictor = build_predictor(args.model_cfg, args.sam2_checkpoint, device=args.device)

    if args.weights_init:
        state = torch.load(args.weights_init, map_location=args.device)
        predictor.model.load_state_dict(state, strict=False)
        print(f"Loaded initial weights: {args.weights_init}")

    set_training_mode(
        predictor,
        train_prompt_encoder=train_prompt_encoder,
        train_mask_decoder=train_mask_decoder,
        freeze_image_encoder=freeze_image_encoder,
    )

    # Optimizer over trainable params only
    trainable_params = [p for p in predictor.model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters. Check your --no_train_* / freeze flags.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
    )

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    # ---------- Helpers for eval ----------
    def _read_item(ent: dict, max_side: int):
        """
        Reads image+mask and chooses a prompt point inside GT mask (distance-transform interior).
        Expects ent like {"image": Path/str, "annotation": Path/str, ...}
        Returns: image_rgb uint8 (H,W,3), gt uint8 (H,W) in {0,1}, point float32 (1,1,2)
        """
        import cv2

        img_path = ent.get("image", None)
        ann_path = ent.get("annotation", None)
        if img_path is None or ann_path is None:
            return None, None, None

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None, None, None
        img_rgb = bgr[..., ::-1]

        m = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None, None, None
        gt = (m > 0).astype(np.uint8)

        img_rgb, gt = _resize_longest_side(img_rgb, gt, max_side=max_side)
        pt = pick_prompt_point_inside_mask(gt)
        if pt is None:
            return None, None, None
        return img_rgb, gt, pt

    @torch.no_grad()
    def _predict_selected_mask(predictor_obj, image_rgb: np.ndarray, point_xy: np.ndarray):
        """
        Deployable selection: pick the mask with highest predicted SAM score among 3 candidates.
        """
        input_label = np.ones((1, 1), dtype=np.int64)

        with torch.amp.autocast("cuda", enabled=is_cuda):
            predictor_obj.set_image(image_rgb)

            _mask_input, unnorm_coords, labels, _ = predictor_obj._prep_prompts(
                point_xy, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0:
                h, w = image_rgb.shape[:2]
                return np.zeros((h, w), np.uint8)

            sparse_embeddings, dense_embeddings = predictor_obj.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )

            high_res_features = [
                feat_level[-1].unsqueeze(0) for feat_level in predictor_obj._features["high_res_feats"]
            ]

            low_res_masks, prd_scores, _, _ = predictor_obj.model.sam_mask_decoder(
                image_embeddings=predictor_obj._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor_obj.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            prd_masks = predictor_obj._transforms.postprocess_masks(low_res_masks, predictor_obj._orig_hw[-1])  # (1,3,H,W)
            scores = prd_scores[0] if prd_scores.ndim == 2 else prd_scores
            k = int(torch.argmax(scores).item())
            prob = torch.sigmoid(prd_masks[0, k]).detach().float().cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)

        return pred

    def _iou_binary(gt: np.ndarray, pred: np.ndarray) -> float:
        gt_b = gt.astype(bool)
        pr_b = pred.astype(bool)
        inter = np.logical_and(gt_b, pr_b).sum()
        union = np.logical_or(gt_b, pr_b).sum()
        return _safe_div(float(inter), float(union))

    @torch.no_grad()
    def eval_mean_iou(entries, max_items: int):
        if len(entries) == 0:
            return 0.0

        rng = np.random.RandomState(args.seed)
        idx = np.arange(len(entries))
        rng.shuffle(idx)
        idx = idx[: min(max_items, len(idx))]

        was_training = predictor.model.training
        predictor.model.eval()

        ious = []
        for i in idx:
            img_rgb, gt, pt = _read_item(entries[i], max_side=args.max_side)
            if img_rgb is None:
                continue
            pred = _predict_selected_mask(predictor, img_rgb, pt)
            ious.append(_iou_binary(gt, pred))

        predictor.model.train(was_training)
        return float(np.mean(ious)) if len(ious) else 0.0

    # Visualize first batch
    if args.visualize_first_batch:
        _img, _mask, _pts, _n = read_batch_fn(train_data, visualize_data=True, max_side=args.max_side, seed=args.seed)

    # ---------- Training loop ----------
    mean_iou = 0.0
    optimizer.zero_grad(set_to_none=True)

    best_val_iou = -1.0
    best_ckpt = out_dir / "fine_tuned_sam2_best.torch"
    no_improve_evals = 0

    for step in range(1, args.steps + 1):
        image, mask, input_point, num_masks = read_batch_fn(
            train_data, visualize_data=False, max_side=args.max_side
        )
        if image is None or mask is None or num_masks == 0:
            continue

        # one positive label per prompt point
        input_label = np.ones((num_masks, 1), dtype=np.int64)
        if input_point.size == 0 or input_label.size == 0:
            continue

        with torch.amp.autocast("cuda", enabled=is_cuda):
            predictor.set_image(image)

            _mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0:
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )

            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # (N,3,H,W) logits

            # ---- GT: (H,W) -> (N,H,W) ----
            gt_np = np.squeeze(mask.astype(np.float32))
            if gt_np.ndim != 2:
                raise ValueError(f"Expected GT mask to be 2D after squeeze, got shape={gt_np.shape}")

            gt = torch.as_tensor(gt_np, device=device)  # (H,W) float
            N = prd_masks.shape[0]
            gt = gt.unsqueeze(0).expand(N, -1, -1)  # (N,H,W)

            # ============================================================
            # Change 1) ORACLE BEST-OF-3 SELECTION (train best candidate)
            # ============================================================
            prd_logits_all = prd_masks                    # (N,3,H,W) logits
            prd_prob_all = torch.sigmoid(prd_logits_all)  # (N,3,H,W)

            # hard IoU per candidate (for selection only)
            prd_bin_all = (prd_prob_all > 0.5).to(gt.dtype)  # (N,3,H,W)
            gt_ = gt.unsqueeze(1)                              # (N,1,H,W)

            inter_k = (prd_bin_all * gt_).sum(dim=(2, 3))       # (N,3)
            union_k = prd_bin_all.sum(dim=(2, 3)) + gt_.sum(dim=(2, 3)) - inter_k
            iou_k = inter_k / (union_k + 1e-6)                  # (N,3)

            k_best = torch.argmax(iou_k, dim=1)                 # (N,)
            idx = torch.arange(N, device=device)

            best_logits = prd_logits_all[idx, k_best]           # (N,H,W)
            best_prob = prd_prob_all[idx, k_best]               # (N,H,W)
            best_iou_hard = iou_k.gather(1, k_best[:, None]).squeeze(1)  # (N,)

            # ============================================================
            # Change 2) SEG LOSS = BCEWithLogits + Soft Dice
            # ============================================================
            bce = F.binary_cross_entropy_with_logits(best_logits, gt, reduction="mean")
            dice = soft_dice_loss(best_prob, gt)
            seg_loss = 0.5 * bce + 0.5 * dice

            # Score calibration aligned to selected candidate
            if prd_scores.ndim == 2:
                best_score = prd_scores.gather(1, k_best[:, None]).squeeze(1)  # (N,)
            else:
                best_score = prd_scores[idx]

            inter_s = (best_prob * gt).sum(dim=(1, 2))
            union_s = best_prob.sum(dim=(1, 2)) + gt.sum(dim=(1, 2)) - inter_s
            iou_soft = inter_s / (union_s + 1e-6)

            score_loss = (best_score - iou_soft.detach()).abs().mean()

            loss = seg_loss + 0.05 * score_loss
            loss = loss / args.accumulation_steps

            # for logging EMA (use best hard IoU)
            iou = best_iou_hard

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        mean_iou = mean_iou * 0.99 + 0.01 * float(iou.mean().detach().cpu().numpy())

        if step % 100 == 0:
            if args.eval_every > 0 and (step % args.eval_every == 0):
                val_iou = eval_mean_iou(eval_data, max_items=args.eval_samples)
                print(f"[Eval] step={step} val_mean_iou={val_iou:.4f} (samples<= {args.eval_samples})")

                if val_iou > best_val_iou + 1e-6:
                    best_val_iou = val_iou
                    no_improve_evals = 0
                    torch.save(predictor.model.state_dict(), best_ckpt)
                    print(f"[Eval] New best checkpoint saved: {best_ckpt} (val_mean_iou={best_val_iou:.4f})")
                else:
                    no_improve_evals += 1
                    if args.early_stop_patience > 0 and no_improve_evals >= args.early_stop_patience:
                        print(f"[Eval] Early stop triggered (no improvement for {no_improve_evals} evals).")
                        break

            lr = float(optimizer.param_groups[0]["lr"])
            print(f"Step {step}: IoU_EMA={mean_iou:.4f} lr={lr:.2e}")

        if step % args.save_every == 0:
            ckpt = out_dir / f"fine_tuned_sam2_{step}.torch"
            torch.save(predictor.model.state_dict(), ckpt)
            print(f"Saved checkpoint: {ckpt}")

    # Save final + pick best for test
    final_ckpt = out_dir / "fine_tuned_sam2_final.torch"
    torch.save(predictor.model.state_dict(), final_ckpt)
    print(f"Done. Final checkpoint: {final_ckpt}")

    ckpt_for_test = best_ckpt if (args.use_best_for_test and best_ckpt.exists()) else final_ckpt
    print(f"[Eval] Using checkpoint for TEST evaluation: {ckpt_for_test}")

    # ============================
    # Test-set evaluation section
    # ============================
    def _accum_counts(gt: np.ndarray, pred: np.ndarray):
        gt_b = gt.astype(bool)
        pr_b = pred.astype(bool)
        tp = int(np.logical_and(pr_b, gt_b).sum())
        fp = int(np.logical_and(pr_b, ~gt_b).sum())
        fn = int(np.logical_and(~pr_b, gt_b).sum())
        return tp, fp, fn

    def _counts_to_metrics(tp: int, fp: int, fn: int):
        iou_ = _safe_div(tp, tp + fp + fn)
        dice_ = _safe_div(2 * tp, 2 * tp + fp + fn)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        return {"iou": iou_, "dice": dice_, "precision": prec, "recall": rec}

    @torch.no_grad()
    def _predict_prob_selected(predictor_obj, image_rgb: np.ndarray, point_xy: np.ndarray, thresh: float = 0.5):
        input_label = np.ones((1, 1), dtype=np.int64)

        with torch.amp.autocast("cuda", enabled=is_cuda):
            predictor_obj.set_image(image_rgb)

            _mask_input, unnorm_coords, labels, _ = predictor_obj._prep_prompts(
                point_xy, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0:
                h, w = image_rgb.shape[:2]
                return np.zeros((h, w), np.uint8), np.zeros((h, w), np.float32), 0, 0.0

            sparse_embeddings, dense_embeddings = predictor_obj.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )

            high_res_features = [
                feat_level[-1].unsqueeze(0) for feat_level in predictor_obj._features["high_res_feats"]
            ]

            low_res_masks, prd_scores, _, _ = predictor_obj.model.sam_mask_decoder(
                image_embeddings=predictor_obj._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor_obj.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            prd_masks = predictor_obj._transforms.postprocess_masks(low_res_masks, predictor_obj._orig_hw[-1])  # (1,3,H,W)

            scores = prd_scores[0] if prd_scores.ndim == 2 else prd_scores
            selected_k = int(torch.argmax(scores).item())
            selected_score = float(scores[selected_k].detach().cpu().item())

            prob = torch.sigmoid(prd_masks[0, selected_k]).detach().float().cpu().numpy()
            pred_bin = (prob > thresh).astype(np.uint8)

        return pred_bin, prob.astype(np.float32), selected_k, selected_score

    print("\n[Eval] Building predictors for zero-shot baseline and fine-tuned model...")
    zs_predictor = build_predictor(args.model_cfg, args.sam2_checkpoint, device=args.device)

    ft_predictor = build_predictor(args.model_cfg, args.sam2_checkpoint, device=args.device)
    ft_state = torch.load(ckpt_for_test, map_location=args.device)
    ft_predictor.model.load_state_dict(ft_state, strict=False)

    zs_predictor.model.eval()
    ft_predictor.model.eval()

    print(
        "\n[Eval] Mask selection rule:\n"
        "  SAM2 returns 3 candidate masks per prompt point (multimask_output=True).\n"
        "  We select the mask with the highest predicted SAM score among the 3.\n"
        "  Rationale: this score is available at deployment time (no GT) and is SAM’s native quality/confidence signal.\n"
    )

    class_order = ["scratch", "oil", "stain"]
    stats = {
        "zero_shot": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0}),
        "fine_tuned": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0}),
    }

    per_item = []
    cached_samples = []

    def _sync_if_cuda():
        if is_cuda:
            torch.cuda.synchronize()

    print(f"[Eval] Running on test set: n={len(test_data)} (max_side={args.max_side})")

    for ent in test_data:
        cls = infer_category(ent)
        img_rgb, gt, pt = _read_item(ent, max_side=args.max_side)
        if img_rgb is None or gt is None or pt is None:
            continue

        if len(cached_samples) < 32:
            cached_samples.append((img_rgb, pt))

        zs_pred, _, _, _ = _predict_prob_selected(zs_predictor, img_rgb, pt)
        tp, fp, fn = _accum_counts(gt, zs_pred)
        stats["zero_shot"][cls]["tp"] += tp
        stats["zero_shot"][cls]["fp"] += fp
        stats["zero_shot"][cls]["fn"] += fn
        stats["zero_shot"][cls]["n"] += 1
        zs_metrics = _counts_to_metrics(tp, fp, fn)

        ft_pred, _, _, _ = _predict_prob_selected(ft_predictor, img_rgb, pt)
        tp, fp, fn = _accum_counts(gt, ft_pred)
        stats["fine_tuned"][cls]["tp"] += tp
        stats["fine_tuned"][cls]["fp"] += fp
        stats["fine_tuned"][cls]["fn"] += fn
        stats["fine_tuned"][cls]["n"] += 1
        ft_metrics = _counts_to_metrics(tp, fp, fn)

        per_item.append(
            {"cls": cls, "img": img_rgb, "gt": gt, "zs_pred": zs_pred, "ft_pred": ft_pred,
             "zs_iou": zs_metrics["iou"], "ft_iou": ft_metrics["iou"], "image_path": str(ent.get("image", ""))}
        )

    def _summarize_model(model_name: str, cls: str):
        d = stats[model_name][cls]
        return _counts_to_metrics(d["tp"], d["fp"], d["fn"]), d["n"]

    rows = []
    all_classes = list(dict.fromkeys(class_order + sorted({r["cls"] for r in per_item})))
    for cls in all_classes:
        n_zs = stats["zero_shot"][cls]["n"]
        n_ft = stats["fine_tuned"][cls]["n"]
        if (n_zs + n_ft) == 0:
            continue
        m_zs, n_zs = _summarize_model("zero_shot", cls)
        m_ft, n_ft = _summarize_model("fine_tuned", cls)
        rows.append(
            {
                "class": cls,
                "n": max(n_zs, n_ft),
                "zs_iou": m_zs["iou"],
                "zs_dice": m_zs["dice"],
                "zs_precision": m_zs["precision"],
                "zs_recall": m_zs["recall"],
                "ft_iou": m_ft["iou"],
                "ft_dice": m_ft["dice"],
                "ft_precision": m_ft["precision"],
                "ft_recall": m_ft["recall"],
                "delta_iou": m_ft["iou"] - m_zs["iou"],
                "delta_dice": m_ft["dice"] - m_zs["dice"],
            }
        )

    metrics_csv =  'SAM_results/test_metrics_per_class.csv'
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["class"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    print("\n[Eval] Per-class metrics (zero-shot vs fine-tuned):")
    header = ["class", "n", "zs_iou", "zs_dice", "zs_precision", "zs_recall",
              "ft_iou", "ft_dice", "ft_precision", "ft_recall", "delta_iou"]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header)) + 5))
    for r in rows:
        print(" | ".join(_fmt(r[h]) for h in header))
    print(f"\n[Eval] Saved per-class CSV: {metrics_csv}")

    # Visual grid (optional)
    vis_path =  "SAM_results/test_visual_grid.png"
    try:
        import matplotlib.pyplot as plt

        # choose best per class + worst overall
        selected_rows = []
        for cls in class_order:
            candidates = [r for r in per_item if r["cls"] == cls]
            if candidates:
                selected_rows.append(max(candidates, key=lambda x: x["ft_iou"]))

        failure = min(per_item, key=lambda x: x["ft_iou"]) if per_item else None
        grid_rows = selected_rows[:3] + ([failure] if failure is not None else [])

        nrows = len(grid_rows)
        if nrows > 0:
            fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(16, 4 * nrows))
            if nrows == 1:
                axes = np.expand_dims(axes, axis=0)

            for i, r in enumerate(grid_rows):
                img, gt, zs, ft = r["img"], r["gt"], r["zs_pred"], r["ft_pred"]
                axes[i, 0].imshow(img); axes[i, 0].set_title(f"Input\n{r['cls']}"); axes[i, 0].axis("off")
                axes[i, 1].imshow(gt, cmap="gray"); axes[i, 1].set_title("GT"); axes[i, 1].axis("off")
                axes[i, 2].imshow(zs, cmap="gray"); axes[i, 2].set_title(f"Zero-shot\nIoU={r['zs_iou']:.3f}"); axes[i, 2].axis("off")
                suffix = " (FAILURE)" if (failure is not None and r is failure) else ""
                axes[i, 3].imshow(ft, cmap="gray"); axes[i, 3].set_title(f"Fine-tuned\nIoU={r['ft_iou']:.3f}{suffix}"); axes[i, 3].axis("off")

            plt.tight_layout()
            fig.savefig(vis_path, dpi=200)
            plt.close(fig)
            print(f"[Eval] Saved visual grid: {vis_path}")
    except Exception as e:
        print(f"[Eval] Warning: could not save visual grid due to: {e}")

    # Latency benchmark
    def _benchmark_latency(predictor_obj, samples, n_warmup=3, n_eval=20):
        if len(samples) == 0:
            return None
        n_eval = min(n_eval, len(samples))

        for i in range(min(n_warmup, len(samples))):
            img, pt = samples[i]
            _predict_prob_selected(predictor_obj, img, pt)
        _sync_if_cuda()

        times_ms = []
        for i in range(n_eval):
            img, pt = samples[i]
            _sync_if_cuda()
            t0 = time.perf_counter()
            _predict_prob_selected(predictor_obj, img, pt)
            _sync_if_cuda()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        times_ms = np.array(times_ms, dtype=np.float32)
        return {
            "mean_ms": float(times_ms.mean()),
            "p50_ms": float(np.percentile(times_ms, 50)),
            "p95_ms": float(np.percentile(times_ms, 95)),
            "n": int(n_eval),
        }

    lat = _benchmark_latency(ft_predictor, cached_samples, n_warmup=3, n_eval=20)
    if lat is not None:
        fps = _safe_div(1000.0, lat["p50_ms"])
        print(
            "\n[Eval] Deployment + latency paragraph:\n"
            f"  Target deployment: automated inspection on a tester/inspection workstation ({args.device}). "
            f"Per-image latency (single positive point prompt) ~{lat['p50_ms']:.1f} ms p50 "
            f"(~{lat['p95_ms']:.1f} ms p95, n={lat['n']}), ~{fps:.1f} FPS at p50.\n"
        )


if __name__ == "__main__":
    main()
