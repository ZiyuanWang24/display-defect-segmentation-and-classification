# train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.utils

from data import load_train_test_data, read_batch
from sam2_utils import build_predictor, set_training_mode


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


def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--data_dir", type=str, default="data/MSD-US/test")
    ap.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "csv", "msd_us"])
    ap.add_argument("--gt_dirname", type=str, default="ground_truth")
    ap.add_argument("--categories", type=str, default="", help="Comma-separated list, e.g. oil,scratch,stain (default: all)")
    ap.add_argument("--test_size", type=float, default=0.2)

    # Model
    ap.add_argument("--sam2_checkpoint", type=str, default="sam_trained_models/sam2_hiera_tiny.pt")
    ap.add_argument("--model_cfg", type=str, default="")  # empty -> infer from ckpt
    ap.add_argument("--weights_init", type=str, default="", help="Optional: start from a fine-tuned .torch state_dict")

    # Train
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    if args.visualize_first_batch:
        _img, _mask, _pts, _n = read_batch(train_data, visualize_data=True, max_side=args.max_side, seed=args.seed)

    mean_iou = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.steps + 1):
        image, mask, input_point, num_masks = read_batch(train_data, visualize_data=False, max_side=args.max_side)
        if image is None or mask is None or num_masks == 0:
            continue

        # one positive label per prompt point
        input_label = np.ones((num_masks, 1), dtype=np.int64)
        if input_point.size == 0 or input_label.size == 0:
            continue

        with torch.amp.autocast("cuda", enabled=is_cuda):
            predictor.set_image(image)

            mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                input_point,
                input_label,
                box=None,
                mask_logits=None,
                normalize_coords=True,
            )
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0:
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels),
                boxes=None,
                masks=None,
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

            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # (N,3,H,W)
            prd_prob = torch.sigmoid(prd_masks[:, 0])  # (N,H,W)

            # GT: input mask is (1,H,W) uint8 -> squeeze -> (H,W) -> expand to (N,H,W)
            gt_np = np.squeeze(mask.astype(np.float32))
            if gt_np.ndim != 2:
                raise ValueError(f"Expected GT mask to be 2D after squeeze, got shape={gt_np.shape}")

            gt = torch.as_tensor(gt_np, device=device)  # (H,W)
            gt = gt.unsqueeze(0).expand(prd_prob.shape[0], -1, -1)  # (N,H,W)

            # Manual BCE (stable)
            seg_loss = (-gt * torch.log(prd_prob + 1e-6) - (1 - gt) * torch.log((1 - prd_prob) + 1e-6)).mean()

            # IoU
            prd_bin = (prd_prob > 0.5).to(gt.dtype)
            inter = (gt * prd_bin).sum(dim=(1, 2))
            union = gt.sum(dim=(1, 2)) + prd_bin.sum(dim=(1, 2)) - inter
            iou = inter / (union + 1e-6)

            # Score calibration loss (small weight)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05
            loss = loss / args.accumulation_steps

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
            lr = float(optimizer.param_groups[0]["lr"])
            print(f"Step {step}: IoU_EMA={mean_iou:.4f} lr={lr:.2e}")

        if step % args.save_every == 0:
            ckpt = out_dir / f"fine_tuned_sam2_{step}.torch"
            torch.save(predictor.model.state_dict(), ckpt)
            print(f"Saved checkpoint: {ckpt}")

    final_ckpt = out_dir / "fine_tuned_sam2_final.torch"
    torch.save(predictor.model.state_dict(), final_ckpt)
    print(f"Done. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
