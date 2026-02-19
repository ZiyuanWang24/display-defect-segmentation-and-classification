# prepare_cnn_dataset.py
"""
Create CNN training patches using fine-tuned SAM2.

Hard-enforces 3 classes: oil, scratch, stain.
Never touches/reads ground_truth for classification.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from sam2_segmenter import load_finetuned_sam2, segment_image_file, crop_defect_patch


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASSES = ("oil", "scratch", "stain")


def iter_images(folder: Path):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--sam2_checkpoint", type=str, required=True)
    ap.add_argument("--model_cfg", type=str, required=True)
    ap.add_argument("--sam_weights", type=str, required=True)

    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--num_points", type=int, default=64)
    ap.add_argument("--score_thr", type=float, default=0.0)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit_per_class", type=int, default=-1)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_finetuned_sam2(
        model_cfg=args.model_cfg,
        base_checkpoint=args.sam2_checkpoint,
        finetuned_weights=args.sam_weights,
        device=args.device,
    )

    for c in CLASSES:
        (out_dir / c).mkdir(parents=True, exist_ok=True)

    for c in CLASSES:
        src = data_dir / c
        if not src.exists():
            raise FileNotFoundError(f"Missing folder: {src}")

        imgs = list(iter_images(src))
        if args.limit_per_class > 0:
            imgs = imgs[: args.limit_per_class]

        print(f"[{c}] {len(imgs)} images")
        for img_path in tqdm(imgs, desc=f"SAM2->{c}"):
            res = segment_image_file(
                str(img_path),
                predictor=predictor,
                max_side=args.max_side,
                num_points=args.num_points,
                seed=0,
                score_thr=args.score_thr,
            )
            patch = crop_defect_patch(res.image_rgb, res.defect_mask, pad=args.pad)
            cv2.imwrite(str(out_dir / c / (img_path.stem + ".png")), patch[..., ::-1])

    print("Done.")


if __name__ == "__main__":
    main()
