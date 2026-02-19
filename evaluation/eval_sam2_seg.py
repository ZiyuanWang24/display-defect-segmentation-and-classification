# eval_sam2_seg.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASSES = ("oil", "scratch", "stain")


def iter_images(folder: Path) -> List[Path]:
    out = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    out.sort()
    return out


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return bgr[..., ::-1]


def read_gt_mask_binary(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read GT mask: {path}")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return (m > 0).astype(np.uint8)


def resize_to_max_side(img: np.ndarray, max_side: int = 1024, interp=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(max_side / w, max_side / h)
    new_w = max(1, int(round(w * r)))
    new_h = max(1, int(round(h * r)))
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def resize_mask_nearest(mask01: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    m = cv2.resize(mask01.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


# ---- metrics
def iou(pred01: np.ndarray, gt01: np.ndarray) -> float:
    p = pred01.astype(bool)
    g = gt01.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter) / float(union + 1e-6)


def dice(pred01: np.ndarray, gt01: np.ndarray) -> float:
    p = pred01.astype(bool)
    g = gt01.astype(bool)
    inter = np.logical_and(p, g).sum()
    return float(2 * inter) / float(p.sum() + g.sum() + 1e-6)


def pixel_prf(pred01: np.ndarray, gt01: np.ndarray) -> Tuple[float, float, float]:
    p = pred01.astype(bool)
    g = gt01.astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    prec = float(tp) / float(tp + fp + 1e-6)
    rec = float(tp) / float(tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return prec, rec, f1


# ---- auto prompting (gradient points)
def auto_points_from_image(image_rgb: np.ndarray, num_points: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    h, w = gray.shape[:2]
    flat = mag.reshape(-1)
    k = min(num_points, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    rng.shuffle(idx)
    ys = idx // w
    xs = idx % w
    pts = np.stack([xs, ys], axis=1).astype(np.int64)

    # add stable points
    pts = np.concatenate([pts, np.array([[w // 2, h // 2]], dtype=np.int64)], axis=0)
    pts = np.unique(pts, axis=0)
    return pts


def normalize_outputs(masks, scores) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(masks)
    s = np.asarray(scores)
    if m.ndim == 4:
        m = m[:, 0]
    if s.ndim == 2:
        s = s[:, 0]
    if m.ndim != 3:
        raise ValueError(f"Unexpected masks shape: {m.shape}")
    if s.ndim != 1:
        raise ValueError(f"Unexpected scores shape: {s.shape}")
    return m.astype(np.uint8), s.astype(np.float32)


def select_mask_by_sam_score(masks3: np.ndarray, scores1: np.ndarray) -> Tuple[np.ndarray, float]:
    j = int(np.argmax(scores1))
    return (masks3[j] > 0).astype(np.uint8), float(scores1[j])


def select_mask_oracle_best_iou(masks3: np.ndarray, gt01: np.ndarray) -> Tuple[np.ndarray, float]:
    # best IoU across returned masks (upper bound diagnostic)
    best_i = -1
    best_v = -1.0
    for i in range(masks3.shape[0]):
        v = iou((masks3[i] > 0).astype(np.uint8), gt01)
        if v > best_v:
            best_v = v
            best_i = i
    return (masks3[best_i] > 0).astype(np.uint8), float(best_v)


def find_gt_for_image(gt_dir: Path, image_path: Path) -> Optional[Path]:
    stem = image_path.stem
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_finetuned_predictor(model_cfg: str, base_ckpt: str, finetuned_weights: str, device: str) -> SAM2ImagePredictor:
    model = build_sam2(model_cfg, base_ckpt, device=device)
    predictor = SAM2ImagePredictor(model)
    sd = torch.load(finetuned_weights, map_location=device)
    predictor.model.load_state_dict(sd, strict=True)
    predictor.model.eval()
    return predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)  # data/MSD-US/test
    ap.add_argument("--sam2_checkpoint", type=str, required=True)
    ap.add_argument("--model_cfg", type=str, required=True)
    ap.add_argument("--sam_weights", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--num_points", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--oracle_best_iou", action="store_true", help="also compute oracle best-IoU mask across returned masks")
    ap.add_argument("--out_csv", type=str, default="sam2_seg_eval.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    gt_dir = data_dir / "ground_truth"

    predictor = load_finetuned_predictor(args.model_cfg, args.sam2_checkpoint, args.sam_weights, args.device)

    rows = []
    for cls in CLASSES:
        img_dir = data_dir / cls
        imgs = iter_images(img_dir)
        for img_path in tqdm(imgs, desc=f"SAM2 seg eval {cls}"):
            gt_path = find_gt_for_image(gt_dir, img_path)
            if gt_path is None:
                continue

            img = read_rgb(img_path)
            gt = read_gt_mask_binary(gt_path)

            img_rs = resize_to_max_side(img, args.max_side, interp=cv2.INTER_LINEAR)
            gt_rs = resize_mask_nearest(gt, img_rs.shape[:2])

            pts = auto_points_from_image(img_rs, args.num_points, args.seed)

            # IMPORTANT: labels must be (N,), not (N,1)
            labels = np.ones((pts.shape[0],), dtype=np.int64)

            with torch.no_grad():
                predictor.set_image(img_rs)
                masks, scores, _ = predictor.predict(point_coords=pts, point_labels=labels)

            masks3, scores1 = normalize_outputs(masks, scores)
            pred, sam_score = select_mask_by_sam_score(masks3, scores1)

            miou = iou(pred, gt_rs)
            mdice = dice(pred, gt_rs)
            prec, rec, f1 = pixel_prf(pred, gt_rs)

            row = {
                "class": cls,
                "image": str(img_path),
                "gt": str(gt_path),
                "sam_score": sam_score,
                "iou": miou,
                "dice": mdice,
                "px_precision": prec,
                "px_recall": rec,
                "px_f1": f1,
                "pred_area": int(pred.sum()),
                "gt_area": int(gt_rs.sum()),
            }

            if args.oracle_best_iou:
                pred_oracle, best_iou = select_mask_oracle_best_iou(masks3, gt_rs)
                row["oracle_best_iou"] = best_iou
                row["oracle_best_dice"] = dice(pred_oracle, gt_rs)

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved -> {args.out_csv}")

    if len(df) == 0:
        print("No samples evaluated. Check GT naming.")
        return

    print("\n=== Overall (SAM-score selected mask) ===")
    print(df[["iou", "dice", "px_precision", "px_recall", "px_f1"]].mean())

    print("\n=== Per-class ===")
    print(df.groupby("class")[["iou", "dice", "px_precision", "px_recall", "px_f1"]].mean().sort_index())


if __name__ == "__main__":
    main()
