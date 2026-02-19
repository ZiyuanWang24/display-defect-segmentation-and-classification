# eval_end2end.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms, models

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
        raise FileNotFoundError(str(path))
    return bgr[..., ::-1]


def resize_to_max_side(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(max_side / w, max_side / h)
    nw = max(1, int(round(w * r)))
    nh = max(1, int(round(h * r)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)


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
    pts = np.unique(np.concatenate([pts, np.array([[w // 2, h // 2]], dtype=np.int64)], axis=0), axis=0)
    return pts


def normalize_outputs(masks, scores) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(masks)
    s = np.asarray(scores)
    if m.ndim == 4:
        m = m[:, 0]
    if s.ndim == 2:
        s = s[:, 0]
    return m.astype(np.uint8), s.astype(np.float32)


def crop_from_mask(image_rgb: np.ndarray, mask01: np.ndarray, pad: int = 16) -> np.ndarray:
    ys, xs = np.where(mask01 > 0)
    h, w = mask01.shape[:2]
    if ys.size == 0:
        # fallback center crop
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return image_rgb[y0:y0+side, x0:x0+side]

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(h - 1, y1 + pad); x1 = min(w - 1, x1 + pad)
    return image_rgb[y0:y1+1, x0:x1+1]


def build_mobilenetv2(num_classes: int) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def prf_from_cm(cm: np.ndarray):
    k = cm.shape[0]
    prec, rec, f1 = [], [], []
    for i in range(k):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        prec.append(p); rec.append(r); f1.append(f)
    return np.array(prec), np.array(rec), np.array(f1)


def load_finetuned_sam2(model_cfg: str, base_ckpt: str, sam_weights: str, device: str) -> SAM2ImagePredictor:
    model = build_sam2(model_cfg, base_ckpt, device=device)
    predictor = SAM2ImagePredictor(model)
    sd = torch.load(sam_weights, map_location=device)
    predictor.model.load_state_dict(sd, strict=True)
    predictor.model.eval()
    return predictor


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)

    ap.add_argument("--sam2_checkpoint", type=str, required=True)
    ap.add_argument("--model_cfg", type=str, required=True)
    ap.add_argument("--sam_weights", type=str, required=True)

    ap.add_argument("--cnn_ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--num_points", type=int, default=64)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_csv", type=str, default="end2end_eval.csv")
    args = ap.parse_args()

    device = torch.device(args.device)

    # SAM2
    predictor = load_finetuned_sam2(args.model_cfg, args.sam2_checkpoint, args.sam_weights, args.device)

    # CNN
    ck = torch.load(args.cnn_ckpt, map_location=device)
    classes = tuple(ck.get("classes", ()))
    if set(classes) != set(CLASSES) or len(classes) != 3:
        raise ValueError(f"CNN checkpoint classes are {classes}, expected {CLASSES}")

    cnn = build_mobilenetv2(3).to(device)
    cnn.load_state_dict(ck["state_dict"], strict=True)
    cnn.eval()

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    y_true, y_pred, paths = [], [], []
    for true_idx, cls in enumerate(CLASSES):
        img_dir = Path(args.data_dir) / cls
        imgs = iter_images(img_dir)
        for img_path in tqdm(imgs, desc=f"End2End {cls}"):
            img = read_rgb(img_path)
            img_rs = resize_to_max_side(img, args.max_side)

            pts = auto_points_from_image(img_rs, args.num_points, args.seed)
            labels = np.ones((pts.shape[0],), dtype=np.int64)  # (N,) !!!

            predictor.set_image(img_rs)
            masks, scores, _ = predictor.predict(point_coords=pts, point_labels=labels)
            masks3, scores1 = normalize_outputs(masks, scores)

            j = int(np.argmax(scores1))
            pred_mask01 = (masks3[j] > 0).astype(np.uint8)

            patch = crop_from_mask(img_rs, pred_mask01, pad=args.pad)
            x = tfm(patch).unsqueeze(0).to(device)

            logits = cnn(x)
            pred_idx = int(logits.argmax(dim=1).item())

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            paths.append(str(img_path))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, 3)
    acc = float((y_true == y_pred).mean())
    prec, rec, f1 = prf_from_cm(cm)

    print("\n=== End-to-end classification (SAM2->crop->CNN) ===")
    print("Accuracy:", acc)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    for i, name in enumerate(CLASSES):
        print(f"{name:8s}  P={prec[i]:.4f}  R={rec[i]:.4f}  F1={f1[i]:.4f}")

    df = pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred})
    df["true_name"] = df["y_true"].map({i: CLASSES[i] for i in range(3)})
    df["pred_name"] = df["y_pred"].map({i: CLASSES[i] for i in range(3)})
    df.to_csv(args.out_csv, index=False)
    print("Saved ->", args.out_csv)


if __name__ == "__main__":
    main()
