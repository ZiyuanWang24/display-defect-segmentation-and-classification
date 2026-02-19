# eval_cnn.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

CLASSES = ("oil", "scratch", "stain")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(folder: Path) -> List[Path]:
    out = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    out.sort()
    return out


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return bgr[..., ::-1]


def center_crop_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = img[y0:y0+s, x0:x0+s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def preprocess_for_cnn(img_rgb: np.ndarray, img_size: int) -> torch.Tensor:
    """
    Returns float tensor (1,3,H,W) in [0,1].
    If your CNN expects ImageNet normalization, enable normalize_imagenet below.
    """
    x = center_crop_square(img_rgb, img_size).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    # Optional: ImageNet normalization (uncomment if you trained that way)
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    # std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    # x = (x - mean) / std

    return x


def confusion_and_prf(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, object]:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    # per-class precision/recall/f1
    prec = np.zeros(num_classes, dtype=np.float64)
    rec = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)
    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec[k] = tp / (tp + fp + 1e-9)
        rec[k] = tp / (tp + fn + 1e-9)
        f1[k] = 2 * prec[k] * rec[k] / (prec[k] + rec[k] + 1e-9)

    acc = (y_true == y_pred).mean() if len(y_true) else 0.0
    macro_f1 = f1.mean() if num_classes else 0.0

    return {"cm": cm, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "macro_f1": macro_f1}


def load_model(model_path: str, device: str, num_classes: int):
    """
    You must adapt this to your CNN architecture import.
    E.g. from src.model import MultiInputMobileNetV2 / build_mobilenetv3 ...
    """
    # ---- CHANGE THIS PART to match your model definition ----
    # Example placeholder: a torchvision MobileNetV3-Large head
    from torchvision import models
    m = models.mobilenet_v3_large(weights=None)
    in_feat = m.classifier[-1].in_features
    m.classifier[-1] = torch.nn.Linear(in_feat, num_classes)
    # ---------------------------------------------

    sd = torch.load(model_path, map_location=device)
    # if checkpoint is {"model": state_dict, ...} handle both
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)  # e.g. data/MSD-US/test
    ap.add_argument("--cnn_weights", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device(args.device)
    num_classes = len(CLASSES)
    model = load_model(args.cnn_weights, args.device, num_classes)

    y_true, y_pred = [], []
    for k, cls in enumerate(CLASSES):
        img_dir = Path(args.data_dir) / cls
        imgs = iter_images(img_dir)
        for p in tqdm(imgs, desc=f"CNN eval {cls}"):
            img = read_rgb(p)
            x = preprocess_for_cnn(img, args.img_size).to(device)
            with torch.no_grad():
                logits = model(x)
                pred = int(torch.argmax(logits, dim=1).item())
            y_true.append(k)
            y_pred.append(pred)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    stats = confusion_and_prf(y_true, y_pred, num_classes)

    print("\n=== CNN results ===")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {stats['acc']:.4f}")
    print(f"Macro-F1: {stats['macro_f1']:.4f}\n")

    print("Per-class:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:8s}  P={stats['prec'][i]:.4f}  R={stats['rec'][i]:.4f}  F1={stats['f1'][i]:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(stats["cm"])


if __name__ == "__main__":
    main()
