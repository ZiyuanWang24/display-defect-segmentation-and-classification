# eval_cnn_patches.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

EXPECTED = ("oil", "scratch", "stain")


class ThreeClassFolder(Dataset):
    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = list(EXPECTED)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = []

        for c in self.classes:
            d = self.root / c
            if not d.exists():
                raise FileNotFoundError(f"Missing required folder: {d}")
            for dp, _dn, fnames in os.walk(d):
                for fn in fnames:
                    if Path(fn).suffix.lower() in IMG_EXTENSIONS:
                        self.samples.append((str(Path(dp) / fn), self.class_to_idx[c]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        x = default_loader(path)  # PIL
        if self.transform is not None:
            x = self.transform(x)
        return x, y, path


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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="cnn_data with oil/scratch/stain subfolders")
    ap.add_argument("--cnn_ckpt", type=str, required=True, help="mobilenetv2_best_3class.pt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out_csv", type=str, default="cnn_patch_eval.csv")
    args = ap.parse_args()

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = ThreeClassFolder(args.data_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(args.device)
    ck = torch.load(args.cnn_ckpt, map_location=device)
    classes = tuple(ck.get("classes", ()))
    if set(classes) != set(EXPECTED) or len(classes) != 3:
        raise ValueError(f"Checkpoint classes are {classes}, expected {EXPECTED}")

    model = build_mobilenetv2(3).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    y_true, y_pred, paths = [], [], []
    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())
        paths.extend(list(p))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, 3)
    acc = float((y_true == y_pred).mean())

    prec, rec, f1 = prf_from_cm(cm)

    print("\n=== CNN on patches ===")
    print("Accuracy:", acc)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    for i, name in enumerate(EXPECTED):
        print(f"{name:8s}  P={prec[i]:.4f}  R={rec[i]:.4f}  F1={f1[i]:.4f}")

    df = pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred})
    df["true_name"] = df["y_true"].map({i: EXPECTED[i] for i in range(3)})
    df["pred_name"] = df["y_pred"].map({i: EXPECTED[i] for i in range(3)})
    df.to_csv(args.out_csv, index=False)
    print("Saved ->", args.out_csv)


if __name__ == "__main__":
    main()
