# train_cnn.py
"""
Train MobileNetV2 classifier on your "ground_truth_seperate" dataset.

✅ Your actual structure (IMPORTANT: group folder includes ".png" in its name):
  data/ground_truth_seperate/
    images/
      oil_0001.png/
        Oil_0001__inst000.jpg
        Oil_0001__inst001.jpg
        ...
      scr_0363.png/
        Scr_0363__inst000.jpg
        ...
      sta_0067.png/
        Sta_0067__inst000.jpg
        ...
    masks/
      oil_0001.png/
        Oil_0001__inst000.png
        Oil_0001__inst001.png
        ...
      scr_0363.png/
        Scr_0363__inst000.jpg   # yes, mask can be .jpg in your data
        ...
      sta_0067.png/
        Sta_0067__inst000.jpg   # yes, mask can be .jpg in your data
        ...

Pairing rule:
  For each mask at masks/<group>/<file>, find image at images/<group>/<same file>.
  If exact path doesn't exist, try swapping extension, then try case-insensitive stem match.

Label rule:
  Infer class from the GROUP folder name (strip extension first):
    oil_0001.png -> "oil"
    scr_0363.png -> "scratch"
    sta_0067.png -> "stain"

Optionally crop image by mask bbox (default ON), then resize to img_size.

Outputs:
  out_dir/
    splits_train.csv / splits_eval.csv / splits_test.csv
    training_log.csv
    mobilenetv2_best.pt
    confusion_matrix.csv (+ optional confusion_matrix.png)
    test_metrics.json / test_metrics.csv
    summary_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS


# ----------------------------
# constants / mapping
# ----------------------------
EXPECTED_CANON = ("oil", "scratch", "stain")
ALIAS = {
    "oil": "oil",
    "scratch": "scratch",
    "scr": "scratch",
    "stain": "stain",
    "sta": "stain",
}

IMG_EXTS = tuple(sorted(set([e.lower() for e in IMG_EXTENSIONS] + [".bmp", ".tif", ".tiff"])))


# ----------------------------
# model
# ----------------------------
def build_mobilenetv2(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------
# utils: class + pairing
# ----------------------------
def infer_class_from_group_folder(group_folder: str) -> Optional[str]:
    """
    group_folder examples (YOUR DATA):
      oil_0001.png
      scr_0363.png
      sta_0067.png

    We must strip the ".png" because it's part of the folder name.
    """
    g = Path(group_folder).stem.lower()  # -> "oil_0001", "scr_0363", "sta_0067"
    prefix = g.split("_", 1)[0]         # -> "oil" / "scr" / "sta" / ...

    if prefix in ALIAS:
        return ALIAS[prefix]

    # fallback: substring match
    for k, v in ALIAS.items():
        if k in g:
            return v
    return None


def find_matching_image(images_root: Path, rel_mask_path: Path) -> Optional[Path]:
    """
    Given rel path under masks/, find corresponding image under images/.
    Tries:
      1) exact same relative path
      2) swap extension (same stem)
      3) case-insensitive stem match in same folder
    """
    p = images_root / rel_mask_path
    if p.exists():
        return p

    # try swap extension (same stem)
    stem = p.with_suffix("")  # remove suffix
    for ext in IMG_EXTS:
        cand = Path(str(stem) + ext)
        if cand.exists():
            return cand

    # case-insensitive stem match inside the directory
    parent = p.parent
    if parent.exists():
        want_stem = p.stem.lower()
        for f in parent.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in IMG_EXTS:
                continue
            if f.stem.lower() == want_stem:
                return f

    return None


def read_mask_binary(mask_path: Path) -> Optional[np.ndarray]:
    """
    Masks can be .png or .jpg in your data.
    For jpg masks, threshold at 127 is safer than >0.
    """
    import cv2

    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 127).astype(np.uint8)


def mask_bbox(mask01: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def crop_with_pad(img_rgb: np.ndarray, bbox: Tuple[int, int, int, int], pad_ratio: float) -> np.ndarray:
    """
    img_rgb: HxWx3 RGB
    bbox: (x0,y0,x1,y1)
    """
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


# ----------------------------
# dataset
# ----------------------------
class ThreeClassSeparatedPairs(Dataset):
    """
    root/
      images/<group_folder>/*.(jpg/png/..)
      masks/<group_folder>/*.(png/jpg/..)

    group_folder includes ".png" in its NAME (your current dataset).
    Label is inferred from group_folder (strip ext, then prefix).
    """

    def __init__(
        self,
        root: str | Path,
        images_dirname: str = "images",
        masks_dirname: str = "masks",
        transform=None,
        crop_by_mask: bool = True,
        pad_ratio: float = 0.25,
        min_mask_area: int = 5,
    ):
        self.root = Path(root)
        self.images_root = self.root / images_dirname
        self.masks_root = self.root / masks_dirname
        self.transform = transform

        self.crop_by_mask = bool(crop_by_mask)
        self.pad_ratio = float(pad_ratio)
        self.min_mask_area = int(min_mask_area)

        if not self.images_root.exists():
            raise FileNotFoundError(f"Missing images folder: {self.images_root}")
        if not self.masks_root.exists():
            raise FileNotFoundError(f"Missing masks folder: {self.masks_root}")

        self.classes = list(EXPECTED_CANON)  # canonical outputs for CNN
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Dict] = []

        self._index_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No samples found. Check folder structure and pairing.\n"
                f"images_root={self.images_root}\n"
                f"masks_root={self.masks_root}\n"
                "Expected masks like: masks/oil_0001.png/Oil_0001__inst000.png\n"
                "Expected images like: images/oil_0001.png/Oil_0001__inst000.jpg"
            )

    def _index_samples(self):
        mask_files = [
            p for p in self.masks_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        mask_files.sort()

        skip_bad_class = 0
        skip_no_img = 0
        skip_bad_mask = 0
        kept = 0
        class_counts = Counter()

        for mp in mask_files:
            rel = mp.relative_to(self.masks_root)  # group/file
            if len(rel.parts) < 2:
                continue

            group_folder = rel.parts[0]  # e.g. "oil_0001.png"
            cls = infer_class_from_group_folder(group_folder)
            if cls is None or cls not in self.class_to_idx:
                skip_bad_class += 1
                continue

            imgp = find_matching_image(self.images_root, rel)
            if imgp is None:
                skip_no_img += 1
                continue

            m01 = read_mask_binary(mp)
            if m01 is None:
                skip_bad_mask += 1
                continue
            area = int(m01.sum())
            if area < self.min_mask_area:
                skip_bad_mask += 1
                continue

            self.samples.append(
                {
                    "image_path": str(imgp),
                    "mask_path": str(mp),
                    "y": int(self.class_to_idx[cls]),
                    "class_name": cls,
                    "rel": str(rel),
                    "mask_area": area,
                }
            )
            kept += 1
            class_counts[cls] += 1

        print(f"[Index] masks found     : {len(mask_files)}")
        print(f"[Index] kept samples   : {kept}")
        print(f"[Index] skip_bad_class : {skip_bad_class}")
        print(f"[Index] skip_no_img    : {skip_no_img}")
        print(f"[Index] skip_bad_mask  : {skip_bad_mask}")
        print(f"[Index] class_counts   : {dict(class_counts)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        import cv2
        from PIL import Image

        s = self.samples[idx]
        img_path = Path(s["image_path"])
        mask_path = Path(s["mask_path"])

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb = bgr[..., ::-1]

        if self.crop_by_mask:
            m01 = read_mask_binary(mask_path)
            if m01 is not None:
                bbox = mask_bbox(m01)
                if bbox is not None:
                    img_rgb = crop_with_pad(img_rgb, bbox, self.pad_ratio)

        pil = Image.fromarray(img_rgb)
        x = self.transform(pil) if self.transform is not None else pil
        y = int(s["y"])
        return x, y


# ----------------------------
# splitting + manifests
# ----------------------------
def count_by_class(ds: ThreeClassSeparatedPairs, indices: List[int]) -> Dict[str, int]:
    counts = {c: 0 for c in ds.classes}
    for i in indices:
        y = ds.samples[i]["y"]
        counts[ds.classes[int(y)]] += 1
    return counts


def stratified_split_indices(
    labels: List[int],
    eval_split: float,
    test_split: float,
    seed: int,
    num_classes: int,
) -> Tuple[List[int], List[int], List[int]]:
    if not (0.0 < eval_split < 1.0) or not (0.0 <= test_split < 1.0):
        raise ValueError("Splits must satisfy: 0<eval_split<1 and 0<=test_split<1")
    if eval_split + test_split >= 1.0:
        raise ValueError("Need eval_split + test_split < 1.0")

    g = torch.Generator().manual_seed(seed)

    by_class: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    for idx, y in enumerate(labels):
        by_class[int(y)].append(idx)

    train_idx: List[int] = []
    eval_idx: List[int] = []
    test_idx: List[int] = []

    for y, idxs in by_class.items():
        idxs = idxs.copy()
        perm = torch.randperm(len(idxs), generator=g).tolist()
        idxs = [idxs[i] for i in perm]

        n = len(idxs)
        n_test = int(round(n * test_split))
        n_eval = int(round(n * eval_split))

        # keep at least 1 train if possible
        if n - (n_eval + n_test) < 1 and n > 0:
            overflow = 1 - (n - (n_eval + n_test))
            take_from_eval = min(n_eval, overflow)
            n_eval -= take_from_eval
            overflow -= take_from_eval
            take_from_test = min(n_test, overflow)
            n_test -= take_from_test

        test_part = idxs[:n_test]
        eval_part = idxs[n_test : n_test + n_eval]
        train_part = idxs[n_test + n_eval :]

        test_idx.extend(test_part)
        eval_idx.extend(eval_part)
        train_idx.extend(train_part)

    def shuffle_list(lst: List[int]) -> List[int]:
        if len(lst) == 0:
            return lst
        perm = torch.randperm(len(lst), generator=g).tolist()
        return [lst[i] for i in perm]

    return shuffle_list(train_idx), shuffle_list(eval_idx), shuffle_list(test_idx)


def save_split_csv(ds: ThreeClassSeparatedPairs, indices: List[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "mask_path", "label", "class_name", "rel", "mask_area"])
        for i in indices:
            s = ds.samples[i]
            w.writerow([s["image_path"], s["mask_path"], s["y"], s["class_name"], s["rel"], s["mask_area"]])


# ----------------------------
# metrics
# ----------------------------
@dataclass
class Metrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class: Dict[str, Dict[str, float]]


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[Metrics, List[List[int]]]:
    model.eval()
    C = len(class_names)
    cm = [[0 for _ in range(C)] for _ in range(C)]
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()
        for yt, yp in zip(y.tolist(), pred.tolist()):
            cm[int(yt)][int(yp)] += 1

    acc = correct / max(1, total)

    per_class: Dict[str, Dict[str, float]] = {}
    supports, precisions, recalls, f1s = [], [], [], []

    for k, name in enumerate(class_names):
        tp = cm[k][k]
        fp = sum(cm[i][k] for i in range(C) if i != k)
        fn = sum(cm[k][j] for j in range(C) if j != k)
        support = sum(cm[k][j] for j in range(C))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        supports.append(support)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = float(sum(precisions) / C)
    macro_recall = float(sum(recalls) / C)
    macro_f1 = float(sum(f1s) / C)

    total_support = sum(supports) if supports else 1
    weighted_precision = float(sum(p * s for p, s in zip(precisions, supports)) / total_support)
    weighted_recall = float(sum(r * s for r, s in zip(recalls, supports)) / total_support)
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_support)

    return Metrics(
        accuracy=float(acc),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        per_class=per_class,
    ), cm


def write_cm_csv(cm: List[List[int]], class_names: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for i, row in enumerate(cm):
            w.writerow([class_names[i]] + row)


def try_plot_confusion_matrix(cm: List[List[int]], class_names: List[str], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    arr = np.array(cm, dtype=float)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(arr)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def benchmark_latency(model: nn.Module, loader: DataLoader, device: torch.device, iters: int = 50) -> Dict[str, float]:
    model.eval()
    times = []
    n_images = 0

    # warmup
    for i, (x, _y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 3:
            break

    for i, (x, _y) in enumerate(loader):
        if i >= iters:
            break
        x = x.to(device, non_blocking=True)
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        n_images += x.shape[0]

    total_s = sum(times) if times else 1e-9
    ms_per_image = (total_s / max(1, n_images)) * 1000.0
    imgs_per_s = max(1e-9, n_images / total_s)
    return {"ms_per_image": float(ms_per_image), "images_per_second": float(imgs_per_s)}


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/ground_truth_seperate")
    ap.add_argument("--images_dirname", type=str, default="images")
    ap.add_argument("--masks_dirname", type=str, default="masks")

    ap.add_argument("--out_dir", type=str, default="PIPELINE_results")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--eval_split", type=float, default=0.2)
    ap.add_argument("--test_split", type=float, default=0.1)
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--no_crop_by_mask", action="store_true", help="Disable mask cropping")
    ap.add_argument("--pad_ratio", type=float, default=0.25)
    ap.add_argument("--min_mask_area", type=int, default=5)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--plot_cm", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    crop_by_mask = not args.no_crop_by_mask

    ds = ThreeClassSeparatedPairs(
        root=args.data_dir,
        images_dirname=args.images_dirname,
        masks_dirname=args.masks_dirname,
        transform=tfm,
        crop_by_mask=crop_by_mask,
        pad_ratio=args.pad_ratio,
        min_mask_area=args.min_mask_area,
    )

    classes = list(EXPECTED_CANON)
    print("Classes (forced):", classes)
    print("Total samples:", len(ds))

    labels = [int(s["y"]) for s in ds.samples]
    train_idx, eval_idx, test_idx = stratified_split_indices(
        labels=labels,
        eval_split=args.eval_split,
        test_split=args.test_split,
        seed=args.seed,
        num_classes=len(classes),
    )

    print("\nSplit sizes:")
    print("  train:", len(train_idx), count_by_class(ds, train_idx))
    print("  eval :", len(eval_idx),  count_by_class(ds, eval_idx))
    print("  test :", len(test_idx),  count_by_class(ds, test_idx))

    save_split_csv(ds, train_idx, out_dir / "splits_train.csv")
    save_split_csv(ds, eval_idx,  out_dir / "splits_eval.csv")
    save_split_csv(ds, test_idx,  out_dir / "splits_test.csv")

    train_ds = Subset(ds, train_idx)
    eval_ds = Subset(ds, eval_idx)
    test_ds = Subset(ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    model = build_mobilenetv2(num_classes=len(classes), pretrained=not args.no_pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_path = out_dir / "mobilenetv2_best.pt"

    log_path = out_dir / "training_log.csv"
    with log_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "eval_acc", "eval_macro_f1"])

        for ep in range(1, args.epochs + 1):
            model.train()
            running = 0.0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                loss = criterion(logits, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.item())

            train_loss = running / max(1, len(train_loader))
            eval_metrics, _ = evaluate_full(model, eval_loader, device, classes)

            score = eval_metrics.macro_f1
            print(
                f"Epoch {ep}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"eval_acc={eval_metrics.accuracy:.4f} | "
                f"eval_macro_f1={eval_metrics.macro_f1:.4f}"
            )

            w.writerow([ep, f"{train_loss:.6f}", f"{eval_metrics.accuracy:.6f}", f"{eval_metrics.macro_f1:.6f}"])
            f.flush()

            if score > best_score:
                best_score = score
                torch.save({"state_dict": model.state_dict(), "classes": classes}, best_path)
                print("Saved:", best_path)

    # test with best
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    test_metrics, cm = evaluate_full(model, test_loader, device, classes)

    write_cm_csv(cm, classes, out_dir / "confusion_matrix.csv")
    if args.plot_cm:
        try_plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix.png")

    metrics_dict = asdict(test_metrics)
    with (out_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics_dict, f, indent=2)

    with (out_dir / "test_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["accuracy", test_metrics.accuracy])
        w.writerow(["macro_precision", test_metrics.macro_precision])
        w.writerow(["macro_recall", test_metrics.macro_recall])
        w.writerow(["macro_f1", test_metrics.macro_f1])
        w.writerow(["weighted_precision", test_metrics.weighted_precision])
        w.writerow(["weighted_recall", test_metrics.weighted_recall])
        w.writerow(["weighted_f1", test_metrics.weighted_f1])
        w.writerow([])
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for c in classes:
            d = test_metrics.per_class[c]
            w.writerow([c, d["precision"], d["recall"], d["f1"], d["support"]])

    summary = {
        "data_dir": args.data_dir,
        "images_dirname": args.images_dirname,
        "masks_dirname": args.masks_dirname,
        "classes": classes,
        "img_size": args.img_size,
        "seed": args.seed,
        "crop_by_mask": crop_by_mask,
        "pad_ratio": args.pad_ratio,
        "min_mask_area": args.min_mask_area,
        "splits": {
            "train": {"n": len(train_idx), "counts": count_by_class(ds, train_idx)},
            "eval":  {"n": len(eval_idx),  "counts": count_by_class(ds, eval_idx)},
            "test":  {"n": len(test_idx),  "counts": count_by_class(ds, test_idx)},
        },
        "best_checkpoint": str(best_path),
        "best_selection_metric": "eval_macro_f1",
        "test_metrics": metrics_dict,
        "model_params": count_parameters(model),
    }

    if args.benchmark:
        summary["latency"] = benchmark_latency(model, test_loader, device)

    with (out_dir / "summary_report.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Test results (best checkpoint) ===")
    print("accuracy :", f"{test_metrics.accuracy:.4f}")
    print("macro_f1 :", f"{test_metrics.macro_f1:.4f}")
    print("per_class:", test_metrics.per_class)
    print("\nSaved artifacts to:", out_dir)


if __name__ == "__main__":
    main()
