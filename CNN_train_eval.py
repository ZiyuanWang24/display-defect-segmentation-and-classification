# train_cnn.py
"""
Train MobileNetV2 classifier on defect images/patches.

✅ Always trains ONLY 3 classes: oil, scratch, stain
✅ Ignores extra folders like: ground_truth (and anything else)
✅ Fails only if one of the required 3 folders is missing

NEW:
✅ Stratified split into train / eval / test
✅ Saves CSV manifests for each split (for reproducibility + report)
✅ Saves training_log.csv + test metrics + confusion matrix (+ optional plot)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

EXPECTED = ("oil", "scratch", "stain")


class ThreeClassFolder(Dataset):
    """
    A strict 3-class folder dataset:
      root/oil/**/*.png
      root/scratch/**/*.png
      root/stain/**/*.png

    Any other folder (e.g., ground_truth) is ignored completely.
    """
    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = list(EXPECTED)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = []

        for c in self.classes:
            class_dir = self.root / c
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing required class folder: {class_dir}")

            for dp, _dn, fnames in os.walk(class_dir):
                for fn in fnames:
                    if Path(fn).suffix.lower() in IMG_EXTENSIONS:
                        self.samples.append((str(Path(dp) / fn), self.class_to_idx[c]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root} for classes {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = default_loader(path)  # PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def build_mobilenetv2(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _count_by_class(dataset: ThreeClassFolder, indices: List[int]) -> Dict[str, int]:
    # dataset.samples: (path, y)
    counts = {c: 0 for c in dataset.classes}
    for i in indices:
        _p, y = dataset.samples[i]
        counts[dataset.classes[y]] += 1
    return counts


def stratified_split_indices(
    dataset: ThreeClassFolder,
    eval_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified split over dataset.samples labels.
    Returns (train_idx, eval_idx, test_idx).
    """
    if not (0.0 < eval_split < 1.0) or not (0.0 <= test_split < 1.0):
        raise ValueError("Splits must satisfy: 0<eval_split<1 and 0<=test_split<1")
    if eval_split + test_split >= 1.0:
        raise ValueError("Need eval_split + test_split < 1.0")

    g = torch.Generator().manual_seed(seed)

    # group indices by class
    by_class: Dict[int, List[int]] = {i: [] for i in range(len(dataset.classes))}
    for idx, (_p, y) in enumerate(dataset.samples):
        by_class[y].append(idx)

    train_idx: List[int] = []
    eval_idx: List[int] = []
    test_idx: List[int] = []

    for y, idxs in by_class.items():
        idxs = idxs.copy()
        # shuffle deterministically using torch
        perm = torch.randperm(len(idxs), generator=g).tolist()
        idxs = [idxs[i] for i in perm]

        n = len(idxs)
        if n < 3 and (eval_split > 0 or test_split > 0):
            # with extremely tiny class counts, any 3-way split is unstable
            # still do something sensible: prioritize train, then eval, then test
            n_test = 1 if (test_split > 0 and n >= 3) else 0
            n_eval = 1 if (eval_split > 0 and n - n_test >= 2) else 0
        else:
            n_test = int(round(n * test_split))
            n_eval = int(round(n * eval_split))

        # ensure at least 1 train sample per class
        if n - (n_eval + n_test) < 1:
            # reduce eval/test to keep train >= 1
            overflow = 1 - (n - (n_eval + n_test))
            # pull back from eval first, then test
            take_from_eval = min(n_eval, overflow)
            n_eval -= take_from_eval
            overflow -= take_from_eval
            take_from_test = min(n_test, overflow)
            n_test -= take_from_test

        # slice
        t0 = 0
        t1 = n_test
        t2 = n_test + n_eval

        test_part = idxs[t0:t1]
        eval_part = idxs[t1:t2]
        train_part = idxs[t2:]

        test_idx.extend(test_part)
        eval_idx.extend(eval_part)
        train_idx.extend(train_part)

    # final shuffle across classes for loaders
    def shuffle_list(lst: List[int]) -> List[int]:
        if len(lst) == 0:
            return lst
        perm = torch.randperm(len(lst), generator=g).tolist()
        return [lst[i] for i in perm]

    return shuffle_list(train_idx), shuffle_list(eval_idx), shuffle_list(test_idx)


def save_split_csv(
    dataset: ThreeClassFolder,
    indices: List[int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "class_name"])
        for i in indices:
            p, y = dataset.samples[i]
            w.writerow([p, y, dataset.classes[y]])


@dataclass
class Metrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class: Dict[str, Dict[str, float]]  # precision/recall/f1/support


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[Metrics, List[List[int]]]:
    """
    Returns metrics + confusion matrix (C x C) where rows=true, cols=pred.
    """
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
            cm[yt][yp] += 1

    acc = correct / max(1, total)

    # per-class precision/recall/f1
    per_class: Dict[str, Dict[str, float]] = {}
    supports = []
    precisions = []
    recalls = []
    f1s = []

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
        import numpy as np
    except Exception:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = __import__("numpy").array(cm, dtype=float)
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def benchmark_latency(model: nn.Module, loader: DataLoader, device: torch.device, iters: int = 50) -> Dict[str, float]:
    """
    Simple latency benchmark (ms/image) over a few batches.
    """
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

    # timed
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/MSD-US/test")
    ap.add_argument("--out_dir", type=str, default="cnn_checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--eval_split", type=float, default=0.2, help="fraction used for eval/validation")
    ap.add_argument("--test_split", type=float, default=0.1, help="fraction used for test")
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--plot_cm", action="store_true", help="save confusion_matrix.png")
    ap.add_argument("--benchmark", action="store_true", help="measure simple latency numbers")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = ThreeClassFolder(args.data_dir, transform=tfm)
    classes = list(EXPECTED)
    print("Classes (forced):", classes)
    print("Total samples:", len(ds))

    train_idx, eval_idx, test_idx = stratified_split_indices(
        ds, eval_split=args.eval_split, test_split=args.test_split, seed=args.seed
    )

    # Print split stats
    print("\nSplit sizes:")
    print("  train:", len(train_idx), _count_by_class(ds, train_idx))
    print("  eval :", len(eval_idx), _count_by_class(ds, eval_idx))
    print("  test :", len(test_idx), _count_by_class(ds, test_idx))

    # Save manifests for report + reproducibility
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

    # training log for report
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
            eval_metrics, _cm_eval = evaluate_full(model, eval_loader, device, classes)

            # choose best by macro-F1 (more robust than accuracy if imbalance)
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

    # --- Final test evaluation with best checkpoint ---
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    test_metrics, cm = evaluate_full(model, test_loader, device, classes)

    # Save confusion matrix + metrics for report
    write_cm_csv(cm, classes, out_dir / "confusion_matrix.csv")
    if args.plot_cm:
        try_plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix.png")

    # Save metrics.json and metrics.csv
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

    # Optional: add model size + latency info to a single summary json for report writing
    summary = {
        "data_dir": args.data_dir,
        "classes": classes,
        "img_size": args.img_size,
        "seed": args.seed,
        "splits": {
            "train": {"n": len(train_idx), "counts": _count_by_class(ds, train_idx)},
            "eval":  {"n": len(eval_idx),  "counts": _count_by_class(ds, eval_idx)},
            "test":  {"n": len(test_idx),  "counts": _count_by_class(ds, test_idx)},
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
    print("accuracy      :", f"{test_metrics.accuracy:.4f}")
    print("macro_f1      :", f"{test_metrics.macro_f1:.4f}")
    print("per_class     :", test_metrics.per_class)
    print("\nSaved artifacts to:", out_dir)


if __name__ == "__main__":
    main()
