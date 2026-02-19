# train_cnn.py
"""
Train MobileNetV2 classifier on defect patches (or raw images).

✅ Always trains ONLY 3 classes: oil, scratch, stain
✅ Ignores extra folders like: ground_truth (and anything else)
✅ Fails only if one of the required 3 folders is missing

Typical workflow:
1) Run prepare_cnn_dataset.py to create cnn_data/{oil,scratch,stain}
2) Run this script with --data_dir cnn_data

But you can also point --data_dir at data/MSD-US/test and it will ignore ground_truth.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/MSD-US/test")
    ap.add_argument("--out_dir", type=str, default="cnn_checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
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

    n_val = int(round(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    if n_val <= 0 or n_train <= 0:
        raise ValueError("Dataset too small for requested val_split.")

    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    model = build_mobilenetv2(num_classes=len(classes), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    best_path = out_dir / "mobilenetv2_best.pt"

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

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {ep}/{args.epochs} | train_loss={running/max(1,len(train_loader)):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, best_path)
            print("Saved:", best_path)

    print("Done. Best val_acc:", best_acc)


if __name__ == "__main__":
    main()
