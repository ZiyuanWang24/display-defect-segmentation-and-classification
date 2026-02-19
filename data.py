# data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Iterable

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split

PathLike = Union[str, Path]


# -----------------------------
# CSV (legacy) dataset support
# -----------------------------
@dataclass(frozen=True)
class DatasetPaths:
    data_dir: Path
    images_dir: Path
    masks_dir: Path
    train_csv: Path


def resolve_dataset_paths(data_dir: PathLike) -> DatasetPaths:
    """
    CSV layout (legacy):
      data_dir/
        train.csv
        images/images/
        masks/masks/
    """
    data_dir = Path(data_dir)
    return DatasetPaths(
        data_dir=data_dir,
        images_dir=data_dir / "images" / "images",
        masks_dir=data_dir / "masks" / "masks",
        train_csv=data_dir / "train.csv",
    )


# -----------------------------
# MSD-US folder dataset support
# -----------------------------
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in _IMG_EXTS


def _iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if _is_image(p):
            yield p


def _candidate_suffixes(prefer: str) -> List[str]:
    """Put preferred suffix first, then common mask suffixes."""
    prefer = prefer.lower()
    out = [prefer] if prefer else []
    for s in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        if s not in out:
            out.append(s)
    return out


def _try_with_suffixes(path_no_suffix: Path, suffixes: List[str]) -> Optional[Path]:
    for suf in suffixes:
        cand = path_no_suffix.with_suffix(suf)
        if cand.exists():
            return cand
    return None


def _find_mask_for_image(
    img_path: Path,
    category_root: Path,
    gt_root: Path,
    category_name: str,
) -> Optional[Path]:
    """
    Robust pairing between an image in a defect folder and a ground-truth mask.

    We try (in order):
      1) Mirror relative path: gt_root/category_name/<relative_path_under_category_root> with various suffixes
      2) gt_root/category_name/<img.name> (various suffixes)
      3) gt_root/<img.name> (various suffixes)
      4) Any mask anywhere under gt_root with matching stem: gt_root/**/<img.stem>.*
    """
    rel = None
    try:
        rel = img_path.relative_to(category_root)
    except Exception:
        rel = img_path.name  # fallback

    # Prefer mask suffix = image suffix (sometimes masks saved with same ext), else .png first.
    suffixes = _candidate_suffixes(img_path.suffix)

    # (1) Mirror relative path under gt_root/category_name/
    if isinstance(rel, Path):
        base = (gt_root / category_name / rel).with_suffix("")
        cand = _try_with_suffixes(base, suffixes)
        if cand is not None:
            return cand

    # (2) Flat under gt_root/category_name/
    base = (gt_root / category_name / img_path.name).with_suffix("")
    cand = _try_with_suffixes(base, suffixes)
    if cand is not None:
        return cand

    # (3) Flat under gt_root/
    base = (gt_root / img_path.name).with_suffix("")
    cand = _try_with_suffixes(base, suffixes)
    if cand is not None:
        return cand

    # (4) Any match by stem under gt_root/**
    hits = list(gt_root.rglob(img_path.stem + ".*"))
    hits = [h for h in hits if h.is_file() and h.suffix.lower() in _MASK_EXTS]
    if len(hits) > 0:
        # Prefer same suffix if possible, else pick smallest path length (often the intended one)
        hits.sort(key=lambda p: (p.suffix.lower() != img_path.suffix.lower(), len(str(p))))
        return hits[0]

    return None


def build_msd_us_pairs(
    data_dir: PathLike,
    gt_dirname: str = "ground_truth",
    categories: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[Dict[str, str]]:
    """
    Build a list of {image, annotation, category} dicts from:
      data_dir/
        ground_truth/
        oil/
        scratch/
        stain/
        ...

    Any folder except ground_truth is treated as an image category.
    """
    data_dir = Path(data_dir)
    gt_root = data_dir / gt_dirname
    if not gt_root.exists():
        raise FileNotFoundError(f"Ground-truth folder not found: {gt_root}")

    all_subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    cat_dirs = [p for p in all_subdirs if p.name != gt_dirname]
    if categories is not None:
        wanted = set([c.strip() for c in categories if c.strip()])
        cat_dirs = [p for p in cat_dirs if p.name in wanted]

    pairs: List[Dict[str, str]] = []
    missing = 0
    total = 0

    for cat_dir in sorted(cat_dirs, key=lambda p: p.name):
        category = cat_dir.name
        for img_path in _iter_images(cat_dir):
            total += 1
            mask_path = _find_mask_for_image(
                img_path=img_path,
                category_root=cat_dir,
                gt_root=gt_root,
                category_name=category,
            )
            if mask_path is None:
                missing += 1
                continue
            pairs.append(
                {
                    "image": str(img_path),
                    "annotation": str(mask_path),
                    "category": category,
                }
            )

    if verbose:
        print(f"[MSD-US] Found pairs={len(pairs)} (missing masks for {missing}/{total} images)")

    if len(pairs) == 0:
        raise RuntimeError(
            f"No (image, mask) pairs found under: {data_dir}\n"
            f"Expected categories like oil/scratch/stain and GT in {gt_root}."
        )
    return pairs


# -----------------------------
# Unified API
# -----------------------------
def load_train_test_data(
    data_dir: PathLike,
    test_size: float = 0.2,
    random_state: int = 42,
    # CSV args
    image_col: str = "ImageId",
    mask_col: str = "MaskId",
    # MSD-US args
    dataset_format: str = "auto",  # auto | csv | msd_us
    gt_dirname: str = "ground_truth",
    categories: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Returns train_data, test_data. Each item is a dict containing:
      {"image": "...", "annotation": "...", "category": "..."}  (category may be absent for CSV datasets)

    dataset_format:
      - "auto": if data_dir/train.csv exists -> CSV, else -> MSD-US folder layout
      - "csv": enforce CSV layout
      - "msd_us": enforce MSD-US folder layout
    """
    data_dir = Path(data_dir)

    if dataset_format not in {"auto", "csv", "msd_us"}:
        raise ValueError("dataset_format must be one of: auto | csv | msd_us")

    if dataset_format == "auto":
        dataset_format = "csv" if (data_dir / "train.csv").exists() else "msd_us"

    if dataset_format == "csv":
        paths = resolve_dataset_paths(data_dir)
        df = pd.read_csv(paths.train_csv)
        if image_col not in df.columns or mask_col not in df.columns:
            raise ValueError(f"train.csv must include '{image_col}' and '{mask_col}'. Got {list(df.columns)}")

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        def _make_list(split_df: pd.DataFrame) -> List[Dict[str, str]]:
            out: List[Dict[str, str]] = []
            for _, row in split_df.iterrows():
                out.append(
                    {
                        "image": str(paths.images_dir / row[image_col]),
                        "annotation": str(paths.masks_dir / row[mask_col]),
                    }
                )
            return out

        return _make_list(train_df), _make_list(test_df)

    # MSD-US
    pairs = build_msd_us_pairs(data_dir, gt_dirname=gt_dirname, categories=categories, verbose=verbose)

    if test_size <= 0:
        return pairs, []

    # Stratify by category if possible (helps keep oil/scratch/stain balanced)
    y = [p.get("category", "unknown") for p in pairs]
    stratify = y if len(set(y)) > 1 else None

    train_list, test_list = train_test_split(
        pairs,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return list(train_list), list(test_list)


# -----------------------------
# IO + batch sampling utilities
# -----------------------------
def read_image(image_path: PathLike, mask_path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read image as RGB uint8 (H,W,3) and mask as grayscale uint8 (H,W).
    """
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    ann = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if ann is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    img_rgb = img_bgr[..., ::-1]
    return img_rgb, ann


def resize_to_max_side(
    img: np.ndarray,
    ann: np.ndarray,
    max_side: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Scale so that max(H,W) ~= max_side.
    Returns resized img, resized ann, and scale factor r.
    """
    h, w = img.shape[:2]
    r = min(max_side / w, max_side / h)
    new_w = max(1, int(round(w * r)))
    new_h = max(1, int(round(h * r)))

    img2 = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ann2 = cv2.resize(ann, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img2, ann2, r


def get_points(mask: np.ndarray, num_samples_per_instance: int = 30, seed: Optional[int] = None) -> np.ndarray:
    """
    Sample positive points (x,y) from the GT mask.
    If mask has multiple instance labels, sample from each label and concat.
    Returns: (N,2) in (x,y)
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(mask)
    ids = ids[ids != 0]

    pts: List[List[int]] = []
    if len(ids) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    for lab in ids:
        coords = np.argwhere(mask == lab)  # (y,x)
        if coords.shape[0] == 0:
            continue
        k = min(num_samples_per_instance, coords.shape[0])
        idx = rng.choice(coords.shape[0], size=k, replace=(coords.shape[0] < k))
        yx = coords[idx]
        xy = np.stack([yx[:, 1], yx[:, 0]], axis=1)
        pts.extend(xy.tolist())

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pts, dtype=np.int64).reshape(-1, 2)


def read_batch(
    data: List[Dict[str, str]],
    visualize_data: bool = False,
    max_side: int = 1024,
    seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Randomly sample one training example from `data`.

    Returns:
      image: RGB uint8 (H,W,3)
      binary_mask_chw: uint8 (1,H,W) with {0,1}
      points_batched: int64 (N,1,2) in (x,y)  (one positive point per instance label)
      num_masks: number of unique labels in ann_map excluding 0
    """
    if len(data) == 0:
        return None, None, None, 0

    rng = np.random.default_rng(seed)

    ent = data[int(rng.integers(0, len(data)))]
    try:
        image, ann_map = read_image(ent["image"], ent["annotation"])
    except Exception as e:
        print(f"Error reading: {e}")
        return None, None, None, 0

    # resize_to_max_side returns (img, ann, scale). We only need img+ann here.
    image, ann_map, _ = resize_to_max_side(image, ann_map, max_side=max_side)

    inds = np.unique(ann_map)
    inds = inds[inds != 0]
    num_masks = int(len(inds))

    binary_mask = (ann_map != 0).astype(np.uint8)
    binary_mask_chw = binary_mask[None, ...].astype(np.uint8)  # (1,H,W)

    if num_masks == 0:
        empty_pts = np.zeros((0, 1, 2), dtype=np.int64)
        return image, binary_mask_chw, empty_pts, 0

    # Erode to avoid boundary points; fallback if erosion wipes mask
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
    coords = np.argwhere(eroded_mask > 0)  # (K,2) in (y,x)
    if coords.shape[0] == 0:
        coords = np.argwhere(binary_mask > 0)
    if coords.shape[0] == 0:
        empty_pts = np.zeros((0, 1, 2), dtype=np.int64)
        return image, binary_mask_chw, empty_pts, 0

    points: List[List[int]] = []
    for _ in inds:
        yx = coords[int(rng.integers(0, coords.shape[0]))]
        points.append([int(yx[1]), int(yx[0])])  # (x,y)

    points_arr = np.asarray(points, dtype=np.int64).reshape(-1, 2) if len(points) else np.zeros((0, 2), dtype=np.int64)

    if visualize_data:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("GT mask (binary)")
        plt.imshow(binary_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Mask + sampled points")
        plt.imshow(binary_mask, cmap="gray")
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, pt in enumerate(points_arr):
            plt.scatter(pt[0], pt[1], c=colors[i % len(colors)], s=80)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    points_batched = points_arr[:, None, :]  # (N,1,2)
    return image, binary_mask_chw, points_batched, num_masks
