# make_instance_dataset.py
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


@dataclass
class Pair:
    image_path: Path
    mask_path: Path


def find_gt_dirs(data_dir: Path, gt_dirname: str) -> List[Path]:
    """Return all directories named gt_dirname under data_dir."""
    return [p for p in data_dir.rglob(gt_dirname) if p.is_dir()]


def iter_masks_from_gt_dirs(gt_dirs: List[Path]) -> List[Path]:
    masks = []
    for gd in gt_dirs:
        for p in gd.rglob("*"):
            if p.is_file() and p.suffix.lower() in MASK_EXTS:
                masks.append(p)
    masks.sort()
    return masks


def _try_find_image_by_replacing_gt_dir(
    data_dir: Path,
    mask_path: Path,
    gt_dirname: str,
    image_dir_candidates: List[str],
) -> Optional[Path]:
    """
    If mask relative path contains gt_dirname, replace that component with each image_dir_candidate
    and try to find an image with same stem.
    """
    try:
        rel = mask_path.relative_to(data_dir)
    except ValueError:
        return None

    parts = list(rel.parts)
    if gt_dirname not in parts:
        return None

    gt_idx = parts.index(gt_dirname)
    mask_stem = mask_path.stem

    for img_dir in image_dir_candidates:
        parts2 = parts.copy()
        parts2[gt_idx] = img_dir
        # keep same filename stem, but allow any image extension
        candidate_dir = data_dir.joinpath(*parts2[:-1])
        if candidate_dir.exists():
            for ext in IMG_EXTS:
                cand = candidate_dir / f"{mask_stem}{ext}"
                if cand.exists():
                    return cand

            # also try matching any file with same stem
            for p in candidate_dir.glob(f"{mask_stem}.*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    return p

    return None


def _try_find_image_in_sibling_images_dir(
    mask_path: Path,
    gt_dirname: str,
    image_dir_candidates: List[str],
) -> Optional[Path]:
    """
    If mask is under .../<gt_dirname>/..., try sibling dirs .../<images_dir>/...
    with same relative subpath from gt_dir.
    """
    # walk up to find the nearest ancestor named gt_dirname
    cur = mask_path.parent
    gt_root = None
    while cur != cur.parent:
        if cur.name == gt_dirname:
            gt_root = cur
            break
        cur = cur.parent

    if gt_root is None:
        return None

    rel_under_gt = mask_path.relative_to(gt_root)  # e.g. class/foo.png
    mask_stem = mask_path.stem
    parent_of_gt = gt_root.parent

    for img_dir in image_dir_candidates:
        img_root = parent_of_gt / img_dir
        candidate_dir = img_root / rel_under_gt.parent
        if candidate_dir.exists():
            for ext in IMG_EXTS:
                cand = candidate_dir / f"{mask_stem}{ext}"
                if cand.exists():
                    return cand
            for p in candidate_dir.glob(f"{mask_stem}.*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    return p

    return None


def _fallback_search_image_by_stem(data_dir: Path, mask_stem: str, max_hits: int = 20) -> Optional[Path]:
    """
    Slow fallback: search entire data_dir for files with same stem and image extension.
    """
    hits = 0
    for p in data_dir.rglob(f"{mask_stem}.*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return p
        hits += 1
        if hits >= max_hits:
            break
    return None


def find_image_for_mask(
    data_dir: Path,
    mask_path: Path,
    gt_dirname: str,
    image_dir_candidates: List[str],
) -> Optional[Path]:
    """
    Try a few common heuristics to locate the corresponding image file.
    """
    # 1) replace 'ground_truth' component in rel path -> 'images'
    img = _try_find_image_by_replacing_gt_dir(data_dir, mask_path, gt_dirname, image_dir_candidates)
    if img is not None:
        return img

    # 2) find sibling images dir next to gt_dir
    img = _try_find_image_in_sibling_images_dir(mask_path, gt_dirname, image_dir_candidates)
    if img is not None:
        return img

    # 3) fallback: search by stem (limited)
    return _fallback_search_image_by_stem(data_dir, mask_path.stem)


def read_mask_binary(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return (m > 0).astype(np.uint8)


def split_connected_components(
    mask_bin: np.ndarray,
    connectivity: int = 8,
    min_area: int = 10,
) -> List[np.ndarray]:
    """
    Returns list of instance masks (uint8 0/1) for each connected component.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    conn = 8 if connectivity == 8 else 4
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=conn)

    inst_masks: List[np.ndarray] = []
    for lab in range(1, num_labels):  # skip background=0
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        inst = (labels == lab).astype(np.uint8)
        inst_masks.append(inst)

    return inst_masks


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, use_symlink: bool) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def save_mask_png(mask01: np.ndarray, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    # save as 0/255 for viewing + compatibility
    m255 = (mask01.astype(np.uint8) * 255)
    ok = cv2.imwrite(str(out_path), m255)
    if not ok:
        raise RuntimeError(f"Failed to write mask: {out_path}")


def infer_class_from_path(mask_path: Path, gt_dirname: str) -> str:
    """
    Heuristic: return the directory name right under gt_dirname if it exists.
    Example: .../ground_truth/scratch/xxx.png -> 'scratch'
    """
    parts = list(mask_path.parts)
    if gt_dirname in parts:
        i = parts.index(gt_dirname)
        if i + 1 < len(parts):
            return parts[i + 1].lower()
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data", help="Root dataset dir")
    ap.add_argument("--gt_dirname", type=str, default="ground_truth", help="GT folder name")
    ap.add_argument("--out_dir", type=str, default="ground_truth_seperate", help="Output dataset dir")
    ap.add_argument("--images_dir_candidates", type=str, default="images,Images,img,IMG",
                    help="Comma-separated folder names to try for images")
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    ap.add_argument("--min_area", type=int, default=10, help="Filter tiny components (noise)")
    ap.add_argument("--only_split_multiples", action="store_true",
                    help="If set: only create new items for masks with >=2 instances; singletons are skipped")
    ap.add_argument("--duplicate_images", action="store_true",
                    help="If set: copy image file for each instance; otherwise keep one copy per instance anyway (same result).")
    ap.add_argument("--symlink_images", action="store_true",
                    help="Use symlinks instead of copying images (recommended on Linux)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_images = out_dir / "images"
    out_masks = out_dir / "masks"
    ensure_dir(out_images)
    ensure_dir(out_masks)

    image_dir_candidates = [s.strip() for s in args.images_dir_candidates.split(",") if s.strip()]

    gt_dirs = find_gt_dirs(data_dir, args.gt_dirname)
    if not gt_dirs:
        raise RuntimeError(f"Could not find any directories named '{args.gt_dirname}' under {data_dir}")

    mask_paths = iter_masks_from_gt_dirs(gt_dirs)
    if not mask_paths:
        raise RuntimeError(f"No mask files found under '{args.gt_dirname}' dirs in {data_dir}")

    csv_path = out_dir / "instances.csv"
    rows = []

    n_total = 0
    n_written = 0
    n_skipped_no_img = 0
    n_skipped_empty = 0

    for mp in mask_paths:
        n_total += 1

        try:
            mask_bin = read_mask_binary(mp)
        except Exception as e:
            print(f"[WARN] skip unreadable mask: {mp} ({e})")
            continue

        if mask_bin.sum() == 0:
            n_skipped_empty += 1
            continue

        inst_masks = split_connected_components(mask_bin, connectivity=args.connectivity, min_area=args.min_area)
        if len(inst_masks) == 0:
            n_skipped_empty += 1
            continue

        if args.only_split_multiples and len(inst_masks) < 2:
            continue

        img_path = find_image_for_mask(data_dir, mp, args.gt_dirname, image_dir_candidates)
        if img_path is None or not img_path.exists():
            n_skipped_no_img += 1
            continue

        cls = infer_class_from_path(mp, args.gt_dirname)
        stem = mp.stem

        # Write one sample per instance
        for inst_id, inst in enumerate(inst_masks):
            new_stem = f"{stem}__inst{inst_id:03d}"
            # image extension preserved
            out_img = out_images / cls / f"{new_stem}{img_path.suffix.lower()}"
            out_msk = out_masks / cls / f"{new_stem}.png"

            # Always create a one-to-one (image,mask) file pair per instance
            copy_or_link(img_path, out_img, use_symlink=args.symlink_images)
            save_mask_png(inst, out_msk)

            rows.append({
                "class": cls,
                "image": str(out_img.relative_to(out_dir)),
                "mask": str(out_msk.relative_to(out_dir)),
                "src_image": str(img_path),
                "src_mask": str(mp),
                "src_stem": stem,
                "instance_id": inst_id,
                "num_instances_in_src": len(inst_masks),
                "area_px": int(inst.sum()),
            })
            n_written += 1

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "class", "image", "mask",
            "src_image", "src_mask", "src_stem",
            "instance_id", "num_instances_in_src", "area_px"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\nDone.")
    print(f"  scanned masks:        {n_total}")
    print(f"  written instances:    {n_written}")
    print(f"  skipped empty masks:  {n_skipped_empty}")
    print(f"  skipped (no image):   {n_skipped_no_img}")
    print(f"  output dataset:       {out_dir}")
    print(f"  index CSV:            {csv_path}")


if __name__ == "__main__":
    main()
