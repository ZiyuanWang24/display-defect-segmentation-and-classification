# sam2_segmenter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch

from sam2_utils import build_predictor


@dataclass
class SegmentationResult:
    image_rgb: np.ndarray          # resized image used for prediction (H,W,3)
    defect_mask: np.ndarray        # uint8 binary (H,W) {0,1}
    seg_map: np.ndarray            # uint16 instance map (H,W) 0=bg, 1..K
    scores: np.ndarray             # float scores per selected instance mask


def resize_to_max_side(img: np.ndarray, max_side: int = 1024) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    r = min(max_side / w, max_side / h)
    new_w = max(1, int(round(w * r)))
    new_h = max(1, int(round(h * r)))
    img2 = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img2, r


def auto_points_from_image(
    image_rgb: np.ndarray,
    num_points: int = 48,
    seed: int = 0,
    border: int = 10,
) -> np.ndarray:
    """
    Heuristic: pick points on high-gradient locations (likely scratches/stains/edges).
    Returns (N,2) int64 in (x,y).
    """
    rng = np.random.default_rng(seed)
    h, w = image_rgb.shape[:2]
    if h < 2 * border + 1 or w < 2 * border + 1:
        border = 0

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Suppress borders
    if border > 0:
        mag[:border, :] = 0
        mag[-border:, :] = 0
        mag[:, :border] = 0
        mag[:, -border:] = 0

    flat = mag.reshape(-1)
    k = min(num_points, flat.size)
    if k <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    idx = np.argpartition(flat, -k)[-k:]
    rng.shuffle(idx)
    ys = idx // w
    xs = idx % w

    pts = np.stack([xs, ys], axis=1).astype(np.int64)

    extra = np.array(
        [
            [w // 2, h // 2],
            [w // 4, h // 4],
            [3 * w // 4, h // 4],
            [w // 4, 3 * h // 4],
            [3 * w // 4, 3 * h // 4],
        ],
        dtype=np.int64,
    )
    pts = np.concatenate([pts, extra], axis=0)
    pts = np.unique(pts, axis=0)
    return pts


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-6)


def _select_nonoverlap_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    overlap_iou: float = 0.15,
    max_instances: int = 20,
) -> Tuple[List[np.ndarray], List[float]]:
    order = np.argsort(scores)[::-1]
    selected: List[np.ndarray] = []
    selected_scores: List[float] = []
    for i in order:
        m = masks[i]
        if m.sum() == 0:
            continue
        ok = True
        for sm in selected:
            if _mask_iou(m, sm) > overlap_iou:
                ok = False
                break
        if ok:
            selected.append(m)
            selected_scores.append(float(scores[i]))
        if len(selected) >= max_instances:
            break
    return selected, selected_scores


def postprocess_binary(mask01: np.ndarray, min_area: int = 50) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() == 0:
        return m

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[lab == i] = 1
    return out.astype(np.uint8)


def predict_defect_mask(
    predictor,
    image_rgb: np.ndarray,
    num_points: int = 64,
    seed: int = 0,
    score_thr: float = 0.0,
    overlap_iou: float = 0.15,
    max_instances: int = 20,
) -> SegmentationResult:
    predictor.set_image(image_rgb)

    pts = auto_points_from_image(image_rgb, num_points=num_points, seed=seed)
    if pts.shape[0] == 0:
        h, w = image_rgb.shape[:2]
        pts = np.array([[w // 2, h // 2]], dtype=np.int64)

    labels = np.ones((pts.shape[0],), dtype=np.int64)  # (N,)

    masks, scores, _logits = predictor.predict(point_coords=pts, point_labels=labels)

    m = np.asarray(masks)
    s = np.asarray(scores)

    # masks: (N,H,W) or (N,1,H,W)
    if m.ndim == 4:
        m = m[:, 0]
    elif m.ndim != 3:
        raise ValueError(f"Unexpected masks shape: {m.shape}")

    # scores: (N,) or (N,1)
    if s.ndim == 2:
        s = s[:, 0]
    elif s.ndim != 1:
        raise ValueError(f"Unexpected scores shape: {s.shape}")

    m = m.astype(np.uint8)
    s = s.astype(np.float32)

    keep = s >= score_thr
    m = m[keep]
    s = s[keep]
    h, w = image_rgb.shape[:2]
    if m.shape[0] == 0:
        return SegmentationResult(image_rgb=image_rgb,
                                  defect_mask=np.zeros((h, w), dtype=np.uint8),
                                  seg_map=np.zeros((h, w), dtype=np.uint16),
                                  scores=np.zeros((0,), dtype=np.float32))

    selected_masks, selected_scores = _select_nonoverlap_masks(
        m, s, overlap_iou=overlap_iou, max_instances=max_instances
    )

    seg_map = np.zeros((h, w), dtype=np.uint16)
    defect = np.zeros((h, w), dtype=np.uint8)

    for idx, sm in enumerate(selected_masks, start=1):
        sm_bool = sm.astype(bool)
        seg_map[sm_bool] = idx
        defect[sm_bool] = 1

    defect = postprocess_binary(defect, min_area=max(20, int(0.00005 * h * w)))

    return SegmentationResult(
        image_rgb=image_rgb,
        defect_mask=defect,
        seg_map=seg_map,
        scores=np.asarray(selected_scores, dtype=np.float32),
    )


def mask_to_bbox(mask01: np.ndarray, pad: int = 16) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = mask01.shape[:2]
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h - 1, y1 + pad)
    x1 = min(w - 1, x1 + pad)
    return x0, y0, x1, y1


def crop_defect_patch(
    image_rgb: np.ndarray,
    defect_mask01: np.ndarray,
    pad: int = 16,
    fallback_center_crop: bool = True,
) -> np.ndarray:
    bbox = mask_to_bbox(defect_mask01, pad=pad)
    h, w = image_rgb.shape[:2]
    if bbox is None:
        if not fallback_center_crop:
            return image_rgb
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return image_rgb[y0:y0+side, x0:x0+side]
    x0, y0, x1, y1 = bbox
    return image_rgb[y0:y1+1, x0:x1+1]


def load_finetuned_sam2(
    model_cfg: str,
    base_checkpoint: str,
    finetuned_weights: str,
    device: str = "cuda",
):
    predictor = build_predictor(model_cfg, base_checkpoint, device=device)
    sd = torch.load(finetuned_weights, map_location=device)
    predictor.model.load_state_dict(sd, strict=True)
    predictor.model.eval()
    return predictor


def segment_image_file(
    image_path: str,
    predictor,
    max_side: int = 1024,
    num_points: int = 64,
    seed: int = 0,
    score_thr: float = 0.0,
) -> SegmentationResult:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = bgr[..., ::-1]
    rgb, _ = resize_to_max_side(rgb, max_side=max_side)
    return predict_defect_mask(
        predictor=predictor,
        image_rgb=rgb,
        num_points=num_points,
        seed=seed,
        score_thr=score_thr,
    )
