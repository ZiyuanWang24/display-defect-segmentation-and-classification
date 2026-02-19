# pipeline_predict.py
"""
End-to-end inference:
  image -> fine-tuned SAM2 -> defect patch -> MobileNetV2 -> class prediction

Hard-enforces 3 classes: oil, scratch, stain.
If the CNN checkpoint contains anything else, it raises an error.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

from sam2_segmenter import load_finetuned_sam2, segment_image_file, crop_defect_patch


EXPECTED = ("oil", "scratch", "stain")


def build_mobilenetv2(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)

    ap.add_argument("--sam2_checkpoint", type=str, required=True)
    ap.add_argument("--model_cfg", type=str, required=True)
    ap.add_argument("--sam_weights", type=str, required=True)

    ap.add_argument("--cnn_ckpt", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--num_points", type=int, default=64)
    ap.add_argument("--score_thr", type=float, default=0.0)
    ap.add_argument("--pad", type=int, default=16)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)

    predictor = load_finetuned_sam2(
        model_cfg=args.model_cfg,
        base_checkpoint=args.sam2_checkpoint,
        finetuned_weights=args.sam_weights,
        device=args.device,
    )

    res = segment_image_file(
        args.image,
        predictor=predictor,
        max_side=args.max_side,
        num_points=args.num_points,
        seed=0,
        score_thr=args.score_thr,
    )
    patch = crop_defect_patch(res.image_rgb, res.defect_mask, pad=args.pad)

    ck = torch.load(args.cnn_ckpt, map_location=device)
    classes = tuple(ck.get("classes", ()))
    if set(classes) != set(EXPECTED) or len(classes) != len(EXPECTED):
        raise ValueError(f"CNN checkpoint classes are {classes}, expected exactly {EXPECTED}. "
                         f"Re-train with train_cnn.py from this 3-class pipeline.")

    classes = EXPECTED  # stable order
    model = build_mobilenetv2(num_classes=len(classes)).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    x = tfm(patch).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    idx = int(probs.argmax())

    print("Prediction:", classes[idx], "confidence:", float(probs[idx]))
    print("Probs:", {classes[i]: float(probs[i]) for i in range(len(classes))})

    if args.show:
        overlay = res.image_rgb.copy()
        m = res.defect_mask.astype(bool)
        overlay[m] = (0.7 * overlay[m] + 0.3 * np.array([255, 0, 0], dtype=np.uint8)).astype(np.uint8)

        cv2.imshow("resized_image_overlay", overlay[..., ::-1])
        cv2.imshow("defect_patch", patch[..., ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
