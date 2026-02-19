# sam2_utils.py
from __future__ import annotations

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_predictor(model_cfg: str, checkpoint: str, device: str = "cuda") -> SAM2ImagePredictor:
    """Build SAM2 model + predictor."""
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


def _set_requires_grad(module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def set_training_mode(
    predictor: SAM2ImagePredictor,
    train_prompt_encoder: bool = True,
    train_mask_decoder: bool = True,
    freeze_image_encoder: bool = True,
) -> None:
    """
    Configure which SAM2 submodules are trainable.

    Typical fine-tuning recipe:
      - Train prompt encoder: optional
      - Train mask decoder: yes
      - Freeze image encoder: yes (saves memory + avoids overfitting)
    """
    model = predictor.model

    # Default: put everything in eval mode, then selectively enable train mode.
    model.train(False)

    # --- Image encoder ---
    # Different SAM2 builds may name this differently; handle common cases.
    img_enc = None
    for name in ("image_encoder", "sam_image_encoder", "vision_encoder"):
        if hasattr(model, name):
            img_enc = getattr(model, name)
            break

    if img_enc is not None:
        if freeze_image_encoder:
            img_enc.train(False)
            _set_requires_grad(img_enc, False)
        else:
            img_enc.train(True)
            _set_requires_grad(img_enc, True)

    # --- Prompt encoder ---
    model.sam_prompt_encoder.train(bool(train_prompt_encoder))
    _set_requires_grad(model.sam_prompt_encoder, bool(train_prompt_encoder))

    # --- Mask decoder ---
    model.sam_mask_decoder.train(bool(train_mask_decoder))
    _set_requires_grad(model.sam_mask_decoder, bool(train_mask_decoder))

    # Safety: if you froze image encoder, keep other non-target parts frozen too
    # (If your model has extra necks/adapters, they stay in eval unless you enable them explicitly.)
