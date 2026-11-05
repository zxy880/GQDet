"""Standalone mathematical kernels for original QCQT contributions (CC BY 4.0).

These tensor interfaces are newly separated from the detector training loop.
They are not a complete replacement for the external D-FINE criterion.
"""
import torch
import torch.nn.functional as F


def qcc_classification(logits, positive_mask, matched_iou, negative_gamma, normalizer, auxiliary_mask=None):
    """Sum weighted BCE / normalizer. Masks and IoUs broadcast to logits.

    Use the same GT normalizer as the corresponding external loss branch.
    auxiliary_mask marks query positions removed from negative supervision.
    """
    positive = positive_mask.to(logits.dtype)
    quality = matched_iou.detach().to(logits.dtype)
    soft_target = positive * quality
    strength = soft_target + (1 - positive) * logits.detach().sigmoid().pow(negative_gamma)
    if auxiliary_mask is not None:
        strength = strength.masked_fill(auxiliary_mask, 0)
        soft_target = soft_target.masked_fill(auxiliary_mask, 0)
    return (F.binary_cross_entropy_with_logits(logits, soft_target, reduction="none") * strength).sum() / normalizer


def rgqs_auxiliary_losses(selected_logits, current_iou, auxiliary_weights, predicted_boxes,
                         target_boxes, matched_giou, target_gamma, normalizer):
    """Single-class selected-query losses before coefficient/schedule weighting.

    Boxes use normalized cxcywh; matched_giou is computed externally for each
    selected pair. normalizer is the batch GT normalizer, not auxiliary count.
    """
    weights = auxiliary_weights.detach()
    soft_target = current_iou.detach().pow(target_gamma)
    bce = F.binary_cross_entropy_with_logits(selected_logits.reshape(-1), soft_target, reduction="none")
    classification = (bce * soft_target.clamp_min(1e-6) * weights).sum() / normalizer
    localization = ((predicted_boxes - target_boxes).abs().sum(-1) * weights).sum() / normalizer
    giou = ((1 - matched_giou) * weights).sum() / normalizer
    return {"loss_brgqa_cls": classification, "loss_brgqa_bbox": localization, "loss_brgqa_giou": giou}
