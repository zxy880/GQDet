"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import math

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dfine_utils import bbox2distance


def _assert_finite_losses(losses, context):
    """Fail fast; loss sanitization would hide an invalid training step."""
    for name, value in losses.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"{context}: non-finite {name}")
    return losses


@register()
class DFINECriterion(nn.Module):
    """This class computes the loss for D-FINE."""

    __share__ = [
        "num_classes",
    ]
    __inject__ = [
        "matcher",
    ]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        # Kept separate from ``gamma``: TrainFix only changes MAL's detached
        # negative modulation and must not alter matching/quality targets.
        self.mal_gamma = float(gamma)
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None

    def set_mal_gamma(self, value):
        self.mal_gamma = float(value)

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        """DEIM-style MAL-only classification loss.

        This keeps D-FINE matching and regression/local losses untouched. The
        matched class target is still IoU-quality aware, while negative samples
        are modulated with the model's detached probability as in focal-style
        dense DETR classification. ``gamma`` is configured separately for MAL.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = src_logits.sigmoid().detach()
        neg_weight = pred_score.pow(self.mal_gamma) * (1 - target)
        pos_weight = target_score
        weight = neg_weight + pos_weight

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_mal": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and "is_dn" in outputs:
                    self.fgl_targets_dn = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )
                if self.fgl_targets is None and "is_dn" not in outputs:
                    self.fgl_targets = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )

            ious = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]), box_cxcywh_to_xyxy(target_boxes)
                )[0]
            )
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            if "teacher_corners" in outputs:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
                if torch.equal(pred_corners, target_corners):
                    losses["loss_ddf"] = pred_corners.sum() * 0
                else:
                    weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype
                    )
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                    )

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    if "is_dn" not in outputs:
                        batch_scale = (
                            8 / outputs["pred_boxes"].shape[0]
                        )  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (
                            (mask.sum() * batch_scale) ** 0.5,
                            ((~mask).sum() * batch_scale) ** 0.5,
                        )
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "mal": self.loss_labels_mal,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)["indices"]
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for i, aux_outputs in enumerate(outputs["aux_outputs"] + [outputs["pre_outputs"]]):
                indices_aux = self.matcher(aux_outputs, targets)["indices"]
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                indices_enc = self.matcher(aux_outputs, targets)["indices"]
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert "aux_outputs" in outputs, ""

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            indices_in = indices_go if loss in ["boxes", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[i]
                    num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_in, num_boxes_in, **meta
                    )

                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer.
        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {
                    k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                }
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    indices_in = indices_go if loss == "boxes" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "boxes" else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses. For dfine
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer.
            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Never mask invalid values: the engine captures the failing batch and
        # aborts so an invalid assignment cannot silently corrupt EMA.
        return _assert_finite_losses(losses, "DFINECriterion")

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == "iou":
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
            )
            iou = torch.diag(iou)
        elif self.boxes_weight_format == "giou":
            iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
                )
            )
        else:
            raise AttributeError()

        if loss in ("boxes",):
            meta = {"boxes_weight": iou}
        elif loss in ("vfl", "mal"):
            meta = {"values": iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step = 0.5 / (num_layers - 1)
        opt_list = [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list


@register()
class DEIMCriterion(DFINECriterion):
    """MAL-only probe criterion.

    It intentionally reuses D-FINE matching, boxes, local/FGL, DDF, GO union
    matching, auxiliary, encoder, pre-output and DN plumbing. The YAML selects
    ``losses: ['mal', 'boxes', 'local']`` and ``loss_mal`` in ``weight_dict``.
    """

    pass


@register()
class RGQACriterion(DEIMCriterion):
    """Refinement-Guided Quality Assignment for decoder layer 1.

    RGQA is a training-only auxiliary assignment. It adds no model parameters
    and never changes inference. Extra positives are selected from decoder
    layer-1 queries according to their layer-0 -> layer-1 localization
    refinement trajectory, while excluding layer-1 O2O positives and the
    D-FINE union regression/local set.
    """

    def __init__(
        self,
        *args,
        rgqa_enabled=True,
        rgqa_aux_index=1,
        rgqa_refinement_weight=0.25,
        rgqa_confidence_weight=0.40,
        rgqa_topk=4,
        rgqa_small_scale_px=32,
        rgqa_small_iou_thr=0.25,
        rgqa_large_iou_thr=0.35,
        rgqa_allow_empty=True,
        rgqa_exclude_o2o=True,
        rgqa_exclude_indices_go=True,
        rgqa_lambda_max=0.50,
        rgqa_start_epoch=10,
        rgqa_full_epoch=20,
        rgqa_decay_epoch=90,
        rgqa_stop_epoch=96,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rgqa_enabled = bool(rgqa_enabled)
        self.rgqa_aux_index = int(rgqa_aux_index)
        self.rgqa_refinement_weight = float(rgqa_refinement_weight)
        self.rgqa_confidence_weight = float(rgqa_confidence_weight)
        self.rgqa_topk = int(rgqa_topk)
        self.rgqa_small_scale_px = float(rgqa_small_scale_px)
        self.rgqa_small_iou_thr = float(rgqa_small_iou_thr)
        self.rgqa_large_iou_thr = float(rgqa_large_iou_thr)
        self.rgqa_allow_empty = bool(rgqa_allow_empty)
        self.rgqa_exclude_o2o = bool(rgqa_exclude_o2o)
        self.rgqa_exclude_indices_go = bool(rgqa_exclude_indices_go)
        self.rgqa_lambda_max = float(rgqa_lambda_max)
        self.rgqa_start_epoch = int(rgqa_start_epoch)
        self.rgqa_full_epoch = int(rgqa_full_epoch)
        self.rgqa_decay_epoch = int(rgqa_decay_epoch)
        self.rgqa_stop_epoch = int(rgqa_stop_epoch)
        self.rgqa_last_stats = {}

    def _rgqa_lambda(self, epoch):
        if not self.rgqa_enabled or self.rgqa_lambda_max <= 0:
            return 0.0
        epoch = 0 if epoch is None else int(epoch)
        if epoch < self.rgqa_start_epoch:
            return 0.0
        if epoch < self.rgqa_full_epoch:
            denom = max(1, self.rgqa_full_epoch - self.rgqa_start_epoch)
            return self.rgqa_lambda_max * (epoch - self.rgqa_start_epoch) / denom
        if epoch < self.rgqa_decay_epoch:
            return self.rgqa_lambda_max
        if epoch < self.rgqa_stop_epoch:
            denom = max(1, self.rgqa_stop_epoch - self.rgqa_decay_epoch)
            return self.rgqa_lambda_max * (self.rgqa_stop_epoch - epoch) / denom
        return 0.0

    def _safe_src_permutation_idx(self, indices, device):
        if indices is None or sum(len(src) for src, _ in indices) == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty
        return self._get_src_permutation_idx(indices)

    def loss_labels_mal(
        self, outputs, targets, indices, num_boxes, values=None, ignore_indices=None
    ):
        """MAL with optional ignored negative query positions.

        RGQA layer-1 extra positives are removed from the original MAL negative
        set, then supervised separately by ``loss_rgqa_cls``.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = src_logits.sigmoid().detach()
        neg_weight = pred_score.pow(self.mal_gamma) * (1 - target)
        pos_weight = target_score
        weight = neg_weight + pos_weight

        if ignore_indices is not None and len(ignore_indices) == 2 and ignore_indices[0].numel() > 0:
            ignore_mask = torch.zeros(src_logits.shape[:2], dtype=torch.bool, device=src_logits.device)
            ignore_mask[ignore_indices] = True
            weight = weight.masked_fill(ignore_mask.unsqueeze(-1), 0)
            target_score = target_score.masked_fill(ignore_mask.unsqueeze(-1), 0)

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_mal": loss}

    @staticmethod
    def _cat_or_empty(tensors, device):
        tensors = [t for t in tensors if t.numel() > 0]
        if not tensors:
            return torch.zeros(0, dtype=torch.long, device=device)
        return torch.cat(tensors)

    def _indices_query_set(self, indices, batch_size, device):
        sets = [set() for _ in range(batch_size)]
        for b, (src, _) in enumerate(indices):
            if src.numel() > 0:
                sets[b].update(int(x) for x in src.detach().cpu().tolist())
        return sets

    def _build_rgqa_indices(self, outputs, targets, indices_o2o, indices_go):
        device = outputs["pred_boxes"].device
        bs, nq = outputs["pred_boxes"].shape[:2]
        if (
            "aux_outputs" not in outputs
            or len(outputs["aux_outputs"]) <= self.rgqa_aux_index
            or self.rgqa_aux_index < 1
        ):
            empty = [(torch.zeros(0, dtype=torch.long, device=device),
                      torch.zeros(0, dtype=torch.long, device=device)) for _ in range(bs)]
            return empty, torch.zeros(0, device=device), torch.zeros(0, dtype=torch.long, device=device), {}

        aux0 = outputs["aux_outputs"][self.rgqa_aux_index - 1]
        aux1 = outputs["aux_outputs"][self.rgqa_aux_index]
        boxes_l0 = aux0["pred_boxes"].detach()
        boxes_l1 = aux1["pred_boxes"].detach()
        logits_l1 = aux1["pred_logits"].detach()
        conf_l1 = logits_l1.sigmoid()[..., 0]

        o2o_sets = self._indices_query_set(indices_o2o, bs, device)
        go_sets = self._indices_query_set(indices_go, bs, device)

        accepted = [[] for _ in range(bs)]
        candidate_pairs = []
        total_gt = 0
        valid_gt = 0
        k_hist = [0, 0, 0, 0, 0]
        small_valid = 0
        large_valid = 0

        iou0_vals, iou1_vals, delta_vals, conf_vals, score_vals = [], [], [], [], []

        for b, target in enumerate(targets):
            gt_boxes = target["boxes"]
            n_gt = gt_boxes.shape[0]
            total_gt += int(n_gt)
            if n_gt == 0:
                continue
            iou0 = box_iou(box_cxcywh_to_xyxy(boxes_l0[b]), box_cxcywh_to_xyxy(gt_boxes))[0]
            iou1 = box_iou(box_cxcywh_to_xyxy(boxes_l1[b]), box_cxcywh_to_xyxy(gt_boxes))[0]
            delta = iou1 - iou0
            refined = (iou1 + self.rgqa_refinement_weight * delta).clamp(min=0, max=1)
            score = refined - self.rgqa_confidence_weight * conf_l1[b].unsqueeze(1)

            wh = gt_boxes[:, 2:4].clamp(min=1e-9)
            scale_px = torch.sqrt(wh[:, 0] * wh[:, 1]) * 640.0
            thr = torch.where(
                scale_px < self.rgqa_small_scale_px,
                torch.full_like(scale_px, self.rgqa_small_iou_thr),
                torch.full_like(scale_px, self.rgqa_large_iou_thr),
            )

            disallowed = set()
            if self.rgqa_exclude_o2o:
                disallowed.update(o2o_sets[b])
            if self.rgqa_exclude_indices_go:
                disallowed.update(go_sets[b])
            if disallowed:
                disallowed_t = torch.tensor(sorted(disallowed), dtype=torch.long, device=device)
            else:
                disallowed_t = torch.zeros(0, dtype=torch.long, device=device)

            for j in range(n_gt):
                valid = (iou1[:, j] >= thr[j]) & (score[:, j] > 0)
                if disallowed_t.numel() > 0:
                    valid[disallowed_t] = False
                valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
                if valid_idx.numel() == 0:
                    k_hist[0] += 1
                    continue

                valid_scores = score[valid_idx, j]
                topn = min(self.rgqa_topk, int(valid_idx.numel()))
                top_scores, order = torch.topk(valid_scores, k=topn)
                pos_scores = top_scores.clamp(min=0)
                k_gt = min(self.rgqa_topk, int(torch.ceil(pos_scores.sum()).item()))
                k_hist[k_gt] += 1
                if k_gt <= 0:
                    continue
                valid_gt += 1
                if scale_px[j] < self.rgqa_small_scale_px:
                    small_valid += 1
                else:
                    large_valid += 1
                for local_rank in order.tolist():
                    q = int(valid_idx[local_rank].item())
                    s = float(score[q, j].item())
                    candidate_pairs.append((s, b, q, int(j), k_gt))

        # Global greedy assignment: each query at most one GT, each GT up to k_gt.
        candidate_pairs.sort(key=lambda x: x[0], reverse=True)
        used_queries = [set() for _ in range(bs)]
        gt_counts = {}
        gt_limits = {}
        conflict_count = 0
        for s, b, q, j, k_gt in candidate_pairs:
            key = (b, j)
            gt_limits[key] = k_gt
            if q in used_queries[b] or gt_counts.get(key, 0) >= k_gt:
                conflict_count += 1
                continue
            used_queries[b].add(q)
            gt_counts[key] = gt_counts.get(key, 0) + 1
            accepted[b].append((q, j))

        rgqa_indices = []
        for b, pairs in enumerate(accepted):
            if pairs:
                src = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
                tgt = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
            else:
                src = torch.zeros(0, dtype=torch.long, device=device)
                tgt = torch.zeros(0, dtype=torch.long, device=device)
            rgqa_indices.append((src, tgt))

            if src.numel() > 0:
                gt = targets[b]["boxes"][tgt]
                iou0 = box_iou(box_cxcywh_to_xyxy(boxes_l0[b, src]), box_cxcywh_to_xyxy(gt))[0]
                iou1 = box_iou(box_cxcywh_to_xyxy(boxes_l1[b, src]), box_cxcywh_to_xyxy(gt))[0]
                diag0 = torch.diag(iou0).detach()
                diag1 = torch.diag(iou1).detach()
                d = (diag1 - diag0).detach()
                c = conf_l1[b, src].detach()
                refined = (diag1 + self.rgqa_refinement_weight * d).clamp(min=0, max=1)
                sc = (refined - self.rgqa_confidence_weight * c).detach()
                iou0_vals.append(diag0)
                iou1_vals.append(diag1)
                delta_vals.append(d)
                conf_vals.append(c)
                score_vals.append(sc)

        pair_count = sum(len(p) for p in accepted)
        pair_count_t = torch.as_tensor([pair_count], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(pair_count_t)
        num_rgqa = torch.clamp(pair_count_t / get_world_size(), min=1).item()

        all_iou0 = torch.cat(iou0_vals) if iou0_vals else torch.zeros(0, device=device)
        all_iou1 = torch.cat(iou1_vals) if iou1_vals else torch.zeros(0, device=device)
        all_delta = torch.cat(delta_vals) if delta_vals else torch.zeros(0, device=device)
        all_conf = torch.cat(conf_vals) if conf_vals else torch.zeros(0, device=device)
        all_score = torch.cat(score_vals) if score_vals else torch.zeros(0, device=device)

        stats = {
            "rgqa_num_pairs": torch.as_tensor(float(pair_count), device=device),
            "rgqa_num_norm": torch.as_tensor(float(num_rgqa), device=device),
            "rgqa_valid_gt_ratio": torch.as_tensor(valid_gt / max(total_gt, 1), device=device),
            "rgqa_extra_positive_per_gt": torch.as_tensor(pair_count / max(total_gt, 1), device=device),
            "rgqa_k0_ratio": torch.as_tensor(k_hist[0] / max(total_gt, 1), device=device),
            "rgqa_k1_ratio": torch.as_tensor(k_hist[1] / max(total_gt, 1), device=device),
            "rgqa_k2_ratio": torch.as_tensor(k_hist[2] / max(total_gt, 1), device=device),
            "rgqa_k3_ratio": torch.as_tensor(k_hist[3] / max(total_gt, 1), device=device),
            "rgqa_k4_ratio": torch.as_tensor(k_hist[4] / max(total_gt, 1), device=device),
            "rgqa_small_valid_gt": torch.as_tensor(float(small_valid), device=device),
            "rgqa_large_valid_gt": torch.as_tensor(float(large_valid), device=device),
            "rgqa_iou_l0_mean": all_iou0.mean() if all_iou0.numel() else torch.zeros((), device=device),
            "rgqa_iou_l1_mean": all_iou1.mean() if all_iou1.numel() else torch.zeros((), device=device),
            "rgqa_delta_iou_mean": all_delta.mean() if all_delta.numel() else torch.zeros((), device=device),
            "rgqa_positive_delta_ratio": (all_delta > 0).float().mean() if all_delta.numel() else torch.zeros((), device=device),
            "rgqa_selected_conf_mean": all_conf.mean() if all_conf.numel() else torch.zeros((), device=device),
            "rgqa_selected_score_mean": all_score.mean() if all_score.numel() else torch.zeros((), device=device),
            "rgqa_conflict_count": torch.as_tensor(float(conflict_count), device=device),
        }
        return rgqa_indices, all_iou1.detach(), torch.zeros(0, dtype=torch.long, device=device), stats

    def loss_rgqa(self, aux_outputs, targets, rgqa_indices, rgqa_ious, num_rgqa):
        device = aux_outputs["pred_boxes"].device
        zero = aux_outputs["pred_boxes"].sum() * 0
        idx = self._safe_src_permutation_idx(rgqa_indices, device)
        if idx[0].numel() == 0:
            return {"loss_rgqa_cls": zero, "loss_rgqa_bbox": zero, "loss_rgqa_giou": zero}

        src_logits = aux_outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, rgqa_indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = rgqa_ious.to(target_score_o.dtype).pow(self.gamma)
        target_score = target_score_o.unsqueeze(-1) * target

        selected_mask = torch.zeros(src_logits.shape[:2], dtype=torch.bool, device=device)
        selected_mask[idx] = True
        weight = selected_mask.unsqueeze(-1).to(src_logits.dtype) * target_score.clamp(min=1e-6)
        loss_cls = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss_cls = loss_cls.mean(1).sum() * src_logits.shape[1] / num_rgqa

        src_boxes = aux_outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, rgqa_indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_rgqa
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou.sum() / num_rgqa
        return {
            "loss_rgqa_cls": loss_cls,
            "loss_rgqa_bbox": loss_bbox,
            "loss_rgqa_giou": loss_giou,
        }

    def forward(self, outputs, targets, **kwargs):
        epoch = kwargs.get("epoch", 0)
        rgqa_lambda = self._rgqa_lambda(epoch)

        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        indices = self.matcher(outputs_without_aux, targets)["indices"]
        self._clear_cache()

        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for aux_outputs in outputs["aux_outputs"] + [outputs["pre_outputs"]]:
                indices_aux = self.matcher(aux_outputs, targets)["indices"]
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for aux_outputs in outputs["enc_aux_outputs"]:
                indices_enc = self.matcher(aux_outputs, targets)["indices"]
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            raise AssertionError("RGQACriterion expects aux_outputs")

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        rgqa_indices = None
        rgqa_ious = None
        num_rgqa = 1.0
        rgqa_stats = {}
        ignore_by_aux = {}
        if self.training and rgqa_lambda > 0:
            rgqa_indices, rgqa_ious, _, rgqa_stats = self._build_rgqa_indices(
                outputs, targets, cached_indices[self.rgqa_aux_index], indices_go
            )
            num_rgqa = float(rgqa_stats["rgqa_num_norm"].detach().item())
            ignore_by_aux[self.rgqa_aux_index] = self._safe_src_permutation_idx(
                rgqa_indices, next(iter(outputs_without_aux.values())).device
            )

        losses = {}
        for loss in self.losses:
            indices_in = indices_go if loss in ["boxes", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        for i, aux_outputs in enumerate(outputs["aux_outputs"]):
            aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[i]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                if i in ignore_by_aux and loss == "mal":
                    meta["ignore_indices"] = ignore_by_aux[i]
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

            if self.training and i == self.rgqa_aux_index and rgqa_lambda > 0:
                rgqa_l = self.loss_rgqa(aux_outputs, targets, rgqa_indices, rgqa_ious, num_rgqa)
                rgqa_l = {
                    k: v * self.weight_dict[k] * rgqa_lambda
                    for k, v in rgqa_l.items()
                    if k in self.weight_dict
                }
                losses.update(rgqa_l)

        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    indices_in = indices_go if loss == "boxes" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "boxes" else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        device = next(iter(outputs.values())).device
        losses["rgqa_lambda"] = torch.as_tensor(float(rgqa_lambda), device=device)
        for k, v in rgqa_stats.items():
            losses[k] = v.detach()
        if self.training and rgqa_lambda > 0:
            rgqa_total = (
                losses.get("loss_rgqa_cls", torch.zeros((), device=device))
                + losses.get("loss_rgqa_bbox", torch.zeros((), device=device))
                + losses.get("loss_rgqa_giou", torch.zeros((), device=device))
            )
            total = sum(v.detach() for k, v in losses.items() if k.startswith("loss_"))
            losses["rgqa_loss_over_total"] = (rgqa_total.detach() / total.clamp(min=1e-6)).detach()

        return _assert_finite_losses(losses, "RGQACriterion")


@register()
class BRGQACriterion(RGQACriterion):
    """Budget-Regularized RGQA.

    BRGQA keeps the dfine54 RGQA score and layer-1-only training path, but
    constrains the auxiliary assignment budget: every GT may receive at most
    ``brgqa_max_positive`` extra positives and the accepted positives for the
    same GT share one normalized supervision mass.  The module adds no model
    parameters and is completely inactive when its lambda is zero.
    """

    def __init__(
        self,
        *args,
        brgqa_enabled=True,
        brgqa_aux_index=1,
        brgqa_candidate_pool=4,
        brgqa_max_positive=2,
        brgqa_refine_rho=0.25,
        brgqa_conf_gamma=0.40,
        brgqa_rank_tau=1.5,
        brgqa_quality_power=1.0,
        brgqa_mass_normalize=True,
        brgqa_small_scale_px=32,
        brgqa_small_iou_thr=0.25,
        brgqa_large_iou_thr=0.35,
        brgqa_allow_empty=True,
        brgqa_exclude_o2o=True,
        brgqa_exclude_indices_go=True,
        brgqa_lambda_max=0.50,
        brgqa_start_epoch=10,
        brgqa_full_epoch=20,
        brgqa_decay_epoch=90,
        brgqa_stop_epoch=96,
        **kwargs,
    ):
        super().__init__(
            *args,
            rgqa_enabled=brgqa_enabled,
            rgqa_aux_index=brgqa_aux_index,
            rgqa_refinement_weight=brgqa_refine_rho,
            rgqa_confidence_weight=brgqa_conf_gamma,
            rgqa_topk=brgqa_candidate_pool,
            rgqa_small_scale_px=brgqa_small_scale_px,
            rgqa_small_iou_thr=brgqa_small_iou_thr,
            rgqa_large_iou_thr=brgqa_large_iou_thr,
            rgqa_allow_empty=brgqa_allow_empty,
            rgqa_exclude_o2o=brgqa_exclude_o2o,
            rgqa_exclude_indices_go=brgqa_exclude_indices_go,
            rgqa_lambda_max=brgqa_lambda_max,
            rgqa_start_epoch=brgqa_start_epoch,
            rgqa_full_epoch=brgqa_full_epoch,
            rgqa_decay_epoch=brgqa_decay_epoch,
            rgqa_stop_epoch=brgqa_stop_epoch,
            **kwargs,
        )
        self.brgqa_enabled = bool(brgqa_enabled)
        self.brgqa_aux_index = int(brgqa_aux_index)
        self.brgqa_candidate_pool = int(brgqa_candidate_pool)
        self.brgqa_max_positive = int(brgqa_max_positive)
        self.brgqa_refine_rho = float(brgqa_refine_rho)
        self.brgqa_conf_gamma = float(brgqa_conf_gamma)
        self.brgqa_rank_tau = float(brgqa_rank_tau)
        self.brgqa_quality_power = float(brgqa_quality_power)
        self.brgqa_mass_normalize = bool(brgqa_mass_normalize)
        self.brgqa_small_scale_px = float(brgqa_small_scale_px)
        self.brgqa_small_iou_thr = float(brgqa_small_iou_thr)
        self.brgqa_large_iou_thr = float(brgqa_large_iou_thr)
        self.brgqa_lambda_max = float(brgqa_lambda_max)
        self.rgqa_enabled = self.brgqa_enabled
        self.rgqa_lambda_max = self.brgqa_lambda_max

    def _rgqa_lambda(self, epoch):
        if not self.brgqa_enabled or self.brgqa_lambda_max <= 0:
            return 0.0
        epoch = 0 if epoch is None else int(epoch)
        max_l = self.brgqa_lambda_max
        if epoch <= 7:
            return 0.0
        if epoch <= 15:
            return max_l * (epoch - 7) / 8.0
        if epoch <= 45:
            return max_l
        if epoch <= 60:
            # Smooth withdrawal: lambda(e45)=max_l, lambda(e60)=0.
            progress = (epoch - 45) / 15.0
            return max_l * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.0

    @staticmethod
    def _scale_id(scale_px):
        # Tiny / Small / Medium / Large diagnostic bins in 640-space.
        if scale_px < 16:
            return 0
        if scale_px < 32:
            return 1
        if scale_px < 96:
            return 2
        return 3

    def _build_brgqa_indices(self, outputs, targets, indices_o2o, indices_go):
        device = outputs["pred_boxes"].device
        bs = outputs["pred_boxes"].shape[0]
        if (
            "aux_outputs" not in outputs
            or len(outputs["aux_outputs"]) <= self.brgqa_aux_index
            or self.brgqa_aux_index < 1
        ):
            empty = [
                (
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                )
                for _ in range(bs)
            ]
            z = torch.zeros(0, device=device)
            return empty, z, z, {}

        aux0 = outputs["aux_outputs"][self.brgqa_aux_index - 1]
        aux1 = outputs["aux_outputs"][self.brgqa_aux_index]
        boxes_l0 = aux0["pred_boxes"].detach()
        boxes_l1 = aux1["pred_boxes"].detach()
        conf_l1 = aux1["pred_logits"].detach().sigmoid()[..., 0]

        o2o_sets = self._indices_query_set(indices_o2o, bs, device)
        go_sets = self._indices_query_set(indices_go, bs, device)

        candidate_edges = []
        total_gt = 0
        gt_scale_counts = [0, 0, 0, 0]

        for b, target in enumerate(targets):
            gt_boxes = target["boxes"]
            n_gt = int(gt_boxes.shape[0])
            total_gt += n_gt
            if n_gt == 0:
                continue

            iou0 = box_iou(box_cxcywh_to_xyxy(boxes_l0[b]), box_cxcywh_to_xyxy(gt_boxes))[0]
            iou1 = box_iou(box_cxcywh_to_xyxy(boxes_l1[b]), box_cxcywh_to_xyxy(gt_boxes))[0]
            delta = iou1 - iou0
            q_hat = (iou1 + self.brgqa_refine_rho * delta).clamp(min=0, max=1)
            score = q_hat - self.brgqa_conf_gamma * conf_l1[b].unsqueeze(1)

            wh = gt_boxes[:, 2:4].clamp(min=1e-9)
            scale_px = torch.sqrt(wh[:, 0] * wh[:, 1]) * 640.0
            thr = torch.where(
                scale_px < self.brgqa_small_scale_px,
                torch.full_like(scale_px, self.brgqa_small_iou_thr),
                torch.full_like(scale_px, self.brgqa_large_iou_thr),
            )

            disallowed = set()
            if self.rgqa_exclude_o2o:
                disallowed.update(o2o_sets[b])
            if self.rgqa_exclude_indices_go:
                disallowed.update(go_sets[b])
            disallowed_t = (
                torch.tensor(sorted(disallowed), dtype=torch.long, device=device)
                if disallowed
                else torch.zeros(0, dtype=torch.long, device=device)
            )

            for j in range(n_gt):
                sid = self._scale_id(float(scale_px[j].item()))
                gt_scale_counts[sid] += 1
                valid = (iou1[:, j] >= thr[j]) & (score[:, j] > 0)
                if disallowed_t.numel() > 0:
                    valid[disallowed_t] = False
                valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
                if valid_idx.numel() == 0:
                    continue
                topn = min(self.brgqa_candidate_pool, int(valid_idx.numel()))
                top_scores, order = torch.topk(score[valid_idx, j], k=topn)
                for local_rank in order.tolist():
                    q = int(valid_idx[local_rank].item())
                    candidate_edges.append(
                        {
                            "sort_score": float(score[q, j].item()),
                            "b": b,
                            "q": q,
                            "j": int(j),
                            "sid": sid,
                            "u0": iou0[q, j].detach(),
                            "u1": iou1[q, j].detach(),
                            "delta": delta[q, j].detach(),
                            "q_hat": q_hat[q, j].detach(),
                            "conf": conf_l1[b, q].detach(),
                            "score": score[q, j].detach(),
                        }
                    )

        candidate_edges.sort(key=lambda x: x["sort_score"], reverse=True)
        used_queries = [set() for _ in range(bs)]
        gt_counts = {}
        gt_items = {}
        conflict_count = 0
        max_pos = max(0, self.brgqa_max_positive)
        for edge in candidate_edges:
            b, q, j = edge["b"], edge["q"], edge["j"]
            key = (b, j)
            if q in used_queries[b] or gt_counts.get(key, 0) >= max_pos:
                conflict_count += 1
                continue
            used_queries[b].add(q)
            gt_counts[key] = gt_counts.get(key, 0) + 1
            gt_items.setdefault(key, []).append(edge)

        k_hist = [0, 0, 0, 0, 0]
        valid_gt = 0
        selected_items = [[] for _ in range(bs)]
        all_weights, all_u0, all_u1, all_delta, all_conf, all_score = [], [], [], [], [], []
        top1_weights, top2_weights, weight_sums = [], [], []
        extra_scale_counts = [0, 0, 0, 0]
        mass_scale_sums = [0.0, 0.0, 0.0, 0.0]

        for b, target in enumerate(targets):
            n_gt = int(target["boxes"].shape[0])
            for j in range(n_gt):
                items = sorted(gt_items.get((b, j), []), key=lambda x: x["sort_score"], reverse=True)
                k = len(items)
                k_hist[min(k, 4)] += 1
                if k == 0:
                    continue
                valid_gt += 1
                qhats = torch.stack([it["q_hat"].to(device) for it in items])
                ranks = torch.arange(k, dtype=qhats.dtype, device=device)
                rank_decay = torch.exp(-ranks / max(self.brgqa_rank_tau, 1e-6))
                quality = qhats.clamp(min=0).pow(self.brgqa_quality_power)
                raw_w = rank_decay * quality
                if self.brgqa_mass_normalize:
                    weights = raw_w / raw_w.sum().clamp(min=1e-6)
                else:
                    weights = torch.ones_like(raw_w)
                weight_sums.append(weights.detach().sum())
                if k >= 1:
                    top1_weights.append(weights[0].detach())
                if k >= 2:
                    top2_weights.append(weights[1].detach())

                for rank, (it, w) in enumerate(zip(items, weights)):
                    selected_items[b].append((it["q"], it["j"], w.detach()))
                    all_weights.append(w.detach())
                    all_u0.append(it["u0"])
                    all_u1.append(it["u1"])
                    all_delta.append(it["delta"])
                    all_conf.append(it["conf"])
                    all_score.append(it["score"])
                    extra_scale_counts[it["sid"]] += 1
                    mass_scale_sums[it["sid"]] += float(w.detach().item())

        brgqa_indices = []
        for b, pairs in enumerate(selected_items):
            if pairs:
                src = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
                tgt = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
            else:
                src = torch.zeros(0, dtype=torch.long, device=device)
                tgt = torch.zeros(0, dtype=torch.long, device=device)
            brgqa_indices.append((src, tgt))

        weights_t = torch.stack(all_weights).detach() if all_weights else torch.zeros(0, device=device)
        u0_t = torch.stack(all_u0).detach() if all_u0 else torch.zeros(0, device=device)
        u1_t = torch.stack(all_u1).detach() if all_u1 else torch.zeros(0, device=device)
        delta_t = torch.stack(all_delta).detach() if all_delta else torch.zeros(0, device=device)
        conf_t = torch.stack(all_conf).detach() if all_conf else torch.zeros(0, device=device)
        score_t = torch.stack(all_score).detach() if all_score else torch.zeros(0, device=device)
        top1_t = torch.stack(top1_weights).detach() if top1_weights else torch.zeros(0, device=device)
        top2_t = torch.stack(top2_weights).detach() if top2_weights else torch.zeros(0, device=device)
        weight_sum_t = torch.stack(weight_sums).detach() if weight_sums else torch.zeros(0, device=device)

        pair_count = int(weights_t.numel())
        scale_names = ["tiny", "small", "medium", "large"]
        stats = {
            "brgqa_num_pairs": torch.as_tensor(float(pair_count), device=device),
            "brgqa_valid_gt_ratio": torch.as_tensor(valid_gt / max(total_gt, 1), device=device),
            "brgqa_extra_positive_per_gt": torch.as_tensor(pair_count / max(total_gt, 1), device=device),
            "brgqa_k0_ratio": torch.as_tensor(k_hist[0] / max(total_gt, 1), device=device),
            "brgqa_k1_ratio": torch.as_tensor(k_hist[1] / max(total_gt, 1), device=device),
            "brgqa_k2_ratio": torch.as_tensor(k_hist[2] / max(total_gt, 1), device=device),
            "brgqa_k3_ratio": torch.as_tensor(k_hist[3] / max(total_gt, 1), device=device),
            "brgqa_k4_ratio": torch.as_tensor(k_hist[4] / max(total_gt, 1), device=device),
            "brgqa_selected_u0_mean": u0_t.mean() if u0_t.numel() else torch.zeros((), device=device),
            "brgqa_selected_u1_mean": u1_t.mean() if u1_t.numel() else torch.zeros((), device=device),
            "brgqa_selected_delta_iou_mean": delta_t.mean() if delta_t.numel() else torch.zeros((), device=device),
            "brgqa_positive_delta_ratio": (delta_t > 0).float().mean() if delta_t.numel() else torch.zeros((), device=device),
            "brgqa_selected_conf_mean": conf_t.mean() if conf_t.numel() else torch.zeros((), device=device),
            "brgqa_selected_score_mean": score_t.mean() if score_t.numel() else torch.zeros((), device=device),
            "brgqa_top1_weight_mean": top1_t.mean() if top1_t.numel() else torch.zeros((), device=device),
            "brgqa_top2_weight_mean": top2_t.mean() if top2_t.numel() else torch.zeros((), device=device),
            "brgqa_weight_sum_per_valid_gt": weight_sum_t.mean() if weight_sum_t.numel() else torch.zeros((), device=device),
            "brgqa_conflict_count": torch.as_tensor(float(conflict_count), device=device),
        }
        for sid, name in enumerate(scale_names):
            stats[f"brgqa_gt_count_share_{name}"] = torch.as_tensor(
                gt_scale_counts[sid] / max(total_gt, 1), device=device
            )
            stats[f"brgqa_extra_count_share_{name}"] = torch.as_tensor(
                extra_scale_counts[sid] / max(pair_count, 1), device=device
            )
            stats[f"brgqa_aux_mass_share_{name}"] = torch.as_tensor(
                mass_scale_sums[sid] / max(float(weights_t.sum().item()) if weights_t.numel() else 0.0, 1e-6),
                device=device,
            )
        return brgqa_indices, u1_t, weights_t, stats

    def loss_brgqa(self, aux_outputs, targets, brgqa_indices, brgqa_ious, brgqa_weights, num_gt):
        device = aux_outputs["pred_boxes"].device
        zero = aux_outputs["pred_boxes"].sum() * 0
        idx = self._safe_src_permutation_idx(brgqa_indices, device)
        if idx[0].numel() == 0:
            return {"loss_brgqa_cls": zero, "loss_brgqa_bbox": zero, "loss_brgqa_giou": zero}

        src_logits = aux_outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, brgqa_indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = brgqa_ious.to(target_score_o.dtype).pow(self.gamma)
        target_score = target_score_o.unsqueeze(-1) * target

        selected_weight = torch.zeros(src_logits.shape[:2], dtype=src_logits.dtype, device=device)
        selected_weight[idx] = brgqa_weights.to(src_logits.dtype)
        weight = selected_weight.unsqueeze(-1) * target_score.clamp(min=1e-6)
        loss_cls = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss_cls = loss_cls.mean(1).sum() * src_logits.shape[1] / num_gt

        src_boxes = aux_outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, brgqa_indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum(-1)
        loss_bbox = (loss_bbox * brgqa_weights.to(loss_bbox.dtype)).sum() / num_gt
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = (loss_giou * brgqa_weights.to(loss_giou.dtype)).sum() / num_gt
        return {
            "loss_brgqa_cls": loss_cls,
            "loss_brgqa_bbox": loss_bbox,
            "loss_brgqa_giou": loss_giou,
        }

    def forward(self, outputs, targets, **kwargs):
        epoch = kwargs.get("epoch", 0)
        brgqa_lambda = self._rgqa_lambda(epoch)

        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        indices = self.matcher(outputs_without_aux, targets)["indices"]
        self._clear_cache()

        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for aux_outputs in outputs["aux_outputs"] + [outputs["pre_outputs"]]:
                indices_aux = self.matcher(aux_outputs, targets)["indices"]
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for aux_outputs in outputs["enc_aux_outputs"]:
                indices_enc = self.matcher(aux_outputs, targets)["indices"]
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            raise AssertionError("BRGQACriterion expects aux_outputs")

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        brgqa_indices = None
        brgqa_ious = None
        brgqa_weights = None
        brgqa_stats = {}
        ignore_by_aux = {}
        if self.training and brgqa_lambda > 0:
            brgqa_indices, brgqa_ious, brgqa_weights, brgqa_stats = self._build_brgqa_indices(
                outputs, targets, cached_indices[self.brgqa_aux_index], indices_go
            )
            ignore_by_aux[self.brgqa_aux_index] = self._safe_src_permutation_idx(
                brgqa_indices, next(iter(outputs_without_aux.values())).device
            )

        losses = {}
        for loss in self.losses:
            indices_in = indices_go if loss in ["boxes", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        for i, aux_outputs in enumerate(outputs["aux_outputs"]):
            aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[i]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                if i in ignore_by_aux and loss == "mal":
                    meta["ignore_indices"] = ignore_by_aux[i]
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

            if self.training and i == self.brgqa_aux_index and brgqa_lambda > 0:
                brgqa_l = self.loss_brgqa(
                    aux_outputs, targets, brgqa_indices, brgqa_ious, brgqa_weights, num_boxes
                )
                brgqa_l = {
                    k: v * self.weight_dict[k] * brgqa_lambda
                    for k, v in brgqa_l.items()
                    if k in self.weight_dict
                }
                losses.update(brgqa_l)

        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    indices_in = indices_go if loss == "boxes" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "boxes" else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        device = next(iter(outputs.values())).device
        losses["brgqa_lambda"] = torch.as_tensor(float(brgqa_lambda), device=device)
        zero = torch.zeros((), device=device)
        if not (self.training and brgqa_lambda > 0):
            brgqa_stats = {
                "brgqa_num_pairs": zero,
                "brgqa_valid_gt_ratio": zero,
                "brgqa_extra_positive_per_gt": zero,
                "brgqa_k0_ratio": zero,
                "brgqa_k1_ratio": zero,
                "brgqa_k2_ratio": zero,
                "brgqa_k3_ratio": zero,
                "brgqa_k4_ratio": zero,
                "brgqa_selected_u0_mean": zero,
                "brgqa_selected_u1_mean": zero,
                "brgqa_selected_delta_iou_mean": zero,
                "brgqa_positive_delta_ratio": zero,
                "brgqa_selected_conf_mean": zero,
                "brgqa_selected_score_mean": zero,
                "brgqa_top1_weight_mean": zero,
                "brgqa_top2_weight_mean": zero,
                "brgqa_weight_sum_per_valid_gt": zero,
                "brgqa_conflict_count": zero,
                "brgqa_gt_count_share_tiny": zero,
                "brgqa_gt_count_share_small": zero,
                "brgqa_gt_count_share_medium": zero,
                "brgqa_gt_count_share_large": zero,
                "brgqa_extra_count_share_tiny": zero,
                "brgqa_extra_count_share_small": zero,
                "brgqa_extra_count_share_medium": zero,
                "brgqa_extra_count_share_large": zero,
                "brgqa_aux_mass_share_tiny": zero,
                "brgqa_aux_mass_share_small": zero,
                "brgqa_aux_mass_share_medium": zero,
                "brgqa_aux_mass_share_large": zero,
            }
            losses["loss_brgqa_cls"] = zero
            losses["loss_brgqa_bbox"] = zero
            losses["loss_brgqa_giou"] = zero
            losses["brgqa_total_loss"] = zero
            losses["brgqa_loss_over_layer1_aux"] = zero
        for k, v in brgqa_stats.items():
            losses[k] = v.detach()
        if self.training and brgqa_lambda > 0:
            brgqa_total = (
                losses.get("loss_brgqa_cls", torch.zeros((), device=device))
                + losses.get("loss_brgqa_bbox", torch.zeros((), device=device))
                + losses.get("loss_brgqa_giou", torch.zeros((), device=device))
            )
            aux1_total = (
                losses.get("loss_mal_aux_1", torch.zeros((), device=device)).detach()
                + losses.get("loss_bbox_aux_1", torch.zeros((), device=device)).detach()
                + losses.get("loss_giou_aux_1", torch.zeros((), device=device)).detach()
                + losses.get("loss_fgl_aux_1", torch.zeros((), device=device)).detach()
                + losses.get("loss_ddf_aux_1", torch.zeros((), device=device)).detach()
            )
            losses["brgqa_total_loss"] = brgqa_total.detach()
            losses["brgqa_loss_over_layer1_aux"] = (
                brgqa_total.detach() / aux1_total.clamp(min=1e-6)
            ).detach()

        return _assert_finite_losses(losses, "BRGQACriterion")
