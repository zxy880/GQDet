"""Original RGQS candidate selection, extracted without the upstream criterion.

The selection and auxiliary-weight methods retain their experiment implementation.
Torch/torchvision are external dependencies and are not redistributed.
"""
import math
import torch
from torchvision.ops import box_convert, box_iou

def box_cxcywh_to_xyxy(boxes):
    return box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

class RGQSSelection:
    def __init__(self, settings):
        # Pass the complete BRGQACriterion section from an experiment config.
        required = ["brgqa_enabled", "brgqa_aux_index", "brgqa_candidate_pool",
                    "brgqa_max_positive", "brgqa_refine_rho", "brgqa_conf_gamma",
                    "brgqa_rank_tau", "brgqa_quality_power", "brgqa_mass_normalize",
                    "brgqa_small_scale_px", "brgqa_small_iou_thr",
                    "brgqa_large_iou_thr", "brgqa_lambda_max"]
        for key in required:
            setattr(self, key, settings[key])
        self.rgqa_exclude_o2o = settings["brgqa_exclude_o2o"]
        self.rgqa_exclude_indices_go = settings["brgqa_exclude_indices_go"]

    def _indices_query_set(self, indices, batch_size, device):
        sets = [set() for _ in range(batch_size)]
        for b, (src, _) in enumerate(indices):
            if src.numel() > 0:
                sets[b].update(int(x) for x in src.detach().cpu().tolist())
        return sets

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

