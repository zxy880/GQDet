"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes


def _recursive_float_cast(value):
    """Cast floating model outputs to FP32 without severing autograd."""
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: _recursive_float_cast(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_recursive_float_cast(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_recursive_float_cast(v) for v in value)
    return value


def _first_nonfinite(value, prefix="outputs"):
    """Return the first non-finite floating tensor path, or None."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():
            return prefix
    elif isinstance(value, dict):
        for key, item in value.items():
            bad = _first_nonfinite(item, f"{prefix}.{key}")
            if bad:
                return bad
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            bad = _first_nonfinite(item, f"{prefix}[{index}]")
            if bad:
                return bad
    return None


def _group_grad_norms(optimizer, device):
    norms = {}
    for group in optimizer.param_groups:
        name = str(group.get("group_name", "unnamed"))
        sq_sum = torch.zeros((), device=device)
        for parameter in group["params"]:
            if parameter.grad is not None:
                sq_sum += parameter.grad.detach().float().pow(2).sum()
        norms[f"grad_norm_{name}"] = sq_sum.sqrt()
    return norms


def _first_nonfinite_gradient(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return name
    return None


def _save_and_raise_nonfinite(output_dir, reason, samples, targets, model, optimizer, scaler, metas):
    """Persist enough state for a deterministic failure audit, then stop."""
    print(f"[NONFINITE] {reason}", flush=True)
    if output_dir is not None and dist_utils.is_main_process():
        failure_path = output_dir / "nonfinite_failure.pth"
        payload = {
            "reason": reason,
            "metas": metas,
            "samples": samples.detach().cpu(),
            "targets": [
                {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
                for t in targets
            ],
            "model": dist_utils.de_parallel(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        }
        torch.save(payload, failure_path)
        print(f"[NONFINITE] saved failure state: {failure_path}", flush=True)
    raise FloatingPointError(reason)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    lr_group_names = sorted(
        {
            pg.get("group_name")
            for pg in optimizer.param_groups
            if pg.get("group_name")
        }
    )
    for group_name in lr_group_names:
        metric_logger.add_meter(
            f"lr_{group_name}", SmoothedValue(window_size=1, fmt="{value:.8f}")
        )

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)
    label2category = kwargs.get("label2category", None)
    dataset = getattr(data_loader, "dataset", None)
    if label2category is None and dataset is not None and hasattr(dataset, "label2category"):
        label2category = dataset.label2category
    train_transforms = getattr(dataset, "transforms", None)
    if train_transforms is None:
        # torchvision's COCO wrapper stores its v2 Compose privately.
        train_transforms = getattr(dataset, "_transforms", None)
    if train_transforms is not None and hasattr(train_transforms, "reset_trigger_stats"):
        train_transforms.reset_trigger_stats()

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        pre_clip_grad_norm = torch.zeros((), device=device)
        post_clip_grad_norm = torch.zeros((), device=device)
        clip_trigger_rate = torch.zeros((), device=device)
        clip_scale_mean = torch.ones((), device=device)
        amp_scale = torch.ones((), device=device)
        optimizer_step_skipped = torch.zeros((), device=device)
        group_grad_norms = {}

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            bad_output = _first_nonfinite(outputs)
            if bad_output:
                _save_and_raise_nonfinite(
                    output_dir, bad_output, samples, targets, model, optimizer, scaler, metas
                )

            # Criterion and Hungarian matching deliberately run in FP32.
            loss_outputs = _recursive_float_cast(outputs)
            try:
                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(loss_outputs, targets, **metas)
            except Exception as exc:
                _save_and_raise_nonfinite(
                    output_dir, f"criterion_or_matcher_exception: {exc}", samples, targets,
                    model, optimizer, scaler, metas,
                )

            loss = sum(v for k, v in loss_dict.items() if k.startswith("loss_"))
            if not torch.isfinite(loss):
                _save_and_raise_nonfinite(
                    output_dir, "nonfinite_total_loss", samples, targets, model, optimizer, scaler, metas
                )
            scaler.scale(loss).backward()

            # Unscale before every gradient audit and before clipping.
            scaler.unscale_(optimizer)
            group_grad_norms = _group_grad_norms(optimizer, device)
            bad_grad = _first_nonfinite_gradient(model)
            if bad_grad is not None:
                _save_and_raise_nonfinite(
                    output_dir, f"nonfinite_unscaled_gradient:{bad_grad}", samples, targets,
                    model, optimizer, scaler, metas,
                )
            if max_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                pre_clip_grad_norm = torch.as_tensor(total_norm, device=device).detach()
                clip_trigger_rate = (pre_clip_grad_norm > max_norm).to(torch.float32)
                clip_scale_mean = (
                    torch.as_tensor(max_norm, device=device) / (pre_clip_grad_norm + 1e-6)
                ).clamp(max=1.0)
                post_sq_sum = torch.zeros((), device=device)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        post_sq_sum += parameter.grad.detach().float().pow(2).sum()
                post_clip_grad_norm = post_sq_sum.sqrt()

            amp_scale = torch.as_tensor(scaler.get_scale(), device=device)
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_skipped = torch.as_tensor(
                float(scaler.get_scale() < amp_scale.item()), device=device
            )
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(v for k, v in loss_dict.items() if k.startswith("loss_"))
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                pre_clip_grad_norm = torch.as_tensor(total_norm, device=device).detach()
                clip_trigger_rate = (pre_clip_grad_norm > max_norm).to(torch.float32)
                clip_scale_mean = (
                    torch.as_tensor(max_norm, device=device) / (pre_clip_grad_norm + 1e-6)
                ).clamp(max=1.0)

            optimizer.step()

        # Skipped AMP steps do not advance EMA or effective-update warmup.
        if ema is not None and not bool(optimizer_step_skipped.item()):
            ema.update(model)

        if lr_warmup_scheduler is not None and not bool(optimizer_step_skipped.item()):
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        p2dia_diagnostics = outputs.get("p2dia_diagnostics", {})
        if p2dia_diagnostics:
            p2dia_diagnostics = dist_utils.reduce_dict(p2dia_diagnostics)
        loss_value = sum(v for k, v in loss_dict_reduced.items() if k.startswith("loss_"))
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(
            pre_clip_grad_norm=pre_clip_grad_norm,
            post_clip_grad_norm=post_clip_grad_norm,
            clip_trigger_rate=clip_trigger_rate,
            clip_scale_mean=clip_scale_mean,
            amp_scale=amp_scale,
            optimizer_step_skipped=optimizer_step_skipped,
            amp_skip_rate=optimizer_step_skipped,
        )
        if group_grad_norms:
            metric_logger.update(**group_grad_norms)
        if p2dia_diagnostics:
            metric_logger.update(**p2dia_diagnostics)
        group_lrs = {
            f"lr_{pg['group_name']}": pg["lr"]
            for pg in optimizer.param_groups
            if pg.get("group_name")
        }
        metric_logger.update(lr=optimizer.param_groups[0]["lr"], **group_lrs)

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    if use_wandb:
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)}
        )
    # gather the stats from all processes
    if train_transforms is not None and hasattr(train_transforms, "trigger_stats"):
        for name, values in train_transforms.trigger_stats().items():
            metric_logger.update(
                **{
                    f"aug_{name}_attempts": values["attempts"],
                    f"aug_{name}_applied": values["applied"],
                    f"aug_{name}_p": values["probability"],
                }
            )
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)
    label2category = kwargs.get("label2category", None)
    dataset = getattr(data_loader, "dataset", None)
    if label2category is None and dataset is not None and hasattr(dataset, "label2category"):
        label2category = dataset.label2category

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        global_step = epoch * len(data_loader) + i

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        p2dia_diagnostics = outputs.get("p2dia_diagnostics", {})
        if p2dia_diagnostics:
            metric_logger.update(**p2dia_diagnostics)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.tensor([[samples.shape[-1], samples.shape[-2]]], device=samples.device)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        coco_results = []
        for output in results:
            coco_output = {k: v for k, v in output.items()}
            if postprocessor.remap_mscoco_category:
                pass
            elif label2category is not None:
                mapped_labels = torch.tensor(
                    [label2category[int(x.item())] for x in output["labels"].flatten()],
                    device=output["labels"].device,
                    dtype=output["labels"].dtype,
                ).reshape(output["labels"].shape)
                coco_output["labels"] = mapped_labels
            coco_results.append(coco_output)

        res = {target["image_id"].item(): output for target, output in zip(targets, coco_results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # validator format for metrics
        for idx, (target, result) in enumerate(zip(targets, results)):
            gt.append(
                {
                    "boxes": scale_boxes(  # from model input size to original img size
                        target["boxes"],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": target["labels"],
                }
            )
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]
            preds.append(
                {"boxes": result["boxes"], "labels": labels, "scores": result["scores"]}
            )

    # Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
