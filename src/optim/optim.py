"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register

__all__ = [
    "AdamW",
    "SGD",
    "Adam",
    "MultiStepLR",
    "CosineAnnealingLR",
    "OneCycleLR",
    "LambdaLR",
    "HoldCosineFlatLR",
    "FlatCosineLRScheduler",
    "GlobalEpochHoldCosineFloor",
]


SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)


@register()
class MultiStepLR(lr_scheduler.MultiStepLR):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        last_epoch=-1,
        flat_epoch=None,
        no_aug_epoch=None,
        epochs=None,
        lr_gamma=None,
        **kwargs,
    ):
        # ``flat_epoch``/``no_aug_epoch``/``epochs``/``lr_gamma`` can remain
        # after recursive YAML overrides from FlatCosine configs. They are
        # intentionally ignored for the explicit MultiStepLR schedule.
        del flat_epoch, no_aug_epoch, epochs, lr_gamma, kwargs
        super().__init__(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)


@register()
class HoldCosineFlatLR(lr_scheduler._LRScheduler):
    """Hold the target LR, cosine-decay it, then keep a fixed LR floor.

    The solver starts stepping the epoch scheduler only after optimizer-step
    warmup finishes. ``warmup_epochs`` maps the scheduler's internal step back
    to the actual training epoch so decay boundaries remain explicit.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs=3,
        decay_start_epoch=20,
        decay_end_epoch=70,
        min_lr_ratio=0.01,
        group_schedules=None,
        milestones=None,
        gamma=None,
        last_epoch=-1,
    ):
        # ``milestones`` and ``gamma`` may remain after recursively merging a
        # MultiStepLR base config. They are intentionally unused here.
        del milestones, gamma
        if decay_end_epoch <= decay_start_epoch:
            raise ValueError("decay_end_epoch must be greater than decay_start_epoch")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")

        self.warmup_epochs = int(warmup_epochs)
        self.decay_start_epoch = int(decay_start_epoch)
        self.decay_end_epoch = int(decay_end_epoch)
        self.min_lr_ratio = float(min_lr_ratio)
        self.group_schedules = {}
        for prefix, schedule in (group_schedules or {}).items():
            schedule = dict(schedule)
            start = int(schedule.get("decay_start_epoch", self.decay_start_epoch))
            end = int(schedule.get("decay_end_epoch", self.decay_end_epoch))
            floor = float(schedule.get("min_lr_ratio", self.min_lr_ratio))
            if end <= start:
                raise ValueError(
                    f"group {prefix!r}: decay_end_epoch must be greater than "
                    "decay_start_epoch"
                )
            if not 0.0 <= floor <= 1.0:
                raise ValueError(f"group {prefix!r}: min_lr_ratio must be in [0, 1]")
            self.group_schedules[str(prefix)] = (start, end, floor)
        super().__init__(optimizer, last_epoch)

    def _factor(self, decay_start_epoch, decay_end_epoch, min_lr_ratio):
        # At the first explicit step (after epoch warmup_epochs - 1),
        # last_epoch is 1 and the returned LR is used by warmup_epochs.
        training_epoch = self.warmup_epochs + self.last_epoch - 1
        if training_epoch <= decay_start_epoch:
            return 1.0
        if training_epoch >= decay_end_epoch:
            return min_lr_ratio

        progress = (
            (training_epoch - decay_start_epoch)
            / (decay_end_epoch - decay_start_epoch)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    def _schedule_for_group(self, group_name):
        matches = [
            (prefix, schedule)
            for prefix, schedule in self.group_schedules.items()
            if group_name.startswith(prefix)
        ]
        if matches:
            return max(matches, key=lambda item: len(item[0]))[1]
        return self.decay_start_epoch, self.decay_end_epoch, self.min_lr_ratio

    def get_lr(self):
        lrs = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            schedule = self._schedule_for_group(str(group.get("group_name", "")))
            lrs.append(base_lr * self._factor(*schedule))
        return lrs


@register()
class GlobalEpochHoldCosineFloor:
    """Explicit global-epoch LR policy used by TrainFix.

    It intentionally does not inherit PyTorch's epoch scheduler semantics:
    resume offsets and ``last_epoch`` cannot shift an LR boundary.  The solver
    calls :meth:`set_epoch` before each epoch once iteration warmup finishes.
    """

    def __init__(self, optimizer, backbone_min_lr=2e-6, main_min_lr=1e-5, scalar_min_lr=5e-6, **kwargs):
        del kwargs
        self.optimizer = optimizer
        self.backbone_min_lr = float(backbone_min_lr)
        self.main_min_lr = float(main_min_lr)
        self.scalar_min_lr = float(scalar_min_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_epoch = -1

    @staticmethod
    def _cosine_floor(epoch, start, end, floor, base_lr):
        if epoch < start:
            return base_lr
        if epoch >= end:
            return floor
        progress = (float(epoch) - float(start)) / float(end - start)
        return floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _lr_for_group(self, group, base_lr, epoch):
        name = str(group.get("group_name", ""))
        if name.startswith("backbone"):
            return self._cosine_floor(epoch, 55, 105, self.backbone_min_lr, base_lr)
        if name.startswith("gpdm_bounded_scalars"):
            return self._cosine_floor(epoch, 75, 118, self.scalar_min_lr, base_lr)
        return self._cosine_floor(epoch, 65, 118, self.main_min_lr, base_lr)

    def set_epoch(self, epoch):
        epoch = int(epoch)
        self.last_epoch = epoch
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._lr_for_group(group, base_lr, epoch)

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self):
        raise RuntimeError("TrainFix LR must be stepped with explicit global epoch")

    def state_dict(self):
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch,
                "backbone_min_lr": self.backbone_min_lr, "main_min_lr": self.main_min_lr,
                "scalar_min_lr": self.scalar_min_lr}

    def load_state_dict(self, state_dict):
        self.base_lrs = list(state_dict.get("base_lrs", self.base_lrs))
        self.last_epoch = int(state_dict.get("last_epoch", -1))
        for key in ("backbone_min_lr", "main_min_lr", "scalar_min_lr"):
            if key in state_dict:
                setattr(self, key, float(state_dict[key]))


@register()
class FlatCosineLRScheduler(lr_scheduler._LRScheduler):
    """Protocol-R flat-cosine scheduler.

    Warmup is handled by ``LinearWarmup`` in iteration space. This scheduler
    holds each parameter group's base LR through ``flat_epoch``, cosine-decays
    until ``epochs - no_aug_epoch``, then keeps ``lr_gamma`` of the base LR.
    """

    def __init__(
        self,
        optimizer,
        flat_epoch=45,
        no_aug_epoch=15,
        epochs=120,
        lr_gamma=0.10,
        milestones=None,
        gamma=None,
        last_epoch=-1,
    ):
        del milestones, gamma
        self.flat_epoch = int(flat_epoch)
        self.no_aug_epoch = int(no_aug_epoch)
        self.epochs = int(epochs)
        self.lr_gamma = float(lr_gamma)
        self.decay_end_epoch = self.epochs - self.no_aug_epoch
        if self.decay_end_epoch <= self.flat_epoch:
            raise ValueError("epochs - no_aug_epoch must be greater than flat_epoch")
        if not 0.0 <= self.lr_gamma <= 1.0:
            raise ValueError("lr_gamma must be in [0, 1]")
        super().__init__(optimizer, last_epoch)

    def _factor(self):
        epoch = self.last_epoch
        if epoch <= self.flat_epoch:
            return 1.0
        if epoch >= self.decay_end_epoch:
            return self.lr_gamma
        progress = (epoch - self.flat_epoch) / (self.decay_end_epoch - self.flat_epoch)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_gamma + (1.0 - self.lr_gamma) * cosine

    def get_lr(self):
        factor = self._factor()
        return [base_lr * factor for base_lr in self.base_lrs]
