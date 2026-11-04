"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T

from ...core import GLOBAL_CONFIG, register
from ._transforms import EmptyTransform

torchvision.disable_beta_transforms_warning()


@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop("type")
                    transform = getattr(
                        GLOBAL_CONFIG[name]["_pymodule"], GLOBAL_CONFIG[name]["_name"]
                    )(**op)
                    transforms.append(transform)
                    op["type"] = name

                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError("")
        else:
            transforms = [
                EmptyTransform(),
            ]

        super().__init__(transforms=transforms)

        if policy is None:
            policy = {"name": "default"}

        self.policy = policy
        self.global_samples = 0

    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy["name"])(*inputs)

    def get_forward(self, name):
        forwards = {
            "default": self.default_forward,
            "stop_epoch": self.stop_epoch_forward,
            "stop_sample": self.stop_sample_forward,
            "protocol_r": self.protocol_r_forward,
            "brgqa_sc": self.brgqa_sc_forward,
            "trainfix": self.trainfix_forward,
        }
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        cur_epoch = dataset.epoch
        policy_ops = self.policy["ops"]
        policy_epoch = self.policy["epoch"]

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                pass
            else:
                sample = transform(sample)

        return sample

    def protocol_r_forward(self, *inputs: Any):
        """Three-stage Protocol-R augmentation policy.

        The listed policy ops are disabled before ``start_epoch`` and from
        ``stop_epoch`` onward. All other transforms stay active.
        """
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        cur_epoch = dataset.epoch
        policy_ops = set(self.policy.get("ops", []))
        start_epoch = int(self.policy.get("start_epoch", 0))
        stop_epoch = int(self.policy.get("stop_epoch", 10**9))

        for transform in self.transforms:
            name = type(transform).__name__
            if name in policy_ops and (cur_epoch < start_epoch or cur_epoch >= stop_epoch):
                pass
            else:
                sample = transform(sample)

        return sample

    def brgqa_sc_forward(self, *inputs: Any):
        """Stagewise SAR-gray policy for dfine56 BRGQA-SC.

        Epoch 0-3: geometry-clean start.
        Epoch 4-59: weak gray intensity + mild crop/zoom.
        Epoch 60-69: weak gray intensity only.
        Epoch 70+: clean consolidation.
        """
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        cur_epoch = int(dataset.epoch)
        gray_ops = set(self.policy.get("gray_ops", []))
        spatial_ops = set(self.policy.get("spatial_ops", []))
        gray_start = int(self.policy.get("gray_start_epoch", 4))
        gray_stop = int(self.policy.get("gray_stop_epoch", 70))
        spatial_start = int(self.policy.get("spatial_start_epoch", 4))
        spatial_stop = int(self.policy.get("spatial_stop_epoch", 60))

        for transform in self.transforms:
            name = type(transform).__name__
            if name in gray_ops and (cur_epoch < gray_start or cur_epoch >= gray_stop):
                continue
            if name in spatial_ops and (cur_epoch < spatial_start or cur_epoch >= spatial_stop):
                continue
            sample = transform(sample)

        return sample

    @staticmethod
    def _cosine_probability(epoch, start_epoch, end_epoch, start_value, end_value):
        if epoch <= start_epoch:
            return float(start_value)
        if epoch >= end_epoch:
            return float(end_value)
        progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
        return float(end_value) + (float(start_value) - float(end_value)) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def _trainfix_probabilities(self, epoch):
        if epoch <= 3:
            return dict(RandomZoomOut=0.0, RandomIoUCrop=0.0, RandomGrayGamma=0.0, RandomGrayGain=0.0)
        if epoch <= 54:
            return dict(RandomZoomOut=0.20, RandomIoUCrop=0.35, RandomGrayGamma=0.20, RandomGrayGain=0.20)
        if epoch <= 79:
            return dict(
                RandomZoomOut=self._cosine_probability(epoch, 55, 79, 0.20, 0.08),
                RandomIoUCrop=self._cosine_probability(epoch, 55, 79, 0.35, 0.12),
                RandomGrayGamma=self._cosine_probability(epoch, 55, 79, 0.20, 0.08),
                RandomGrayGain=self._cosine_probability(epoch, 55, 79, 0.20, 0.08),
            )
        if epoch <= 104:
            return dict(RandomZoomOut=0.05, RandomIoUCrop=0.08, RandomGrayGamma=0.05, RandomGrayGain=0.05)
        if epoch <= 109:
            return dict(RandomZoomOut=0.0, RandomIoUCrop=0.0, RandomGrayGamma=0.03, RandomGrayGain=0.03)
        return dict(RandomZoomOut=0.0, RandomIoUCrop=0.0, RandomGrayGamma=0.0, RandomGrayGain=0.0)

    def reset_trigger_stats(self):
        for transform in self.transforms:
            if hasattr(transform, "reset_stats"):
                transform.reset_stats()

    def trigger_stats(self):
        result = {}
        for transform in self.transforms:
            if hasattr(transform, "trigger_stats"):
                result[type(transform).__name__] = transform.trigger_stats()
        return result

    def trainfix_forward(self, *inputs: Any):
        """Global-epoch SAR augmentation schedule for dfine59 TrainFix."""
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        cur_epoch = int(dataset.epoch)
        probabilities = self._trainfix_probabilities(cur_epoch)
        for transform in self.transforms:
            name = type(transform).__name__
            if name in probabilities and hasattr(transform, "set_probability"):
                transform.set_probability(probabilities[name])
            sample = transform(sample)
        return sample

    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy["ops"]
        policy_sample = self.policy["sample"]

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample
