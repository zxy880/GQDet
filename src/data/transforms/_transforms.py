"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import multiprocessing as mp
from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from ...core import register
from .._misc import (
    BoundingBoxes,
    Image,
    Mask,
    SanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomVerticalFlip = register()(T.RandomVerticalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


class _TrackedProbability:
    """Worker-safe counters for epoch-level augmentation auditing."""
    def _init_probability_tracking(self, p):
        self.p = float(p)
        self._configured_p = float(p)
        self._attempts = mp.Value("q", 0)
        self._applied = mp.Value("q", 0)

    def set_probability(self, p):
        self.p = float(p)

    def reset_stats(self):
        with self._attempts.get_lock():
            self._attempts.value = 0
        with self._applied.get_lock():
            self._applied.value = 0

    def trigger_stats(self):
        return {
            "attempts": int(self._attempts.value),
            "applied": int(self._applied.value),
            "probability": float(self.p),
        }

    def _record_attempt(self, applied):
        with self._attempts.get_lock():
            self._attempts.value += 1
        if applied:
            with self._applied.get_lock():
                self._applied.value += 1


@register()
class RandomZoomOut(_TrackedProbability, T.RandomZoomOut):
    def __init__(self, fill=0, side_range=(1.0, 4.0), p=0.5, **kwargs):
        # We sample externally so the base transform always applies once chosen.
        T.RandomZoomOut.__init__(self, fill=fill, side_range=side_range, p=1.0, **kwargs)
        self._init_probability_tracking(p)

    def __call__(self, *inputs: Any) -> Any:
        applied = bool(torch.rand(()) < self.p)
        self._record_attempt(applied)
        if not applied:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


@register()
class RandomHorizontalFlip(_TrackedProbability, T.RandomHorizontalFlip):
    def __init__(self, p=0.5, **kwargs):
        T.RandomHorizontalFlip.__init__(self, p=1.0, **kwargs)
        self._init_probability_tracking(p)

    def __call__(self, *inputs: Any) -> Any:
        applied = bool(torch.rand(()) < self.p)
        self._record_attempt(applied)
        if not applied:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


class _RandomGrayIntensity(_TrackedProbability, T.Transform):
    _transformed_types = (Image,)

    def __init__(
        self,
        value_range=(0.90, 1.10),
        p=0.20,
        late_p=0.10,
        late_start_epoch=60,
        stop_epoch=70,
    ) -> None:
        super().__init__()
        self.value_range = tuple(value_range)
        self._init_probability_tracking(p)
        self.late_p = float(late_p)
        self.late_start_epoch = int(late_start_epoch)
        self.stop_epoch = int(stop_epoch)

    def _prob_for_epoch(self, epoch):
        if epoch >= self.stop_epoch:
            return 0.0
        if epoch >= self.late_start_epoch:
            return self.late_p
        return self.p

    def _epoch_from_inputs(self, inputs):
        seq = inputs[0] if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)) else inputs
        if len(seq) > 1 and hasattr(seq[-1], "epoch"):
            return int(seq[-1].epoch)
        return 0

    def _gray(self, img):
        gray = img[:1] if img.shape[0] > 1 else img
        return gray.clamp(0.0, 1.0)

    def __call__(self, *inputs: Any) -> Any:
        packed = len(inputs) == 1 and isinstance(inputs[0], (tuple, list))
        outputs = list(inputs[0]) if packed else list(inputs)
        epoch = self._epoch_from_inputs(inputs)
        p = self._prob_for_epoch(epoch)
        applied = bool(torch.rand(()) < p)
        self._record_attempt(applied)
        if not applied:
            return tuple(outputs) if packed or len(outputs) > 1 else outputs[0]

        img = outputs[0]
        gray = self._gray(img)
        gray = self._apply(gray)
        outputs[0] = Image(gray.repeat(3, 1, 1).clamp(0.0, 1.0))
        return tuple(outputs) if packed or len(outputs) > 1 else outputs[0]

    def _apply(self, gray):
        raise NotImplementedError


@register()
class RandomGrayGamma(_RandomGrayIntensity):
    def __init__(self, gamma_range=(0.90, 1.10), **kwargs) -> None:
        super().__init__(value_range=gamma_range, **kwargs)

    def _apply(self, gray):
        lo, hi = self.value_range
        gamma = torch.empty((), device=gray.device).uniform_(float(lo), float(hi))
        return gray.clamp(min=1e-6).pow(gamma).clamp(0.0, 1.0)


@register()
class RandomGrayGain(_RandomGrayIntensity):
    def __init__(self, gain_range=(0.90, 1.10), **kwargs) -> None:
        super().__init__(value_range=gain_range, **kwargs)

    def _apply(self, gray):
        lo, hi = self.value_range
        gain = torch.empty((), device=gray.device).uniform_(float(lo), float(hi))
        return (gray * gain).clamp(0.0, 1.0)


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(_TrackedProbability, T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self._init_probability_tracking(p)

    def __call__(self, *inputs: Any) -> Any:
        applied = bool(torch.rand(()) < self.p)
        self._record_attempt(applied)
        if not applied:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
            )

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt
