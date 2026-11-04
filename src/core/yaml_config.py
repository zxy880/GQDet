"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict


class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)

        for k in super().__dict__:
            if not k.startswith("_") and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None and "model" in self.yaml_cfg:
            self._model = create(self.yaml_cfg["model"], self.global_cfg)
        return super().model

    @property
    def postprocessor(self) -> torch.nn.Module:
        if self._postprocessor is None and "postprocessor" in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg["postprocessor"], self.global_cfg)
        return super().postprocessor

    @property
    def criterion(self) -> torch.nn.Module:
        if self._criterion is None and "criterion" in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg["criterion"], self.global_cfg)
        return super().criterion

    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer is None and "optimizer" in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg["optimizer"], self.model)
            global_cfg = self.global_cfg
            if isinstance(global_cfg.get("optimizer"), dict):
                for key in (
                    "protocol_r_groups",
                    "protocol_r_backbone_lr",
                    "protocol_r_main_lr",
                    "protocol_r_scalar_lr",
                    "protocol_r_main_weight_decay",
                ):
                    global_cfg["optimizer"].pop(key, None)
            self._optimizer = create("optimizer", global_cfg, params=params)
        return super().optimizer

    @property
    def lr_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and "lr_scheduler" in self.yaml_cfg:
            self._lr_scheduler = create("lr_scheduler", self.global_cfg, optimizer=self.optimizer)
            print(f"Initial lr: {self._lr_scheduler.get_last_lr()}")
        return super().lr_scheduler

    @property
    def lr_warmup_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and "lr_warmup_scheduler" in self.yaml_cfg:
            self._lr_warmup_scheduler = create(
                "lr_warmup_scheduler", self.global_cfg, lr_scheduler=self.lr_scheduler
            )
        return super().lr_warmup_scheduler

    @property
    def train_dataloader(self) -> DataLoader:
        if self._train_dataloader is None and "train_dataloader" in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader("train_dataloader")
        return super().train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        if self._val_dataloader is None and "val_dataloader" in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader("val_dataloader")
        return super().val_dataloader

    @property
    def ema(self) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get("use_ema", False):
            self._ema = create("ema", self.global_cfg, model=self.model)
        return super().ema

    @property
    def scaler(self):
        if self._scaler is None and self.yaml_cfg.get("use_amp", False):
            self._scaler = create("scaler", self.global_cfg)
        return super().scaler

    @property
    def evaluator(self):
        if self._evaluator is None and "evaluator" in self.yaml_cfg:
            if self.yaml_cfg["evaluator"]["type"] == "CocoEvaluator":
                from ..data import get_coco_api_from_dataset

                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                self._evaluator = create("evaluator", self.global_cfg, coco_gt=base_ds)
            else:
                raise NotImplementedError(f"{self.yaml_cfg['evaluator']['type']}")
        return super().evaluator

    @property
    def use_wandb(self) -> bool:
        return self.yaml_cfg.get("use_wandb", False)

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        assert "type" in cfg, ""
        cfg = copy.deepcopy(cfg)

        if cfg.get("protocol_r_groups", False):
            cfg.pop("protocol_r_groups", None)
            return YAMLConfig.get_protocol_r_optim_params(cfg, model)

        if "params" not in cfg:
            return model.parameters()

        assert isinstance(cfg["params"], list), ""

        param_groups = []
        visited = []
        for pg in cfg["params"]:
            pattern = pg["params"]
            params = {
                k: v
                for k, v in model.named_parameters()
                if v.requires_grad and len(re.findall(pattern, k)) > 0
            }
            pg["params"] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))
            # print(params.keys())

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({"params": params.values()})
            visited.extend(list(params.keys()))
            # print(params.keys())

        assert len(visited) == len(names), ""

        return param_groups

    @staticmethod
    def get_protocol_r_optim_params(cfg: dict, model: nn.Module):
        """Build and verify the five Protocol-R AdamW parameter groups."""

        # These values preserve the verified full-model recipe by default.
        # The optional keys only expose optimizer hyperparameters to the
        # isolated Optuna copy; they do not change parameter membership.
        backbone_lr = float(cfg.pop("protocol_r_backbone_lr", 1e-4))
        main_lr = float(cfg.pop("protocol_r_main_lr", 2e-4))
        scalar_lr = float(cfg.pop("protocol_r_scalar_lr", 5e-5))
        main_weight_decay = float(cfg.pop("protocol_r_main_weight_decay", 1e-4))

        def is_norm_bn_bias(name):
            lname = name.lower()
            return (
                ("norm" in lname)
                or ("bn" in lname)
                or lname.endswith(".bias")
                or lname.endswith("_bias")
                or lname.endswith("bias")
            )

        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        scalar_names = {
            "p2dia_gpdm.lambda_p2",
            "p2dia_gpdm.beta_param",
        }

        def is_gpdm_scalar(name):
            return any(name.endswith(s) for s in scalar_names)

        groups = [
            (
                "backbone_main",
                backbone_lr,
                main_weight_decay,
                lambda n: "backbone" in n and not is_norm_bn_bias(n),
            ),
            (
                "backbone_norm_bn_bias",
                backbone_lr,
                0.0,
                lambda n: "backbone" in n and is_norm_bn_bias(n),
            ),
            (
                "gpdm_bounded_scalars",
                scalar_lr,
                0.0,
                is_gpdm_scalar,
            ),
            (
                "enc_dec_head_gpdm_main",
                main_lr,
                main_weight_decay,
                lambda n: "backbone" not in n and not is_gpdm_scalar(n) and not is_norm_bn_bias(n),
            ),
            (
                "enc_dec_gpdm_norm_bn_bias",
                main_lr,
                0.0,
                lambda n: "backbone" not in n and not is_gpdm_scalar(n) and is_norm_bn_bias(n),
            ),
        ]

        assigned = {}
        param_groups = []
        for group_name, lr, weight_decay, predicate in groups:
            names = [n for n, _ in named if predicate(n)]
            duplicate = sorted(set(names).intersection(assigned))
            if duplicate:
                raise RuntimeError(
                    f"Protocol-R optimizer duplicate params in {group_name}: {duplicate[:20]}"
                )
            name_set = set(names)
            params = [p for n, p in named if n in name_set]
            for n in names:
                assigned[n] = group_name
            n_params = sum(p.numel() for p in params)
            print(
                f"[Protocol-R optimizer] group={group_name} tensors={len(params)} "
                f"params={n_params} lr={lr} weight_decay={weight_decay}"
            )
            print(f"[Protocol-R optimizer] {group_name} first20={names[:20]}")
            param_groups.append(
                {
                    "params": params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "group_name": group_name,
                }
            )

        all_names = [n for n, _ in named]
        missing = sorted(set(all_names) - set(assigned))
        if missing:
            raise RuntimeError(f"Protocol-R optimizer missed params: {missing[:50]}")
        if len(assigned) != len(all_names):
            raise RuntimeError("Protocol-R optimizer assignment count mismatch")

        scalars = sorted([n for n in assigned if is_gpdm_scalar(n)])
        print(f"[Protocol-R optimizer] low-lr scalars={scalars}")
        if len(scalars) != 2:
            raise RuntimeError(f"Expected 2 GPDM bounded scalars, got {scalars}")

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided."""
        assert ("total_batch_size" in cfg or "batch_size" in cfg) and not (
            "total_batch_size" in cfg and "batch_size" in cfg
        ), "`batch_size` or `total_batch_size` should be choosed one"

        total_batch_size = cfg.get("total_batch_size", None)
        if total_batch_size is None:
            bs = cfg.get("batch_size")
        else:
            from ..misc import dist_utils

            assert (
                total_batch_size % dist_utils.get_world_size() == 0
            ), "total_batch_size should be divisible by world size"
            bs = total_batch_size // dist_utils.get_world_size()
        return bs

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg
        if "total_batch_size" in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop("total_batch_size")
        print(f"building {name} with batch_size={bs}...")
        loader = create(name, global_cfg, batch_size=bs)
        loader.shuffle = self.yaml_cfg[name].get("shuffle", False)
        return loader
