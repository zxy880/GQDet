"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register
from .p2dia_gpdm import P2DIAGPDM
from .p2dia_lite import P2DIALite
from .p2dia_main import P2DIAMain

__all__ = [
    "DFINE",
]


@register()
class DFINE(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        use_p2dia_lite=False,
        p2dia_c2=64,
        p2dia_c3=256,
        p2dia_hidden=64,
        p2dia_init_lambda=0.04,
        use_p2dia_main=False,
        p2dia_c4=512,
        p2dia_gate_hidden=32,
        use_p2dia_gpdm=False,
        p2dia_beta_init=0.0,
        p2dia_beta_max=0.25,
        p2dia_gate_max=0.60,
        p2dia_alpha_base_max=0.27,
        p2dia_alpha_res_ratio=0.50,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.use_p2dia_lite = use_p2dia_lite
        self.use_p2dia_main = use_p2dia_main
        self.use_p2dia_gpdm = use_p2dia_gpdm
        if sum(
            [
                bool(self.use_p2dia_lite),
                bool(self.use_p2dia_main),
                bool(self.use_p2dia_gpdm),
            ]
        ) > 1:
            raise ValueError("Only one P2DIA variant can be enabled.")
        self.p2dia_lite = (
            P2DIALite(
                c2=p2dia_c2,
                c3=p2dia_c3,
                hidden=p2dia_hidden,
                init_lambda=p2dia_init_lambda,
            )
            if use_p2dia_lite
            else None
        )
        self.p2dia_main = (
            P2DIAMain(
                c2=p2dia_c2,
                c3=p2dia_c3,
                c4=p2dia_c4,
                hidden=p2dia_hidden,
                gate_hidden=p2dia_gate_hidden,
                init_alpha=p2dia_init_lambda,
            )
            if use_p2dia_main
            else None
        )
        self.p2dia_gpdm = (
            P2DIAGPDM(
                c2=p2dia_c2,
                c3=p2dia_c3,
                c4=p2dia_c4,
                hidden=p2dia_hidden,
                gate_hidden=p2dia_gate_hidden,
                init_alpha=p2dia_init_lambda,
                beta_init=p2dia_beta_init,
                beta_max=p2dia_beta_max,
                gate_max=p2dia_gate_max,
                alpha_base_max=p2dia_alpha_base_max,
                alpha_res_ratio=p2dia_alpha_res_ratio,
            )
            if use_p2dia_gpdm
            else None
        )

    def forward(self, x, targets=None):
        features = self.backbone(x)
        p2dia_diagnostics = None
        if self.use_p2dia_lite or self.use_p2dia_main or self.use_p2dia_gpdm:
            if len(features) != 4:
                raise ValueError(
                    "P2DIA requires HGNetv2 return_idx=[0, 1, 2, 3]."
                )
            p2, p3, p4, p5 = features
            if self.use_p2dia_gpdm:
                p3, p2dia_diagnostics = self.p2dia_gpdm(p2, p3, p4)
            elif self.use_p2dia_main:
                p3, p2dia_diagnostics = self.p2dia_main(p2, p3, p4)
            else:
                p3 = self.p2dia_lite(p2, p3)
            features = [p3, p4, p5]
        x = self.encoder(features)
        x = self.decoder(x, targets)
        if p2dia_diagnostics is not None:
            x["p2dia_diagnostics"] = p2dia_diagnostics

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
