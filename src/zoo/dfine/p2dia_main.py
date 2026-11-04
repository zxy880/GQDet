"""Directional P2 detail injection with P4 context gating."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        act=True,
    ):
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(size // 2 for size in kernel_size)
            else:
                padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class P2DIAMain(nn.Module):
    """Inject directional P2 detail into P3 under P4 semantic context."""

    def __init__(
        self,
        c2=64,
        c3=256,
        c4=512,
        hidden=64,
        gate_hidden=32,
        init_alpha=0.04,
    ):
        super().__init__()
        self.p2_reduce = ConvBNAct(c2, hidden, kernel_size=1)
        self.p2_dw_down = ConvBNAct(
            hidden,
            hidden,
            kernel_size=3,
            stride=2,
            groups=hidden,
        )
        self.p2_to_p3 = ConvBNAct(hidden, c3, kernel_size=1, act=False)

        self.detail_local = ConvBNAct(
            c3, c3, kernel_size=3, groups=c3
        )
        self.detail_horizontal = ConvBNAct(
            c3,
            c3,
            kernel_size=(1, 5),
            padding=(0, 2),
            groups=c3,
        )
        self.detail_vertical = ConvBNAct(
            c3,
            c3,
            kernel_size=(5, 1),
            padding=(2, 0),
            groups=c3,
        )
        self.detail_fuse = ConvBNAct(
            c3 * 3, c3, kernel_size=1, act=False
        )

        self.p4_to_p3 = ConvBNAct(c4, c3, kernel_size=1, act=False)
        self.gate_reduce = ConvBNAct(
            c3 * 3, gate_hidden, kernel_size=1
        )
        self.gate_spatial = ConvBNAct(
            gate_hidden,
            gate_hidden,
            kernel_size=3,
            groups=gate_hidden,
        )
        self.gate_out = nn.Conv2d(gate_hidden, 1, kernel_size=1, bias=True)
        nn.init.normal_(self.gate_out.weight, std=0.001)
        nn.init.zeros_(self.gate_out.bias)

        self.lambda_p2 = nn.Parameter(torch.tensor(float(init_alpha)))

    @staticmethod
    def _rms(x):
        return x.float().square().mean().sqrt()

    def forward(self, p2, p3, p4):
        d2 = self.p2_to_p3(self.p2_dw_down(self.p2_reduce(p2)))
        if d2.shape[-2:] != p3.shape[-2:]:
            d2 = F.interpolate(
                d2,
                size=p3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        d_local = self.detail_local(d2)
        d_horizontal = self.detail_horizontal(d2)
        d_vertical = self.detail_vertical(d2)
        d_dir = self.detail_fuse(
            torch.cat([d_local, d_horizontal, d_vertical], dim=1)
        )

        u4 = self.p4_to_p3(p4)
        if u4.shape[-2:] != p3.shape[-2:]:
            u4 = F.interpolate(
                u4,
                size=p3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        gate_features = self.gate_reduce(torch.cat([p3, d_dir, u4], dim=1))
        gate = torch.sigmoid(self.gate_out(self.gate_spatial(gate_features)))
        injection = self.lambda_p2 * gate * torch.tanh(d_dir)
        p3_new = p3 + injection

        with torch.no_grad():
            gate_float = gate.detach().float()
            detail_float = d_dir.detach().float()
            diagnostics = {
                "p2dia_gate_mean": gate_float.mean(),
                "p2dia_gate_std": gate_float.std(unbiased=False),
                "p2dia_gate_min": gate_float.min(),
                "p2dia_gate_max": gate_float.max(),
                "p2dia_dir_detail_mean": detail_float.mean(),
                "p2dia_dir_detail_std": detail_float.std(unbiased=False),
                "p2dia_inject_norm": self._rms(injection.detach()),
                "p3_norm_before": self._rms(p3.detach()),
                "p3_norm_after": self._rms(p3_new.detach()),
            }
        return p3_new, diagnostics
