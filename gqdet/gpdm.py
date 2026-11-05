"""Geometry-preserved directional modulation for P2-to-P3 injection."""

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


class P2DIAGPDM(nn.Module):
    """Preserve direct D2 geometry and add bounded directional modulation."""

    def __init__(
        self,
        c2=64,
        c3=256,
        c4=512,
        hidden=64,
        gate_hidden=32,
        init_alpha=0.04,
        beta_init=0.0,
        beta_max=0.25,
        gate_max=0.60,
        alpha_base_max=0.27,
        alpha_res_ratio=0.50,
    ):
        super().__init__()
        if not 0.0 < beta_max <= 0.30:
            raise ValueError("beta_max must be in (0, 0.30].")
        if not 0.0 <= beta_init <= beta_max:
            raise ValueError("beta_init must be in [0, beta_max].")
        if not 0.0 < gate_max <= 1.0:
            raise ValueError("gate_max must be in (0, 1].")
        if alpha_base_max <= 0.0:
            raise ValueError("alpha_base_max must be positive.")
        if not 0.0 <= alpha_res_ratio <= 1.0:
            raise ValueError("alpha_res_ratio must be in [0, 1].")

        self.beta_max = float(beta_max)
        self.gate_max = float(gate_max)
        self.alpha_base_max = float(alpha_base_max)
        self.alpha_res_ratio = float(alpha_res_ratio)

        # The complete dfine28 direct-detail path.
        self.p2_reduce = ConvBNAct(c2, hidden, kernel_size=1)
        self.p2_dw_down = ConvBNAct(
            hidden,
            hidden,
            kernel_size=3,
            stride=2,
            groups=hidden,
        )
        self.p2_to_p3 = ConvBNAct(hidden, c3, kernel_size=1, act=False)
        self.base_gate = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.lambda_p2 = nn.Parameter(torch.tensor(float(init_alpha)))

        # Directional evidence only modulates the residual path.
        self.detail_local = ConvBNAct(c3, c3, kernel_size=3, groups=c3)
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
        self.detail_fuse = ConvBNAct(c3 * 3, c3, kernel_size=1, act=False)
        self.mod_conv = nn.Conv2d(c3 * 2, c3, kernel_size=1, bias=True)
        nn.init.normal_(self.mod_conv.weight, std=0.01)
        nn.init.zeros_(self.mod_conv.bias)
        self.beta_param = nn.Parameter(torch.tensor(float(beta_init)))

        self.p4_to_p3 = ConvBNAct(c4, c3, kernel_size=1, act=False)
        self.res_gate_reduce = ConvBNAct(
            c3 * 4, gate_hidden, kernel_size=1
        )
        self.res_gate_spatial = ConvBNAct(
            gate_hidden,
            gate_hidden,
            kernel_size=3,
            groups=gate_hidden,
        )
        self.res_gate_out = nn.Conv2d(
            gate_hidden, 1, kernel_size=1, bias=True
        )
        nn.init.normal_(self.res_gate_out.weight, std=0.001)
        nn.init.zeros_(self.res_gate_out.bias)

    @staticmethod
    def _rms(x):
        return x.float().square().mean().sqrt()

    @staticmethod
    def _straight_through_clamp(value, minimum, maximum):
        clamped = value.clamp(min=minimum, max=maximum)
        return value + (clamped - value).detach()

    def _align_d2(self, p2, p3):
        d2 = self.p2_to_p3(self.p2_dw_down(self.p2_reduce(p2)))
        if d2.shape[-2:] != p3.shape[-2:]:
            d2 = F.interpolate(
                d2,
                size=p3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return d2

    def forward(self, p2, p3, p4):
        d2 = self._align_d2(p2, p3)

        base_gate_input = torch.cat(
            [
                d2.mean(dim=1, keepdim=True),
                d2.amax(dim=1, keepdim=True),
                p3.mean(dim=1, keepdim=True),
                p3.amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        g_base = self.base_gate(base_gate_input)
        alpha_base = self._straight_through_clamp(
            self.lambda_p2, 0.0, self.alpha_base_max
        )
        base_inject = alpha_base * g_base * torch.tanh(d2)

        d_local = self.detail_local(d2)
        d_horizontal = self.detail_horizontal(d2)
        d_vertical = self.detail_vertical(d2)
        d_fuse = self.detail_fuse(
            torch.cat([d_local, d_horizontal, d_vertical], dim=1)
        )
        modulation = torch.tanh(
            self.mod_conv(torch.cat([d2, d_fuse], dim=1))
        )
        effective_beta = self._straight_through_clamp(
            self.beta_param, 0.0, self.beta_max
        )
        d_mod = d2 * (1.0 + effective_beta * modulation)
        d_delta = d_mod - d2

        u4 = self.p4_to_p3(p4)
        if u4.shape[-2:] != p3.shape[-2:]:
            u4 = F.interpolate(
                u4,
                size=p3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        gate_features = self.res_gate_reduce(
            torch.cat([p3, d2, d_fuse, u4], dim=1)
        )
        raw_gate_res = self.res_gate_out(
            self.res_gate_spatial(gate_features)
        )
        g_res = self.gate_max * torch.sigmoid(raw_gate_res)
        alpha_res = self.alpha_res_ratio * alpha_base
        res_inject = alpha_res * g_res * torch.tanh(d_delta)

        total_inject = base_inject + res_inject
        p3_new = p3 + total_inject

        with torch.no_grad():
            p3_norm = self._rms(p3.detach()).clamp_min(1e-12)
            d2_norm = self._rms(d2.detach()).clamp_min(1e-12)
            base_norm = self._rms(base_inject.detach())
            res_norm = self._rms(res_inject.detach())
            total_norm = self._rms(total_inject.detach())
            diagnostics = {
                "p2dia_beta_dir": self.beta_param.detach().float(),
                "p2dia_effective_beta_dir": effective_beta.detach().float(),
                "p2dia_g_res_mean": g_res.detach().float().mean(),
                "p2dia_g_res_std": g_res.detach().float().std(unbiased=False),
                "p2dia_g_res_max": g_res.detach().float().max(),
                "p2dia_base_inject_norm": base_norm,
                "p2dia_res_inject_norm": res_norm,
                "p2dia_total_inject_norm": total_norm,
                "p2dia_base_inject_over_p3": base_norm / p3_norm,
                "p2dia_res_inject_over_p3": res_norm / p3_norm,
                "p2dia_total_inject_over_p3": total_norm / p3_norm,
                "p2dia_d_delta_over_d2": self._rms(d_delta.detach()) / d2_norm,
                "p2dia_d_fuse_over_d2": self._rms(d_fuse.detach()) / d2_norm,
                "p2dia_p3_norm_before": p3_norm,
                "p2dia_p3_norm_after": self._rms(p3_new.detach()),
                "p2dia_alpha_base": alpha_base.detach().float(),
                "p2dia_alpha_res": alpha_res.detach().float(),
                "p2dia_base_gate_mean": g_base.detach().float().mean(),
            }
        return p3_new, diagnostics
