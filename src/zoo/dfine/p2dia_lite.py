"""Lightweight P2-to-P3 detail injection for D-FINE-S."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, groups=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class P2DIALite(nn.Module):
    """Inject stride-4 P2 detail into stride-8 P3 through a gated residual."""

    def __init__(self, c2=64, c3=256, hidden=64, init_lambda=0.04):
        super().__init__()
        self.p2_reduce = ConvBNAct(c2, hidden, k=1)
        self.p2_dw_down = ConvBNAct(hidden, hidden, k=3, s=2, groups=hidden)
        self.p2_to_p3 = ConvBNAct(hidden, c3, k=1, act=False)

        self.gate = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.lambda_p2 = nn.Parameter(torch.tensor(float(init_lambda)))

    def forward(self, p2, p3):
        detail = self.p2_reduce(p2)
        detail = self.p2_dw_down(detail)
        if detail.shape[-2:] != p3.shape[-2:]:
            detail = F.interpolate(
                detail,
                size=p3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        detail = self.p2_to_p3(detail)

        gate_input = torch.cat(
            [
                detail.mean(dim=1, keepdim=True),
                detail.amax(dim=1, keepdim=True),
                p3.mean(dim=1, keepdim=True),
                p3.amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        gate = self.gate(gate_input)
        return p3 + self.lambda_p2 * gate * torch.tanh(detail)
