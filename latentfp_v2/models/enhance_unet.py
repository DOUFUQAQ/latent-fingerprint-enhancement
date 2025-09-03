from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = nn.functional.pad(x, (0, skip.size(-1)-x.size(-1), 0, skip.size(-2)-x.size(-2)))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EnhanceUNetDualHead(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = base, base*2, base*4, base*8, base*16
        self.inc = DoubleConv(in_channels, c1)
        self.d1 = Down(c1, c2)
        self.d2 = Down(c2, c3)
        self.d3 = Down(c3, c4)
        self.d4 = Down(c4, c5)
        self.u1 = Up(c5 + c4, c4)
        self.u2 = Up(c4 + c3, c3)
        self.u3 = Up(c3 + c2, c2)
        self.u4 = Up(c2 + c1, c1)
        self.head_restore = nn.Conv2d(c1, 1, 1)
        self.head_ridge = nn.Conv2d(c1, 1, 1)
    def forward(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.head_restore(x), self.head_ridge(x)


