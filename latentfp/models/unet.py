from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class Down(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.pool = nn.MaxPool2d(2)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.pool(x)
		return self.conv(x)


class Up(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
		x1 = self.up(x1)
		# Pad if needed
		diff_y = x2.size(2) - x1.size(2)
		diff_x = x2.size(3) - x1.size(3)
		x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class UNetDualHead(nn.Module):
	def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
		super().__init__()
		c1, c2, c3, c4, c5 = (
			base_channels,
			base_channels * 2,
			base_channels * 4,
			base_channels * 8,
			base_channels * 16,
		)
		self.inc = DoubleConv(in_channels, c1)
		self.down1 = Down(c1, c2)
		self.down2 = Down(c2, c3)
		self.down3 = Down(c3, c4)
		self.down4 = Down(c4, c5)
		self.up1 = Up(c5 + c4, c4)
		self.up2 = Up(c4 + c3, c3)
		self.up3 = Up(c3 + c2, c2)
		self.up4 = Up(c2 + c1, c1)
		# Two heads: restoration and ridge segmentation (logits)
		self.head_restoration = nn.Conv2d(c1, 1, kernel_size=1)
		self.head_ridge = nn.Conv2d(c1, 1, kernel_size=1)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		restored = self.head_restoration(x)
		ridge_logits = self.head_ridge(x)
		return restored, ridge_logits


