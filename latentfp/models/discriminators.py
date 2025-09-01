from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def conv_block(in_c: int, out_c: int, k: int = 4, s: int = 2, p: int = 1, use_bn: bool = True) -> nn.Sequential:
	layers = [nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=not use_bn)]
	if use_bn:
		layers.append(nn.BatchNorm2d(out_c))
	layers.append(nn.LeakyReLU(0.2, inplace=True))
	return nn.Sequential(*layers)


class PatchDiscriminator(nn.Module):
	def __init__(self, in_channels: int = 2, base_channels: int = 64) -> None:
		super().__init__()
		c1, c2, c3, c4, c5 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 8
		self.net = nn.Sequential(
			conv_block(in_channels, c1, use_bn=False),
			conv_block(c1, c2),
			conv_block(c2, c3),
			conv_block(c3, c4),
			nn.Conv2d(c4, 1, kernel_size=4, stride=1, padding=1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class MultiScaleDiscriminator(nn.Module):
	def __init__(self, in_channels: int = 2, base_channels: int = 64, num_scales: int = 3) -> None:
		super().__init__()
		self.discriminators = nn.ModuleList([PatchDiscriminator(in_channels, base_channels) for _ in range(num_scales)])
		self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

	def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
		outputs: List[torch.Tensor] = []
		inp = x
		for disc in self.discriminators:
			outputs.append(disc(inp))
			inp = self.downsample(inp)
		return outputs


