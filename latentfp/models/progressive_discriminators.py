from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveDiscriminator(nn.Module):
	"""Progressive discriminator at multiple resolutions with spectral normalization for stability."""
	def __init__(self, in_channels: int = 2, base_channels: int = 64) -> None:
		super().__init__()
		c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
		
		# Main discriminator path (full resolution)
		self.conv1 = self._conv_block(in_channels, c1, use_bn=False, spectral=True)
		self.conv2 = self._conv_block(c1, c2, spectral=True)
		self.conv3 = self._conv_block(c2, c3, spectral=True)
		self.conv4 = self._conv_block(c3, c4, spectral=True)
		self.final = nn.utils.spectral_norm(nn.Conv2d(c4, 1, kernel_size=4, stride=1, padding=1))
		
		# Progressive discriminators at 1/2 and 1/4 resolution
		self.disc_half = self._make_small_discriminator(in_channels, base_channels // 2)
		self.disc_quarter = self._make_small_discriminator(in_channels, base_channels // 4)
		
	def _conv_block(self, in_c: int, out_c: int, k: int = 4, s: int = 2, p: int = 1, use_bn: bool = False, spectral: bool = True) -> nn.Sequential:
		conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=True)
		if spectral:
			conv = nn.utils.spectral_norm(conv)
		layers = [conv]
		# Remove BN for better stability with small batch sizes
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		return nn.Sequential(*layers)
	
	def _make_small_discriminator(self, in_channels: int, base_c: int) -> nn.Sequential:
		return nn.Sequential(
			self._conv_block(in_channels, base_c),
			self._conv_block(base_c, base_c * 2),
			self._conv_block(base_c * 2, base_c * 4),
			nn.utils.spectral_norm(nn.Conv2d(base_c * 4, 1, kernel_size=4, stride=1, padding=1)),
		)
	
	def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
		outputs = []
		
		# Main path (full resolution)
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.final(out)
		outputs.append(out)
		
		# 1/2 resolution
		x_half = F.avg_pool2d(x, kernel_size=2, stride=2)
		out_half = self.disc_half(x_half)
		outputs.append(out_half)
		
		# 1/4 resolution  
		x_quarter = F.avg_pool2d(x, kernel_size=4, stride=4)
		out_quarter = self.disc_quarter(x_quarter)
		outputs.append(out_quarter)
		
		return outputs
