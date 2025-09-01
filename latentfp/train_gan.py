from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from latentfp.data.sd4_dataset import SD4Fingerprints
from latentfp.models.unet import UNetDualHead
from latentfp.models.discriminators import MultiScaleDiscriminator


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
	C1 = 0.01 ** 2
	C2 = 0.03 ** 2
	mu_x = torch.nn.functional.avg_pool2d(pred, window_size, 1, window_size // 2)
	mu_y = torch.nn.functional.avg_pool2d(target, window_size, 1, window_size // 2)
	sigma_x = torch.nn.functional.avg_pool2d(pred * pred, window_size, 1, window_size // 2) - mu_x * mu_x
	sigma_y = torch.nn.functional.avg_pool2d(target * target, window_size, 1, window_size // 2) - mu_y * mu_y
	sigma_xy = torch.nn.functional.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu_x * mu_y
	ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
	ssim_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
	ssim_map = ssim_n / (ssim_d + 1e-8)
	return (1.0 - ssim_map.clamp(0, 1)).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	probs = torch.sigmoid(logits)
	intersection = (probs * targets).sum(dim=(1, 2, 3))
	sum_probs = probs.sum(dim=(1, 2, 3))
	sum_targets = targets.sum(dim=(1, 2, 3))
	dice = (2 * intersection + eps) / (sum_probs + sum_targets + eps)
	return 1.0 - dice.mean()


def gan_train(
	data_root: Path,
	batch_size: int = 8,
	epochs: int = 2,
	lr_g: float = 2e-4,
	lr_d: float = 2e-4,
	adv_lambda: float = 1.0,
	l1_lambda: float = 100.0,
	ssim_lambda: float = 20.0,
	ridge_lambda: float = 1.0,
	device: str = "cuda" if torch.cuda.is_available() else "cpu",
	resume_g: Optional[Path] = None,
	limit_train_batches: Optional[int] = 50,
	limit_val_batches: Optional[int] = 10,
) -> None:
	dataset = SD4Fingerprints(root_dir=data_root, crop_size=256, training=True)
	n_val = max(1, int(0.05 * len(dataset)))
	n_train = len(dataset) - n_val
	train_set, val_set = random_split(dataset, [n_train, n_val])
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

	G = UNetDualHead(in_channels=1, base_channels=32).to(device)
	if resume_g is not None and Path(resume_g).exists():
		ckpt = torch.load(str(resume_g), map_location=device)
		state = ckpt.get("model", ckpt)
		G.load_state_dict(state)
	D = MultiScaleDiscriminator(in_channels=2, base_channels=64, num_scales=3).to(device)

	opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
	opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

	l1 = nn.L1Loss()
	bce_logits = nn.BCEWithLogitsLoss()
	scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda") and torch.cuda.is_available()))

	best_val = float("inf")

	for epoch in range(1, epochs + 1):
		pbar = tqdm(train_loader, desc=f"GAN Epoch {epoch}/{epochs}")
		G.train(); D.train()
		train_step = 0
		for batch in pbar:
			latent = batch["latent"].to(device)
			target = batch["target"].to(device)
			ridge = batch["ridge"].to(device)

			# 1) Update D
			opt_D.zero_grad(set_to_none=True)
			with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
				restored_fake, _ = G(latent)
				fake_pair = torch.cat([latent, torch.sigmoid(restored_fake).detach()], dim=1)
				real_pair = torch.cat([latent, target], dim=1)
				out_real = D(real_pair)
				out_fake = D(fake_pair)
				loss_D_real = sum(bce_logits(x, torch.ones_like(x)) for x in out_real) / len(out_real)
				loss_D_fake = sum(bce_logits(x, torch.zeros_like(x)) for x in out_fake) / len(out_fake)
				loss_D = (loss_D_real + loss_D_fake) * 0.5
			scaler.scale(loss_D).backward()
			scaler.step(opt_D)

			# 2) Update G
			opt_G.zero_grad(set_to_none=True)
			with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
				restored, ridge_logits = G(latent)
				fake_pair = torch.cat([latent, torch.sigmoid(restored)], dim=1)
				out_fake = D(fake_pair)
				adv_loss = sum(bce_logits(x, torch.ones_like(x)) for x in out_fake) / len(out_fake)
				rec_loss = l1_lambda * l1(restored, target) + ssim_lambda * ssim_loss(torch.sigmoid(restored), target)
				ridge_loss = ridge_lambda * (nn.BCEWithLogitsLoss()(ridge_logits, ridge) + 0.5 * dice_loss(ridge_logits, ridge))
				loss_G = adv_lambda * adv_loss + rec_loss + ridge_loss
			scaler.scale(loss_G).backward()
			scaler.step(opt_G)
			scaler.update()

			pbar.set_postfix({"D": float(loss_D.item()), "G": float(loss_G.item())})
			train_step += 1
			if limit_train_batches is not None and train_step >= limit_train_batches:
				break

		# Validation
		G.eval()
		val_loss = 0.0
		with torch.no_grad():
			val_step = 0
			for batch in val_loader:
				latent = batch["latent"].to(device)
				target = batch["target"].to(device)
				ridge = batch["ridge"].to(device)
				restored, ridge_logits = G(latent)
				rec_loss = l1_lambda * l1(restored, target) + ssim_lambda * ssim_loss(torch.sigmoid(restored), target)
				ridge_loss = ridge_lambda * (nn.BCEWithLogitsLoss()(ridge_logits, ridge) + 0.5 * dice_loss(ridge_logits, ridge))
				val_loss += float((rec_loss + ridge_loss).item())
				val_step += 1
				if limit_val_batches is not None and val_step >= limit_val_batches:
					break
		val_loss /= max(1, len(val_loader))
		print(f"Val: {val_loss:.4f}")
		if val_loss < best_val:
			best_val = val_loss
			ckpt = {"model": G.state_dict(), "meta": {"in_channels": 1, "base_channels": 32}}
			torch.save(ckpt, str(Path("checkpoint_gan_best.pt")))
			print("Saved G best to checkpoint_gan_best.pt")

	# Save last
	ckpt = {"model": G.state_dict(), "meta": {"in_channels": 1, "base_channels": 32}}
	torch.save(ckpt, str(Path("checkpoint_gan_last.pt")))
	print("Saved G last to checkpoint_gan_last.pt")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_root", type=Path, required=True)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--epochs", type=int, default=2)
	parser.add_argument("--resume_g", type=Path, default=None)
	parser.add_argument("--limit_train_batches", type=int, default=50)
	parser.add_argument("--limit_val_batches", type=int, default=10)
	args = parser.parse_args()

	gan_train(
		data_root=args.data_root,
		batch_size=args.batch_size,
		epochs=args.epochs,
		resume_g=args.resume_g,
		limit_train_batches=args.limit_train_batches,
		limit_val_batches=args.limit_val_batches,
	)


if __name__ == "__main__":
	main()


