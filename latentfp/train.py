from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from latentfp.data.sd4_dataset import SD4Fingerprints
from latentfp.models.unet import UNetDualHead


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
	# Very lightweight SSIM approximation using avg pooling for brevity
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


def train(
	data_root: Path,
	batch_size: int = 8,
	epochs: int = 5,
	lr: float = 1e-3,
	device: str = "cuda" if torch.cuda.is_available() else "cpu",
	resume_ckpt: Optional[Path] = None,
	limit_train_batches: Optional[int] = None,
	limit_val_batches: Optional[int] = None,
) -> None:
	dataset = SD4Fingerprints(root_dir=data_root, crop_size=256, training=True)
	n_val = max(1, int(0.05 * len(dataset)))
	n_train = len(dataset) - n_val
	train_set, val_set = random_split(dataset, [n_train, n_val])
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

	model = UNetDualHead(in_channels=1, base_channels=32).to(device)
	if resume_ckpt is not None and Path(resume_ckpt).exists():
		ckpt = torch.load(str(resume_ckpt), map_location=device)
		state = ckpt.get("model", ckpt)
		model.load_state_dict(state)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	l1 = nn.L1Loss()
	bce = nn.BCEWithLogitsLoss()
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
	scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda") and torch.cuda.is_available()))
	best_val_loss = float("inf")

	for epoch in range(1, epochs + 1):
		model.train()
		pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
		train_step = 0
		for batch in pbar:
			latent = batch["latent"].to(device)
			target = batch["target"].to(device)
			ridge = batch["ridge"].to(device)

			optimizer.zero_grad(set_to_none=True)
			with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
				restored, ridge_logits = model(latent)
				loss_rec = l1(restored, target) + 0.2 * ssim_loss(torch.sigmoid(restored), target)
				loss_ridge = bce(ridge_logits, ridge) + 0.5 * dice_loss(ridge_logits, ridge)
				loss = loss_rec + 0.5 * loss_ridge
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			pbar.set_postfix({"loss": float(loss.item())})
			train_step += 1
			if limit_train_batches is not None and train_step >= limit_train_batches:
				break

		# Eval (light)
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			val_step = 0
			for batch in val_loader:
				latent = batch["latent"].to(device)
				target = batch["target"].to(device)
				ridge = batch["ridge"].to(device)
				with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
					restored, ridge_logits = model(latent)
					loss_rec = l1(restored, target) + 0.2 * ssim_loss(torch.sigmoid(restored), target)
					loss_ridge = bce(ridge_logits, ridge) + 0.5 * dice_loss(ridge_logits, ridge)
					val_loss += float((loss_rec + 0.5 * loss_ridge).item())
				val_step += 1
				if limit_val_batches is not None and val_step >= limit_val_batches:
					break
		val_loss /= max(1, len(val_loader))
		print(f"Validation loss: {val_loss:.4f}")
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			ckpt_best = {
				"model": model.state_dict(),
				"meta": {"in_channels": 1, "base_channels": 32},
			}
			torch.save(ckpt_best, str(Path("checkpoint_unet_dualhead_best.pt")))
			print("Saved best checkpoint to checkpoint_unet_dualhead_best.pt")
		scheduler.step()

	# Save checkpoint
	ckpt = {
		"model": model.state_dict(),
		"meta": {"in_channels": 1, "base_channels": 32},
	}
	torch.save(ckpt, str(Path("checkpoint_unet_dualhead.pt")))
	print("Saved checkpoint to checkpoint_unet_dualhead.pt")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_root", type=Path, required=True)
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--resume_ckpt", type=Path, default=None)
	parser.add_argument("--limit_train_batches", type=int, default=None)
	parser.add_argument("--limit_val_batches", type=int, default=None)
	args = parser.parse_args()

	train(
		data_root=args.data_root,
		batch_size=args.batch_size,
		epochs=args.epochs,
		lr=args.lr,
		resume_ckpt=args.resume_ckpt,
		limit_train_batches=args.limit_train_batches,
		limit_val_batches=args.limit_val_batches,
	)


if __name__ == "__main__":
	main()


