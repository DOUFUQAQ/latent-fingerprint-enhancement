from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from latentfp_v2.data.enhance_dataset import LatentEnhanceDataset
from latentfp_v2.models.enhance_unet import EnhanceUNetDualHead


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum()
        denom = pred.sum() + target.sum() + self.eps
        return 1.0 - (2.0 * inter + self.eps) / denom


def train(data_root: str, epochs: int = 5, batch_size: int = 4, lr: float = 1e-3, device: str = "cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    ds = LatentEnhanceDataset(data_root, crop=256, training=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    model = EnhanceUNetDualHead(1, 32).to(device)
    l1 = nn.L1Loss()
    dice = DiceLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dl, desc=f"EnhanceV2 Epoch {epoch+1}/{epochs}")
        loss_avg = 0.0
        for batch in pbar:
            lat = batch["latent"].to(device)
            tgt = batch["clean"].to(device)
            rid = batch["ridge"].to(device)

            opt.zero_grad()
            rec, rid_log = model(lat)
            loss = l1(torch.sigmoid(rec), tgt) + dice(rid_log, rid)
            loss.backward()
            opt.step()
            loss_avg += float(loss.detach().cpu().item())
            pbar.set_postfix({"loss": f"{loss_avg/ (pbar.n+1):.4f}"})
        torch.save({"generator": model.state_dict()}, "checkpoint_enhance_v2_last.pt")
        print(f"Epoch {epoch+1} done. Avg loss: {loss_avg/len(dl):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    train(args.data_root, args.epochs, args.batch_size, args.lr, args.device)

if __name__ == "__main__":
    main()


