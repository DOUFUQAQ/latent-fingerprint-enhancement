"""
论文级别的增强GAN训练
整合所有论文中提到的技术要点：
1. 多级渐进式判别器
2. 感知损失和风格损失
3. 方向场约束损失
4. 改进的数据合成策略
5. U-Net双头输出
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path

# 导入我们的模块
from latentfp.data.sd4_dataset import SD4Dataset
from latentfp.models.unet import UNetDualHead
from latentfp.models.progressive_discriminators import ProgressiveDiscriminator
from latentfp.losses import VGGPerceptualLoss, OrientationFieldLoss, TotalVariationLoss
from latentfp.transforms.advanced_synthesis import paper_style_latent_synthesis

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice

def enhanced_gan_train(
    data_root: str,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 2e-4,
    resume_g: str = None,
    resume_d: str = None,
    device: str = "cuda"
):
    """增强的GAN训练，集成论文中的所有技术"""
    
    # 设备健壮性：CUDA不可用则回退CPU
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    # 数据加载
    print(f"Loading dataset from {data_root}")
    dataset = SD4Dataset(data_root, crop_size=512, latent_transform=paper_style_latent_synthesis)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 模型初始化
    print("Initializing models...")
    G = UNetDualHead(in_channels=1).to(device)
    D = ProgressiveDiscriminator(in_channels=2).to(device)  # 输入图像+脊线
    
    # 损失函数
    # 使用Hinge GAN 更稳健
    def d_hinge_loss(real_logits, fake_logits):
        loss_real = torch.relu(1.0 - real_logits).mean()
        loss_fake = torch.relu(1.0 + fake_logits).mean()
        return loss_real + loss_fake
    
    def g_hinge_loss(fake_logits):
        return (-fake_logits).mean()
    criterion_l1 = nn.L1Loss()
    criterion_dice = DiceLoss()
    
    # 高级损失函数
    perceptual_loss = VGGPerceptualLoss().to(device)
    perceptual_loss.eval()
    orientation_loss = OrientationFieldLoss().to(device)
    tv_loss = TotalVariationLoss().to(device)
    
    # 优化器
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 学习率调度器
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs)
    
    # 恢复检查点
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_g and Path(resume_g).exists():
        print(f"Resuming generator from {resume_g}")
        checkpoint = torch.load(resume_g, map_location=device)
        G.load_state_dict(checkpoint['generator'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    if resume_d and Path(resume_d).exists():
        print(f"Resuming discriminator from {resume_d}")
        checkpoint_d = torch.load(resume_d, map_location=device)
        D.load_state_dict(checkpoint_d['discriminator'])
    
    # 混合精度训练（分别为D、G使用独立的scaler，兼容老版本PyTorch步进约束）
    scaler_D = torch.cuda.amp.GradScaler()
    scaler_G = torch.cuda.amp.GradScaler()
    
    # 损失权重
    lambda_l1 = 100.0
    lambda_perceptual = 10.0
    lambda_style = 10.0
    lambda_orientation = 5.0
    lambda_tv = 0.1
    lambda_dice = 10.0
    
    print("Starting enhanced GAN training...")
    r1_gamma = 10.0
    adv_warmup_epochs = max(1, epochs // 10)  # 前10%epoch进行对抗权重warmup
    
    for epoch in range(start_epoch, epochs):
        # 训练模式
        G.train()
        D.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        with tqdm(dataloader, desc=f"Enhanced GAN Epoch {epoch+1}/{epochs}") as pbar:
            for i, batch in enumerate(pbar):
                latent = batch['latent'].to(device).float()
                clean = batch['clean'].to(device).float()
                ridge = batch['ridge'].to(device).float()
                
                batch_size_cur = latent.size(0)
                
                # ================== 训练判别器 ==================
                optimizer_D.zero_grad()
                
                with torch.cuda.amp.autocast():
                    # 生成假图像
                    with torch.no_grad():
                        restored_fake, ridge_fake_logits = G(latent)
                        ridge_fake = torch.sigmoid(ridge_fake_logits)
                    
                    # 真实和虚假样本的判别
                    real_input = torch.nan_to_num(torch.cat([clean, ridge], dim=1))
                    fake_input = torch.nan_to_num(torch.cat([restored_fake, ridge_fake], dim=1))
                    
                    # 多级判别器损失
                    d_real_losses = D(real_input)
                    d_fake_losses = D(fake_input)
                    
                    d_loss = 0.0
                    num_scales = float(len(d_real_losses))
                    for d_real, d_fake in zip(d_real_losses, d_fake_losses):
                        loss = d_hinge_loss(d_real, d_fake)
                        d_loss = d_loss + loss
                    d_loss = d_loss / num_scales

                    # R1 正则（只对真实样本）
                    r1_loss = 0.0
                    real_input.requires_grad_(True)
                    r1_real_losses = D(real_input)
                    r1_sum = 0.0
                    for r1_real in r1_real_losses:
                        r1_sum = r1_sum + r1_real.sum()
                    grads = torch.autograd.grad(r1_sum, real_input, create_graph=True, retain_graph=True, only_inputs=True)[0]
                    r1_penalty = grads.pow(2).view(grads.size(0), -1).sum(1).mean()
                    r1_loss = (r1_gamma * 0.5) * r1_penalty
                    real_input = real_input.detach()  # 释放图
                    d_total = d_loss + r1_loss
                    # 对抗warmup
                    adv_weight = min(1.0, (epoch + 1) / adv_warmup_epochs)
                    d_total = adv_weight * d_total
                
                # 数值安全与步进：每次最多调用一次step/update
                skip_d = bool(torch.isnan(d_total) or torch.isinf(d_total))
                used_scale_d = False
                if not skip_d:
                    scaler_D.scale(d_total).backward()
                    used_scale_d = True
                    # 梯度裁剪，避免梯度爆炸
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                    scaler_D.step(optimizer_D)
                else:
                    optimizer_D.zero_grad(set_to_none=True)
                if used_scale_d:
                    scaler_D.update()
                
                # ================== 训练生成器 ==================
                optimizer_G.zero_grad()
                
                with torch.cuda.amp.autocast():
                    # 前向传播
                    restored, ridge_logits = G(latent)
                    ridge_pred = torch.sigmoid(ridge_logits)
                    
                    # 对抗损失
                    fake_input = torch.nan_to_num(torch.cat([restored, ridge_pred], dim=1))
                    g_fake_losses = D(fake_input)
                    
                    g_adv_loss = 0.0
                    num_scales = float(len(g_fake_losses))
                    for g_fake in g_fake_losses:
                        g_adv_loss = g_adv_loss + g_hinge_loss(g_fake)
                    g_adv_loss = g_adv_loss / num_scales
                    
                    # L1重建损失
                    l1_loss = criterion_l1(restored, clean)
                    
                    # 脊线预测损失
                    dice_loss = criterion_dice(ridge_logits, ridge)
                
                # 重要：在float32下计算感知/风格/方向场/TV，避免半精度溢出
                with torch.cuda.amp.autocast(enabled=False):
                    restored_f32 = torch.clamp(restored.detach(), 0.0, 1.0).float()
                    clean_f32 = torch.clamp(clean.detach(), 0.0, 1.0).float()
                    
                    perceptual_l, style_l = perceptual_loss(restored_f32, clean_f32)
                    orient_loss = orientation_loss(restored_f32, clean_f32)
                    tv_l = tv_loss(restored_f32)
                    
                    # 汇总到float32再求和
                    g_loss = (
                        (adv_weight * g_adv_loss).float() +
                        lambda_l1 * l1_loss.float() +
                        lambda_perceptual * perceptual_l +
                        lambda_style * style_l +
                        lambda_orientation * orient_loss +
                        lambda_tv * tv_l +
                        lambda_dice * dice_loss.float()
                    )
                
                skip_g = bool(torch.isnan(g_loss) or torch.isinf(g_loss))
                used_scale_g = False
                if not skip_g:
                    scaler_G.scale(g_loss).backward()
                    used_scale_g = True
                    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                    scaler_G.step(optimizer_G)
                else:
                    optimizer_G.zero_grad(set_to_none=True)
                if used_scale_g:
                    scaler_G.update()
                
                # 记录损失
                epoch_g_loss += float(g_loss.detach().cpu().item())
                epoch_d_loss += float(d_total.detach().cpu().item())
                
                # 更新进度条
                pbar.set_postfix({
                    'D': f'{d_loss.item():.3f}',
                    'G': f'{g_loss.item():.3f}', 
                    'L1': f'{l1_loss.item():.3f}',
                    'Perceptual': f'{perceptual_l.item():.3f}',
                    'Orientation': f'{orient_loss.item():.3f}',
                    'Dice': f'{dice_loss.item():.3f}'
                })
        
        # 学习率调度
        scheduler_G.step()
        scheduler_D.step()
        
        # 验证
        G.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if val_batches >= 20:  # 限制验证批次
                    break
                    
                latent = batch['latent'].to(device).float()
                clean = batch['clean'].to(device).float()
                ridge = batch['ridge'].to(device).float()
                
                restored, ridge_logits = G(latent)
                
                # 计算验证损失（主要指标）
                l1_val = criterion_l1(restored, clean)
                dice_val = criterion_dice(ridge_logits, ridge)
                perceptual_val, style_val = perceptual_loss(restored, clean)
                orient_val = orientation_loss(restored, clean)
                
                batch_val_loss = (l1_val + 0.1 * dice_val + 
                                0.1 * perceptual_val + 0.1 * style_val + 
                                0.05 * orient_val)
                val_loss += batch_val_loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        
        print(f"Epoch {epoch+1} - Train G: {epoch_g_loss/len(dataloader):.4f}, "
              f"Train D: {epoch_d_loss/len(dataloader):.4f}, Val: {avg_val_loss:.4f}")
        
        # 保存检查点
        checkpoint = {
            'generator': G.state_dict(),
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'optimizer_G': optimizer_G.state_dict(),
            'scheduler_G': scheduler_G.state_dict(),
        }
        
        torch.save(checkpoint, 'checkpoint_enhanced_gan_last.pt')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, 'checkpoint_enhanced_gan_best.pt')
            print("Saved enhanced GAN best checkpoint")
        
        # 保存判别器
        if (epoch + 1) % 5 == 0:
            torch.save({
                'discriminator': D.state_dict(),
                'epoch': epoch + 1,
                'optimizer_D': optimizer_D.state_dict(),
                'scheduler_D': scheduler_D.state_dict(),
            }, f'checkpoint_enhanced_discriminator_epoch_{epoch+1}.pt')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--resume_g', type=str, default=None)
    parser.add_argument('--resume_d', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    enhanced_gan_train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_g=args.resume_g,
        resume_d=args.resume_d,
        device=args.device
    )

if __name__ == '__main__':
    main()
