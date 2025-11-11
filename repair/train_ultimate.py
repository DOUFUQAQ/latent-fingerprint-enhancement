#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
终极版训练脚本
综合所有改进：大模型 + 感知损失 + 数据增强
目标：将验证损失降到50以下
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from repair.train_with_augmentation import EnhancedFingerprintDataset, AdvancedUNet
from repair.perceptual_loss import EnhancedLoss

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_DIR = 'logs_ultimate'

# 损失权重
LAMBDA_SEGMENT = 200.0
LAMBDA_ORIENT = 50.0
LAMBDA_RIDGE = 100.0

class UltimateLoss(nn.Module):
    """终极损失函数：分割 + 方向 + 修复（含感知损失）"""
    
    def __init__(self):
        super(UltimateLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.enhanced_loss = EnhancedLoss(lambda_mse=1.0, lambda_perceptual=0.1)
    
    def forward(self, outputs, targets):
        # 分割损失（BCE）
        segment_loss = self.bce_loss(outputs['segment'], targets['target_mask'])
        
        # 方向损失（MSE）
        orient_loss = self.mse_loss(outputs['orient'], targets['target_orient'])
        
        # 修复损失（MSE + 感知）
        repair_loss, repair_dict = self.enhanced_loss(outputs['repair'], targets['target_ridge'])
        
        # 加权总损失
        total_loss = (
            LAMBDA_SEGMENT * segment_loss +
            LAMBDA_ORIENT * orient_loss +
            LAMBDA_RIDGE * repair_loss
        )
        
        return {
            'total': total_loss,
            'segment': segment_loss,
            'orient': orient_loss,
            'repair': repair_loss,
            'repair_mse': repair_dict['mse'],
            'repair_perceptual': repair_dict['perceptual']
        }

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    segment_loss = 0.0
    orient_loss = 0.0
    repair_loss = 0.0
    
    pbar = tqdm(dataloader, desc="训练")
    for batch in pbar:
        # 准备数据
        inputs = batch['input'].to(device)
        targets = {
            'target_mask': batch['target_mask'].to(device),
            'target_orient': batch['target_orient'].to(device),
            'target_ridge': batch['target_ridge'].to(device)
        }
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 计算损失
        losses = criterion(outputs, targets)
        
        # 反向传播
        losses['total'].backward()
        optimizer.step()
        
        # 统计
        total_loss += losses['total'].item()
        segment_loss += losses['segment'].item()
        orient_loss += losses['orient'].item()
        repair_loss += losses['repair'].item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': losses['total'].item(),
            'seg': losses['segment'].item(),
            'ori': losses['orient'].item(),
            'rep': losses['repair'].item(),
            'rep_mse': losses['repair_mse'],
            'rep_percep': losses['repair_perceptual']
        })
    
    num_batches = len(dataloader)
    return {
        'total': total_loss / num_batches,
        'segment': segment_loss / num_batches,
        'orient': orient_loss / num_batches,
        'repair': repair_loss / num_batches
    }

def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    
    total_loss = 0.0
    segment_loss = 0.0
    orient_loss = 0.0
    repair_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = {
                'target_mask': batch['target_mask'].to(device),
                'target_orient': batch['target_orient'].to(device),
                'target_ridge': batch['target_ridge'].to(device)
            }
            
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            
            total_loss += losses['total'].item()
            segment_loss += losses['segment'].item()
            orient_loss += losses['orient'].item()
            repair_loss += losses['repair'].item()
    
    num_batches = len(dataloader)
    return {
        'total': total_loss / num_batches,
        'segment': segment_loss / num_batches,
        'orient': orient_loss / num_batches,
        'repair': repair_loss / num_batches
    }

def find_latest_checkpoint(output_dir):
    """查找最新的检查点"""
    import glob
    
    # 查找所有检查点
    checkpoints = glob.glob(os.path.join(output_dir, 'model_ultimate_epoch_*.pth'))
    if not checkpoints:
        # 检查是否有最佳模型
        best_model = os.path.join(output_dir, 'best_model_ultimate.pth')
        if os.path.exists(best_model):
            return best_model
        return None
    
    # 按epoch排序，返回最新的
    checkpoints.sort(key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0]))
    return checkpoints[-1]

def main(args):
    print("\n" + "=" * 70)
    print("  🚀 终极版训练")
    print("  综合所有改进，目标：验证损失 < 50")
    print("=" * 70)
    print("\n配置:")
    print("  数据目录: %s" % args.data_root)
    print("  批大小: %d" % args.batch_size)
    print("  训练轮数: %d" % args.num_epochs)
    print("  学习率: %.6f" % args.lr)
    print("  图像尺寸: %d" % args.image_size)
    print("  基础通道数: %d (参数量: ~%.0fM)" % (args.base_channels, (args.base_channels/64)**2 * 32))
    print("  设备: %s" % DEVICE)
    print("  自动恢复: %s" % ("是" if args.auto_resume else "否"))
    print("\n改进点:")
    print("  ✓ 数据增强: 遮挡 + 模糊 + 噪声")
    print("  ✓ 感知损失: VGG特征匹配")
    print("  ✓ 大模型: base_channels=%d" % args.base_channels)
    print("  ✓ 断点续训: 自动保存和恢复")
    print("\n损失权重:")
    print("  分割: %.1f" % LAMBDA_SEGMENT)
    print("  修复: %.1f (MSE + 0.1×感知)" % LAMBDA_RIDGE)
    print("  方向: %.1f" % LAMBDA_ORIENT)
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = EnhancedFingerprintDataset(
        root_dir=args.data_root,
        image_size=args.image_size,
        split='train',
        augment=True,
        strong_augment=True
    )
    val_dataset = EnhancedFingerprintDataset(
        root_dir=args.data_root,
        image_size=args.image_size,
        split='val',
        augment=False,
        strong_augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print("✓ 训练集: %d 张" % len(train_dataset))
    print("✓ 验证集: %d 张\n" % len(val_dataset))
    
    # 创建模型
    print("创建模型...")
    model = AdvancedUNet(in_channels=1, base_channels=args.base_channels).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print("✓ 模型参数量: %.2fM\n" % (num_params / 1e6))
    
    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    criterion = UltimateLoss().to(DEVICE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 加载检查点（自动恢复或手动指定）
    start_epoch = 0
    best_val_loss = float('inf')
    resume_path = None
    
    # 1. 优先使用手动指定的检查点
    if args.resume and os.path.exists(args.resume):
        resume_path = args.resume
        print(f"📂 手动指定检查点: {resume_path}")
    
    # 2. 自动查找最新检查点
    elif args.auto_resume:
        resume_path = find_latest_checkpoint(args.output_dir)
        if resume_path:
            print(f"📂 自动检测到检查点: {resume_path}")
    
    # 3. 加载检查点
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 正在恢复训练...")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✓ 恢复成功")
        print(f"  起始轮次: Epoch {start_epoch}")
        print(f"  当前最佳验证损失: {best_val_loss:.6f}")
        if 'val_loss' in checkpoint:
            val_loss = checkpoint['val_loss']
            print(f"  上次验证损失:")
            print(f"    - 总损失: {val_loss['total']:.6f}")
            print(f"    - 分割: {val_loss['segment']:.6f}")
            print(f"    - 方向: {val_loss['orient']:.6f}")
            print(f"    - 修复: {val_loss['repair']:.6f}")
        print()
    else:
        print("🆕 从头开始训练\n")
    
    # 训练
    print("开始训练...\n")
    patience = 20
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(start_epoch, args.num_epochs):
        print("Epoch [%d/%d]" % (epoch + 1, args.num_epochs))
        
        # 训练
        train_losses = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # 验证
        val_losses = validate(model, val_loader, criterion, DEVICE)
        
        # 学习率调度
        scheduler.step()
        
        # 记录
        history['train'].append(train_losses)
        history['val'].append(val_losses)
        
        # 打印
        print("  [训练] 总损失: %.6f | 分割: %.6f | 方向: %.6f | 修复: %.6f" % 
              (train_losses['total'], train_losses['segment'], 
               train_losses['orient'], train_losses['repair']))
        print("  [验证] 总损失: %.6f | 分割: %.6f | 方向: %.6f | 修复: %.6f" % 
              (val_losses['total'], val_losses['segment'], 
               val_losses['orient'], val_losses['repair']))
        print("  学习率: %.6f\n" % scheduler.get_last_lr()[0])
        
        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            checkpoint_path = os.path.join(args.output_dir, 'best_model_ultimate.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses,
                'train_loss': train_losses,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print("  ✓ 保存最佳模型 (验证损失: %.6f)\n" % val_losses['total'])
        else:
            patience_counter += 1
            print("  ⚠ 验证损失未改善 (%d/%d)\n" % (patience_counter, patience))
            
            if patience_counter >= patience:
                print("=" * 70)
                print("  ⚠ 早停触发")
                print("  最佳验证损失: %.6f" % best_val_loss)
                print("=" * 70)
                break
        
        # 定期保存（用于断点续训）
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, 'model_ultimate_epoch_%d.pth' % (epoch + 1))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses,
                'train_loss': train_losses,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print("  💾 保存检查点: epoch_%d.pth\n" % (epoch + 1))
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print("最佳验证损失: %.6f" % best_val_loss)
    print("模型保存于: %s" % args.output_dir)
    print("=" * 70)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='终极版训练（支持断点续训）')
    parser.add_argument('--data_root', type=str, default='../processed_sd04_dataset')
    parser.add_argument('--output_dir', type=str, default='checkpoints_ultimate')
    parser.add_argument('--batch_size', type=int, default=4, help='减小以适应大模型')
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--base_channels', type=int, default=128, help='128=大模型, 64=标准')
    parser.add_argument('--resume', type=str, default='', help='手动指定恢复检查点路径')
    parser.add_argument('--auto_resume', action='store_true', default=True, help='自动从最新检查点恢复（默认开启）')
    parser.add_argument('--no_auto_resume', dest='auto_resume', action='store_false', help='禁用自动恢复')
    parser.add_argument('--save_freq', type=int, default=10, help='保存检查点频率（每N轮保存一次）')
    
    args = parser.parse_args()
    main(args)
