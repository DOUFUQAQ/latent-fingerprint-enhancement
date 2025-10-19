#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指纹脊线修复模型训练脚本
支持单任务（仅脊线修复）和多任务（脊线+方向场）
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import glob
from tqdm import tqdm
import argparse
from datetime import datetime
import json

# ==================== 配置 ====================

# 训练模式
TASK_MODE = 'single'  # 'single': 仅脊线修复, 'multi': 脊线+方向场

# 数据路径
DATA_ROOT = '../processed_sd04_dataset'

# 训练参数
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 0.0002
IMAGE_SIZE = 256

# 损失函数权重
LAMBDA_L1 = 100.0
LAMBDA_ORIENT = 10.0  # 方向场损失权重（多任务模式）

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 输出目录
OUTPUT_DIR = 'checkpoints'
LOG_DIR = 'logs'

# ==================== 数据集 ====================

class FingerprintDataset(Dataset):
    """指纹修复数据集"""
    
    def __init__(self, root_dir, image_size=256, mode='single', split='train', train_ratio=0.8):
        """
        Args:
            root_dir: 数据根目录
            image_size: 图像尺寸
            mode: 'single' 或 'multi'
            split: 'train' 或 'val'
            train_ratio: 训练集比例
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode
        
        # 获取所有文件
        all_files = sorted(glob.glob(os.path.join(root_dir, 'original', '*.png')))
        
        # 划分训练集和验证集
        n_train = int(len(all_files) * train_ratio)
        if split == 'train':
            self.files = all_files[:n_train]
        else:
            self.files = all_files[n_train:]
        
        print("  [%s] 加载 %d 张图像" % (split.upper(), len(self.files)))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = os.path.basename(self.files[idx])
        
        # 读取输入图像（模糊、断裂）
        input_path = os.path.join(self.root_dir, 'original', filename)
        input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        # 读取目标图像（清晰脊线）
        target_path = os.path.join(self.root_dir, 'enhanced_binary', filename)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        
        # 调整尺寸
        if input_img.shape[0] != self.image_size:
            input_img = cv2.resize(input_img, (self.image_size, self.image_size))
            target_img = cv2.resize(target_img, (self.image_size, self.image_size))
        
        # 归一化到 [-1, 1]
        input_img = (input_img / 255.0 - 0.5) * 2.0
        target_img = (target_img / 255.0 - 0.5) * 2.0
        
        # 转换为 Tensor
        input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_img).float().unsqueeze(0)
        
        sample = {
            'input': input_tensor,
            'target': target_tensor,
            'filename': filename
        }
        
        # 多任务模式：加载方向场
        if self.mode == 'multi':
            orient_path = os.path.join(self.root_dir, 'orientation', filename)
            if os.path.exists(orient_path):
                orient_img = cv2.imread(orient_path, cv2.IMREAD_GRAYSCALE)
                if orient_img.shape[0] != self.image_size:
                    orient_img = cv2.resize(orient_img, (self.image_size, self.image_size))
                orient_img = (orient_img / 255.0 - 0.5) * 2.0
                sample['orientation'] = torch.from_numpy(orient_img).float().unsqueeze(0)
            else:
                # 如果方向场不存在，使用零张量
                sample['orientation'] = torch.zeros(1, self.image_size, self.image_size)
        
        return sample

# ==================== 模型定义 ====================

class UNet(nn.Module):
    """U-Net 模型（支持单任务和多任务）"""
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_tasks=1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 每个任务的输出通道数
            base_channels: 基础通道数
            num_tasks: 任务数量（1=单任务, 2=多任务）
        """
        super(UNet, self).__init__()
        self.num_tasks = num_tasks
        
        # 编码器
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        
        # 瓶颈层
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
        
        # 解码器（共享）
        self.dec4 = self.upconv_block(base_channels * 16, base_channels * 8)
        self.dec3 = self.upconv_block(base_channels * 8, base_channels * 4)
        self.dec2 = self.upconv_block(base_channels * 4, base_channels * 2)
        self.dec1 = self.upconv_block(base_channels * 2, base_channels)
        
        # 输出层
        self.out_ridge = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        if num_tasks == 2:
            # 多任务：额外的方向场输出
            self.out_orient = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.tanh = nn.Tanh()
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # 瓶颈
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # 解码（跳跃连接）
        dec4 = self.dec4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.conv_block(dec4.size(1), dec4.size(1) // 2)(dec4)
        
        dec3 = self.dec3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.conv_block(dec3.size(1), dec3.size(1) // 2)(dec3)
        
        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.conv_block(dec2.size(1), dec2.size(1) // 2)(dec2)
        
        dec1 = self.dec1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.conv_block(dec1.size(1), dec1.size(1) // 2)(dec1)
        
        # 输出
        out_ridge = self.tanh(self.out_ridge(dec1))
        
        if self.num_tasks == 2:
            out_orient = self.tanh(self.out_orient(dec1))
            return out_ridge, out_orient
        else:
            return out_ridge

# ==================== 训练函数 ====================

def train_epoch(model, train_loader, optimizer, criterion, device, mode='single'):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        if mode == 'single':
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        else:
            # 多任务
            out_ridge, out_orient = model(inputs)
            loss_ridge = criterion(out_ridge, targets)
            
            if 'orientation' in batch:
                orient_targets = batch['orientation'].to(device)
                loss_orient = criterion(out_orient, orient_targets)
                loss = loss_ridge + LAMBDA_ORIENT / LAMBDA_L1 * loss_orient
            else:
                loss = loss_ridge
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device, mode='single'):
    """验证"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            if mode == 'single':
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            else:
                out_ridge, out_orient = model(inputs)
                loss_ridge = criterion(out_ridge, targets)
                
                if 'orientation' in batch:
                    orient_targets = batch['orientation'].to(device)
                    loss_orient = criterion(out_orient, orient_targets)
                    loss = loss_ridge + LAMBDA_ORIENT / LAMBDA_L1 * loss_orient
                else:
                    loss = loss_ridge
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# ==================== 主训练流程 ====================

def main(args):
    print("\n" + "=" * 70)
    print("  指纹脊线修复模型训练")
    print("=" * 70)
    print("\n配置:")
    print("  数据目录: %s" % args.data_root)
    print("  训练模式: %s" % ('单任务（仅脊线）' if args.mode == 'single' else '多任务（脊线+方向场）'))
    print("  批大小: %d" % args.batch_size)
    print("  训练轮数: %d" % args.num_epochs)
    print("  学习率: %.6f" % args.lr)
    print("  图像尺寸: %d" % args.image_size)
    print("  设备: %s" % DEVICE)
    print("")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = FingerprintDataset(
        args.data_root,
        image_size=args.image_size,
        mode=args.mode,
        split='train',
        train_ratio=0.8
    )
    
    val_dataset = FingerprintDataset(
        args.data_root,
        image_size=args.image_size,
        mode=args.mode,
        split='val',
        train_ratio=0.8
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
    num_tasks = 1 if args.mode == 'single' else 2
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        num_tasks=num_tasks
    ).to(DEVICE)
    
    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    print("✓ 模型参数量: %.2fM\n" % (num_params / 1e6))
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 训练
    print("开始训练...\n")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        print("Epoch [%d/%d]" % (epoch + 1, args.num_epochs))
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, args.mode)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, DEVICE, args.mode)
        
        # 学习率调度
        scheduler.step()
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print("  训练损失: %.6f" % train_loss)
        print("  验证损失: %.6f" % val_loss)
        print("  学习率: %.6f\n" % scheduler.get_last_lr()[0])
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'mode': args.mode
            }, checkpoint_path)
            print("  ✓ 保存最佳模型 (验证损失: %.6f)\n" % val_loss)
        
        # 定期保存
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(args.output_dir, 'model_epoch_%d.pth' % (epoch + 1))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'mode': args.mode
            }, checkpoint_path)
            print("  ✓ 保存检查点: epoch_%d\n" % (epoch + 1))
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': vars(args)
    }
    
    history_path = os.path.join(args.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print("最佳验证损失: %.6f" % best_val_loss)
    print("模型保存于: %s" % args.output_dir)
    print("=" * 70 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='指纹脊线修复模型训练')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT, help='数据根目录')
    parser.add_argument('--mode', type=str, default=TASK_MODE, choices=['single', 'multi'], 
                       help='训练模式: single=仅脊线, multi=脊线+方向场')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='训练轮数')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='学习率')
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE, help='图像尺寸')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='模型输出目录')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='日志目录')
    
    args = parser.parse_args()
    
    main(args)
