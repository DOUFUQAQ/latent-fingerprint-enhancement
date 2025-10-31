#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版指纹脊线修复训练脚本
三任务联合训练：脊线识别 + 方向场预测 + 脊线修复
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# 数据路径
DATA_ROOT = '../processed_sd04_dataset'

# 训练参数
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 0.0002
IMAGE_SIZE = 256

# 多任务损失权重（已优化：大幅增加分割权重）
LAMBDA_SEGMENT = 200.0    # 脊线分割损失（提高4倍 - 优先学习分割）
LAMBDA_RIDGE = 100.0      # 脊线修复损失
LAMBDA_ORIENT = 50.0      # 方向场损失（适当增加）
LAMBDA_EDGE = 20.0        # 边缘损失（强调脊线边界）

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 输出目录
OUTPUT_DIR = 'checkpoints_advanced'
LOG_DIR = 'logs_advanced'

# ==================== 数据集 ====================

class AdvancedFingerprintDataset(Dataset):
    """增强版指纹数据集 - 包含分割掩码"""
    
    def __init__(self, root_dir, image_size=256, split='train', train_ratio=0.8, augment=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment and (split == 'train')  # 仅在训练时增强
        
        # 获取所有文件
        all_files = sorted(glob.glob(os.path.join(root_dir, 'original', '*.png')))
        
        # 划分数据集
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
        
        # 1. 输入：原始潜指纹（模糊、断裂、有背景噪声）
        input_path = os.path.join(self.root_dir, 'original', filename)
        input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. 目标1：清晰完整的脊线
        target_path = os.path.join(self.root_dir, 'enhanced_binary', filename)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        
        # 3. 目标2：分割掩码（前景/背景）
        mask_path = os.path.join(self.root_dir, 'segmentation', filename)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 4. 目标3：方向场
        orient_path = os.path.join(self.root_dir, 'orientation', filename)
        orient_img = cv2.imread(orient_path, cv2.IMREAD_GRAYSCALE)
        
        # 调整尺寸
        if input_img.shape[0] != self.image_size:
            input_img = cv2.resize(input_img, (self.image_size, self.image_size))
            target_img = cv2.resize(target_img, (self.image_size, self.image_size))
            mask_img = cv2.resize(mask_img, (self.image_size, self.image_size))
            orient_img = cv2.resize(orient_img, (self.image_size, self.image_size))
        
        # 数据增强（仅训练集）
        if self.augment and np.random.rand() > 0.5:
            # 水平翻转
            if np.random.rand() > 0.5:
                input_img = cv2.flip(input_img, 1)
                target_img = cv2.flip(target_img, 1)
                mask_img = cv2.flip(mask_img, 1)
                orient_img = cv2.flip(orient_img, 1)
            
            # 垂直翻转
            if np.random.rand() > 0.5:
                input_img = cv2.flip(input_img, 0)
                target_img = cv2.flip(target_img, 0)
                mask_img = cv2.flip(mask_img, 0)
                orient_img = cv2.flip(orient_img, 0)
            
            # 随机旋转（小角度）
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10)
                h, w = input_img.shape
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                input_img = cv2.warpAffine(input_img, M, (w, h))
                target_img = cv2.warpAffine(target_img, M, (w, h))
                mask_img = cv2.warpAffine(mask_img, M, (w, h))
                orient_img = cv2.warpAffine(orient_img, M, (w, h))
        
        # 归一化
        input_img = (input_img / 255.0 - 0.5) * 2.0     # [-1, 1]
        target_img = (target_img / 255.0 - 0.5) * 2.0   # [-1, 1]
        mask_img = mask_img / 255.0                      # [0, 1]
        orient_img = (orient_img / 255.0 - 0.5) * 2.0   # [-1, 1]
        
        # 转换为 Tensor
        sample = {
            'input': torch.from_numpy(input_img).float().unsqueeze(0),
            'target_ridge': torch.from_numpy(target_img).float().unsqueeze(0),
            'target_mask': torch.from_numpy(mask_img).float().unsqueeze(0),
            'target_orient': torch.from_numpy(orient_img).float().unsqueeze(0),
            'filename': filename
        }
        
        return sample

# ==================== 注意力机制 ====================

class AttentionBlock(nn.Module):
    """注意力模块 - 让模型聚焦于脊线区域"""
    
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch, C, H, W = x.size()
        
        # Query, Key, Value
        query = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)  # [B, HW, C']
        key = self.key(x).view(batch, -1, H * W)                       # [B, C', HW]
        value = self.value(x).view(batch, -1, H * W)                   # [B, C, HW]
        
        # Attention
        attention = torch.softmax(torch.bmm(query, key), dim=-1)       # [B, HW, HW]
        out = torch.bmm(value, attention.permute(0, 2, 1))             # [B, C, HW]
        out = out.view(batch, C, H, W)
        
        # Residual
        out = self.gamma * out + x
        return out

# ==================== 增强版 U-Net ====================

class AdvancedUNet(nn.Module):
    """
    增强版 U-Net - 三任务输出
    1. 脊线分割（识别脊线位置）
    2. 方向场预测（理解脊线方向）
    3. 脊线修复（生成完整清晰脊线）
    """
    
    def __init__(self, in_channels=1, base_channels=64):
        super(AdvancedUNet, self).__init__()
        
        # ==================== 编码器 ====================
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        
        # 瓶颈层 + 注意力
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
        self.attention = AttentionBlock(base_channels * 16)
        
        # ==================== 共享解码器 ====================
        self.dec4 = self.upconv_block(base_channels * 16, base_channels * 8)
        self.dec4_conv = self.conv_block(base_channels * 16, base_channels * 8)
        
        self.dec3 = self.upconv_block(base_channels * 8, base_channels * 4)
        self.dec3_conv = self.conv_block(base_channels * 8, base_channels * 4)
        
        self.dec2 = self.upconv_block(base_channels * 4, base_channels * 2)
        self.dec2_conv = self.conv_block(base_channels * 4, base_channels * 2)
        
        self.dec1 = self.upconv_block(base_channels * 2, base_channels)
        self.dec1_conv = self.conv_block(base_channels * 2, base_channels)
        
        # ==================== 任务专用输出头 ====================
        
        # 任务1：脊线分割（识别哪里是脊线）
        self.segment_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid()  # [0, 1] 分割概率
        )
        
        # 任务2：方向场预测（理解脊线方向）
        self.orient_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Tanh()  # [-1, 1]
        )
        
        # 任务3：脊线修复（生成完整清晰脊线）
        self.repair_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Tanh()  # [-1, 1]
        )
        
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_ch, out_ch):
        """卷积块（带残差连接）"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_ch, out_ch):
        """上采样块"""
        return nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    
    def forward(self, x):
        # ==================== 编码 ====================
        enc1 = self.enc1(x)                    # [B, 64, H, W]
        enc2 = self.enc2(self.pool(enc1))      # [B, 128, H/2, W/2]
        enc3 = self.enc3(self.pool(enc2))      # [B, 256, H/4, W/4]
        enc4 = self.enc4(self.pool(enc3))      # [B, 512, H/8, W/8]
        
        # ==================== 瓶颈 + 注意力 ====================
        bottleneck = self.bottleneck(self.pool(enc4))  # [B, 1024, H/16, W/16]
        bottleneck = self.attention(bottleneck)         # 注意力聚焦脊线
        
        # ==================== 解码（共享） ====================
        dec4 = self.dec4(bottleneck)                    # [B, 512, H/8, W/8]
        dec4 = torch.cat([dec4, enc4], dim=1)           # 跳跃连接
        dec4 = self.dec4_conv(dec4)
        
        dec3 = self.dec3(dec4)                          # [B, 256, H/4, W/4]
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3_conv(dec3)
        
        dec2 = self.dec2(dec3)                          # [B, 128, H/2, W/2]
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2_conv(dec2)
        
        dec1 = self.dec1(dec2)                          # [B, 64, H, W]
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1_conv(dec1)
        
        # ==================== 三个任务输出 ====================
        segment = self.segment_head(dec1)    # 脊线分割 [B, 1, H, W] ∈ [0, 1]
        orient = self.orient_head(dec1)      # 方向场 [B, 1, H, W] ∈ [-1, 1]
        repair = self.repair_head(dec1)      # 脊线修复 [B, 1, H, W] ∈ [-1, 1]
        
        return {
            'segment': segment,   # 任务1：识别脊线位置
            'orient': orient,     # 任务2：预测脊线方向
            'repair': repair      # 任务3：修复完整脊线
        }

# ==================== 损失函数 ====================

class EdgeLoss(nn.Module):
    """边缘损失 - 强调脊线边界"""
    
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Sobel 算子
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        device = pred.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        
        # 数值稳定性：clamp 输入
        pred = torch.clamp(pred, min=-1.0, max=1.0)
        target = torch.clamp(target, min=-1.0, max=1.0)
        
        # 计算梯度
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-8)  # 添加 epsilon 防止 NaN
        
        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-8)
        
        # L1 损失
        return F.l1_loss(pred_edge, target_edge)

class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.edge_loss = EdgeLoss()
    
    def forward(self, outputs, targets):
        """
        outputs: {'segment': ..., 'orient': ..., 'repair': ...}
        targets: {'target_mask': ..., 'target_orient': ..., 'target_ridge': ...}
        """
        # 任务1：脊线分割损失（二值交叉熵）
        segment_pred = outputs['segment']
        segment_target = targets['target_mask']
        
        # 数值稳定性：确保在 [0, 1] 范围内
        segment_pred = torch.clamp(segment_pred, min=1e-7, max=1.0 - 1e-7)
        segment_target = torch.clamp(segment_target, min=0.0, max=1.0)
        
        loss_segment = self.bce_loss(segment_pred, segment_target)
        
        # 任务2：方向场损失（L1）
        orient_pred = outputs['orient']
        orient_target = targets['target_orient']
        loss_orient = self.l1_loss(orient_pred, orient_target)
        
        # 任务3：脊线修复损失（L1 + Edge）
        repair_pred = outputs['repair']
        repair_target = targets['target_ridge']
        loss_repair_l1 = self.l1_loss(repair_pred, repair_target)
        loss_repair_edge = self.edge_loss(repair_pred, repair_target)
        loss_repair = loss_repair_l1 + 0.5 * loss_repair_edge
        
        # 加权组合
        total_loss = (
            LAMBDA_SEGMENT * loss_segment +
            LAMBDA_ORIENT * loss_orient +
            LAMBDA_RIDGE * loss_repair
        )
        
        # 返回详细损失
        losses = {
            'total': total_loss,
            'segment': loss_segment,
            'orient': loss_orient,
            'repair': loss_repair,
            'repair_l1': loss_repair_l1,
            'repair_edge': loss_repair_edge
        }
        
        return losses

# ==================== 训练函数 ====================

def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_losses = {
        'total': 0.0,
        'segment': 0.0,
        'orient': 0.0,
        'repair': 0.0
    }
    
    pbar = tqdm(train_loader, desc="Training")
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
        
        # 检查损失是否为 NaN
        if torch.isnan(losses['total']) or torch.isinf(losses['total']):
            print("\n⚠ 检测到 NaN/Inf 损失，跳过此批次")
            continue
        
        # 反向传播
        losses['total'].backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累积损失
        for key in total_losses.keys():
            total_losses[key] += losses[key].item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': losses['total'].item(),
            'seg': losses['segment'].item(),
            'ori': losses['orient'].item(),
            'rep': losses['repair'].item()
        })
    
    # 平均损失
    for key in total_losses.keys():
        total_losses[key] /= len(train_loader)
    
    return total_losses

def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_losses = {
        'total': 0.0,
        'segment': 0.0,
        'orient': 0.0,
        'repair': 0.0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            targets = {
                'target_mask': batch['target_mask'].to(device),
                'target_orient': batch['target_orient'].to(device),
                'target_ridge': batch['target_ridge'].to(device)
            }
            
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            
            # 跳过 NaN/Inf 损失
            if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                continue
            
            for key in total_losses.keys():
                total_losses[key] += losses[key].item()
    
    for key in total_losses.keys():
        total_losses[key] /= len(val_loader)
    
    return total_losses

# ==================== 主训练流程 ====================

def main(args):
    print("\n" + "=" * 70)
    print("  增强版指纹脊线修复训练")
    print("  三任务联合：脊线识别 + 方向场 + 脊线修复")
    print("=" * 70)
    print("\n配置:")
    print("  数据目录: %s" % args.data_root)
    print("  批大小: %d" % args.batch_size)
    print("  训练轮数: %d" % args.num_epochs)
    print("  学习率: %.6f" % args.lr)
    print("  图像尺寸: %d" % args.image_size)
    print("  设备: %s" % DEVICE)
    print("\n任务权重:")
    print("  脊线修复: %.1f" % LAMBDA_RIDGE)
    print("  脊线分割: %.1f" % LAMBDA_SEGMENT)
    print("  方向场: %.1f" % LAMBDA_ORIENT)
    print("")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = AdvancedFingerprintDataset(
        args.data_root,
        image_size=args.image_size,
        split='train',
        train_ratio=0.8
    )
    
    val_dataset = AdvancedFingerprintDataset(
        args.data_root,
        image_size=args.image_size,
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
    print("创建增强版 U-Net 模型...")
    model = AdvancedUNet(in_channels=1, base_channels=64).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print("✓ 模型参数量: %.2fM\n" % (num_params / 1e6))
    
    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)  # 添加 L2 正则化
    criterion = CombinedLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 训练
    print("开始训练...\n")
    best_val_loss = float('inf')
    patience = 15  # 早停：验证损失 15 轮不下降就停止
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(args.num_epochs):
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
        
        # 保存最佳模型 + 早停
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0  # 重置计数器
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses,
                'train_loss': train_losses
            }, checkpoint_path)
            print("  ✓ 保存最佳模型 (验证损失: %.6f)\n" % val_losses['total'])
        else:
            patience_counter += 1
            print("  ⚠ 验证损失未改善 (%d/%d)\n" % (patience_counter, patience))
            
            # 早停检查
            if patience_counter >= patience:
                print("=" * 70)
                print("  ⚠ 早停触发：验证损失连续 %d 轮未改善" % patience)
                print("  最佳验证损失: %.6f (Epoch %d)" % (best_val_loss, epoch - patience + 1))
                print("=" * 70)
                break
        
        # 定期保存
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(args.output_dir, 'model_epoch_%d.pth' % (epoch + 1))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses,
                'train_loss': train_losses
            }, checkpoint_path)
            print("  ✓ 保存检查点: epoch_%d\n" % (epoch + 1))
    
    # 保存训练历史
    history_path = os.path.join(args.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # 转换为可序列化格式
        history_json = {
            'train': [{'total': float(h['total']), 'segment': float(h['segment']), 
                      'orient': float(h['orient']), 'repair': float(h['repair'])} 
                     for h in history['train']],
            'val': [{'total': float(h['total']), 'segment': float(h['segment']), 
                    'orient': float(h['orient']), 'repair': float(h['repair'])} 
                   for h in history['val']],
            'config': vars(args)
        }
        json.dump(history_json, f, indent=4)
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print("最佳验证损失: %.6f" % best_val_loss)
    print("模型保存于: %s" % args.output_dir)
    print("=" * 70 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='增强版指纹脊线修复训练')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--log_dir', type=str, default=LOG_DIR)
    
    args = parser.parse_args()
    main(args)
