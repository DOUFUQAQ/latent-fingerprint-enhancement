#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进版训练脚本
添加强化数据增强：遮挡、模糊、噪声
专门训练处理严重损坏图像的能力
"""

import os
import sys
import cv2
import numpy as np
import random

# 复制原训练脚本的所有导入和类
from repair.train_advanced_repair import *

class EnhancedFingerprintDataset(AdvancedFingerprintDataset):
    """增强版数据集 - 添加遮挡和模糊增强"""
    
    def __init__(self, root_dir, image_size=256, split='train', train_ratio=0.8, augment=True, 
                 strong_augment=True):
        super().__init__(root_dir, image_size, split, train_ratio, augment)
        self.strong_augment = strong_augment and (split == 'train')
    
    def apply_occlusion(self, img):
        """应用随机遮挡"""
        occluded = img.copy()
        h, w = img.shape
        
        # 随机选择遮挡类型
        occlusion_type = random.choice(['blocks', 'stripes', 'noise', 'large_area'])
        
        if occlusion_type == 'blocks':
            # 随机方块遮挡
            num_blocks = random.randint(2, 6)
            for _ in range(num_blocks):
                size = random.randint(15, 50)
                x = random.randint(0, max(1, w - size))
                y = random.randint(0, max(1, h - size))
                occluded[y:y+size, x:x+size] = 0
        
        elif occlusion_type == 'stripes':
            # 条纹遮挡
            num_stripes = random.randint(2, 5)
            for _ in range(num_stripes):
                width = random.randint(3, 12)
                if random.choice([True, False]):
                    y = random.randint(0, h - width)
                    occluded[y:y+width, :] = 0
                else:
                    x = random.randint(0, w - width)
                    occluded[:, x:x+width] = 0
        
        elif occlusion_type == 'noise':
            # 随机噪声遮挡
            noise_ratio = random.uniform(0.1, 0.4)
            mask = np.random.rand(h, w) < noise_ratio
            occluded[mask] = 0
        
        elif occlusion_type == 'large_area':
            # 大面积遮挡
            num_areas = random.randint(1, 2)
            for _ in range(num_areas):
                size_h = random.randint(h//5, h//3)
                size_w = random.randint(w//5, w//3)
                x = random.randint(0, w - size_w)
                y = random.randint(0, h - size_h)
                occluded[y:y+size_h, x:x+size_w] = 0
        
        return occluded
    
    def apply_blur(self, img):
        """应用随机模糊"""
        blur_type = random.choice(['gaussian', 'motion', 'median'])
        
        if blur_type == 'gaussian':
            # 高斯模糊
            kernel_size = random.choice([3, 5, 7])
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        elif blur_type == 'motion':
            # 运动模糊
            kernel_size = random.choice([5, 7, 9])
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            blurred = cv2.filter2D(img, -1, kernel)
        
        else:  # median
            # 中值滤波
            kernel_size = random.choice([3, 5])
            blurred = cv2.medianBlur(img, kernel_size)
        
        return blurred
    
    def apply_noise(self, img):
        """应用随机噪声"""
        noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
        
        if noise_type == 'gaussian':
            # 高斯噪声
            noise = np.random.randn(*img.shape) * random.uniform(5, 15)
            noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        elif noise_type == 'salt_pepper':
            # 椒盐噪声
            prob = random.uniform(0.01, 0.05)
            noisy = img.copy()
            salt_mask = np.random.rand(*img.shape) < prob
            pepper_mask = np.random.rand(*img.shape) < prob
            noisy[salt_mask] = 255
            noisy[pepper_mask] = 0
        
        else:  # speckle
            # 斑点噪声
            noise = np.random.randn(*img.shape) * img * random.uniform(0.05, 0.15)
            noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return noisy
    
    def __getitem__(self, idx):
        filepath = self.files[idx]
        filename = os.path.basename(filepath)
        
        # 读取所有数据
        input_path = os.path.join(self.root_dir, 'original', filename)
        target_path = os.path.join(self.root_dir, 'enhanced_binary', filename)
        mask_path = os.path.join(self.root_dir, 'segmentation', filename)  # 修正：使用segmentation而不是mask
        orient_path = os.path.join(self.root_dir, 'orientation', filename)
        
        input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        orient_img = cv2.imread(orient_path, cv2.IMREAD_GRAYSCALE)
        
        # 错误处理：检查是否读取成功
        if input_img is None:
            raise ValueError(f"无法读取: {input_path}")
        if target_img is None:
            raise ValueError(f"无法读取: {target_path}")
        if mask_img is None:
            raise ValueError(f"无法读取: {mask_path}")
        if orient_img is None:
            raise ValueError(f"无法读取: {orient_path}")
        
        # 调整尺寸
        if input_img.shape[0] != self.image_size:
            input_img = cv2.resize(input_img, (self.image_size, self.image_size))
            target_img = cv2.resize(target_img, (self.image_size, self.image_size))
            mask_img = cv2.resize(mask_img, (self.image_size, self.image_size))
            orient_img = cv2.resize(orient_img, (self.image_size, self.image_size))
        
        # 原有的基础数据增强
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
            
            # 随机旋转
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10)
                h, w = input_img.shape
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                input_img = cv2.warpAffine(input_img, M, (w, h))
                target_img = cv2.warpAffine(target_img, M, (w, h))
                mask_img = cv2.warpAffine(mask_img, M, (w, h))
                orient_img = cv2.warpAffine(orient_img, M, (w, h))
        
        # 强化数据增强（新增）
        if self.strong_augment:
            # 50% 概率应用遮挡
            if np.random.rand() > 0.5:
                input_img = self.apply_occlusion(input_img)
            
            # 30% 概率应用模糊
            if np.random.rand() > 0.7:
                input_img = self.apply_blur(input_img)
            
            # 20% 概率应用噪声
            if np.random.rand() > 0.8:
                input_img = self.apply_noise(input_img)
        
        # 归一化
        input_img = (input_img / 255.0 - 0.5) * 2.0
        target_img = (target_img / 255.0 - 0.5) * 2.0
        mask_img = mask_img / 255.0
        orient_img = (orient_img / 255.0 - 0.5) * 2.0
        
        # 转换为 Tensor
        sample = {
            'input': torch.from_numpy(input_img).float().unsqueeze(0),
            'target_ridge': torch.from_numpy(target_img).float().unsqueeze(0),
            'target_mask': torch.from_numpy(mask_img).float().unsqueeze(0),
            'target_orient': torch.from_numpy(orient_img).float().unsqueeze(0),
            'filename': filename
        }
        
        return sample

def main(args):
    print("\n" + "=" * 70)
    print("  增强版指纹脊线修复训练")
    print("  新增：遮挡 + 模糊 + 噪声增强")
    print("=" * 70)
    print("\n配置:")
    print("  数据目录: %s" % args.data_root)
    print("  批大小: %d" % args.batch_size)
    print("  训练轮数: %d" % args.num_epochs)
    print("  学习率: %.6f" % args.lr)
    print("  图像尺寸: %d" % args.image_size)
    print("  设备: %s" % DEVICE)
    print("\n数据增强:")
    print("  ✓ 基础增强: 翻转、旋转")
    print("  ✓ 遮挡增强: 方块、条纹、大面积 (50%概率)")
    print("  ✓ 模糊增强: 高斯、运动模糊 (30%概率)")
    print("  ✓ 噪声增强: 高斯、椒盐噪声 (20%概率)")
    print("\n损失权重:")
    print("  分割: %.1f" % LAMBDA_SEGMENT)
    print("  修复: %.1f" % LAMBDA_RIDGE)
    print("  方向: %.1f" % LAMBDA_ORIENT)
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 加载数据集（使用增强版）
    print("加载数据集...")
    train_dataset = EnhancedFingerprintDataset(
        root_dir=args.data_root,
        image_size=args.image_size,
        split='train',
        augment=True,
        strong_augment=True  # 启用强化增强
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
    print("创建增强版 U-Net 模型...")
    model = AdvancedUNet(in_channels=1, base_channels=args.base_channels).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print("✓ 模型参数量: %.2fM\n" % (num_params / 1e6))
    
    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    criterion = CombinedLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 加载检查点（自动恢复或手动指定）
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 查找最新检查点（自动恢复）
    if not args.resume and os.path.exists(args.output_dir):
        import glob
        checkpoints = glob.glob(os.path.join(args.output_dir, 'model_augmented_epoch_*.pth'))
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0]))
            args.resume = checkpoints[-1]
            print(f"📂 自动检测到检查点: {args.resume}")
    
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 正在恢复训练...")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✓ 恢复成功")
        print(f"  起始轮次: Epoch {start_epoch}")
        if 'val_loss' in checkpoint:
            val_loss = checkpoint['val_loss']
            print(f"  上次验证损失: {val_loss['total']:.6f}")
        print()
    else:
        print("🆕 从头开始训练\n")
    
    # 训练
    print("开始训练...\n")
    patience = 15
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
        
        # 保存最佳模型 + 早停
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            checkpoint_path = os.path.join(args.output_dir, 'best_model_augmented.pth')
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
                print("  ⚠ 早停触发：验证损失连续 %d 轮未改善" % patience)
                print("  最佳验证损失: %.6f (Epoch %d)" % (best_val_loss, epoch - patience + 1))
                print("=" * 70)
                break
        
        # 定期保存（用于断点续训）
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, 'model_augmented_epoch_%d.pth' % (epoch + 1))
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
    
    parser = argparse.ArgumentParser(description='增强版指纹脊线修复训练')
    parser.add_argument('--data_root', type=str, default='../processed_sd04_dataset')
    parser.add_argument('--output_dir', type=str, default='checkpoints_augmented')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--resume', type=str, default='')
    
    args = parser.parse_args()
    main(args)
