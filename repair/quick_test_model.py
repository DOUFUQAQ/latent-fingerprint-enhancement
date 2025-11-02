#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试训练好的模型效果
从验证集随机选择图像，展示三任务输出
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from repair.train_advanced_repair import AdvancedUNet, AdvancedFingerprintDataset
import random
import argparse

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model(checkpoint_path, device, base_channels=None):
    """加载训练好的模型（自动检测模型尺寸）"""
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 自动检测 base_channels（从第一层参数形状推断）
    if base_channels is None:
        first_layer_shape = checkpoint['model_state_dict']['enc1.0.weight'].shape
        base_channels = first_layer_shape[0]  # [out_channels, in_channels, h, w]
        print(f"✓ 自动检测模型尺寸: base_channels={base_channels}")
    
    # 创建对应尺寸的模型
    model = AdvancedUNet(in_channels=1, base_channels=base_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型加载成功")
    print(f"  模型参数量: {num_params / 1e6:.2f}M")
    print(f"  训练轮数: Epoch {checkpoint['epoch'] + 1}")
    if 'val_loss' in checkpoint:
        val_loss = checkpoint['val_loss']
        print(f"  验证损失: {val_loss['total']:.6f}")
        print(f"    - 分割: {val_loss['segment']:.6f}")
        print(f"    - 方向: {val_loss['orient']:.6f}")
        print(f"    - 修复: {val_loss['repair']:.6f}")
    
    return model

def visualize_results(model, dataset, num_samples=6, device='cuda', save_path='model_results.png'):
    """可视化模型在验证集上的结果"""
    
    # 随机选择样本
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # 创建画布
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            # 获取数据
            sample = dataset[sample_idx]
            input_img = sample['input'].unsqueeze(0).to(device)
            
            # 推理
            outputs = model(input_img)
            
            # 转换为numpy用于显示
            input_np = sample['input'].squeeze().cpu().numpy()
            target_ridge = sample['target_ridge'].squeeze().cpu().numpy()
            target_mask = sample['target_mask'].squeeze().cpu().numpy()
            
            pred_segment = outputs['segment'].squeeze().cpu().numpy()
            pred_orient = outputs['orient'].squeeze().cpu().numpy()
            pred_repair = outputs['repair'].squeeze().cpu().numpy()
            
            # 反归一化到 [0, 1]
            input_np = (input_np + 1) / 2
            target_ridge = (target_ridge + 1) / 2
            pred_orient = (pred_orient + 1) / 2
            pred_repair = (pred_repair + 1) / 2
            
            # 显示
            # 列1: 输入（原始潜指纹）
            axes[idx, 0].imshow(input_np, cmap='gray', vmin=0, vmax=1)
            axes[idx, 0].set_title('输入\n(模糊、断裂)', fontsize=10, fontweight='bold')
            axes[idx, 0].axis('off')
            
            # 列2: 脊线分割
            axes[idx, 1].imshow(pred_segment, cmap='gray', vmin=0, vmax=1)
            axes[idx, 1].set_title('任务1：脊线分割\n(识别脊线位置)', fontsize=10, fontweight='bold')
            axes[idx, 1].axis('off')
            
            # 列3: 方向场
            axes[idx, 2].imshow(pred_orient, cmap='gray', vmin=0, vmax=1)
            axes[idx, 2].set_title('任务2：方向场\n(脊线方向)', fontsize=10, fontweight='bold')
            axes[idx, 2].axis('off')
            
            # 列4: 修复结果
            axes[idx, 3].imshow(pred_repair, cmap='gray', vmin=0, vmax=1)
            axes[idx, 3].set_title('任务3：脊线修复\n(完整脊线)', fontsize=10, fontweight='bold')
            axes[idx, 3].axis('off')
            
            # 列5: 真实标签（对比）
            axes[idx, 4].imshow(target_ridge, cmap='gray', vmin=0, vmax=1)
            axes[idx, 4].set_title('真实标签\n(Ground Truth)', fontsize=10, fontweight='bold')
            axes[idx, 4].axis('off')
            
            # 添加行标签
            axes[idx, 0].text(-0.1, 0.5, f'样本 {idx+1}', 
                            transform=axes[idx, 0].transAxes,
                            fontsize=12, fontweight='bold',
                            verticalalignment='center',
                            rotation=90)
    
    plt.suptitle('指纹脊线修复模型 - 效果展示', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化结果已保存: {save_path}")
    
    # 显示
    plt.show()
    
    return fig

def compare_with_ground_truth(model, dataset, num_samples=4, device='cuda', save_path='comparison_detail.png'):
    """详细对比预测和真实标签"""
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            input_img = sample['input'].unsqueeze(0).to(device)
            
            # 推理
            outputs = model(input_img)
            
            # 数据
            input_np = ((sample['input'].squeeze().cpu().numpy() + 1) / 2)
            target_ridge = ((sample['target_ridge'].squeeze().cpu().numpy() + 1) / 2)
            target_mask = sample['target_mask'].squeeze().cpu().numpy()
            target_orient = ((sample['target_orient'].squeeze().cpu().numpy() + 1) / 2)
            
            pred_segment = outputs['segment'].squeeze().cpu().numpy()
            pred_orient = ((outputs['orient'].squeeze().cpu().numpy() + 1) / 2)
            pred_repair = ((outputs['repair'].squeeze().cpu().numpy() + 1) / 2)
            
            # 计算差异
            diff_segment = np.abs(pred_segment - target_mask)
            diff_orient = np.abs(pred_orient - target_orient)
            diff_repair = np.abs(pred_repair - target_ridge)
            
            # 显示
            axes[idx, 0].imshow(input_np, cmap='gray')
            axes[idx, 0].set_title('输入', fontsize=10)
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(pred_segment, cmap='gray')
            axes[idx, 1].set_title('分割预测', fontsize=10)
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(target_mask, cmap='gray')
            axes[idx, 2].set_title('分割真值', fontsize=10)
            axes[idx, 2].axis('off')
            
            axes[idx, 3].imshow(pred_repair, cmap='gray')
            axes[idx, 3].set_title('修复预测', fontsize=10)
            axes[idx, 3].axis('off')
            
            axes[idx, 4].imshow(target_ridge, cmap='gray')
            axes[idx, 4].set_title('修复真值', fontsize=10)
            axes[idx, 4].axis('off')
            
            axes[idx, 5].imshow(diff_repair, cmap='hot')
            axes[idx, 5].set_title('修复差异', fontsize=10)
            axes[idx, 5].axis('off')
    
    plt.suptitle('预测 vs 真值 对比', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 对比结果已保存: {save_path}")
    plt.show()

def save_individual_results(model, dataset, output_dir='test_outputs', num_samples=10, device='cuda'):
    """保存单独的结果文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    model.eval()
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            input_img = sample['input'].unsqueeze(0).to(device)
            filename = sample['filename']
            
            # 推理
            outputs = model(input_img)
            
            # 转换为图像
            input_np = ((sample['input'].squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            pred_segment = (outputs['segment'].squeeze().cpu().numpy() * 255).astype(np.uint8)
            pred_orient = ((outputs['orient'].squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            pred_repair = ((outputs['repair'].squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            
            # 保存
            name = os.path.splitext(filename)[0]
            cv2.imwrite(os.path.join(output_dir, f'{name}_input.png'), input_np)
            cv2.imwrite(os.path.join(output_dir, f'{name}_segment.png'), pred_segment)
            cv2.imwrite(os.path.join(output_dir, f'{name}_orient.png'), pred_orient)
            cv2.imwrite(os.path.join(output_dir, f'{name}_repair.png'), pred_repair)
            
            # 创建对比图
            comparison = np.hstack([input_np, pred_segment, pred_orient, pred_repair])
            cv2.imwrite(os.path.join(output_dir, f'{name}_comparison.png'), comparison)
    
    print(f"✓ 已保存 {num_samples} 个样本到: {output_dir}")

def main():
    print("=" * 70)
    print("  指纹脊线修复模型 - 快速效果测试")
    print("=" * 70)
    print()

    # 参数
    parser = argparse.ArgumentParser(description='快速测试模型效果')
    parser.add_argument('--checkpoint', type=str, default='', help='模型权重路径')
    parser.add_argument('--num_samples', type=int, default=6, help='可视化样本数')
    parser.add_argument('--base_channels', type=int, default=None, help='模型基础通道数（默认自动检测）')
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载模型
    # 优先级：命令行参数 > 增强版检查点 > 旧版检查点
    checkpoint_path = args.checkpoint.strip()
    if not checkpoint_path:
        if os.path.exists('checkpoints_augmented/best_model_augmented.pth'):
            checkpoint_path = 'checkpoints_augmented/best_model_augmented.pth'
        else:
            checkpoint_path = 'checkpoints_advanced/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"✗ 错误: 找不到模型文件 {checkpoint_path}")
        print("  请确保训练已完成并且模型已保存")
        return
    
    model = load_model(checkpoint_path, device, base_channels=args.base_channels)
    print()
    
    # 加载验证集
    print("加载验证集...")
    dataset = AdvancedFingerprintDataset(
        root_dir='../processed_sd04_dataset',
        split='val',
        augment=False  # 验证集不增强
    )
    print(f"✓ 验证集: {len(dataset)} 张图像\n")
    
    # 1. 可视化效果
    print("=" * 70)
    print(f"1. 生成效果展示图（{args.num_samples}个样本）...")
    print("=" * 70)
    visualize_results(model, dataset, num_samples=args.num_samples, device=device, save_path='model_results.png')
    
    # 2. 详细对比
    print("\n" + "=" * 70)
    print("2. 生成预测 vs 真值对比图（4个样本）...")
    print("=" * 70)
    compare_with_ground_truth(model, dataset, num_samples=4, device=device, save_path='comparison_detail.png')
    
    # 3. 保存单独文件
    print("\n" + "=" * 70)
    print("3. 保存单独结果文件（10个样本）...")
    print("=" * 70)
    save_individual_results(model, dataset, output_dir='test_outputs', num_samples=10, device=device)
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. model_results.png        - 6个样本的完整效果展示")
    print("  2. comparison_detail.png    - 4个样本的详细对比")
    print("  3. test_outputs/            - 10个样本的单独文件")
    print("\n请打开图像文件查看模型效果！\n")

if __name__ == '__main__':
    main()
