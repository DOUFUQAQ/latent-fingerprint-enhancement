#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多尺度迭代Refinement
基于论文: Latent Fingerprint Enhancement via Multi-Scale Patch Based Sparse Representation

实现论文Algorithm 1: 基于多尺度块的稀疏表示的指纹增强方法
"""

import numpy as np
import cv2
from tqdm import tqdm
from repair.gabor_sparse_coding import (build_gabor_dictionary, sparse_coding_patch,
                                        reconstruct_patch)
from repair.tv_decomposition import compute_coherence

def quality_assessment(image, window_size=16):
    """评估图像质量（论文公式10）
    
    使用方向一致性作为质量指标
    
    Args:
        image: 输入图像 (H, W)
        window_size: 窗口大小
    
    Returns:
        quality_map: 质量图，范围[0, 1]，值越大质量越好
    """
    
    return compute_coherence(image, window_size=window_size)


def identify_low_quality_patches(quality_map, patch_size, threshold=0.5):
    """识别低质量区域（论文Algorithm 1步骤3.2）
    
    Args:
        quality_map: 质量图
        patch_size: 块大小
        threshold: 质量阈值
    
    Returns:
        low_quality_positions: 低质量块的位置列表 [(y, x), ...]
    """
    
    h, w = quality_map.shape
    low_quality_positions = []
    
    stride = patch_size // 2  # 重叠步长
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # 提取块的平均质量
            patch_quality = quality_map[y:y+patch_size, x:x+patch_size]
            avg_quality = np.mean(patch_quality)
            
            # 如果质量低于阈值，标记为低质量
            if avg_quality < threshold:
                low_quality_positions.append((y, x))
    
    return low_quality_positions


def multi_scale_iterative_enhancement(image, initial_patch_size=24, max_patch_size=48,
                                     patch_size_increment=4, max_iterations=20,
                                     quality_threshold=0.5, lambda_reg=0.05, gamma=0.8,
                                     method='lasso'):
    """多尺度迭代指纹增强（论文Algorithm 1）
    
    Args:
        image: 输入图像（纹理部分，来自TV分解）
        initial_patch_size: 初始小块大小（论文: 24）
        max_patch_size: 最大块大小（论文: 逐步增加到48+）
        patch_size_increment: 块大小增量（论文: 4）
        max_iterations: 最大迭代次数（论文: 20）
        quality_threshold: 质量阈值（论文中自适应）
        lambda_reg: L1正则化参数（论文: 0.5）
        gamma: 大块权重（论文: 0.8）
        method: 稀疏编码方法
    
    Returns:
        enhanced: 增强后的指纹图像
        history: 迭代历史（用于可视化）
    """
    
    print("\n" + "=" * 70)
    print("  多尺度迭代指纹增强（论文Algorithm 1）")
    print("=" * 70)
    print(f"初始块大小: {initial_patch_size}×{initial_patch_size}")
    print(f"最大块大小: {max_patch_size}×{max_patch_size}")
    print(f"最大迭代次数: {max_iterations}")
    print(f"质量阈值: {quality_threshold}")
    print()
    
    h, w = image.shape
    history = []
    
    # ===== 步骤1: 使用小块进行初始重建 =====
    print("[步骤1] 使用小块初始重建...")
    small_patch_size = initial_patch_size
    stride = small_patch_size // 3  # 重叠步长
    
    # 构建小块字典
    dict_small, _ = build_gabor_dictionary(patch_size=small_patch_size)
    
    # 初始化
    enhanced = np.zeros_like(image, dtype=np.float64)
    weight = np.zeros_like(image, dtype=np.float64)
    img_normalized = image.astype(np.float64) / 255.0
    
    # 扫描并重建
    positions = []
    for y in range(0, h - small_patch_size + 1, stride):
        for x in range(0, w - small_patch_size + 1, stride):
            positions.append((y, x))
    
    for y, x in tqdm(positions, desc="初始重建"):
        # 提取块
        patch = img_normalized[y:y+small_patch_size, x:x+small_patch_size]
        
        # 稀疏编码
        alpha, mean, norm = sparse_coding_patch(patch, dict_small, method=method, lambda_reg=lambda_reg)
        
        # 重建
        reconstructed = reconstruct_patch(alpha, dict_small, mean, norm)
        
        # 累加
        enhanced[y:y+small_patch_size, x:x+small_patch_size] += reconstructed
        weight[y:y+small_patch_size, x:x+small_patch_size] += 1
    
    # 平均
    enhanced = enhanced / (weight + 1e-8)
    enhanced_uint8 = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    
    # 保存初始结果
    history.append({
        'iteration': 0,
        'patch_size': small_patch_size,
        'image': enhanced_uint8.copy(),
        'low_quality_count': 0
    })
    
    print(f"✓ 初始重建完成\n")
    
    # ===== 步骤2: 多尺度迭代修复低质量区域 =====
    print("[步骤2] 多尺度迭代修复...")
    
    current_large_size = small_patch_size + patch_size_increment
    iteration = 0
    
    while iteration < max_iterations and current_large_size <= max_patch_size:
        iteration += 1
        
        print(f"\n--- 迭代 {iteration}/{max_iterations} ---")
        print(f"小块大小: {small_patch_size}×{small_patch_size}")
        print(f"大块大小: {current_large_size}×{current_large_size}")
        
        # 评估质量
        quality_map = quality_assessment(enhanced_uint8, window_size=16)
        
        # 识别低质量区域
        low_quality_positions = identify_low_quality_patches(
            quality_map, small_patch_size, threshold=quality_threshold
        )
        
        print(f"检测到 {len(low_quality_positions)} 个低质量块")
        
        # 如果没有低质量区域，停止迭代
        if len(low_quality_positions) == 0:
            print("✓ 所有区域质量良好，停止迭代")
            break
        
        # 构建大块字典
        dict_large, _ = build_gabor_dictionary(patch_size=current_large_size)
        
        # 修复低质量区域
        temp_enhanced = enhanced.copy()
        temp_weight = np.zeros_like(weight)
        
        for y, x in tqdm(low_quality_positions, desc=f"修复 (iter {iteration})"):
            # 计算大块的中心位置
            offset = (current_large_size - small_patch_size) // 2
            y_large = max(0, y - offset)
            x_large = max(0, x - offset)
            
            # 边界检查
            if y_large + current_large_size > h or x_large + current_large_size > w:
                continue
            
            # 提取小块和大块
            patch_small = img_normalized[y:y+small_patch_size, x:x+small_patch_size]
            patch_large = img_normalized[y_large:y_large+current_large_size, 
                                        x_large:x_large+current_large_size]
            
            # 预处理小块和大块（不做任何归一化）
            ys_flat = patch_small.flatten()
            yl_flat = patch_large.flatten()
            
            # 多尺度稀疏编码（论文公式9）
            y_combined = np.concatenate([ys_flat, np.sqrt(gamma) * yl_flat])
            dict_combined = np.vstack([dict_small, np.sqrt(gamma) * dict_large])
            
            # 稀疏编码（使用fit_intercept学习直流分量）
            from sklearn.linear_model import Lasso
            adjusted_lambda = lambda_reg * 0.01  # 调整lambda
            model = Lasso(alpha=adjusted_lambda, max_iter=1000, fit_intercept=True)
            model.fit(dict_combined, y_combined)
            alpha = model.coef_
            intercept = model.intercept_
            
            # 用小字典重建
            reconstructed = dict_small @ alpha + intercept  # 加上截距
            reconstructed = reconstructed.reshape(small_patch_size, small_patch_size)
            
            # 累加到临时结果
            temp_enhanced[y:y+small_patch_size, x:x+small_patch_size] += reconstructed
            temp_weight[y:y+small_patch_size, x:x+small_patch_size] += 1
        
        # 更新：只更新修复过的区域
        mask = temp_weight > 0
        enhanced[mask] = (enhanced[mask] * weight[mask] + temp_enhanced[mask]) / (weight[mask] + temp_weight[mask])
        weight[mask] += temp_weight[mask]
        
        # 转换为uint8
        enhanced_uint8 = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        # 保存历史
        history.append({
            'iteration': iteration,
            'patch_size': current_large_size,
            'image': enhanced_uint8.copy(),
            'low_quality_count': len(low_quality_positions),
            'quality_map': quality_map.copy()
        })
        
        print(f"✓ 迭代 {iteration} 完成")
        
        # 增加块大小
        current_large_size += patch_size_increment
    
    print(f"\n" + "=" * 70)
    print(f"  多尺度迭代增强完成！")
    print(f"  总迭代次数: {iteration}")
    print("=" * 70 + "\n")
    
    return enhanced_uint8, history


def visualize_refinement_process(history, save_path=None):
    """可视化refinement过程
    
    Args:
        history: 迭代历史
        save_path: 保存路径
    """
    
    import matplotlib.pyplot as plt
    
    n_iterations = len(history)
    
    # 创建子图
    if n_iterations <= 4:
        fig, axes = plt.subplots(1, n_iterations, figsize=(5*n_iterations, 5))
    else:
        rows = 2
        cols = (n_iterations + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.flatten()
    
    # 确保axes是数组
    if n_iterations == 1:
        axes = [axes]
    
    for idx, record in enumerate(history):
        axes[idx].imshow(record['image'], cmap='gray')
        
        if record['iteration'] == 0:
            title = f"初始重建\n块大小: {record['patch_size']}×{record['patch_size']}"
        else:
            title = f"迭代 {record['iteration']}\n块大小: {record['patch_size']}×{record['patch_size']}\n低质量块: {record['low_quality_count']}"
        
        axes[idx].set_title(title, fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    # 隐藏多余的子图
    for idx in range(n_iterations, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('多尺度迭代Refinement过程', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Refinement过程已保存: {save_path}")
    
    plt.show()


def visualize_quality_improvement(history, save_path=None):
    """可视化质量改善过程
    
    Args:
        history: 迭代历史
        save_path: 保存路径
    """
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 低质量块数量变化
    iterations = [h['iteration'] for h in history]
    low_quality_counts = [h['low_quality_count'] for h in history]
    
    axes[0].plot(iterations, low_quality_counts, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Iteration', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Low Quality Patch Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Low Quality Regions Reduction', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 平均质量变化
    if 'quality_map' in history[0]:
        avg_qualities = []
        for h in history:
            if 'quality_map' in h:
                avg_qualities.append(np.mean(h['quality_map']))
        
        axes[1].plot(iterations[:len(avg_qualities)], avg_qualities, 's-', 
                    linewidth=2, markersize=8, color='green')
        axes[1].set_xlabel('Iteration', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Average Quality', fontsize=12, fontweight='bold')
        axes[1].set_title('Quality Improvement', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 质量改善图已保存: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='多尺度迭代Refinement')
    parser.add_argument('--input', type=str, required=True, help='输入图像（纹理部分）')
    parser.add_argument('--output', type=str, default='refinement_results', help='输出目录')
    parser.add_argument('--initial_size', type=int, default=24, help='初始块大小')
    parser.add_argument('--max_size', type=int, default=48, help='最大块大小')
    parser.add_argument('--increment', type=int, default=4, help='块大小增量')
    parser.add_argument('--max_iter', type=int, default=20, help='最大迭代次数')
    parser.add_argument('--quality_threshold', type=float, default=0.5, help='质量阈值')
    parser.add_argument('--visualize', action='store_true', help='可视化过程')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  多尺度迭代Refinement")
    print("=" * 70)
    print()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 读取图像
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    
    # 多尺度迭代增强
    enhanced, history = multi_scale_iterative_enhancement(
        img,
        initial_patch_size=args.initial_size,
        max_patch_size=args.max_size,
        patch_size_increment=args.increment,
        max_iterations=args.max_iter,
        quality_threshold=args.quality_threshold
    )
    
    # 保存结果
    cv2.imwrite(os.path.join(args.output, 'enhanced_final.png'), enhanced)
    print(f"\n✓ 最终结果已保存: {os.path.join(args.output, 'enhanced_final.png')}")
    
    # 可视化
    if args.visualize:
        visualize_refinement_process(history, os.path.join(args.output, 'refinement_process.png'))
        visualize_quality_improvement(history, os.path.join(args.output, 'quality_improvement.png'))
