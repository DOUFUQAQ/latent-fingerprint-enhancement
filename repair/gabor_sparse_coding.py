#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gabor字典稀疏表示
基于论文: Latent Fingerprint Enhancement via Multi-Scale Patch Based Sparse Representation

构建Gabor字典并进行稀疏编码，用于指纹增强
"""

import numpy as np
try:  # OpenCV is optional: the pure-numpy helpers below import without it
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # OpenCV-backed helpers will raise if called

try:  # scikit-learn is only needed by the two sparse-coding solvers
    from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
except ImportError:  # pragma: no cover
    Lasso = OrthogonalMatchingPursuit = None

try:  # tqdm is only used to draw the progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def build_gabor_dictionary(patch_size=24, n_orientations=16, n_frequencies=9, n_phases=6):
    """构建Gabor字典（论文方法）
    
    Args:
        patch_size: 图像块大小
        n_orientations: 方向数量（论文: 16）
        n_frequencies: 频率数量（论文: 9）
        n_phases: 相位数量（论文: 6）
    
    Returns:
        dictionary: Gabor字典 (patch_size*patch_size, n_atoms)
        params: 每个原子的参数 (theta, freq, phase)
    """
    
    print(f"构建Gabor字典...")
    print(f"  块大小: {patch_size}×{patch_size}")
    print(f"  方向数: {n_orientations}")
    print(f"  频率数: {n_frequencies}")
    print(f"  相位数: {n_phases}")
    
    # 参数范围（论文设定）
    thetas = np.linspace(0, 15*np.pi/16, n_orientations)  # 0 到 15π/16
    frequencies = np.arange(5, 22, 2)[:n_frequencies]      # 5, 7, 9, ..., 21
    phases = np.linspace(0, 5*np.pi/6, n_phases)          # 0 到 5π/6
    
    # 总原子数
    n_atoms = n_orientations * n_frequencies * n_phases
    
    # 初始化字典
    dictionary = np.zeros((patch_size * patch_size, n_atoms))
    params = []
    
    # 标准差（论文：设置为块大小）
    sigma_x = patch_size / 2.0
    sigma_y = patch_size / 2.0
    
    idx = 0
    for theta in thetas:
        for freq in frequencies:
            for phase in phases:
                # 生成Gabor原子
                gabor_atom = create_gabor_kernel(
                    size=patch_size,
                    theta=theta,
                    freq=freq / patch_size,  # 归一化频率
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    phase=phase
                )
                
                # 预处理：零均值，L2范数为1
                gabor_atom = gabor_atom - np.mean(gabor_atom)
                gabor_norm = np.linalg.norm(gabor_atom)
                if gabor_norm > 0:
                    gabor_atom = gabor_atom / gabor_norm
                
                # 展平并添加到字典
                dictionary[:, idx] = gabor_atom.flatten()
                params.append((theta, freq, phase))
                idx += 1
    
    print(f"✓ 字典构建完成: {n_atoms} 个原子\n")
    
    return dictionary, params


def create_gabor_kernel(size, theta, freq, sigma_x, sigma_y, phase=0):
    """创建Gabor核（论文公式4-6）
    
    Args:
        size: 核大小
        theta: 方向（弧度）
        freq: 频率
        sigma_x, sigma_y: 高斯包络的标准差
        phase: 相位偏移
    
    Returns:
        gabor_kernel: Gabor核 (size, size)
    """
    
    # 创建坐标网格
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    X, Y = np.meshgrid(x, y)
    
    # 旋转坐标（公式5-6）
    x_theta = X * np.cos(theta) + Y * np.sin(theta)
    y_theta = -X * np.sin(theta) + Y * np.cos(theta)
    
    # Gabor函数（公式4）
    gaussian = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
    sinusoid = np.cos(2 * np.pi * freq * x_theta + phase)
    
    gabor = gaussian * sinusoid
    
    return gabor


def sparse_coding_patch(patch, dictionary, method='lasso', lambda_reg=0.05, n_nonzero=None):
    """对单个图像块进行稀疏编码（论文公式3）
    
    Args:
        patch: 图像块 (patch_size, patch_size)
        dictionary: Gabor字典 (n_features, n_atoms)
        method: 'lasso' 或 'omp'（正交匹配追踪）
        lambda_reg: L1正则化参数（论文: 0.5）
        n_nonzero: OMP的非零系数数量
    
    Returns:
        alpha: 稀疏系数 (n_atoms,)
        mean: 原始均值（用于重建）
        norm: 原始范数（用于重建）
    """
    
    # 展平（不做任何归一化，直接用原始值）
    y = patch.flatten()
    # 不做L2归一化，不做零均值
    y_norm = 1.0  # 不归一化
    
    # 稀疏编码
    if method == 'lasso':
        # L1正则化最小二乘（论文公式3）
        # y = Φα + b, Φ:(n_pixels, n_atoms), y:(n_pixels,), α:(n_atoms,)
        # 使用fit_intercept=True来学习直流分量（均值）
        # 调整lambda：因为不归一化，值域是[0,1]，所以lambda要相应减小
        adjusted_lambda = lambda_reg * 0.01  # 从0.05调整到0.0005
        model = Lasso(alpha=adjusted_lambda, max_iter=1000, fit_intercept=True)
        model.fit(dictionary, y)  # sklearn期望 X:(n_samples, n_features), y:(n_samples,)
        alpha = model.coef_
        y_mean = model.intercept_  # 学习到的截距（直流分量）
    
    elif method == 'omp':
        # 正交匹配追踪
        if n_nonzero is None:
            n_nonzero = max(1, int(dictionary.shape[1] * 0.1))  # 10%稀疏度
        model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero, fit_intercept=True)
        model.fit(dictionary, y)
        alpha = model.coef_
        y_mean = model.intercept_ if hasattr(model, 'intercept_') else 0.0  # 学习到的截距
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return alpha, y_mean, y_norm


def reconstruct_patch(alpha, dictionary, mean=0.0, norm=1.0):
    """从稀疏系数重建图像块
    
    Args:
        alpha: 稀疏系数 (n_atoms,)
        dictionary: Gabor字典 (n_features, n_atoms)
        mean: 截距项（直流分量），从fit_intercept学习得到
        norm: 原始范数（恢复用）
    
    Returns:
        patch: 重建的图像块 (patch_size, patch_size)
    """
    
    # y_normalized = Φα + mean (mean是截距项)
    y_reconstructed = dictionary @ alpha + mean
    
    # 反归一化：恢复原始尺度
    y_reconstructed = y_reconstructed * norm
    
    # 重塑为方形块
    patch_size = int(np.sqrt(dictionary.shape[0]))
    patch = y_reconstructed.reshape(patch_size, patch_size)
    
    return patch


def enhance_image_with_sparse_coding(image, patch_size=24, stride=8, 
                                     dictionary=None, method='lasso', lambda_reg=0.05):
    """使用稀疏编码增强整张图像
    
    Args:
        image: 输入图像 (H, W)
        patch_size: 块大小
        stride: 步长
        dictionary: Gabor字典（如果为None则自动构建）
        method: 稀疏编码方法
        lambda_reg: 正则化参数
    
    Returns:
        enhanced: 增强后的图像
    """
    
    h, w = image.shape
    
    # 构建字典（如果未提供）
    if dictionary is None:
        dictionary, _ = build_gabor_dictionary(patch_size=patch_size)
    
    # 初始化
    enhanced = np.zeros_like(image, dtype=np.float64)
    weight = np.zeros_like(image, dtype=np.float64)
    
    # 归一化输入
    img_normalized = image.astype(np.float64) / 255.0
    
    # 扫描图像（从左到右，从上到下）
    positions = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            positions.append((y, x))
    
    print(f"处理 {len(positions)} 个图像块...")
    
    for y, x in tqdm(positions, desc="稀疏编码"):
        # 提取块
        patch = img_normalized[y:y+patch_size, x:x+patch_size]
        
        # 稀疏编码
        alpha, mean, norm = sparse_coding_patch(patch, dictionary, method=method, lambda_reg=lambda_reg)
        
        # 重建（恢复原始均值和范数）
        reconstructed = reconstruct_patch(alpha, dictionary, mean, norm)
        
        # 累加到输出
        enhanced[y:y+patch_size, x:x+patch_size] += reconstructed
        weight[y:y+patch_size, x:x+patch_size] += 1
    
    # 平均
    enhanced = enhanced / (weight + 1e-8)
    
    # 转换回[0, 255]
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    
    return enhanced


def multi_scale_sparse_coding(image, small_size=24, large_size=36, 
                              stride=8, method='lasso', lambda_reg=0.05, gamma=0.8):
    """多尺度稀疏编码（论文公式7-9）
    
    Args:
        image: 输入图像
        small_size: 小块尺寸
        large_size: 大块尺寸
        stride: 步长
        method: 稀疏编码方法
        lambda_reg: 正则化参数
        gamma: 大块权重（论文: 0.8）
    
    Returns:
        enhanced: 增强后的图像
    """
    
    h, w = image.shape
    
    # 构建两个尺度的字典
    print(f"构建多尺度字典...")
    dict_small, _ = build_gabor_dictionary(patch_size=small_size)
    dict_large, _ = build_gabor_dictionary(patch_size=large_size)
    
    # 初始化
    enhanced = np.zeros_like(image, dtype=np.float64)
    weight = np.zeros_like(image, dtype=np.float64)
    
    # 归一化
    img_normalized = image.astype(np.float64) / 255.0
    
    # 扫描位置
    positions = []
    for y in range(0, h - large_size + 1, stride):
        for x in range(0, w - large_size + 1, stride):
            positions.append((y, x))
    
    print(f"多尺度稀疏编码...")
    
    for y, x in tqdm(positions, desc="处理"):
        # 提取小块（居中）
        offset = (large_size - small_size) // 2
        ys, xs = y + offset, x + offset
        if ys + small_size <= h and xs + small_size <= w:
            patch_small = img_normalized[ys:ys+small_size, xs:xs+small_size]
        else:
            continue
        
        # 提取大块
        patch_large = img_normalized[y:y+large_size, x:x+large_size]
        
        # 预处理小块和大块（不做任何归一化）
        ys_flat = patch_small.flatten()
        yl_flat = patch_large.flatten()
        
        # 多尺度稀疏编码（论文公式9）
        # 最小化: ||yₛ - Φₛα - b||² + γ||yₗ - Φₗα - b||² + λ||α||₁
        
        # 组合字典和数据
        y_combined = np.concatenate([ys_flat, np.sqrt(gamma) * yl_flat])
        dict_combined = np.vstack([dict_small, np.sqrt(gamma) * dict_large])
        
        # 稀疏编码（使用fit_intercept学习直流分量）
        # 调整lambda：因为不归一化，值域是[0,1]
        adjusted_lambda = lambda_reg * 0.01
        model = Lasso(alpha=adjusted_lambda, max_iter=1000, fit_intercept=True)
        model.fit(dict_combined, y_combined)
        alpha = model.coef_
        intercept = model.intercept_
        
        # 用小字典重建（论文方法）
        reconstructed = dict_small @ alpha + intercept  # 加上截距
        reconstructed = reconstructed.reshape(small_size, small_size)
        
        # 累加
        enhanced[ys:ys+small_size, xs:xs+small_size] += reconstructed
        weight[ys:ys+small_size, xs:xs+small_size] += 1
    
    # 平均
    enhanced = enhanced / (weight + 1e-8)
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    
    return enhanced


def compute_sparsity(alpha, threshold=0.001):
    """计算稀疏系数的稀疏度
    
    Args:
        alpha: 稀疏系数
        threshold: 阈值
    
    Returns:
        sparsity: 稀疏度（非零系数占比）
        l1_norm: L1范数
    """
    
    nonzero = np.sum(np.abs(alpha) > threshold)
    sparsity = nonzero / len(alpha)
    l1_norm = np.sum(np.abs(alpha))
    
    return sparsity, l1_norm


def visualize_gabor_dictionary(dictionary, params, save_path=None, n_samples=64):
    """可视化Gabor字典
    
    Args:
        dictionary: Gabor字典
        params: 参数列表
        save_path: 保存路径
        n_samples: 显示的原子数量
    """
    
    import matplotlib.pyplot as plt
    
    patch_size = int(np.sqrt(dictionary.shape[0]))
    n_atoms = min(n_samples, dictionary.shape[1])
    
    # 网格大小
    grid_size = int(np.ceil(np.sqrt(n_atoms)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    for idx in range(n_atoms):
        row = idx // grid_size
        col = idx % grid_size
        
        # 提取并重塑原子
        atom = dictionary[:, idx].reshape(patch_size, patch_size)
        
        # 显示
        axes[row, col].imshow(atom, cmap='gray')
        axes[row, col].axis('off')
        
        # 标题（参数）
        if params:
            theta, freq, phase = params[idx]
            axes[row, col].set_title(f'θ={theta:.2f}\nf={freq:.1f}', fontsize=6)
    
    # 隐藏多余的子图
    for idx in range(n_atoms, grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        axes[row, col].axis('off')
    
    plt.suptitle(f'Gabor Dictionary ({dictionary.shape[1]} atoms)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 字典可视化已保存: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Gabor字典稀疏编码')
    parser.add_argument('--input', type=str, help='输入图像')
    parser.add_argument('--output', type=str, default='sparse_coding_results', help='输出目录')
    parser.add_argument('--patch_size', type=int, default=24, help='块大小')
    parser.add_argument('--stride', type=int, default=8, help='步长')
    parser.add_argument('--method', type=str, default='lasso', choices=['lasso', 'omp'], help='方法')
    parser.add_argument('--lambda_reg', type=float, default=0.5, help='正则化参数')
    parser.add_argument('--multi_scale', action='store_true', help='使用多尺度')
    parser.add_argument('--visualize_dict', action='store_true', help='可视化字典')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Gabor字典稀疏编码")
    print("=" * 70)
    print()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 可视化字典
    if args.visualize_dict:
        dictionary, params = build_gabor_dictionary(patch_size=args.patch_size)
        visualize_gabor_dictionary(dictionary, params, 
                                   os.path.join(args.output, 'gabor_dictionary.png'))
    
    # 处理图像
    if args.input:
        img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        
        if args.multi_scale:
            enhanced = multi_scale_sparse_coding(img, small_size=24, large_size=36,
                                                stride=args.stride, method=args.method,
                                                lambda_reg=args.lambda_reg)
        else:
            enhanced = enhance_image_with_sparse_coding(img, patch_size=args.patch_size,
                                                       stride=args.stride, method=args.method,
                                                       lambda_reg=args.lambda_reg)
        
        # 保存
        cv2.imwrite(os.path.join(args.output, 'enhanced.png'), enhanced)
        print(f"\n✓ 增强结果已保存: {os.path.join(args.output, 'enhanced.png')}")
