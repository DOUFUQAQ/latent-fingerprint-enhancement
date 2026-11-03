#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TV模型图像分解
基于论文: Latent Fingerprint Enhancement via Multi-Scale Patch Based Sparse Representation
将图像分解为卡通部分（结构噪声）和纹理部分（指纹脊线）
"""

import numpy as np
try:  # OpenCV is optional: the pure-numpy helpers below import without it
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # OpenCV-backed helpers will raise if called


def tv_decomposition(image, lambda_tv=1.5, num_iter=150, tau=0.125):
    """使用Total Variation模型分解图像
    
    Args:
        image: 输入图像 (H, W)，灰度图，范围[0, 255]
        lambda_tv: TV正则化参数，控制分解程度（越大越平滑）
        num_iter: 迭代次数
        tau: 时间步长
    
    Returns:
        cartoon: 卡通部分（结构噪声）
        texture: 纹理部分（指纹脊线）
    """
    
    # 归一化到[0, 1]
    img = image.astype(np.float64) / 255.0
    
    # 初始化
    u = img.copy()  # 卡通部分
    
    # TV分解：最小化 ||u - img||² + λ * TV(u)
    for _ in range(num_iter):
        # 计算梯度
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u
        
        # 计算TV
        grad_magnitude = np.sqrt(ux**2 + uy**2 + 1e-8)
        
        # 归一化梯度
        nx = ux / grad_magnitude
        ny = uy / grad_magnitude
        
        # 计算散度
        nxx = nx - np.roll(nx, 1, axis=1)
        nyy = ny - np.roll(ny, 1, axis=0)
        div_n = nxx + nyy
        
        # 更新u（卡通部分）
        u = u + tau * (lambda_tv * div_n - (u - img))
    
    # 纹理部分 = 原图 - 卡通部分
    v = img - u
    
    # 转换回[0, 255]
    # 注意：texture使用 (v + 1.0) 使背景为白色（255），脊线为暗色
    cartoon = np.clip(u * 255, 0, 255).astype(np.uint8)
    texture = np.clip((v + 1.0) * 255, 0, 255).astype(np.uint8)
    
    return cartoon, texture


def rudin_osher_fatemi_tv(image, lambda_tv=1.5, num_iter=150):
    """ROF (Rudin-Osher-Fatemi) TV模型
    
    更经典的TV分解实现
    
    Args:
        image: 输入图像 (H, W)
        lambda_tv: TV正则化参数
        num_iter: 迭代次数
    
    Returns:
        cartoon: 卡通部分
        texture: 纹理部分
    """
    
    # 归一化
    img = image.astype(np.float64) / 255.0
    
    # 初始化
    u = img.copy()
    
    # 时间步长
    dt = 0.25
    
    for _ in range(num_iter):
        # 梯度
        ux = np.gradient(u, axis=1)
        uy = np.gradient(u, axis=0)
        
        # TV散度
        norm = np.sqrt(ux**2 + uy**2 + 1e-8)
        div_x = np.gradient(ux / norm, axis=1)
        div_y = np.gradient(uy / norm, axis=0)
        div_p = div_x + div_y
        
        # 更新
        u = u + dt * (div_p - (u - img) / lambda_tv)
    
    # 分解
    v = img - u
    
    # 注意：texture使用 (v + 1.0) 使背景为白色（255），脊线为暗色
    cartoon = np.clip(u * 255, 0, 255).astype(np.uint8)
    texture = np.clip((v + 1.0) * 255, 0, 255).astype(np.uint8)
    
    return cartoon, texture


def adaptive_tv_decomposition(image, quality_map=None, lambda_base=1.5, num_iter=150):
    """自适应TV分解
    
    根据图像质量调整TV参数
    
    Args:
        image: 输入图像
        quality_map: 质量图（可选），高质量区域使用较小的λ
        lambda_base: 基础TV参数
        num_iter: 迭代次数
    
    Returns:
        cartoon, texture
    """
    
    if quality_map is None:
        # 自动计算质量图（基于梯度一致性）
        quality_map = compute_coherence(image)
    
    # 归一化质量图到[0.5, 1.5]
    quality_map = 0.5 + quality_map
    
    # 归一化图像
    img = image.astype(np.float64) / 255.0
    u = img.copy()
    
    dt = 0.25
    
    for _ in range(num_iter):
        # 自适应λ
        lambda_adaptive = lambda_base * quality_map
        
        # 梯度和TV
        ux = np.gradient(u, axis=1)
        uy = np.gradient(u, axis=0)
        norm = np.sqrt(ux**2 + uy**2 + 1e-8)
        
        div_x = np.gradient(ux / norm, axis=1)
        div_y = np.gradient(uy / norm, axis=0)
        div_p = div_x + div_y
        
        # 更新
        u = u + dt * (div_p - (u - img) / lambda_adaptive)
    
    v = img - u
    
    # 注意：texture使用 (v + 1.0) 使背景为白色（255），脊线为暗色
    cartoon = np.clip(u * 255, 0, 255).astype(np.uint8)
    texture = np.clip((v + 1.0) * 255, 0, 255).astype(np.uint8)
    
    return cartoon, texture


def compute_coherence(image, window_size=16):
    """计算方向一致性（论文公式10）
    
    Args:
        image: 输入图像
        window_size: 窗口大小
    
    Returns:
        coherence: 一致性图，范围[0, 1]
    """
    
    # 归一化
    img = image.astype(np.float64) / 255.0
    
    # 计算梯度
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度的平方
    gxx = gx * gx
    gyy = gy * gy
    gxy = gx * gy
    
    # 窗口平均
    kernel = np.ones((window_size, window_size)) / (window_size**2)
    Gxx = cv2.filter2D(gxx, -1, kernel)
    Gyy = cv2.filter2D(gyy, -1, kernel)
    Gxy = cv2.filter2D(gxy, -1, kernel)
    
    # 一致性 = sqrt((Gxx-Gyy)^2 + 4*Gxy^2) / (Gxx+Gyy)
    numerator = np.sqrt((Gxx - Gyy)**2 + 4 * Gxy**2)
    denominator = Gxx + Gyy + 1e-8
    coherence = numerator / denominator
    
    return coherence


def visualize_decomposition(original, cartoon, texture, save_path=None):
    """可视化TV分解结果
    
    Args:
        original: 原始图像
        cartoon: 卡通部分
        texture: 纹理部分
        save_path: 保存路径（可选）
    """
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cartoon, cmap='gray')
    axes[1].set_title('Cartoon (Structural Noise)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(texture, cmap='gray')
    axes[2].set_title('Texture (Fingerprint)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存到: {save_path}")
    
    plt.show()


def batch_tv_decomposition(input_dir, output_dir, method='rof', lambda_tv=0.15):
    """批量TV分解
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        method: 'rof' 或 'adaptive'
        lambda_tv: TV参数
    """
    
    import os
    import glob
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'cartoon'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'texture'), exist_ok=True)
    
    # 查找图像
    image_files = glob.glob(os.path.join(input_dir, '*.png')) + \
                  glob.glob(os.path.join(input_dir, '*.jpg'))
    
    print(f"找到 {len(image_files)} 张图像")
    print(f"方法: {method}")
    print(f"λ: {lambda_tv}\n")
    
    for img_path in tqdm(image_files, desc="TV分解"):
        filename = os.path.basename(img_path)
        name = os.path.splitext(filename)[0]
        
        # 读取
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # 分解
        if method == 'rof':
            cartoon, texture = rudin_osher_fatemi_tv(img, lambda_tv=lambda_tv)
        elif method == 'adaptive':
            cartoon, texture = adaptive_tv_decomposition(img, lambda_base=lambda_tv)
        else:
            cartoon, texture = tv_decomposition(img, lambda_tv=lambda_tv)
        
        # 保存
        cv2.imwrite(os.path.join(output_dir, 'cartoon', f'{name}_cartoon.png'), cartoon)
        cv2.imwrite(os.path.join(output_dir, 'texture', f'{name}_texture.png'), texture)
    
    print(f"\n✓ 完成！输出目录: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TV模型图像分解')
    parser.add_argument('--input', type=str, required=True, help='输入图像或目录')
    parser.add_argument('--output', type=str, default='tv_decomposition_results', help='输出目录')
    parser.add_argument('--method', type=str, default='rof', choices=['basic', 'rof', 'adaptive'],
                       help='TV方法')
    parser.add_argument('--lambda_tv', type=float, default=0.15, help='TV参数')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  TV模型图像分解")
    print("=" * 70)
    print()
    
    if os.path.isfile(args.input):
        # 单张图像
        img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        
        if args.method == 'rof':
            cartoon, texture = rudin_osher_fatemi_tv(img, lambda_tv=args.lambda_tv)
        elif args.method == 'adaptive':
            cartoon, texture = adaptive_tv_decomposition(img, lambda_base=args.lambda_tv)
        else:
            cartoon, texture = tv_decomposition(img, lambda_tv=args.lambda_tv)
        
        # 保存
        os.makedirs(args.output, exist_ok=True)
        cv2.imwrite(os.path.join(args.output, 'cartoon.png'), cartoon)
        cv2.imwrite(os.path.join(args.output, 'texture.png'), texture)
        
        if args.visualize:
            visualize_decomposition(img, cartoon, texture, 
                                   os.path.join(args.output, 'decomposition.png'))
    
    else:
        # 批量处理
        batch_tv_decomposition(args.input, args.output, method=args.method, lambda_tv=args.lambda_tv)
