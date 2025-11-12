#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在test2数据集上测试模型效果
评估指标：PSNR, SSIM
"""

import os
import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from repair.train_advanced_repair import AdvancedUNet

def load_model(checkpoint_path, device, base_channels=None):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 自动检测 base_channels
    if base_channels is None:
        first_layer_shape = checkpoint['model_state_dict']['enc1.0.weight'].shape
        base_channels = first_layer_shape[0]
        print(f"✓ 自动检测模型尺寸: base_channels={base_channels}")
    
    # 创建模型
    model = AdvancedUNet(in_channels=1, base_channels=base_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型加载成功")
    print(f"  模型参数量: {num_params / 1e6:.2f}M")
    
    if 'epoch' in checkpoint:
        print(f"  训练轮数: Epoch {checkpoint['epoch'] + 1}")
    
    if 'val_loss' in checkpoint:
        val_loss = checkpoint['val_loss']
        if isinstance(val_loss, dict):
            print(f"  验证损失: {val_loss['total']:.6f}")
        else:
            print(f"  验证损失: {val_loss:.6f}")
    
    return model

def preprocess_image(img, image_size=256):
    """预处理图像"""
    # 调整尺寸
    if img.shape[0] != image_size or img.shape[1] != image_size:
        img_resized = cv2.resize(img, (image_size, image_size))
    else:
        img_resized = img
    
    # 归一化到[-1, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.5) * 2
    
    # 转换为tensor
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
    
    return img_tensor, img.shape

def postprocess_image(tensor, original_shape, binarize=False, threshold=127):
    """后处理输出
    
    Args:
        tensor: 输出张量
        original_shape: 原始图像尺寸
        binarize: 是否二值化
        threshold: 二值化阈值（0-255）或'otsu'使用大津法
    """
    # 转换为numpy
    img = tensor.squeeze().cpu().numpy()
    
    # 反归一化到[0, 255]
    img = (img + 1) / 2
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # 恢复原始尺寸
    if img.shape != original_shape:
        img = cv2.resize(img, (original_shape[1], original_shape[0]))
    
    # 二值化
    if binarize:
        if threshold == 'otsu':
            # 使用大津法自动阈值
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # 使用固定阈值
            _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    return img

def calculate_metrics(pred, gt):
    """计算PSNR和SSIM"""
    # 确保数据范围一致
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    
    # PSNR
    psnr_value = psnr(gt, pred, data_range=255)
    
    # SSIM
    ssim_value = ssim(gt, pred, data_range=255)
    
    return psnr_value, ssim_value

def test_on_test2(model, test2_dir, output_dir, device, output_mode='binary', binary_threshold='otsu'):
    """在test2数据集上测试
    
    Args:
        model: 模型
        test2_dir: 测试目录
        output_dir: 输出目录
        device: 设备
        output_mode: 输出模式 ('binary'=二值化, 'gray'=灰度, 'both'=两者都保存)
        binary_threshold: 二值化阈值 (整数0-255 或 'otsu')
    """
    
    # 查找所有图像
    image_files = glob.glob(os.path.join(test2_dir, '*.png')) + \
                  glob.glob(os.path.join(test2_dir, '*.jpg')) + \
                  glob.glob(os.path.join(test2_dir, '*.bmp'))
    
    if len(image_files) == 0:
        print(f"✗ 错误: {test2_dir} 中没有找到图像文件")
        return
    
    print(f"\n找到 {len(image_files)} 张测试图像")
    print(f"输出目录: {output_dir}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'repaired'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparison'), exist_ok=True)
    
    results = []
    
    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="处理图像"):
            filename = os.path.basename(img_path)
            name = os.path.splitext(filename)[0]
            
            # 读取图像
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"✗ 跳过: 无法读取 {filename}")
                continue
            
            # 预处理
            img_tensor, original_shape = preprocess_image(img)
            img_tensor = img_tensor.to(device)
            
            # 推理
            outputs = model(img_tensor)
            
            # 后处理
            # 根据output_mode决定是否二值化
            if output_mode == 'binary':
                repaired = postprocess_image(outputs['repair'], original_shape, binarize=True, threshold=binary_threshold)
                segment = postprocess_image(outputs['segment'], original_shape, binarize=True, threshold=binary_threshold)
            elif output_mode == 'gray':
                repaired = postprocess_image(outputs['repair'], original_shape, binarize=False)
                segment = postprocess_image(outputs['segment'], original_shape, binarize=False)
            else:  # both
                repaired = postprocess_image(outputs['repair'], original_shape, binarize=False)
                repaired_binary = postprocess_image(outputs['repair'], original_shape, binarize=True, threshold=binary_threshold)
                segment = postprocess_image(outputs['segment'], original_shape, binarize=False)
                segment_binary = postprocess_image(outputs['segment'], original_shape, binarize=True, threshold=binary_threshold)
            
            orient = postprocess_image(outputs['orient'], original_shape, binarize=False)  # 方向场始终保持灰度
            
            # 保存结果
            cv2.imwrite(os.path.join(output_dir, 'input', f'{name}_input.png'), img)
            
            if output_mode == 'both':
                # 保存灰度和二值化两个版本
                cv2.imwrite(os.path.join(output_dir, 'repaired', f'{name}_repaired_gray.png'), repaired)
                cv2.imwrite(os.path.join(output_dir, 'repaired', f'{name}_repaired_binary.png'), repaired_binary)
                # 对比图使用二值化版本
                repaired_for_comparison = repaired_binary
            else:
                cv2.imwrite(os.path.join(output_dir, 'repaired', f'{name}_repaired.png'), repaired)
                repaired_for_comparison = repaired
            
            # 创建对比图（4列：输入、分割、方向、修复）
            h, w = original_shape
            comparison = np.zeros((h, w * 4), dtype=np.uint8)
            comparison[:, 0:w] = img
            comparison[:, w:w*2] = segment
            comparison[:, w*2:w*3] = orient
            comparison[:, w*3:w*4] = repaired_for_comparison
            
            cv2.imwrite(os.path.join(output_dir, 'comparison', f'{name}_comparison.png'), comparison)
            
            # 如果存在ground truth，计算指标
            # 注意：test2可能没有ground truth，这里假设修复结果本身就是评估对象
            # 可以与输入对比，看改善程度
            
            results.append({
                'filename': filename,
                'input_path': img_path,
                'repaired_path': os.path.join(output_dir, 'repaired', f'{name}_repaired.png')
            })
    
    return results

def create_summary_visualization(results, output_dir, num_samples=9):
    """创建总结可视化"""
    
    # 随机选择样本
    import random
    sample_results = random.sample(results, min(num_samples, len(results)))
    
    # 创建网格
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
    
    for idx, result in enumerate(sample_results):
        row = idx // cols
        col = (idx % cols) * 2
        
        # 读取图像
        input_img = cv2.imread(result['input_path'], cv2.IMREAD_GRAYSCALE)
        repaired_img = cv2.imread(result['repaired_path'], cv2.IMREAD_GRAYSCALE)
        
        # 显示
        if rows == 1:
            axes[col].imshow(input_img, cmap='gray')
            axes[col].set_title(f'输入 - {result["filename"]}', fontsize=8)
            axes[col].axis('off')
            
            axes[col + 1].imshow(repaired_img, cmap='gray')
            axes[col + 1].set_title('修复结果', fontsize=8)
            axes[col + 1].axis('off')
        else:
            axes[row, col].imshow(input_img, cmap='gray')
            axes[row, col].set_title(f'输入 - {result["filename"]}', fontsize=8)
            axes[row, col].axis('off')
            
            axes[row, col + 1].imshow(repaired_img, cmap='gray')
            axes[row, col + 1].set_title('修复结果', fontsize=8)
            axes[row, col + 1].axis('off')
    
    plt.suptitle('test2数据集 - 修复效果总览', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, 'summary_visualization.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 总结可视化已保存: {summary_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='在test2数据集上测试模型')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints_ultimate/best_model_ultimate.pth',
                       help='模型权重路径')
    parser.add_argument('--test2_dir', type=str, 
                       default='../test2',
                       help='test2数据集目录')
    parser.add_argument('--output_dir', type=str, 
                       default='test2_results',
                       help='输出目录')
    parser.add_argument('--base_channels', type=int, default=None, 
                       help='模型基础通道数（默认自动检测）')
    parser.add_argument('--output_mode', type=str, default='binary',
                       choices=['binary', 'gray', 'both'],
                       help='输出模式: binary=纯黑白二值化, gray=保持灰度, both=同时保存两种')
    parser.add_argument('--binary_threshold', type=str, default='otsu',
                       help='二值化阈值: otsu=自动阈值, 或指定0-255的整数')
    args = parser.parse_args()
    
    # 处理binary_threshold参数
    if args.binary_threshold != 'otsu':
        try:
            args.binary_threshold = int(args.binary_threshold)
            if args.binary_threshold < 0 or args.binary_threshold > 255:
                print("✗ 错误: binary_threshold必须在0-255之间")
                return
        except ValueError:
            print("✗ 错误: binary_threshold必须是'otsu'或0-255的整数")
            return
    
    print("=" * 70)
    print("  test2数据集测试")
    print("=" * 70)
    print()
    
    # 检查路径
    if not os.path.exists(args.checkpoint):
        print(f"✗ 错误: 找不到模型文件 {args.checkpoint}")
        return
    
    if not os.path.exists(args.test2_dir):
        print(f"✗ 错误: 找不到test2目录 {args.test2_dir}")
        return
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载模型
    model = load_model(args.checkpoint, device, base_channels=args.base_channels)
    print()
    
    # 测试
    print("=" * 70)
    print("开始测试...")
    print("=" * 70)
    print("输出设置:")
    if args.output_mode == 'binary':
        print(f"  - 输出模式: 二值化（纯黑白）")
        print(f"  - 二值化阈值: {args.binary_threshold}")
    elif args.output_mode == 'gray':
        print(f"  - 输出模式: 灰度（保持原模型输出）")
    else:
        print(f"  - 输出模式: 同时保存灰度和二值化")
        print(f"  - 二值化阈值: {args.binary_threshold}")
    print("  - 方向场: 始终保持灰度\n")
    results = test_on_test2(model, args.test2_dir, args.output_dir, device, 
                           output_mode=args.output_mode, 
                           binary_threshold=args.binary_threshold)
    
    if results:
        # 创建可视化
        print("\n" + "=" * 70)
        print("创建总结可视化...")
        print("=" * 70)
        create_summary_visualization(results, args.output_dir)
        
        # 输出统计
        print("\n" + "=" * 70)
        print("测试完成！")
        print("=" * 70)
        print(f"\n成功处理: {len(results)} 张图像")
        print(f"\n输出目录: {args.output_dir}")
        print("  - input/         输入图像（原始灰度）")
        if args.output_mode == 'binary':
            print("  - repaired/      修复结果（二值化，纯黑白）")
        elif args.output_mode == 'gray':
            print("  - repaired/      修复结果（灰度，原模型输出）")
        else:
            print("  - repaired/      修复结果（*_gray.png和*_binary.png两个版本）")
        print("  - comparison/    对比图（输入|分割|方向|修复）")
        print("  - summary_visualization.png  总览")
        
        if args.output_mode == 'binary':
            print(f"\n✓ 修复结果已二值化（阈值: {args.binary_threshold}）")
            print("✓ 输出为纯黑白图像，无灰色")
        elif args.output_mode == 'gray':
            print("\n✓ 修复结果保持灰度（原模型输出）")
            print("✓ 输出包含灰度信息")
        else:
            print(f"\n✓ 同时保存灰度和二值化版本（阈值: {args.binary_threshold}）")
            print("✓ 可对比两种输出效果")
        
        print("\n请查看输出文件评估模型效果！\n")
    else:
        print("\n✗ 没有成功处理任何图像")

if __name__ == '__main__':
    main()
