#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
保存最佳方法（UNet → TV → Gabor）的结果
"""

import cv2
import numpy as np
from repair.hybrid_enhancement_pipeline import HybridFingerprintEnhancer
import os
import argparse

def adaptive_binarization(image, block_size=15):
    """自适应二值化"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯自适应阈值
    binary = cv2.adaptiveThreshold(
        image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 2
    )
    
    return binary

def save_best_method_results(input_image_path, output_dir='best_method_results'):
    """保存最佳方法的结果
    
    Args:
        input_image_path: 输入图像路径
        output_dir: 输出目录
    """
    
    print("=" * 70)
    print("  保存最佳方法结果（UNet → TV → Gabor）")
    print("=" * 70)
    print()
    print(f"输入图像: {input_image_path}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 输出目录已创建: {output_dir}/")
    print()
    
    # 读取图像
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 无法读取图像: {input_image_path}")
        return
    
    # 调整大小（可选）
    target_size = 256
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img_resized = cv2.resize(img, (target_size, target_size))
        print(f"图像调整为: {target_size}×{target_size}")
    else:
        img_resized = img
    
    print(f"原图: {img_resized.shape}, 均值={np.mean(img_resized):.1f}")
    print()
    
    # 创建增强器
    enhancer = HybridFingerprintEnhancer(
        unet_checkpoint='checkpoints_ultimate/best_model_ultimate.pth',
        device='cpu'
    )
    
    # 运行最佳方法：UNet → TV(λ=0.15) → Gabor
    print("运行最佳方法：UNet → TV(λ=0.15) → Gabor")
    print("⚠️  使用论文推荐参数：lambda_tv=0.15, lambda_reg=0.5")
    print()
    
    result_gray, intermediates = enhancer.enhance(
        img_resized, 
        method='unet_tv_gabor', 
        lambda_tv=1.5,  # ✅ 论文推荐值
        save_intermediates=False
    )
    
    # 二值化
    print("应用自适应二值化...")
    result_binary = adaptive_binarization(result_gray, block_size=15)
    
    # 统计信息
    print()
    print("=" * 70)
    print("结果统计:")
    print(f"  灰度图 - 均值: {np.mean(result_gray):.1f}, 标准差: {np.std(result_gray):.1f}")
    print(f"  灰度图 - 范围: [{np.min(result_gray)}, {np.max(result_gray)}]")
    print(f"  二值图 - 白色像素: {np.sum(result_binary == 255)}, 黑色像素: {np.sum(result_binary == 0)}")
    print("=" * 70)
    print()
    
    # 保存结果
    # 1. 原图
    orig_path = os.path.join(output_dir, '01_original.png')
    cv2.imwrite(orig_path, img_resized)
    print(f"✓ 保存原图: {orig_path}")
    
    # 2. 灰度增强结果
    gray_path = os.path.join(output_dir, '02_enhanced_gray.png')
    cv2.imwrite(gray_path, result_gray)
    print(f"✓ 保存灰度结果: {gray_path}")
    
    # 3. 二值化结果
    binary_path = os.path.join(output_dir, '03_enhanced_binary.png')
    cv2.imwrite(binary_path, result_binary)
    print(f"✓ 保存二值化结果: {binary_path}")
    
    # 4. 保存中间结果（如果有）
    if 'unet_repair' in intermediates:
        unet_path = os.path.join(output_dir, '04_intermediate_unet.png')
        cv2.imwrite(unet_path, intermediates['unet_repair'])
        print(f"✓ 保存UNet输出: {unet_path}")
    
    if 'texture_after_unet' in intermediates:
        texture_path = os.path.join(output_dir, '05_intermediate_tv_texture.png')
        cv2.imwrite(texture_path, intermediates['texture_after_unet'])
        print(f"✓ 保存TV纹理: {texture_path}")
    
    if 'gabor_enhanced' in intermediates:
        gabor_path = os.path.join(output_dir, '06_intermediate_gabor.png')
        cv2.imwrite(gabor_path, intermediates['gabor_enhanced'])
        print(f"✓ 保存Gabor输出: {gabor_path}")
    
    # 5. 创建对比图
    print()
    print("创建对比图...")
    comparison = np.hstack([
        cv2.resize(img_resized, (256, 256)),
        cv2.resize(result_gray, (256, 256)),
        cv2.resize(result_binary, (256, 256))
    ])
    
    # 添加标签
    comparison_labeled = comparison.copy()
    cv2.putText(comparison_labeled, 'Original', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    cv2.putText(comparison_labeled, 'Enhanced Gray', (266, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    cv2.putText(comparison_labeled, 'Binary', (522, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    
    comparison_path = os.path.join(output_dir, '00_comparison.png')
    cv2.imwrite(comparison_path, comparison_labeled)
    print(f"✓ 保存对比图: {comparison_path}")
    
    print()
    print("=" * 70)
    print(f"✅ 所有结果已保存到: {output_dir}/")
    print("=" * 70)
    print()
    print("文件列表:")
    print("  00_comparison.png - 三图对比")
    print("  01_original.png - 原始图像")
    print("  02_enhanced_gray.png - 灰度增强结果（方法2）")
    print("  03_enhanced_binary.png - 二值化结果（论文效果）")
    print("  04_intermediate_unet.png - UNet修复输出")
    print("  05_intermediate_tv_texture.png - TV纹理提取")
    print("  06_intermediate_gabor.png - Gabor增强输出")
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='保存最佳方法（UNet → TV → Gabor）的增强结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python save_best_method.py -i input.jpg
  python save_best_method.py -i input.jpg -o my_output_dir
  python save_best_method.py --input ../test2/1.jpg --output results
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='../test2/1.jpg',
        help='输入图像路径（默认: ../test2/1.jpg）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='best_method_results',
        help='输出目录路径（默认: best_method_results）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入图像不存在: {args.input}")
        print(f"   请检查文件路径是否正确")
        exit(1)
    
    # 运行处理
    save_best_method_results(args.input, output_dir=args.output)
