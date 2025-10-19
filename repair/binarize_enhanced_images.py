#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强图像二值化与背景去除
对 FingerNet 生成的增强图像进行二值化处理，并应用分割掩码去除背景
让脊线更加清晰，适合用于训练
"""

import os
import cv2
import numpy as np
import glob
from time import time

# ==================== 配置参数 ====================
INPUT_DIR = '../processed_sd04_dataset/enhanced'        # 输入：增强图像
MASK_DIR = '../processed_sd04_dataset/segmentation'    # 输入：分割掩码
OUTPUT_DIR = '../processed_sd04_dataset/enhanced_binary'  # 输出：二值化图像

# 二值化方法选择
# 'otsu' - Otsu 自动阈值（推荐）
# 'adaptive' - 自适应阈值
# 'threshold' - 固定阈值
BINARIZATION_METHOD = 'adaptive'  # 使用 Adaptive 方法（效果最好）

# 固定阈值（仅当 method='threshold' 时使用）
FIXED_THRESHOLD = 127

# 形态学处理（可选）
ENABLE_MORPHOLOGY = True  # 是否启用形态学处理
MORPH_KERNEL_SIZE = 3     # 形态学核大小

# 处理选项
APPLY_MASK = True         # 是否应用分割掩码去除背景
INVERT_OUTPUT = True      # 反转输出：黑色脊线，白色背景

# 掩码边缘处理（避免分割边缘噪声）
ERODE_MASK_EDGE = True    # 是否腐蚀掩码边缘
ERODE_KERNEL_SIZE = 15    # 腐蚀核大小（加倍腐蚀：7→11）
ERODE_ITERATIONS = 3      # 腐蚀迭代次数（加倍腐蚀：2→3）

# ==================== 辅助函数 ====================

def create_output_dir():
    """创建输出目录"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print("✓ 创建输出目录: %s" % OUTPUT_DIR)

def binarize_image(img, method='otsu', threshold=127):
    """
    二值化图像
    
    Args:
        img: 输入灰度图像
        method: 二值化方法 ('otsu', 'adaptive', 'threshold')
        threshold: 固定阈值（仅用于 method='threshold'）
    
    Returns:
        binary_img: 二值化图像
    """
    if method == 'otsu':
        # Otsu 自动阈值
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    elif method == 'adaptive':
        # 自适应阈值（适合光照不均匀的情况）
        binary = cv2.adaptiveThreshold(
            img, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            blockSize=11, 
            C=2
        )
        
    elif method == 'threshold':
        # 固定阈值
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    else:
        raise ValueError("不支持的二值化方法: %s" % method)
    
    return binary

def morphology_process(img, kernel_size=3):
    """
    形态学处理：去除噪点，连接断裂
    
    Args:
        img: 二值图像
        kernel_size: 核大小
    
    Returns:
        processed: 处理后的图像
    """
    # 创建形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 先开运算去除小噪点
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 再闭运算连接断裂
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return closed

def erode_mask_edge(mask, kernel_size=5, iterations=2):
    """
    腐蚀掩码边缘，去除边界噪声
    
    Args:
        mask: 分割掩码（255=前景，0=背景）
        kernel_size: 腐蚀核大小
        iterations: 腐蚀迭代次数
    
    Returns:
        eroded_mask: 腐蚀后的掩码
    """
    # 确保掩码是二值的
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 创建腐蚀核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 腐蚀操作：缩小前景区域
    eroded = cv2.erode(mask_binary, kernel, iterations=iterations)
    
    return eroded

def apply_segmentation_mask(img, mask, background_value=0, erode_edge=False, 
                           erode_kernel=5, erode_iter=2):
    """
    应用分割掩码，去除背景
    
    Args:
        img: 二值化图像
        mask: 分割掩码（255=前景，0=背景）
        background_value: 背景填充值 (0=黑色, 255=白色)
        erode_edge: 是否腐蚀掩码边缘
        erode_kernel: 腐蚀核大小
        erode_iter: 腐蚀迭代次数
    
    Returns:
        masked_img: 应用掩码后的图像
    """
    # 确保掩码是二值的
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 腐蚀掩码边缘（可选）
    if erode_edge:
        mask_binary = erode_mask_edge(mask_binary, erode_kernel, erode_iter)
    
    # 创建输出图像，背景填充指定值
    masked = np.full_like(img, background_value, dtype=np.uint8)
    
    # 只复制前景区域
    masked[mask_binary > 0] = img[mask_binary > 0]
    
    return masked

def process_single_image(enhanced_path, mask_path, output_path):
    """
    处理单张图像
    
    Args:
        enhanced_path: 增强图像路径
        mask_path: 分割掩码路径
        output_path: 输出路径
    
    Returns:
        success: 是否成功
        info: 处理信息
    """
    try:
        # 读取增强图像
        enhanced = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)
        if enhanced is None:
            return False, "无法读取增强图像"
        
        # 1. 二值化
        binary = binarize_image(enhanced, method=BINARIZATION_METHOD, threshold=FIXED_THRESHOLD)
        
        # 2. 形态学处理（可选）
        if ENABLE_MORPHOLOGY:
            binary = morphology_process(binary, MORPH_KERNEL_SIZE)
        
        # 3. 应用分割掩码和反转处理（组合处理）
        if APPLY_MASK:
            if not os.path.exists(mask_path):
                return False, "分割掩码不存在"
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return False, "无法读取分割掩码"
            
            # 调整掩码尺寸（如果不匹配）
            if mask.shape != binary.shape:
                mask = cv2.resize(mask, (binary.shape[1], binary.shape[0]))
            
            # 如果需要反转输出（黑色脊线，白色背景）
            if INVERT_OUTPUT:
                # 先反转前景区域（脊线变黑）
                binary = cv2.bitwise_not(binary)
                # 然后应用掩码，背景填充白色
                binary = apply_segmentation_mask(
                    binary, mask, 
                    background_value=255, 
                    erode_edge=ERODE_MASK_EDGE,
                    erode_kernel=ERODE_KERNEL_SIZE,
                    erode_iter=ERODE_ITERATIONS
                )
            else:
                # 白色脊线，背景填充黑色
                binary = apply_segmentation_mask(
                    binary, mask, 
                    background_value=0,
                    erode_edge=ERODE_MASK_EDGE,
                    erode_kernel=ERODE_KERNEL_SIZE,
                    erode_iter=ERODE_ITERATIONS
                )
        
        elif INVERT_OUTPUT:
            # 4. 如果没有掩码但需要反转
            binary = cv2.bitwise_not(binary)
        
        # 5. 保存结果
        cv2.imwrite(output_path, binary)
        
        # 计算统计信息
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.shape[0] * binary.shape[1]
        white_ratio = white_pixels / float(total_pixels)
        
        return True, {
            'white_ratio': white_ratio,
            'size': binary.shape
        }
        
    except Exception as e:
        return False, str(e)

def main():
    """主处理流程"""
    print("\n" + "=" * 70)
    print("  增强图像二值化与背景去除")
    print("=" * 70)
    print("\n配置信息:")
    print("  输入目录 (增强): %s" % INPUT_DIR)
    print("  输入目录 (掩码): %s" % MASK_DIR)
    print("  输出目录: %s" % OUTPUT_DIR)
    print("  二值化方法: %s" % BINARIZATION_METHOD)
    print("  应用分割掩码: %s" % APPLY_MASK)
    print("  形态学处理: %s" % ENABLE_MORPHOLOGY)
    print("  反转输出: %s (黑色脊线 + 白色背景)" % INVERT_OUTPUT if INVERT_OUTPUT else "  反转输出: %s (白色脊线 + 黑色背景)" % INVERT_OUTPUT)
    print("  腐蚀掩码边缘: %s" % ERODE_MASK_EDGE)
    if ERODE_MASK_EDGE:
        print("    - 核大小: %d" % ERODE_KERNEL_SIZE)
        print("    - 迭代次数: %d" % ERODE_ITERATIONS)
    print("")
    
    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        print("✗ 错误: 输入目录不存在 - %s" % INPUT_DIR)
        return
    
    if APPLY_MASK and not os.path.exists(MASK_DIR):
        print("✗ 错误: 掩码目录不存在 - %s" % MASK_DIR)
        return
    
    # 创建输出目录
    create_output_dir()
    
    # 获取所有增强图像
    print("扫描增强图像...")
    enhanced_files = glob.glob(os.path.join(INPUT_DIR, '*.png'))
    enhanced_files.sort()
    
    if len(enhanced_files) == 0:
        print("✗ 错误: 未找到增强图像")
        return
    
    print("找到 %d 张增强图像\n" % len(enhanced_files))
    
    # 批量处理
    print("=" * 70)
    print("开始批量处理...")
    print("=" * 70 + "\n")
    
    success_count = 0
    fail_count = 0
    start_time = time()
    
    for idx, enhanced_path in enumerate(enhanced_files):
        # 获取文件名
        filename = os.path.basename(enhanced_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # 构建路径
        mask_path = os.path.join(MASK_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # 进度显示
        progress = (idx + 1) / float(len(enhanced_files)) * 100
        elapsed = time() - start_time
        eta = elapsed / (idx + 1) * (len(enhanced_files) - idx - 1) if idx > 0 else 0
        
        print("[%d/%d] (%.1f%%) %s | 用时: %.1fs | 剩余: %.1fs" % 
              (idx + 1, len(enhanced_files), progress, name_without_ext, elapsed, eta))
        
        # 处理图像
        success, info = process_single_image(enhanced_path, mask_path, output_path)
        
        if success:
            success_count += 1
            print("  ✓ 成功 | 白色像素比例: %.2f%%" % (info['white_ratio'] * 100))
        else:
            fail_count += 1
            print("  ✗ 失败: %s" % info)
    
    total_time = time() - start_time
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print("统计信息:")
    print("  总计: %d 张" % len(enhanced_files))
    print("  成功: %d 张 (%.1f%%)" % (success_count, success_count / float(len(enhanced_files)) * 100))
    print("  失败: %d 张 (%.1f%%)" % (fail_count, fail_count / float(len(enhanced_files)) * 100))
    print("  总用时: %.1f 秒 (%.2f 秒/张)" % (total_time, total_time / len(enhanced_files)))
    print("\n输出目录: %s" % OUTPUT_DIR)
    print("\n提示: 请检查输出图像，确认脊线清晰且背景已去除")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n处理被用户中断")
    except Exception as e:
        print("\n\n✗ 处理过程中发生错误:")
        import traceback
        traceback.print_exc()
