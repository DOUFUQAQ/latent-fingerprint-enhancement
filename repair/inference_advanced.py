#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版指纹脊线修复推理脚本
三任务输出：脊线分割 + 方向场 + 完整脊线
"""

import os
import sys
import torch
import cv2
import numpy as np
import argparse
from repair.train_advanced_repair import AdvancedUNet

def load_model(checkpoint_path, device, base_channels=None):
    """加载训练好的模型（自动检测模型尺寸）"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 自动检测 base_channels
    if base_channels is None:
        first_layer_shape = checkpoint['model_state_dict']['enc1.0.weight'].shape
        base_channels = first_layer_shape[0]
        print("✓ 自动检测模型尺寸: base_channels=%d" % base_channels)
    
    # 创建模型
    model = AdvancedUNet(in_channels=1, base_channels=base_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print("✓ 模型加载成功")
    print("  模型参数量: %.2fM" % (num_params / 1e6))
    if 'val_loss' in checkpoint:
        val_loss = checkpoint['val_loss']
        print("  验证损失: %.6f" % val_loss.get('total', val_loss))
    
    return model

def preprocess_image(img_path, image_size=256):
    """预处理输入图像"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法读取图像: %s" % img_path)
    
    original_shape = img.shape
    
    # 调整尺寸
    if img.shape[0] != image_size:
        img = cv2.resize(img, (image_size, image_size))
    
    # 归一化到 [-1, 1]
    img = (img / 255.0 - 0.5) * 2.0
    
    # 转换为 Tensor
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    
    return img_tensor, original_shape

def postprocess_image(output_tensor, original_shape):
    """后处理输出图像"""
    output = output_tensor.squeeze().cpu().numpy()
    
    # 反归一化到 [0, 255]
    output = ((output + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    
    # 恢复原始尺寸
    if output.shape != original_shape:
        output = cv2.resize(output, (original_shape[1], original_shape[0]))
    
    return output

def postprocess_mask(mask_tensor, original_shape):
    """后处理分割掩码"""
    mask = mask_tensor.squeeze().cpu().numpy()
    
    # 转换到 [0, 255]
    mask = (mask * 255.0).clip(0, 255).astype(np.uint8)
    
    # 恢复原始尺寸
    if mask.shape != original_shape:
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
    
    return mask

def repair_image(model, img_path, output_dir, image_size=256, device='cuda'):
    """
    修复单张图像
    
    输出三个结果:
    1. segment: 脊线分割掩码（识别出的脊线位置）
    2. orient: 方向场（脊线方向）
    3. repair: 完整修复的脊线图像
    """
    # 预处理
    img_tensor, original_shape = preprocess_image(img_path, image_size)
    img_tensor = img_tensor.to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # 后处理
    segment_mask = postprocess_mask(outputs['segment'], original_shape)
    orient_img = postprocess_image(outputs['orient'], original_shape)
    repair_img = postprocess_image(outputs['repair'], original_shape)
    
    # 生成文件名
    basename = os.path.basename(img_path)
    name_without_ext = os.path.splitext(basename)[0]
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 脊线分割掩码
    segment_path = os.path.join(output_dir, '%s_segment.png' % name_without_ext)
    cv2.imwrite(segment_path, segment_mask)
    
    # 2. 方向场
    orient_path = os.path.join(output_dir, '%s_orientation.png' % name_without_ext)
    cv2.imwrite(orient_path, orient_img)
    
    # 3. 修复后的脊线
    repair_path = os.path.join(output_dir, '%s_repaired.png' % name_without_ext)
    cv2.imwrite(repair_path, repair_img)
    
    # 4. 对比图（原始 vs 修复）
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if original_img.shape != (original_shape[0], original_shape[1]):
        original_img = cv2.resize(original_img, (original_shape[1], original_shape[0]))
    
    # 创建对比图
    comparison = np.hstack([
        original_img,
        np.ones((original_shape[0], 10), dtype=np.uint8) * 255,  # 分隔线
        segment_mask,
        np.ones((original_shape[0], 10), dtype=np.uint8) * 255,
        orient_img,
        np.ones((original_shape[0], 10), dtype=np.uint8) * 255,
        repair_img
    ])
    
    comparison_path = os.path.join(output_dir, '%s_comparison.png' % name_without_ext)
    cv2.imwrite(comparison_path, comparison)
    
    print("✓ 处理完成: %s" % basename)
    print("  - 分割掩码: %s" % segment_path)
    print("  - 方向场: %s" % orient_path)
    print("  - 修复结果: %s" % repair_path)
    print("  - 对比图: %s" % comparison_path)
    
    return {
        'segment': segment_mask,
        'orient': orient_img,
        'repair': repair_img
    }

def _collect_image_files(input_dir, recursive=True):
    """收集目录中的所有图像文件"""
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    files = []
    input_dir = os.path.abspath(input_dir)
    if recursive:
        for root, _, filenames in os.walk(input_dir):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in valid_exts:
                    files.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_dir):
            path = os.path.join(input_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in valid_exts:
                files.append(path)
    return sorted(files)

def batch_repair(model, input_dir, output_dir, image_size=256, device='cuda', recursive=True):
    """批量修复图像（支持多后缀与递归）"""
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像
    image_files = _collect_image_files(input_dir, recursive=recursive)
    print("\n扫描目录: %s" % os.path.abspath(input_dir))
    print("找到 %d 张图像 (支持: png/jpg/jpeg/bmp/tif/tiff/webp)\n" % len(image_files))
    if len(image_files) == 0:
        print("提示: 若目录包含子文件夹或非PNG，请使用 --recursive（已默认开启）并确保后缀正确。")
    
    # 批量处理
    for img_path in tqdm(image_files, desc="修复图像"):
        try:
            repair_image(model, img_path, output_dir, image_size, device)
        except Exception as e:
            filename = os.path.basename(img_path)
            print("✗ 处理失败 %s: %s" % (filename, str(e)))

def create_visualization(outputs, original_img):
    """创建可视化结果"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原始图像
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title('输入：原始指纹\n(模糊、断裂、有背景)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 脊线分割
    axes[0, 1].imshow(outputs['segment'], cmap='gray')
    axes[0, 1].set_title('任务1：脊线识别\n(从复杂背景中提取脊线)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 方向场
    axes[1, 0].imshow(outputs['orient'], cmap='gray')
    axes[1, 0].set_title('任务2：方向场预测\n(理解脊线走向)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 修复结果
    axes[1, 1].imshow(outputs['repair'], cmap='gray')
    axes[1, 1].set_title('任务3：脊线修复\n(完整清晰的脊线)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='增强版指纹脊线修复推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像或目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    parser.add_argument('--recursive', action='store_true', help='递归扫描子目录中的图片')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    print("\n使用设备: %s\n" % device)
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    print("")
    
    # 判断是单张图像还是批量处理
    if os.path.isfile(args.input):
        # 单张图像
        print("处理单张图像...\n")
        outputs = repair_image(model, args.input, args.output, args.image_size, device)
        
        # 可视化
        if args.visualize:
            import matplotlib.pyplot as plt
            original_img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
            fig = create_visualization(outputs, original_img)
            viz_path = os.path.join(args.output, 'visualization.png')
            fig.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("✓ 可视化结果: %s" % viz_path)
    else:
        # 批量处理
        print("批量处理图像...\n")
        batch_repair(model, args.input, args.output, args.image_size, device, recursive=(args.recursive or True))
    
    print("\n完成！\n")

if __name__ == '__main__':
    main()
