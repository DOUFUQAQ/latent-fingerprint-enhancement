#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合指纹增强Pipeline
结合论文技术和深度学习模型

Pipeline:
1. TV模型分解 → 去除结构噪声
2. 深度学习增强（UNet） → 初步修复
3. Gabor稀疏表示 → 细节增强
4. 多尺度迭代refinement → 最终优化
"""

import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入各模块
from repair.tv_decomposition import rudin_osher_fatemi_tv, adaptive_tv_decomposition
from repair.gabor_sparse_coding import (enhance_image_with_sparse_coding,
                                       multi_scale_sparse_coding)
from repair.multi_scale_refinement import (multi_scale_iterative_enhancement,
                                           quality_assessment)
from repair.train_advanced_repair import AdvancedUNet


class HybridFingerprintEnhancer:
    """混合指纹增强器
    
    结合传统方法和深度学习
    """
    
    def __init__(self, unet_checkpoint=None, device='auto'):
        """初始化
        
        Args:
            unet_checkpoint: UNet模型路径（可选）
            device: 设备 ('auto', 'cuda', 'cpu')
        """
        
        # 自动检测设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        self.unet_model = None
        
        # 加载UNet模型（如果提供）
        if unet_checkpoint and os.path.exists(unet_checkpoint):
            self.load_unet(unet_checkpoint)
    
    def load_unet(self, checkpoint_path):
        """加载UNet模型
        
        Args:
            checkpoint_path: 检查点路径
        """
        
        print(f"加载UNet模型: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 自动检测base_channels
        first_layer_shape = checkpoint['model_state_dict']['enc1.0.weight'].shape
        base_channels = first_layer_shape[0]
        
        # 创建模型
        self.unet_model = AdvancedUNet(in_channels=1, base_channels=base_channels)
        self.unet_model.load_state_dict(checkpoint['model_state_dict'])
        self.unet_model.to(self.device)
        self.unet_model.eval()
        
        print(f"✓ UNet加载成功 (base_channels={base_channels})")
    
    def preprocess_for_unet(self, image, image_size=256):
        """UNet预处理
        
        Args:
            image: 输入图像 (H, W)
            image_size: 目标大小
        
        Returns:
            tensor: 预处理后的张量
            original_shape: 原始形状
        """
        
        original_shape = image.shape
        
        # 调整大小
        if image.shape[0] != image_size or image.shape[1] != image_size:
            img_resized = cv2.resize(image, (image_size, image_size))
        else:
            img_resized = image
        
        # 归一化到[-1, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_normalized = (img_normalized - 0.5) * 2
        
        # 转换为tensor
        tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
        
        return tensor, original_shape
    
    def postprocess_from_unet(self, tensor, original_shape):
        """UNet后处理
        
        Args:
            tensor: 输出张量
            original_shape: 原始形状
        
        Returns:
            image: 后处理后的图像 (H, W)
        """
        
        # 转换为numpy
        img = tensor.squeeze().cpu().numpy()
        
        # 反归一化到[0, 255]
        img = (img + 1) / 2
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # 恢复原始尺寸
        if img.shape != original_shape:
            img = cv2.resize(img, (original_shape[1], original_shape[0]))
        
        return img
    
    def enhance(self, image, method='full', save_intermediates=False, output_dir=None, lambda_tv=1.5):
        """增强指纹图像
        
        Args:
            image: 输入图像 (H, W)
            method: 增强方法
                - 'tv_only': 仅TV分解
                - 'unet_only': 仅UNet
                - 'gabor_only': 仅Gabor稀疏表示
                - 'tv_gabor': TV + Gabor
                - 'unet_gabor': UNet + Gabor
                - 'unet_tv_gabor': UNet + TV + Gabor（新推荐）
                - 'full': 完整pipeline（推荐）
            save_intermediates: 是否保存中间结果
            output_dir: 输出目录
        
        Returns:
            enhanced: 增强后的图像
            intermediates: 中间结果字典
        """
        
        intermediates = {'original': image.copy()}
        
        print("\n" + "=" * 70)
        print(f"  混合指纹增强Pipeline (方法: {method})")
        print("=" * 70)
        print()
        
        # ===== 步骤1: UNet深度学习增强（如果unet_tv_gabor先做UNet） =====
        if method == 'unet_tv_gabor' and self.unet_model:
            print("[步骤1] UNet深度学习修复...")
            tensor, original_shape = self.preprocess_for_unet(image)
            tensor = tensor.to(self.device)
            with torch.no_grad():
                outputs = self.unet_model(tensor)
            unet_repair = self.postprocess_from_unet(outputs['repair'], original_shape)
            intermediates['unet_repair'] = unet_repair
            current = unet_repair
            print(f"✓ UNet修复完成\n")
        
        # ===== 步骤2: TV分解（可选） =====
        elif method in ['tv_only', 'tv_gabor', 'full']:
            print("[步骤1] TV模型分解...")
            print(f"  使用参数: λ={lambda_tv}, 迭代={150}次")
            cartoon, texture = adaptive_tv_decomposition(image, lambda_base=lambda_tv, num_iter=150)
            intermediates['cartoon'] = cartoon
            intermediates['texture'] = texture
            current = texture
            print(f"✓ TV分解完成")
            print(f"  卡通部分（结构噪声）: 已去除")
            print(f"  纹理部分（指纹脊线）: 已提取\n")
        else:
            current = image
        
        # ===== 步骤2/3: TV分解（如果unet_tv_gabor在UNet后做TV） =====
        if method == 'unet_tv_gabor':
            print("[步骤2] TV分解（去除UNet输出的背景噪声）...")
            print(f"  使用参数: λ={lambda_tv}, 迭代={150}次")
            cartoon, texture = adaptive_tv_decomposition(current, lambda_base=lambda_tv, num_iter=150)
            intermediates['cartoon_after_unet'] = cartoon
            intermediates['texture_after_unet'] = texture
            current = texture
            print(f"✓ TV分解完成")
            print(f"  Texture统计: 均值={np.mean(texture):.1f}, 范围=[{np.min(texture)}, {np.max(texture)}]\n")
        
        # ===== 步骤3: UNet深度学习增强（传统流程） =====
        if method in ['unet_only', 'unet_gabor', 'full'] and self.unet_model:
            print("[步骤3] UNet深度学习增强...")
            
            # 预处理
            tensor, original_shape = self.preprocess_for_unet(current)
            tensor = tensor.to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.unet_model(tensor)
            
            # 后处理
            unet_repair = self.postprocess_from_unet(outputs['repair'], original_shape)
            unet_segment = self.postprocess_from_unet(outputs['segment'], original_shape)
            unet_orient = self.postprocess_from_unet(outputs['orient'], original_shape)
            
            intermediates['unet_repair'] = unet_repair
            intermediates['unet_segment'] = unet_segment
            intermediates['unet_orient'] = unet_orient
            current = unet_repair
            
            print(f"✓ UNet增强完成")
            print(f"  修复输出: 已生成")
            print(f"  分割输出: 已生成")
            print(f"  方向输出: 已生成\n")
        
        # ===== 步骤3/4: Gabor稀疏表示增强（可选） =====
        if method in ['gabor_only', 'tv_gabor', 'unet_gabor', 'unet_tv_gabor', 'full']:
            step_num = 3 if method == 'unet_tv_gabor' else (3 if method in ['tv_gabor', 'gabor_only'] else 4)
            print(f"[步骤{step_num}] Gabor稀疏表示增强...")
            
            # 对比度增强预处理（如果输入对比度不足）
            current_max = np.max(current)
            if current_max < 200:
                print(f"  检测到低对比度输入 (max={current_max})，应用对比度拉伸...")
                # 使用2%和98%分位数进行拉伸
                p2, p98 = np.percentile(current, (2, 98))
                if p98 > p2:
                    current_stretched = np.clip((current.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
                    print(f"  拉伸后范围: [{np.min(current_stretched)}, {np.max(current_stretched)}]")
                    intermediates['contrast_stretched'] = current_stretched
                    current = current_stretched
            
            # 使用多尺度稀疏编码
            # 注意: lambda_reg=0.5是论文推荐值，保证正确的稀疏度
            gabor_enhanced = multi_scale_sparse_coding(
                current, 
                small_size=24, 
                large_size=36,
                stride=8,
                method='lasso',
                lambda_reg=0.5,  # ✅ 论文值，正确的稀疏度
                gamma=0.8
            )
            
            intermediates['gabor_enhanced'] = gabor_enhanced
            current = gabor_enhanced
            
            print(f"✓ Gabor增强完成\n")
        
        # ===== 步骤5: 多尺度迭代refinement（仅full模式） =====
        if method == 'full':
            print("[步骤5] 多尺度迭代Refinement...")
            
            refined, history = multi_scale_iterative_enhancement(
                current,
                initial_patch_size=24,
                max_patch_size=48,
                patch_size_increment=4,
                max_iterations=10,  # 减少迭代次数以加快速度
                quality_threshold=0.5,
                lambda_reg=0.5,
                gamma=0.8
            )
            
            intermediates['refined'] = refined
            intermediates['refinement_history'] = history
            current = refined
            
            print(f"✓ 迭代refinement完成\n")
        
        # 最终结果
        enhanced = current
        
        # 保存中间结果
        if save_intermediates and output_dir:
            self.save_intermediates(intermediates, output_dir)
        
        print("=" * 70)
        print("  增强完成！")
        print("=" * 70 + "\n")
        
        return enhanced, intermediates
    
    def save_intermediates(self, intermediates, output_dir):
        """保存中间结果
        
        Args:
            intermediates: 中间结果字典
            output_dir: 输出目录
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, img in intermediates.items():
            if isinstance(img, np.ndarray) and img.ndim == 2:
                cv2.imwrite(os.path.join(output_dir, f'{name}.png'), img)
        
        print(f"✓ 中间结果已保存到: {output_dir}")
    
    def visualize_pipeline(self, intermediates, save_path=None):
        """可视化pipeline
        
        Args:
            intermediates: 中间结果字典
            save_path: 保存路径
        """
        
        # 选择要显示的关键步骤
        keys_to_show = []
        titles = []
        
        if 'original' in intermediates:
            keys_to_show.append('original')
            titles.append('原始输入')
        
        if 'texture' in intermediates:
            keys_to_show.append('texture')
            titles.append('TV分解\n(纹理部分)')
        
        if 'unet_repair' in intermediates:
            keys_to_show.append('unet_repair')
            titles.append('UNet\n(修复输出)')
        
        if 'gabor_enhanced' in intermediates:
            keys_to_show.append('gabor_enhanced')
            titles.append('Gabor\n(稀疏表示)')
        
        if 'refined' in intermediates:
            keys_to_show.append('refined')
            titles.append('最终输出\n(多尺度refinement)')
        elif len(keys_to_show) > 0:
            # 如果没有refined，最后一个就是最终输出
            titles[-1] = '最终输出'
        
        n = len(keys_to_show)
        
        if n == 0:
            print("没有中间结果可显示")
            return
        
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        
        if n == 1:
            axes = [axes]
        
        for idx, (key, title) in enumerate(zip(keys_to_show, titles)):
            img = intermediates[key]
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        plt.suptitle('混合指纹增强Pipeline', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Pipeline可视化已保存: {save_path}")
        
        plt.show()


def batch_enhance(input_dir, output_dir, enhancer, method='full'):
    """批量增强
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        enhancer: HybridFingerprintEnhancer实例
        method: 增强方法
    """
    
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找图像
    image_files = glob.glob(os.path.join(input_dir, '*.png')) + \
                  glob.glob(os.path.join(input_dir, '*.jpg'))
    
    print(f"找到 {len(image_files)} 张图像")
    print(f"方法: {method}\n")
    
    for img_path in tqdm(image_files, desc="批量增强"):
        filename = os.path.basename(img_path)
        name = os.path.splitext(filename)[0]
        
        # 读取
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # 增强
        enhanced, _ = enhancer.enhance(img, method=method, save_intermediates=False)
        
        # 保存
        cv2.imwrite(os.path.join(output_dir, f'{name}_enhanced.png'), enhanced)
    
    print(f"\n✓ 批量增强完成！输出目录: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='混合指纹增强Pipeline')
    parser.add_argument('--input', type=str, required=True, help='输入图像或目录')
    parser.add_argument('--output', type=str, default='hybrid_results', help='输出目录')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints_ultimate/best_model_ultimate.pth',
                       help='UNet模型路径')
    parser.add_argument('--method', type=str, default='full',
                       choices=['tv_only', 'unet_only', 'gabor_only', 'tv_gabor', 'unet_gabor', 'full'],
                       help='增强方法')
    parser.add_argument('--save_intermediates', action='store_true', help='保存中间结果')
    parser.add_argument('--visualize', action='store_true', help='可视化pipeline')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], 
                       help='设备 (auto=自动检测)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  混合指纹增强Pipeline")
    print("=" * 70)
    print()
    
    # 创建增强器
    enhancer = HybridFingerprintEnhancer(
        unet_checkpoint=args.checkpoint if os.path.exists(args.checkpoint) else None,
        device=args.device
    )
    
    # 处理
    if os.path.isfile(args.input):
        # 单张图像
        img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        
        enhanced, intermediates = enhancer.enhance(
            img,
            method=args.method,
            save_intermediates=args.save_intermediates,
            output_dir=args.output
        )
        
        # 保存最终结果
        os.makedirs(args.output, exist_ok=True)
        cv2.imwrite(os.path.join(args.output, 'final_enhanced.png'), enhanced)
        
        # 可视化
        if args.visualize:
            enhancer.visualize_pipeline(intermediates, 
                                       os.path.join(args.output, 'pipeline_visualization.png'))
    
    else:
        # 批量处理
        batch_enhance(args.input, args.output, enhancer, method=args.method)
