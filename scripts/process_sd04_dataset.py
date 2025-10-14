#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 FingerNet 批量处理 SD04 数据集
提取脊线增强图像，生成配对数据集用于训练

输入: SD04 原始潜指纹图像 (PNG)
输出: 
  - 原始图像 (用于训练的输入)
  - 增强脊线图像 (用于训练的目标)
  - 分割掩码 (前景/背景)
  - 方向场图像
"""

import os
import sys
import cv2
import numpy as np
from scipy import misc, ndimage
import glob
import shutil
from time import time
from datetime import datetime

# 获取脚本绝对路径
SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
SCRIPT_NAME = os.path.basename(SCRIPT_PATH)

# 设置 matplotlib 后端
import matplotlib
matplotlib.use('Agg')

# 导入 FingerNet 相关模块
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU，如果只有CPU则自动使用CPU

# 配置导入环境
# train_test_deploy.py 中有: copy_file(sys.path[0]+'/'+sys.argv[0], output_dir+'/')
# 需要: sys.path[0] = 脚本目录, sys.argv[0] = 脚本文件名
src_dir = os.path.join(SCRIPT_DIR, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
sys.path.insert(0, SCRIPT_DIR)  # 临时放在最前面，让 copy_file 能找到文件
sys.argv = [SCRIPT_NAME, '0', 'deploy']  # 只用文件名，不用绝对路径

from keras import backend as K
config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
sess = K.tf.Session(config=config)
K.set_session(sess)

from train_test_deploy import get_main_net, ori_highest_peak
from utils import mkdir

# ==================== 配置参数 ====================
# SCRIPT_DIR 已在文件开头定义
PROJECT_ROOT = SCRIPT_DIR

# 数据集和模型路径配置
SD04_ROOT = os.path.join(PROJECT_ROOT, '..', 'sd04', 'png_txt')  # SD04 数据集根目录
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, '..', 'processed_sd04_dataset')  # 输出数据集根目录
PRETRAINED_MODEL = os.path.join(PROJECT_ROOT, 'models', 'released_version', 'Model.model')  # 预训练模型路径

# 路径归一化
SD04_ROOT = os.path.normpath(SD04_ROOT)
OUTPUT_ROOT = os.path.normpath(OUTPUT_ROOT)
PRETRAINED_MODEL = os.path.normpath(PRETRAINED_MODEL)

# 输出子目录
OUTPUT_ORIGINAL = os.path.join(OUTPUT_ROOT, 'original')      # 原始图像
OUTPUT_ENHANCED = os.path.join(OUTPUT_ROOT, 'enhanced')      # 增强图像（脊线）
OUTPUT_SEGMENTATION = os.path.join(OUTPUT_ROOT, 'segmentation')  # 分割掩码
OUTPUT_ORIENTATION = os.path.join(OUTPUT_ROOT, 'orientation')    # 方向场
OUTPUT_PAIRS = os.path.join(OUTPUT_ROOT, 'paired_list.txt')  # 配对文件列表

# 处理参数
CONVERT_TO_BMP = True  # 是否转换为BMP格式（FingerNet需要）
TARGET_SIZE = None  # 目标尺寸，None表示保持原始尺寸
BATCH_PROCESS = True  # 批量处理模式

# ==================== 辅助函数 ====================

def create_output_dirs():
    """创建输出目录结构"""
    print("创建输出目录...")
    mkdir(OUTPUT_ROOT)
    mkdir(OUTPUT_ORIGINAL)
    mkdir(OUTPUT_ENHANCED)
    mkdir(OUTPUT_SEGMENTATION)
    mkdir(OUTPUT_ORIENTATION)
    print("输出目录创建完成: %s" % OUTPUT_ROOT)

def get_all_fingerprint_images():
    """获取所有SD04指纹图像路径"""
    print("\n扫描 SD04 数据集...")
    image_paths = []
    
    # 遍历所有子文件夹
    for subdir in ['figs_0', 'figs_1', 'figs_2', 'figs_3', 
                   'figs_4', 'figs_5', 'figs_6', 'figs_7']:
        subdir_path = os.path.join(SD04_ROOT, subdir)
        if not os.path.exists(subdir_path):
            print("警告: 目录不存在 - %s" % subdir_path)
            continue
        
        # 查找所有PNG图像
        pattern = os.path.join(subdir_path, '*.png')
        images = glob.glob(pattern)
        image_paths.extend(images)
    
    print("找到 %d 张指纹图像" % len(image_paths))
    return sorted(image_paths)

def preprocess_image(img_path, target_size=None):
    """
    预处理图像
    
    Args:
        img_path: 图像路径
        target_size: 目标尺寸 (height, width)，None表示保持原始尺寸
    
    Returns:
        image: 归一化后的图像 [0, 1]
        original: 原始图像 [0, 255]
        img_size: 图像尺寸 (height, width)
    """
    # 读取图像（灰度）
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None:
        raise ValueError("无法读取图像: %s" % img_path)
    
    # 调整尺寸（如果需要）
    if target_size is not None:
        original = cv2.resize(original, (target_size[1], target_size[0]))
    
    # 确保尺寸是8的倍数（FingerNet要求）
    h, w = original.shape
    h_new = (h // 8) * 8
    w_new = (w // 8) * 8
    
    if h != h_new or w != w_new:
        original = original[:h_new, :w_new]
    
    # 归一化到 [0, 1]
    image = original.astype(np.float32) / 255.0
    image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])
    
    return image, original, (h_new, w_new)

def save_image(img, save_path, is_float=False):
    """
    保存图像
    
    Args:
        img: 图像数据
        save_path: 保存路径
        is_float: 是否为浮点图像 [0, 1]
    """
    # 确保目录存在
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 转换数据类型
    if is_float:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    # 保存图像
    cv2.imwrite(save_path, img)

def process_single_image(model, img_path, image_name):
    """
    处理单张图像
    
    Args:
        model: FingerNet 模型
        img_path: 输入图像路径
        image_name: 图像名称（不含扩展名）
    
    Returns:
        success: 是否成功
        info: 处理信息字典
    """
    try:
        # 预处理
        image, original, img_size = preprocess_image(img_path, TARGET_SIZE)
        
        # FingerNet 推理
        enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = model.predict(image)
        
        # 处理输出
        # 1. 增强图像（脊线）
        enhanced = np.squeeze(enhance_img)
        
        # 2. 分割掩码
        seg_mask = np.squeeze(seg_out)
        seg_mask_round = np.round(seg_mask)
        
        # 形态学处理：开运算去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_mask_clean = cv2.morphologyEx(seg_mask_round, cv2.MORPH_OPEN, kernel)
        
        # 上采样到原始分辨率
        seg_mask_full = ndimage.zoom(seg_mask_clean, [8, 8], order=0)
        
        # 3. 方向场
        ori = sess.run(ori_highest_peak(ori_out_1))
        ori = (np.argmax(ori, axis=-1) * 2 - 90) / 180.0 * np.pi
        ori_full = ndimage.zoom(np.squeeze(ori), [8, 8], order=0)
        
        # 应用分割掩码到增强图像（只保留前景区域）
        enhanced_masked = enhanced * seg_mask_full
        
        # 归一化增强图像到 [0, 255]
        # 脊线为白色，背景为黑色
        enhanced_normalized = np.zeros_like(enhanced_masked)
        if np.max(enhanced_masked) > np.min(enhanced_masked):
            enhanced_normalized = (enhanced_masked - np.min(enhanced_masked)) / (np.max(enhanced_masked) - np.min(enhanced_masked))
        
        # 保存结果
        # 1. 原始图像
        save_image(original, os.path.join(OUTPUT_ORIGINAL, image_name + '.png'))
        
        # 2. 增强图像（脊线）
        save_image(enhanced_normalized, os.path.join(OUTPUT_ENHANCED, image_name + '.png'), is_float=True)
        
        # 3. 分割掩码
        save_image(seg_mask_full * 255, os.path.join(OUTPUT_SEGMENTATION, image_name + '.png'))
        
        # 4. 方向场（保存为可视化图像）
        # 将方向编码为颜色
        ori_vis = ((ori_full + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        save_image(ori_vis, os.path.join(OUTPUT_ORIENTATION, image_name + '.png'))
        
        return True, {
            'name': image_name,
            'size': img_size,
            'foreground_ratio': np.sum(seg_mask_full) / (seg_mask_full.shape[0] * seg_mask_full.shape[1])
        }
        
    except Exception as e:
        print("  ✗ 处理失败: %s" % str(e))
        return False, {'name': image_name, 'error': str(e)}

def main():
    """主处理流程"""
    print("\n" + "=" * 70)
    print("  SD04 数据集批量处理 - 使用 FingerNet 提取脊线特征")
    print("=" * 70)
    print("\n配置信息:")
    print("  当前工作目录: %s" % os.getcwd())
    print("  脚本目录: %s" % SCRIPT_DIR)
    print("  输入目录: %s" % SD04_ROOT)
    print("  输出目录: %s" % OUTPUT_ROOT)
    print("  预训练模型: %s" % PRETRAINED_MODEL)
    print("\n")
    
    # 检查 SD04 数据集目录
    if not os.path.exists(SD04_ROOT):
        print("✗ 错误: SD04 数据集目录不存在")
        print("  期望路径: %s" % SD04_ROOT)
        print("\n请确认:")
        print("  1. SD04 数据集已解压")
        print("  2. 目录结构为: sd04/png_txt/figs_X/*.png")
        print("  3. 与 FingerNet-master 在同一父目录下")
        print("\n当前目录结构应为:")
        print("  lanyun-tmp/")
        print("  ├── sd04/")
        print("  │   └── png_txt/")
        print("  │       ├── figs_0/")
        print("  │       ├── figs_1/")
        print("  │       └── ...")
        print("  └── FingerNet-master/")
        print("      └── process_sd04_dataset.py (当前脚本)")
        return
    
    # 检查模型文件
    if not os.path.exists(PRETRAINED_MODEL):
        print("✗ 错误: 预训练模型不存在")
        print("  期望路径: %s" % PRETRAINED_MODEL)
        print("\n请确认:")
        print("  1. 模型文件在 models/released_version/Model.model")
        print("  2. 文件大小约 19MB")
        return
    
    # 创建输出目录
    create_output_dirs()
    
    # 获取所有图像
    image_paths = get_all_fingerprint_images()
    if len(image_paths) == 0:
        print("✗ 错误: 未找到图像文件")
        return
    
    # 加载 FingerNet 模型
    print("\n加载 FingerNet 模型...")
    # 使用第一张图像确定输入尺寸
    first_img = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    h, w = first_img.shape
    h = (h // 8) * 8
    w = (w // 8) * 8
    
    print("  输入尺寸: %d x %d" % (h, w))
    model = get_main_net((h, w, 1), PRETRAINED_MODEL)
    print("✓ 模型加载完成\n")
    
    # 批量处理
    print("=" * 70)
    print("开始批量处理...")
    print("=" * 70)
    
    success_count = 0
    fail_count = 0
    pair_list = []
    start_time = time()
    
    for idx, img_path in enumerate(image_paths):
        # 获取图像名称
        base_name = os.path.basename(img_path)
        image_name = os.path.splitext(base_name)[0]
        
        # 进度显示
        progress = (idx + 1) / float(len(image_paths)) * 100
        elapsed = time() - start_time
        eta = elapsed / (idx + 1) * (len(image_paths) - idx - 1)
        
        print("[%d/%d] (%.1f%%) 处理: %s | 用时: %.1fs | 预计剩余: %.1fs" % 
              (idx + 1, len(image_paths), progress, image_name, elapsed, eta))
        
        # 处理图像
        success, info = process_single_image(model, img_path, image_name)
        
        if success:
            success_count += 1
            # 记录配对信息
            pair_list.append({
                'original': os.path.join('original', image_name + '.png'),
                'enhanced': os.path.join('enhanced', image_name + '.png'),
                'segmentation': os.path.join('segmentation', image_name + '.png'),
                'orientation': os.path.join('orientation', image_name + '.png'),
                'foreground_ratio': info['foreground_ratio']
            })
            print("  ✓ 成功 | 前景比例: %.2f%%" % (info['foreground_ratio'] * 100))
        else:
            fail_count += 1
    
    total_time = time() - start_time
    
    # 保存配对列表
    print("\n保存配对列表...")
    with open(OUTPUT_PAIRS, 'w') as f:
        f.write("# SD04 数据集配对列表\n")
        f.write("# 生成时间: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.write("# 总数: %d\n" % len(pair_list))
        f.write("# 格式: 原始图像|增强图像|分割掩码|方向场|前景比例\n")
        f.write("#" + "-" * 68 + "\n")
        for pair in pair_list:
            f.write("%s|%s|%s|%s|%.4f\n" % (
                pair['original'], 
                pair['enhanced'], 
                pair['segmentation'],
                pair['orientation'],
                pair['foreground_ratio']
            ))
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print("统计信息:")
    print("  总计: %d 张" % len(image_paths))
    print("  成功: %d 张 (%.1f%%)" % (success_count, success_count / float(len(image_paths)) * 100))
    print("  失败: %d 张 (%.1f%%)" % (fail_count, fail_count / float(len(image_paths)) * 100))
    print("  总用时: %.1f 秒 (%.2f 秒/张)" % (total_time, total_time / len(image_paths)))
    print("\n输出目录:")
    print("  原始图像: %s" % OUTPUT_ORIGINAL)
    print("  增强图像: %s" % OUTPUT_ENHANCED)
    print("  分割掩码: %s" % OUTPUT_SEGMENTATION)
    print("  方向场: %s" % OUTPUT_ORIENTATION)
    print("  配对列表: %s" % OUTPUT_PAIRS)
    print("\n数据集已准备好用于训练！")
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
