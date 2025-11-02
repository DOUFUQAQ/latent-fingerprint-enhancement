#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大容量模型训练
base_channels=128，参数量约130M
更强的表达能力，突破当前瓶颈
"""

import sys
from repair.train_with_augmentation import *

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='大容量模型训练')
    parser.add_argument('--data_root', type=str, default='../processed_sd04_dataset')
    parser.add_argument('--output_dir', type=str, default='checkpoints_large')
    parser.add_argument('--batch_size', type=int, default=4, help='减小batch_size以适应更大模型')
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0001, help='稍微降低学习率')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--base_channels', type=int, default=128, help='增加到128')
    parser.add_argument('--resume', type=str, default='')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  🚀 大容量模型训练")
    print("  目标：突破 base_channels=64 的能力上限")
    print("=" * 70)
    print("\n配置:")
    print("  基础通道数: %d (提升 2x)" % args.base_channels)
    print("  预计参数量: ~130M (提升 4x)")
    print("  批大小: %d (减小以节省显存)" % args.batch_size)
    print("  学习率: %.6f" % args.lr)
    print("  训练轮数: %d" % args.num_epochs)
    print("\n预期:")
    print("  ✓ 更强的特征表达能力")
    print("  ✓ 更精细的细节重建")
    print("  ✓ 验证损失可能降到 50-60")
    print("  ⚠ 训练时间增加 1.5-2x")
    print("  ⚠ 显存需求增加 2-3x")
    print()
    
    main(args)
