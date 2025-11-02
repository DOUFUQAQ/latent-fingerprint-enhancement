#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
感知损失（Perceptual Loss）
使用预训练VGG提取特征，比较高层语义相似性
对结构性差异更敏感
"""

import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """VGG感知损失"""
    
    def __init__(self, feature_layers=[3, 8, 15, 22], use_normalization=True):
        """
        Args:
            feature_layers: VGG特征层索引（relu1_2, relu2_2, relu3_3, relu4_3）
            use_normalization: 是否对特征归一化
        """
        super(VGGPerceptualLoss, self).__init__()
        
        # 加载预训练VGG16
        vgg = models.vgg16(pretrained=True)
        
        # 提取特征层
        self.feature_blocks = nn.ModuleList()
        self.feature_layers = feature_layers
        
        prev_layer = 0
        for layer_idx in feature_layers:
            block = nn.Sequential(*list(vgg.features.children())[prev_layer:layer_idx+1])
            self.feature_blocks.append(block)
            prev_layer = layer_idx + 1
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
        
        self.use_normalization = use_normalization
        self.criterion = nn.L1Loss()
        
        # VGG归一化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x):
        """归一化到ImageNet标准"""
        # x: [-1, 1] -> [0, 1]
        x = (x + 1) / 2
        # 转换为RGB（复制灰度通道）
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # ImageNet归一化
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测图像 [-1, 1]
            target: 目标图像 [-1, 1]
        
        Returns:
            loss: 感知损失
        """
        # 归一化
        pred = self._normalize(pred)
        target = self._normalize(target)
        
        # 提取多层特征并计算损失
        loss = 0.0
        x_pred = pred
        x_target = target
        
        for block in self.feature_blocks:
            x_pred = block(x_pred)
            x_target = block(x_target)
            
            if self.use_normalization:
                # 归一化特征
                x_pred_norm = x_pred / (x_pred.norm(dim=1, keepdim=True) + 1e-8)
                x_target_norm = x_target / (x_target.norm(dim=1, keepdim=True) + 1e-8)
                loss += self.criterion(x_pred_norm, x_target_norm)
            else:
                loss += self.criterion(x_pred, x_target)
        
        # 平均到每层
        loss = loss / len(self.feature_blocks)
        
        return loss

class EnhancedLoss(nn.Module):
    """增强版损失：MSE + 感知损失"""
    
    def __init__(self, lambda_mse=1.0, lambda_perceptual=0.1):
        super(EnhancedLoss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual
        
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = VGGPerceptualLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测图像 [-1, 1]
            target: 目标图像 [-1, 1]
        
        Returns:
            loss: 总损失
            loss_dict: 各项损失
        """
        # MSE损失
        mse = self.mse_loss(pred, target)
        
        # 感知损失
        perceptual = self.perceptual_loss(pred, target)
        
        # 总损失
        total = self.lambda_mse * mse + self.lambda_perceptual * perceptual
        
        return total, {
            'mse': mse.item(),
            'perceptual': perceptual.item(),
            'total': total.item()
        }

# 测试代码
if __name__ == '__main__':
    print("测试感知损失...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建损失函数
    criterion = EnhancedLoss(lambda_mse=1.0, lambda_perceptual=0.1).to(device)
    
    # 测试
    pred = torch.randn(2, 1, 256, 256).to(device)
    target = torch.randn(2, 1, 256, 256).to(device)
    
    loss, loss_dict = criterion(pred, target)
    
    print("✓ 感知损失测试成功")
    print(f"  MSE损失: {loss_dict['mse']:.6f}")
    print(f"  感知损失: {loss_dict['perceptual']:.6f}")
    print(f"  总损失: {loss_dict['total']:.6f}")
