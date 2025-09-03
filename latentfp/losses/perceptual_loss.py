"""
感知损失和风格损失实现
基于VGG16预训练特征
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    """基于VGG16的感知损失"""
    def __init__(self, feature_layers=[4, 9, 16], use_style_loss=True):
        super(VGGPerceptualLoss, self).__init__()
        
        # 加载预训练VGG16
        vgg = models.vgg16(pretrained=True).features
        self.feature_layers = feature_layers
        self.use_style_loss = use_style_loss
        
        # 分离VGG各层
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(feature_layers[0] + 1):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(feature_layers[0] + 1, feature_layers[1] + 1):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(feature_layers[1] + 1, feature_layers[2] + 1):
            self.slice3.add_module(str(x), vgg[x])
        
        # 冻结VGG参数
        for param in self.parameters():
            param.requires_grad = False
        
        # VGG标准化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_input(self, x):
        """标准化输入到VGG期望的范围"""
        if x.size(1) == 1:  # 如果是灰度图，复制到3通道
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return x
    
    def gram_matrix(self, features):
        """计算Gram矩阵用于风格损失"""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, input, target):
        # 标准化输入
        input = self.normalize_input(input)
        target = self.normalize_input(target)
        
        # 提取特征
        input_h1 = self.slice1(input)
        input_h2 = self.slice2(input_h1)
        input_h3 = self.slice3(input_h2)
        
        target_h1 = self.slice1(target)
        target_h2 = self.slice2(target_h1)
        target_h3 = self.slice3(target_h2)
        
        # 计算感知损失
        perceptual_loss = F.l1_loss(input_h1, target_h1) + \
                         F.l1_loss(input_h2, target_h2) + \
                         F.l1_loss(input_h3, target_h3)
        
        # 计算风格损失
        style_loss = 0
        if self.use_style_loss:
            style_loss = F.l1_loss(self.gram_matrix(input_h1), self.gram_matrix(target_h1)) + \
                        F.l1_loss(self.gram_matrix(input_h2), self.gram_matrix(target_h2)) + \
                        F.l1_loss(self.gram_matrix(input_h3), self.gram_matrix(target_h3))
        
        return perceptual_loss, style_loss

class OrientationFieldLoss(nn.Module):
    """指纹方向场约束损失"""
    def __init__(self):
        super(OrientationFieldLoss, self).__init__()
        
        # Sobel算子用于计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # 用于平滑的卷积核：使用15x15并配合padding=7以保持尺寸不变
        self.register_buffer('smooth_kernel',
                           torch.ones(1, 1, 15, 15, dtype=torch.float32) / (15 * 15))
    
    def compute_orientation_field(self, image):
        """计算指纹方向场"""
        # 计算梯度
        gx = F.conv2d(image, self.sobel_x, padding=1)
        gy = F.conv2d(image, self.sobel_y, padding=1)
        
        # 计算平方项
        gxx = gx * gx
        gyy = gy * gy
        gxy = gx * gy
        
        # 平滑处理（padding=7 以保持与输入同尺寸）
        gxx_smooth = F.conv2d(gxx, self.smooth_kernel, padding=7)
        gyy_smooth = F.conv2d(gyy, self.smooth_kernel, padding=7)
        gxy_smooth = F.conv2d(gxy, self.smooth_kernel, padding=7)
        
        # 计算方向角
        # theta = 0.5 * atan2(2*Gxy, Gxx - Gyy)
        numerator = 2 * gxy_smooth
        denominator = gxx_smooth - gyy_smooth
        orientation = 0.5 * torch.atan2(numerator, denominator + 1e-8)
        
        return orientation
    
    def orientation_coherence(self, orientation):
        """计算方向场一致性"""
        # 计算局部方向差异（分别在x、y方向）
        diff_x = orientation[:, :, 1:, :] - orientation[:, :, :-1, :]
        diff_y = orientation[:, :, :, 1:] - orientation[:, :, :, :-1]

        # 处理角度的周期性，将差异限制到[-pi/2, pi/2]
        diff_x = torch.atan2(torch.sin(2 * diff_x), torch.cos(2 * diff_x)) / 2
        diff_y = torch.atan2(torch.sin(2 * diff_y), torch.cos(2 * diff_y)) / 2

        # 分别计算x、y方向的平均方差，再组合为一致性分数（标量）
        var_x = torch.mean(diff_x ** 2)
        var_y = torch.mean(diff_y ** 2)
        coherence = torch.exp(-0.5 * (var_x + var_y))
        return coherence
    
    def forward(self, pred_image, target_image):
        """计算方向场损失"""
        pred_orient = self.compute_orientation_field(pred_image)
        target_orient = self.compute_orientation_field(target_image)
        
        # L1损失
        orientation_loss = F.l1_loss(pred_orient, target_orient)
        
        # 一致性损失
        coherence_loss = 1.0 - self.orientation_coherence(pred_orient)
        
        return orientation_loss + 0.1 * coherence_loss

class TotalVariationLoss(nn.Module):
    """总变分损失，用于平滑图像"""
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
    
    def forward(self, image):
        diff_x = image[:, :, 1:, :] - image[:, :, :-1, :]
        diff_y = image[:, :, :, 1:] - image[:, :, :, :-1]
        
        tv_loss = torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))
        return tv_loss
