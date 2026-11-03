"""
多尺度稀疏表示增强模块
基于论文: Latent Fingerprint Enhancement via Multi-Scale Patch Based Sparse Representation
"""

import warnings
from typing import Tuple, Optional, List

import numpy as np
try:  # OpenCV is optional: the pure-numpy helpers below import without it
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # OpenCV-backed helpers will raise if called

try:  # scikit-learn is only needed by the sparse-coding solvers
    from sklearn.linear_model import OrthogonalMatchingPursuit, LassoLars
except ImportError:  # pragma: no cover
    OrthogonalMatchingPursuit = LassoLars = None


class TVDecomposition:
    """TV模型分解器"""
    
    def __init__(self, lambda_tv: float = 0.1, max_iter: int = 100):
        self.lambda_tv = lambda_tv
        self.max_iter = max_iter
    
    def decompose(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        TV模型分解
        返回: (texture, cartoon)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = image.astype(np.float32) / 255.0
        h, w = image.shape
        
        # 初始化
        u = np.copy(image)  # cartoon部分
        v = np.zeros_like(image)  # texture部分
        
        # 梯度算子
        def gradient(img):
            grad_x = np.diff(img, axis=1, prepend=img[:, :1])
            grad_y = np.diff(img, axis=0, prepend=img[:1, :])
            return grad_x, grad_y
        
        def divergence(grad_x, grad_y):
            div_x = -np.diff(grad_x, axis=1, append=grad_x[:, -1:])
            div_y = -np.diff(grad_y, axis=0, append=grad_y[-1:, :])
            return div_x + div_y
        
        # 迭代优化
        for i in range(self.max_iter):
            u_old = np.copy(u)
            
            # 计算梯度
            grad_x, grad_y = gradient(u)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            
            # 更新u
            div_term = divergence(grad_x / grad_magnitude, grad_y / grad_magnitude)
            u = (image + self.lambda_tv * div_term) / (1 + self.lambda_tv)
            
            # 检查收敛
            if np.mean(np.abs(u - u_old)) < 1e-4:
                break
        
        v = image - u  # texture = original - cartoon
        
        return v, u

class GaborDictionary:
    """Gabor字典构建器"""
    
    def __init__(self, patch_size: int, orientations: int = 16, frequencies: Optional[List[float]] = None,
                 phases: Optional[List[float]] = None):
        self.patch_size = patch_size
        self.orientations = orientations
        self.frequencies = frequencies if frequencies is not None else [5,7,9,11,13,15,17,19,21]
        self.phases = phases if phases is not None else [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
        self.atom_orientations: Optional[np.ndarray] = None  # 与字典列一一对应的方向（弧度）
        self.atom_wavelengths: Optional[np.ndarray] = None   # 与字典列一一对应的像素波长
        self.atom_phases: Optional[np.ndarray] = None        # 与字典列一一对应的相位
    
    def build_dictionary(self) -> np.ndarray:
        """构建并返回字典矩阵，列为原子，行展平为patch"""
        size = self.patch_size
        atoms = []
        atom_oris = []
        atom_lams = []
        atom_phs = []
        for o in range(self.orientations):
            theta = o * np.pi / self.orientations
            for f in self.frequencies:
                for p in self.phases:
                    g = self._create_gabor_kernel(size, f, theta, p)
                    g = g - np.mean(g)
                    n = np.linalg.norm(g)
                    if n > 1e-8:
                        g = g / n
                        atoms.append(g.flatten())
                        atom_oris.append(theta)
                        atom_lams.append(float(f))
                        atom_phs.append(float(p))
        dictionary = np.stack(atoms, axis=1) if atoms else np.empty((size*size, 0))
        self.atom_orientations = np.array(atom_oris, dtype=np.float32) if atom_oris else np.empty((0,), dtype=np.float32)
        self.atom_wavelengths = np.array(atom_lams, dtype=np.float32) if atom_lams else np.empty((0,), dtype=np.float32)
        self.atom_phases = np.array(atom_phs, dtype=np.float32) if atom_phs else np.empty((0,), dtype=np.float32)
        return dictionary
    
    def _create_gabor_kernel(self, size: int, frequency: float, 
                           orientation: float, phase: float) -> np.ndarray:
        """创建Gabor核"""
        # 使用居中坐标系，确保输出尺寸严格为 (size, size)
        # 之前用对称范围 [-size//2, size//2] 会在偶数尺寸时得到 (size+1) 的边长
        coords = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
        x, y = np.meshgrid(coords, coords)
        
        # 旋转坐标
        x_rot = x * np.cos(orientation) + y * np.sin(orientation)
        y_rot = -x * np.sin(orientation) + y * np.cos(orientation)
        
        # Gabor函数（按像素波长参数化）
        # 略收紧高斯包络，提升局部对比
        sigma_x, sigma_y = size/4, size/4
        gaussian = np.exp(-(x_rot**2/(2*sigma_x**2) + y_rot**2/(2*sigma_y**2)))
        # 将 frequency 解释为像素波长 lambda_pix，避免与补丁尺寸绑定
        lambda_pix = max(1e-6, float(frequency))
        sinusoid = np.cos(2 * np.pi / lambda_pix * x_rot + phase)
        
        gabor = gaussian * sinusoid
        return gabor

class MultiScaleSparseEnhancer:
    """多尺度稀疏表示增强器"""
    
    def __init__(self, initial_patch_size: int = 24, max_patch_size: int = 40, 
                 sparsity_lambda: float = 0.5, max_iterations: int = 20,
                 gamma_large: float = 0.8, quality_threshold: float = 0.3,
                 tv_lambda: float = 0.1,
                 texture_mix: float = 0.5,
                 solver: str = 'omp',
                 debug: bool = False):
        self.initial_patch_size = initial_patch_size
        self.max_patch_size = max_patch_size
        self.sparsity_lambda = sparsity_lambda
        self.max_iterations = max_iterations
        self.gamma_large = gamma_large
        self.quality_threshold = quality_threshold
        self.texture_mix = texture_mix
        self.solver = solver  # 'omp' or 'lars'
        self.debug_enabled = debug
        self.debug = {}
        
        self.tv_decomposer = TVDecomposition(lambda_tv=tv_lambda)
        self.dictionaries = {}
        self.dict_orientations = {}
        self.dict_wavelengths = {}
    
    def enhance(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        多尺度稀疏表示增强
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: TV分解
        texture, cartoon = self.tv_decomposer.decompose(image)
        # 将原始图归一化到[0,1]用于混合，避免TV过强导致纹理过弱
        img_norm = image.astype(np.float32)
        if img_norm.max() > 1.5:  # 假定输入为0-255
            img_norm = img_norm / 255.0
        mix = np.clip(self.texture_mix, 0.0, 1.0)
        texture = mix * texture + (1.0 - mix) * img_norm
        
        # Step 2: 初始稀疏重建
        enhanced = self._sparse_reconstruction(
            texture, self.initial_patch_size, roi_mask
        )
        if self.debug_enabled:
            self.debug['initial'] = enhanced.copy()
        
        # Step 3: 多尺度迭代增强
        current_patch_size = self.initial_patch_size
        
        for iteration in range(self.max_iterations):
            # 计算图像质量
            quality_map = self._compute_quality(enhanced)
            if self.debug_enabled and iteration == 0:
                self.debug['quality'] = quality_map.copy()
            
            # 找到低质量区域
            low_quality_mask = quality_map < self.quality_threshold
            
            if np.sum(low_quality_mask) == 0:
                break
            
            # 增加patch大小
            current_patch_size = min(current_patch_size + 4, self.max_patch_size)
            
            # 对低质量区域进行增强
            enhanced = self._multiscale_reconstruction(
                enhanced, texture, current_patch_size, 
                low_quality_mask, roi_mask
            )
        
        return enhanced
    
    def _sparse_reconstruction(self, texture: np.ndarray, patch_size: int, 
                             roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """稀疏重建"""
        h, w = texture.shape
        enhanced = np.zeros_like(texture)
        weight = np.zeros_like(texture)
        coeff_l1_map = np.zeros_like(texture)
        coeff_w = np.zeros_like(texture)
        ori_resp_map = np.zeros_like(texture)
        
        # 获取或构建字典
        if patch_size not in self.dictionaries:
            gabor_dict = GaborDictionary(patch_size)
            D = gabor_dict.build_dictionary()
            self.dictionaries[patch_size] = D
            self.dict_orientations[patch_size] = gabor_dict.atom_orientations
            self.dict_wavelengths[patch_size] = gabor_dict.atom_wavelengths
        
        dictionary = self.dictionaries[patch_size]
        atom_oris = self.dict_orientations.get(patch_size, None)
        atom_lams = self.dict_wavelengths.get(patch_size, None)
        
        # 滑动窗口处理（使用固定步长8，覆盖更密集，提升可见细节）
        stride = 6
        # 忽略由OMP引发的线性相关 RuntimeWarning
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                # 提取patch
                patch = texture[i:i+patch_size, j:j+patch_size]
                patch_mean = float(np.mean(patch))
                
                # 检查ROI
                if roi_mask is not None:
                    roi_patch = roi_mask[i:i+patch_size, j:j+patch_size]
                    if np.mean(roi_patch) < 0.5:
                        continue
                
                # 标准化patch
                patch_flat = patch.flatten()
                patch_flat = patch_flat - patch_mean
                patch_norm = np.linalg.norm(patch_flat)

                # 跳过过于平坦的patch
                if patch_norm > 1e-4:
                    patch_flat = patch_flat / patch_norm
                    # 稀疏编码：优先使用 L1（LassoLars），失败时回退到 OMP
                    # 将 sparsity_lambda 映射为一个较小的 alpha，避免过强正则
                    # 更小的正则，避免系数过于稀疏导致重建为近常数
                    alpha = max(5e-5, min(2e-3, self.sparsity_lambda * 5e-4))
                    coeffs = None

                    # 方向+频率 引导的子字典选择
                    D_use = dictionary
                    if atom_oris is not None and atom_oris.size == dictionary.shape[1]:
                        # 估计补丁主方向（结构张量）
                        # 修复OpenCV 4.11.0兼容性问题：确保patch是连续的float32数组
                        patch_for_sobel = np.ascontiguousarray(patch, dtype=np.float32)
                        gx = cv2.Sobel(patch_for_sobel, cv2.CV_32F, 1, 0, ksize=3)
                        gy = cv2.Sobel(patch_for_sobel, cv2.CV_32F, 0, 1, ksize=3)
                        gxx = np.mean(gx*gx)
                        gyy = np.mean(gy*gy)
                        gxy = np.mean(gx*gy)
                        theta = 0.5 * np.arctan2(2*gxy, gxx - gyy)
                        # 方向差异归一到 [-pi/2, pi/2]
                        dtheta = np.abs(((atom_oris - theta + np.pi/2) % np.pi) - np.pi/2)
                        band_ori = dtheta <= (15.0/180.0*np.pi)
                        # 简易频率估计：沿垂直于主方向的法向做投影并FFT求主峰
                        lam_est = None
                        try:
                            # 旋转到局部坐标
                            c, s = np.cos(theta), np.sin(theta)
                            # 法向方向单位向量 (cos(theta+pi/2), sin(...)) = (-s, c)
                            nx, ny = -s, c
                            # 在补丁中心采样一条长度为 patch_size 的剖面
                            xs = np.linspace(-(patch_size-1)/2, (patch_size-1)/2, patch_size)
                            ys = np.zeros_like(xs)
                            # 坐标映射到图像内索引
                            cx = (patch_size-1)/2; cy = (patch_size-1)/2
                            xi = np.clip((cx + nx*xs).astype(np.float32), 0, patch_size-1)
                            yi = np.clip((cy + ny*xs).astype(np.float32), 0, patch_size-1)
                            profile = cv2.remap(patch.astype(np.float32), xi.reshape(1,-1), yi.reshape(1,-1), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).flatten()
                            profile = profile - np.mean(profile)
                            if np.linalg.norm(profile) > 1e-6:
                                # 频谱
                                spec = np.abs(np.fft.rfft(profile))
                                freqs = np.fft.rfftfreq(profile.size, d=1.0)
                                # 排除直流
                                if spec.size > 1:
                                    spec[0] = 0
                                idx = int(np.argmax(spec))
                                f = freqs[idx]
                                if f > 1e-6:
                                    lam_est = float(1.0 / f)
                        except Exception:
                            lam_est = None

                        if lam_est is not None and atom_lams is not None and atom_lams.size == dictionary.shape[1]:
                            band_lam = np.abs(atom_lams - lam_est) <= 2.0
                            band = band_ori & band_lam
                            if not np.any(band):
                                # 放宽
                                band_lam = np.abs(atom_lams - lam_est) <= 4.0
                                band = band_ori & band_lam
                        else:
                            band = band_ori

                        if np.any(band):
                            D_use = dictionary[:, band]
                        else:
                            # 如果没有命中，退化到仅方向±30°
                            band = dtheta <= (30.0/180.0*np.pi)
                            if np.any(band):
                                D_use = dictionary[:, band]

                    if self.solver.lower() == 'omp':
                        # 固定稀疏度，强调结构重构
                        k = max(16, min(36, D_use.shape[1] // 20 if D_use.shape[1] > 0 else 16))
                        try:
                            omp = OrthogonalMatchingPursuit(
                                n_nonzero_coefs=k, fit_intercept=False, precompute=False
                            )
                            omp.fit(D_use, patch_flat)
                            coeffs = omp.coef_
                        except Exception:
                            pass
                    else:
                        try:
                            lars = LassoLars(alpha=alpha, fit_intercept=False)
                            lars.fit(D_use, patch_flat)
                            coeffs = lars.coef_
                            if np.sum(np.abs(coeffs)) < 2e-3:
                                coeffs = None
                        except Exception:
                            coeffs = None
                    if coeffs is None:
                        continue
                    
                    # 重建
                    # 若使用了子字典，需要映射回全量系数以便一致的重建；这里直接用子字典重建即可
                    reconstructed = D_use @ coeffs
                    reconstructed = reconstructed.reshape(patch_size, patch_size)
                    
                    # 叠加写入并计权，最后统一平均，提升细节可见度
                    enhanced[i:i+patch_size, j:j+patch_size] += reconstructed * patch_norm + patch_mean
                    weight[i:i+patch_size, j:j+patch_size] += 1.0

                    # 调试：系数L1与方向响应
                    if self.debug_enabled:
                        coeff_l1 = float(np.sum(np.abs(coeffs)))
                        coeff_l1_map[i:i+patch_size, j:j+patch_size] += coeff_l1
                        coeff_w[i:i+patch_size, j:j+patch_size] += 1.0
                        # 方向响应（若可用方向标签，则取各方向abs系数和的最大值；否则用L1替代）
                        try:
                            # 方向个数近似 self.dictionaries 的 orientations
                            # 由于无法可靠反推每列方向，这里简化为用L1代替
                            ori_resp_map[i:i+patch_size, j:j+patch_size] += coeff_l1
                        except Exception:
                            ori_resp_map[i:i+patch_size, j:j+patch_size] += coeff_l1

        # 防止除零
        mask = weight > 0
        out = np.copy(texture)
        out[mask] = enhanced[mask] / (weight[mask] + 1e-8)
        if self.debug_enabled:
            m2 = coeff_w > 0
            coeff_l1_out = np.zeros_like(texture)
            coeff_l1_out[m2] = coeff_l1_map[m2] / (coeff_w[m2] + 1e-8)
            ori_resp_out = np.zeros_like(texture)
            ori_resp_out[m2] = ori_resp_map[m2] / (coeff_w[m2] + 1e-8)
            self.debug['coeff_l1'] = coeff_l1_out
            self.debug['ori_resp'] = ori_resp_out
        return out
    
    def _multiscale_reconstruction(self, current_enhanced: np.ndarray, 
                                 texture: np.ndarray, large_patch_size: int,
                                 low_quality_mask: np.ndarray, 
                                 roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """多尺度重建"""
        h, w = texture.shape
        enhanced = np.zeros_like(current_enhanced)
        weight = np.zeros_like(current_enhanced)
        
        # 获取大尺度字典
        if large_patch_size not in self.dictionaries:
            gabor_dict = GaborDictionary(large_patch_size)
            self.dictionaries[large_patch_size] = gabor_dict.build_dictionary()
        
        large_dict = self.dictionaries[large_patch_size]
        small_dict = self.dictionaries[self.initial_patch_size]
        small_oris = self.dict_orientations.get(self.initial_patch_size, None)
        small_lams = self.dict_wavelengths.get(self.initial_patch_size, None)
        
        # 只处理低质量区域
        stride = max(8, large_patch_size // 2)
        for i in range(0, h - large_patch_size + 1, stride):
            for j in range(0, w - large_patch_size + 1, stride):
                # 检查是否为低质量区域
                quality_patch = low_quality_mask[i:i+large_patch_size, j:j+large_patch_size]
                if np.mean(quality_patch) < 0.5:
                    continue
                
                # 检查ROI
                if roi_mask is not None:
                    roi_patch = roi_mask[i:i+large_patch_size, j:j+large_patch_size]
                    if np.mean(roi_patch) < 0.5:
                        continue
                
                # 提取小patch和大patch
                small_patch = current_enhanced[
                    i:i+self.initial_patch_size, 
                    j:j+self.initial_patch_size
                ]
                large_patch = texture[i:i+large_patch_size, j:j+large_patch_size]
                
                # 多尺度稀疏编码
                reconstructed = self._multiscale_sparse_coding(
                    small_patch, large_patch, small_dict, large_dict, self.gamma_large,
                    small_oris, small_lams
                )
                
                # 叠加写入并计权
                ii = i
                jj = j
                enhanced[ii:ii+self.initial_patch_size, jj:jj+self.initial_patch_size] += reconstructed
                weight[ii:ii+self.initial_patch_size, jj:jj+self.initial_patch_size] += 1.0
        
        return enhanced
    
    def _multiscale_sparse_coding(self, small_patch: np.ndarray, large_patch: np.ndarray,
                                small_dict: np.ndarray, large_dict: np.ndarray,
                                gamma: float, small_oris: Optional[np.ndarray], small_lams: Optional[np.ndarray]) -> np.ndarray:
        """多尺度稀疏编码"""
        # 标准化patches
        small_flat = small_patch.flatten()
        large_flat = large_patch.flatten()
        
        small_flat = small_flat - np.mean(small_flat)
        large_flat = large_flat - np.mean(large_flat)
        
        small_norm = np.linalg.norm(small_flat)
        large_norm = np.linalg.norm(large_flat)
        
        if small_norm > 1e-6:
            small_flat = small_flat / small_norm
        if large_norm > 1e-6:
            large_flat = large_flat / large_norm
        
        # 构建联合优化问题（式8/9）：堆叠小块与大块，列一致为同一组原子
        # 方向引导：为小尺度选择子字典
        D_small = small_dict
        if small_oris is not None and small_oris.size == small_dict.shape[1]:
            # 修复OpenCV 4.11.0兼容性问题
            small_patch_for_sobel = np.ascontiguousarray(small_patch, dtype=np.float32)
            gx = cv2.Sobel(small_patch_for_sobel, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(small_patch_for_sobel, cv2.CV_32F, 0, 1, ksize=3)
            gxx = np.mean(gx*gx); gyy = np.mean(gy*gy); gxy = np.mean(gx*gy)
            theta = 0.5 * np.arctan2(2*gxy, gxx - gyy)
            dtheta = np.abs(((small_oris - theta + np.pi/2) % np.pi) - np.pi/2)
            band_ori = dtheta <= (15.0/180.0*np.pi)
            # 频率估计（同上）
            lam_est = None
            try:
                c, s = np.cos(theta), np.sin(theta)
                nx, ny = -s, c
                xs = np.linspace(-(small_patch.shape[0]-1)/2, (small_patch.shape[0]-1)/2, small_patch.shape[0])
                cx = (small_patch.shape[0]-1)/2; cy = (small_patch.shape[1]-1)/2
                xi = np.clip((cx + nx*xs).astype(np.float32), 0, small_patch.shape[0]-1)
                yi = np.clip((cy + ny*xs).astype(np.float32), 0, small_patch.shape[1]-1)
                profile = cv2.remap(small_patch.astype(np.float32), xi.reshape(1,-1), yi.reshape(1,-1), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).flatten()
                profile = profile - np.mean(profile)
                if np.linalg.norm(profile) > 1e-6:
                    spec = np.abs(np.fft.rfft(profile))
                    freqs = np.fft.rfftfreq(profile.size, d=1.0)
                    if spec.size > 1:
                        spec[0] = 0
                    idx = int(np.argmax(spec))
                    f = freqs[idx]
                    if f > 1e-6:
                        lam_est = float(1.0 / f)
            except Exception:
                lam_est = None

            if lam_est is not None and small_lams is not None and small_lams.size == small_dict.shape[1]:
                band_lam = np.abs(small_lams - lam_est) <= 2.0
                band = band_ori & band_lam
                if not np.any(band):
                    band_lam = np.abs(small_lams - lam_est) <= 4.0
                    band = band_ori & band_lam
            else:
                band = band_ori

            if np.any(band):
                D_small = small_dict[:, band]
            else:
                band = dtheta <= (30.0/180.0*np.pi)
                if np.any(band):
                    D_small = small_dict[:, band]

        if self.solver.lower() == 'omp':
            try:
                k = max(16, min(36, D_small.shape[1] // 20 if D_small.shape[1] > 0 else 16))
                omp_small = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
                omp_small.fit(D_small, small_flat)
                coeffs_small = omp_small.coef_
                reconstructed = D_small @ coeffs_small
                reconstructed = reconstructed.reshape(small_patch.shape)
                reconstructed = reconstructed * small_norm
            except Exception:
                reconstructed = small_patch
        else:
            try:
                alpha = max(5e-5, min(2e-3, self.sparsity_lambda * 5e-4))
                A = np.vstack([D_small, np.sqrt(max(gamma, 0.0)) * large_dict])
                b = np.concatenate([small_flat, np.sqrt(max(gamma, 0.0)) * large_flat])
                lars = LassoLars(alpha=alpha, fit_intercept=False)
                lars.fit(A, b)
                coeffs = lars.coef_
                if np.sum(np.abs(coeffs)) < 1e-2:
                    raise ValueError('coeff too small')
                reconstructed = D_small @ coeffs
                reconstructed = reconstructed.reshape(small_patch.shape)
                reconstructed = reconstructed * small_norm
            except Exception:
                reconstructed = small_patch
        
        return reconstructed
    
    def _compute_quality(self, image: np.ndarray, window_size: int = 16) -> np.ndarray:
        """计算图像质量（方向一致性）"""
        # 计算梯度
        # 修复OpenCV 4.11.0兼容性问题
        image_for_sobel = np.ascontiguousarray(image, dtype=np.float32)
        grad_x = cv2.Sobel(image_for_sobel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_for_sobel, cv2.CV_32F, 0, 1, ksize=3)
        
        # 计算结构张量
        Gxx = cv2.boxFilter(grad_x * grad_x, cv2.CV_32F, (window_size, window_size))
        Gyy = cv2.boxFilter(grad_y * grad_y, cv2.CV_32F, (window_size, window_size))
        Gxy = cv2.boxFilter(grad_x * grad_y, cv2.CV_32F, (window_size, window_size))
        
        # 计算一致性
        coherence = np.sqrt((Gxx - Gyy)**2 + 4*Gxy**2) / (Gxx + Gyy + 1e-8)
        
        return coherence
    
    def segment_fingerprint(self, image: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
        """自动分割指纹前景"""
        # 基于稀疏系数的分割
        h, w = image.shape
        segmentation = np.zeros((h, w), dtype=np.uint8)
        
        if self.initial_patch_size not in self.dictionaries:
            return segmentation
        
        dictionary = self.dictionaries[self.initial_patch_size]
        patch_size = self.initial_patch_size
        
        # 滑动窗口计算稀疏系数
        stride = 8
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = enhanced[i:i+patch_size, j:j+patch_size]
                patch_flat = patch.flatten()
                patch_flat = patch_flat - np.mean(patch_flat)
                patch_norm = np.linalg.norm(patch_flat)
                
                if patch_norm > 1e-6:
                    patch_flat = patch_flat / patch_norm
                    
                    # 稀疏编码
                    try:
                        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10)
                        omp.fit(dictionary, patch_flat)
                        coeffs = omp.coef_
                        
                        # 如果稀疏系数和足够大，认为是前景
                        if np.sum(np.abs(coeffs)) > 0.001:
                            segmentation[i:i+patch_size, j:j+patch_size] = 255
                    except Exception:
                        continue
        
        # 形态学后处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        segmentation = cv2.morphologyEx(segmentation, cv2.MORPH_CLOSE, kernel)
        segmentation = cv2.morphologyEx(segmentation, cv2.MORPH_OPEN, kernel)
        
        return segmentation
