import cv2
import numpy as np
from skimage.morphology import thin
from scipy import ndimage
import os
import matplotlib.pyplot as plt


class MinutiaeExtractor:
    def __init__(self):
        self.min_ridge_length = 10
    
    def extract(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []
        
        binary = self._binarize(img)
        skeleton = self._get_skeleton(binary)
        minutiae = self._detect_minutiae(skeleton)
        minutiae = self._filter_border_minutiae(minutiae, img.shape)
        
        return minutiae
    
    def _binarize(self, img):
        binary = cv2.adaptiveThreshold(
            img, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary
    
    def _get_skeleton(self, binary):
        skeleton = thin(binary // 255).astype(np.uint8) * 255
        return skeleton
    
    def _detect_minutiae(self, skeleton):
        minutiae = []
        h, w = skeleton.shape
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 0:
                    continue
                
                neighbors = self._get_neighbors(skeleton, x, y)
                cn = self._crossing_number(neighbors)
                
                if cn == 1:
                    angle = self._compute_angle(skeleton, x, y, neighbors)
                    minutiae.append((x, y, angle, 0))
                elif cn == 3:
                    angle = self._compute_angle(skeleton, x, y, neighbors)
                    minutiae.append((x, y, angle, 1))
        
        return minutiae
    
    def _get_neighbors(self, skeleton, x, y):
        neighbors = [
            skeleton[y-1, x-1], skeleton[y-1, x], skeleton[y-1, x+1],
            skeleton[y, x+1], skeleton[y+1, x+1], skeleton[y+1, x],
            skeleton[y+1, x-1], skeleton[y, x-1]
        ]
        return [1 if p > 0 else 0 for p in neighbors]
    
    def _crossing_number(self, neighbors):
        cn = 0
        for i in range(8):
            cn += abs(neighbors[i] - neighbors[(i+1) % 8])
        return cn // 2
    
    def _compute_angle(self, skeleton, x, y, neighbors):
        angles = []
        positions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1), (1, 0),
            (1, -1), (0, -1)
        ]
        
        for i, (dx, dy) in enumerate(positions):
            if neighbors[i] == 1:
                angle = np.arctan2(dy, dx)
                angles.append(angle)
        
        if len(angles) > 0:
            return np.mean(angles)
        return 0.0
    
    def _filter_border_minutiae(self, minutiae, shape, border=20):
        h, w = shape
        filtered = []
        for x, y, angle, mtype in minutiae:
            if border < x < w-border and border < y < h-border:
                filtered.append((x, y, angle, mtype))
        return filtered
    
    def save_minutiae(self, minutiae, output_path):
        with open(output_path, 'w') as f:
            f.write(f"# Minutiae count: {len(minutiae)}\n")
            f.write("# x y angle type\n")
            for x, y, angle, mtype in minutiae:
                f.write(f"{x} {y} {angle:.4f} {mtype}\n")
    
    def load_minutiae(self, input_path):
        minutiae = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 4:
                    x, y = int(parts[0]), int(parts[1])
                    angle = float(parts[2])
                    mtype = int(parts[3])
                    minutiae.append((x, y, angle, mtype))
        return minutiae


def visualize_minutiae(image_path, minutiae, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    
    for x, y, angle, mtype in minutiae:
        color = 'r' if mtype == 0 else 'b'
        plt.plot(x, y, color+'o', markersize=8)
        dx = 15 * np.cos(angle)
        dy = 15 * np.sin(angle)
        plt.arrow(x, y, dx, dy, color=color, width=1, head_width=3)
    
    plt.title(f'Minutiae Detection (Total: {len(minutiae)})')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
