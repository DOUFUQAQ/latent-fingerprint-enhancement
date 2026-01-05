import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class FingerprintMatcher:
    def __init__(self, distance_threshold=20, angle_threshold=np.pi/6):
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
    
    def match(self, minutiae1, minutiae2):
        if len(minutiae1) == 0 or len(minutiae2) == 0:
            return 0.0, 0
        
        mnt1 = np.array(minutiae1)
        mnt2 = np.array(minutiae2)
        
        best_score = 0
        best_matched = 0
        
        for i in range(min(5, len(mnt1))):
            for j in range(min(5, len(mnt2))):
                tx = mnt2[j, 0] - mnt1[i, 0]
                ty = mnt2[j, 1] - mnt1[i, 1]
                
                mnt1_aligned = mnt1.copy()
                mnt1_aligned[:, 0] += tx
                mnt1_aligned[:, 1] += ty
                
                matched = self._count_matches(mnt1_aligned, mnt2)
                score = self._compute_score(matched, len(mnt1), len(mnt2))
                
                if score > best_score:
                    best_score = score
                    best_matched = matched
        
        return best_score, best_matched
    
    def _count_matches(self, mnt1, mnt2):
        if len(mnt1) == 0 or len(mnt2) == 0:
            return 0
        
        pos1 = mnt1[:, :2]
        pos2 = mnt2[:, :2]
        dist_matrix = cdist(pos1, pos2, metric='euclidean')
        
        angle1 = mnt1[:, 2]
        angle2 = mnt2[:, 2]
        angle_diff = np.abs(angle1[:, np.newaxis] - angle2[np.newaxis, :])
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        
        type1 = mnt1[:, 3].astype(int)
        type2 = mnt2[:, 3].astype(int)
        type_match = (type1[:, np.newaxis] == type2[np.newaxis, :]).astype(float)
        
        valid_matches = (
            (dist_matrix < self.distance_threshold) &
            (angle_diff < self.angle_threshold) &
            (type_match > 0)
        )
        
        cost_matrix = dist_matrix.copy()
        cost_matrix[~valid_matches] = 1e6
        
        if cost_matrix.min() < 1e6:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_count = np.sum(cost_matrix[row_ind, col_ind] < 1e6)
        else:
            matched_count = 0
        
        return matched_count
    
    def _compute_score(self, matched, count1, count2):
        if count1 == 0 or count2 == 0:
            return 0.0
        
        avg_count = (count1 + count2) / 2.0
        similarity = matched / avg_count
        score = min(100.0, similarity * 100.0)
        
        return score
    
    def batch_match(self, query_minutiae, gallery_minutiae_list):
        scores = []
        matched_counts = []
        
        for gallery_mnt in gallery_minutiae_list:
            score, matched = self.match(query_minutiae, gallery_mnt)
            scores.append(score)
            matched_counts.append(matched)
        
        return np.array(scores), np.array(matched_counts)
    
    def get_rank(self, scores, true_index):
        sorted_indices = np.argsort(scores)[::-1]
        rank = np.where(sorted_indices == true_index)[0][0] + 1
        return rank


class MatcherWithTransform(FingerprintMatcher):
    def match(self, minutiae1, minutiae2):
        if len(minutiae1) == 0 or len(minutiae2) == 0:
            return 0.0, 0
        
        mnt1 = np.array(minutiae1)
        mnt2 = np.array(minutiae2)
        
        best_score = 0
        best_matched = 0
        
        rotation_angles = np.linspace(0, 2*np.pi, 12)
        
        for angle in rotation_angles:
            mnt1_rotated = self._rotate_minutiae(mnt1, angle)
            
            for i in range(min(3, len(mnt1_rotated))):
                for j in range(min(3, len(mnt2))):
                    tx = mnt2[j, 0] - mnt1_rotated[i, 0]
                    ty = mnt2[j, 1] - mnt1_rotated[i, 1]
                    
                    mnt1_aligned = mnt1_rotated.copy()
                    mnt1_aligned[:, 0] += tx
                    mnt1_aligned[:, 1] += ty
                    
                    matched = self._count_matches(mnt1_aligned, mnt2)
                    score = self._compute_score(matched, len(mnt1), len(mnt2))
                    
                    if score > best_score:
                        best_score = score
                        best_matched = matched
        
        return best_score, best_matched
    
    def _rotate_minutiae(self, minutiae, angle):
        rotated = minutiae.copy()
        
        center_x = np.mean(minutiae[:, 0])
        center_y = np.mean(minutiae[:, 1])
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        x = minutiae[:, 0] - center_x
        y = minutiae[:, 1] - center_y
        rotated[:, 0] = x * cos_a - y * sin_a + center_x
        rotated[:, 1] = x * sin_a + y * cos_a + center_y
        
        rotated[:, 2] = (minutiae[:, 2] + angle) % (2 * np.pi)
        
        return rotated
