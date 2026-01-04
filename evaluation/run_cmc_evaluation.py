#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import json
from evaluation.minutiae_extractor import MinutiaeExtractor
from evaluation.fingerprint_matcher import FingerprintMatcher, MatcherWithTransform
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class CMCEvaluator:
    
    def __init__(self, dataset_dir, methods_config, cache_dir='cmc_cache'):
        self.dataset_dir = dataset_dir
        self.methods_config = methods_config
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.extractor = MinutiaeExtractor()
        self.matcher = MatcherWithTransform(distance_threshold=25, angle_threshold=np.pi/4)
        
        self.load_dataset_info()
    
    def load_dataset_info(self):
        print("Loading dataset info...")
        
        original_dir = os.path.join(self.dataset_dir, 'original')
        all_files = os.listdir(original_dir)
        
        self.latent_files = sorted([f for f in all_files if f.startswith('f') and not 'manual' in f])
        self.rolled_files = sorted([f for f in all_files if f.startswith('s') and not 'manual' in f])
        
        self.latent_to_rolled = {}
        for latent in self.latent_files:
            fid = latent.split('_')[0][1:]
            rolled_match = f"s{fid}_"
            
            for rolled in self.rolled_files:
                if rolled.startswith(rolled_match):
                    self.latent_to_rolled[latent] = rolled
                    break
        
        print(f"Dataset loaded:")
        print(f"  Latent prints: {len(self.latent_files)}")
        print(f"  Rolled prints: {len(self.rolled_files)}")
        print(f"  Pairs: {len(self.latent_to_rolled)}")
    
    def extract_all_minutiae(self, method_name, image_dir, force=False):
        cache_file = os.path.join(self.cache_dir, f'{method_name}_minutiae.pkl')
        
        if os.path.exists(cache_file) and not force:
            print(f"Loading cached minutiae for {method_name}...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"Extracting minutiae for {method_name}...")
        minutiae_dict = {}
        
        if os.path.exists(image_dir):
            image_files = [f for f in os.listdir(image_dir) 
                          if f.endswith(('.png', '.jpg', '.bmp'))]
        else:
            print(f"Warning: directory not found {image_dir}")
            return minutiae_dict
        
        for img_file in tqdm(image_files, desc=f"Extract {method_name}"):
            img_path = os.path.join(image_dir, img_file)
            try:
                minutiae = self.extractor.extract(img_path)
                minutiae_dict[img_file] = minutiae
            except Exception as e:
                print(f"Failed to extract {img_file}: {e}")
                minutiae_dict[img_file] = []
        
        with open(cache_file, 'wb') as f:
            pickle.dump(minutiae_dict, f)
        
        return minutiae_dict
    
    def compute_cmc(self, query_minutiae_dict, gallery_minutiae_dict, max_rank=20):
        print("Computing CMC curve...")
        
        ranks = []
        total_queries = 0
        
        for latent_file in tqdm(self.latent_files, desc="Matching"):
            if latent_file not in self.latent_to_rolled:
                continue
            
            true_rolled = self.latent_to_rolled[latent_file]
            
            query_mnt = query_minutiae_dict.get(latent_file, [])
            if len(query_mnt) == 0:
                continue
            
            gallery_files = []
            gallery_mnts = []
            true_index = -1
            
            for i, rolled_file in enumerate(self.rolled_files):
                gallery_mnt = gallery_minutiae_dict.get(rolled_file, [])
                if len(gallery_mnt) > 0:
                    gallery_files.append(rolled_file)
                    gallery_mnts.append(gallery_mnt)
                    if rolled_file == true_rolled:
                        true_index = len(gallery_mnts) - 1
            
            if true_index == -1:
                continue
            
            scores, _ = self.matcher.batch_match(query_mnt, gallery_mnts)
            
            rank = self.matcher.get_rank(scores, true_index)
            ranks.append(rank)
            total_queries += 1
        
        cmc = np.zeros(max_rank)
        for rank in ranks:
            for k in range(rank, max_rank+1):
                cmc[k-1] += 1
        
        if total_queries > 0:
            cmc = cmc / total_queries * 100
        
        print(f"Done! Valid queries: {total_queries}")
        print(f"Rank-1 accuracy: {cmc[0]:.2f}%")
        
        return cmc, ranks
    
    def evaluate_all_methods(self, max_rank=20, force_extract=False):
        results = {}
        
        print("\n=== Step 1: Extract gallery minutiae ===")
        original_dir = os.path.join(self.dataset_dir, 'original')
        gallery_minutiae = self.extract_all_minutiae('gallery', original_dir, force_extract)
        
        for method_name, image_dir in self.methods_config.items():
            print(f"\n=== Step 2: Evaluate method [{method_name}] ===")
            
            query_minutiae = self.extract_all_minutiae(method_name, image_dir, force_extract)
            
            cmc, ranks = self.compute_cmc(query_minutiae, gallery_minutiae, max_rank)
            
            results[method_name] = {
                'cmc': cmc,
                'ranks': ranks,
                'rank1': cmc[0] if len(cmc) > 0 else 0.0
            }
        
        return results
    
    def save_results(self, results, output_file='cmc_results.json'):
        results_serializable = {}
        for method, data in results.items():
            results_serializable[method] = {
                'cmc': data['cmc'].tolist(),
                'ranks': data['ranks'],
                'rank1': float(data['rank1'])
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
    
    def plot_cmc_curves(self, results, output_file='cmc_curves.png'):
        plt.figure(figsize=(10, 7))
        
        colors = ['r', 'b', 'g', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'v', 'D', 'p']
        
        max_rank = 0
        for i, (method, data) in enumerate(results.items()):
            cmc = data['cmc']
            max_rank = max(max_rank, len(cmc))
            ranks = np.arange(1, len(cmc)+1)
            
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            plt.plot(ranks, cmc, 
                    label=f"{method} (Rank-1: {data['rank1']:.1f}%)",
                    color=color, marker=marker, markevery=2,
                    linewidth=2, markersize=6)
        
        plt.xlabel('Rank', fontsize=12, fontweight='bold')
        plt.ylabel('Identification Rate (%)', fontsize=12, fontweight='bold')
        plt.title('Cumulative Match Characteristic (CMC) Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([1, max_rank])
        plt.ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"CMC curve saved to: {output_file}")
        plt.close()
    
    def print_summary(self, results):
        print("\n" + "="*60)
        print("CMC Evaluation Results Summary")
        print("="*60)
        
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['rank1'], reverse=True)
        
        print(f"\n{'Method':<20} {'Rank-1':<10} {'Rank-5':<10} {'Rank-10':<10}")
        print("-"*60)
        
        for method, data in sorted_methods:
            cmc = data['cmc']
            rank1 = cmc[0] if len(cmc) > 0 else 0
            rank5 = cmc[4] if len(cmc) > 4 else 0
            rank10 = cmc[9] if len(cmc) > 9 else 0
            
            print(f"{method:<20} {rank1:>6.2f}%   {rank5:>6.2f}%   {rank10:>6.2f}%")
        
        print("="*60)


def main():
    print("="*60)
    print("Fingerprint Enhancement CMC Evaluation System")
    print("="*60)
    
    dataset_dir = os.environ.get(
        "LATENTFP_DATASET_DIR", os.path.join("datasets", "processed_sd04_dataset"))
    
    methods_config = {
        'Original': os.path.join(dataset_dir, 'original'),
        'Proposed': os.environ.get(
            "LATENTFP_PROPOSED_DIR", os.path.join("datasets", "proposed")),
        'Gabor': os.environ.get(
            "LATENTFP_GABOR_DIR", os.path.join("datasets", "gabor")),
        'STFT': os.environ.get(
            "LATENTFP_STFT_DIR", os.path.join("datasets", "stft")),
    }
    
    evaluator = CMCEvaluator(dataset_dir, methods_config)
    
    results = evaluator.evaluate_all_methods(max_rank=20, force_extract=False)
    
    evaluator.save_results(results, 'cmc_results.json')
    
    evaluator.plot_cmc_curves(results, 'cmc_curves.png')
    
    evaluator.print_summary(results)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
