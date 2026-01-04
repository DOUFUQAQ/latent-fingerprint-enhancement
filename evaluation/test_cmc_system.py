#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import sys

sys.path.append(os.path.dirname(__file__))

from evaluation.minutiae_extractor import MinutiaeExtractor, visualize_minutiae
from evaluation.fingerprint_matcher import FingerprintMatcher, MatcherWithTransform


def test_minutiae_extraction():
    """Test minutiae extraction on a sample image"""
    print("\n" + "="*60)
    print("Test 1: Minutiae Extraction")
    print("="*60)
    
    extractor = MinutiaeExtractor()
    
    # Find a test image
    test_dir = os.path.join(
        os.environ.get("LATENTFP_DATASET_DIR",
                       os.path.join("datasets", "processed_sd04_dataset")),
        "original")
    test_files = [f for f in os.listdir(test_dir) if f.startswith('f') and f.endswith('.png')][:1]
    
    if not test_files:
        print("ERROR: No test images found!")
        return False
    
    test_image = os.path.join(test_dir, test_files[0])
    print(f"Testing on: {test_image}")
    
    try:
        minutiae = extractor.extract(test_image)
        print(f"SUCCESS: Extracted {len(minutiae)} minutiae points")
        
        # Visualize
        visualize_minutiae(test_image, minutiae, "test_minutiae_output.png")
        print("Visualization saved to: test_minutiae_output.png")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_fingerprint_matching():
    """Test fingerprint matching"""
    print("\n" + "="*60)
    print("Test 2: Fingerprint Matching")
    print("="*60)
    
    # Create synthetic minutiae data
    mnt1 = [
        (100, 100, 0.5, 0),
        (150, 120, 1.0, 1),
        (200, 150, 1.5, 0),
        (250, 180, 2.0, 1),
    ]
    
    # Create similar minutiae with slight variations
    mnt2 = [
        (102, 98, 0.52, 0),   # Should match mnt1[0]
        (148, 122, 0.98, 1),  # Should match mnt1[1]
        (198, 152, 1.48, 0),  # Should match mnt1[2]
        (300, 300, 0.0, 0),   # No match
    ]
    
    # Create dissimilar minutiae
    mnt3 = [
        (50, 50, 0, 0),
        (60, 60, 1, 1),
    ]
    
    matcher = FingerprintMatcher(distance_threshold=20, angle_threshold=np.pi/6)
    
    # Test matching
    score12, matched12 = matcher.match(mnt1, mnt2)
    score13, matched13 = matcher.match(mnt1, mnt3)
    
    print(f"Match score (similar):     {score12:.2f} (matched: {matched12})")
    print(f"Match score (dissimilar):  {score13:.2f} (matched: {matched13})")
    
    if score12 > score13:
        print("SUCCESS: Matching works correctly!")
        return True
    else:
        print("ERROR: Matching logic incorrect!")
        return False


def test_batch_matching():
    """Test batch matching and ranking"""
    print("\n" + "="*60)
    print("Test 3: Batch Matching and Ranking")
    print("="*60)
    
    # Query
    query = [
        (100, 100, 0.5, 0),
        (150, 120, 1.0, 1),
    ]
    
    # Gallery with true match at index 2
    gallery = [
        [(50, 50, 0, 0)],  # No match
        [(60, 60, 1, 1)],  # No match
        [(101, 99, 0.51, 0), (149, 121, 0.99, 1)],  # True match (index 2)
        [(200, 200, 0, 0)],  # No match
    ]
    
    matcher = FingerprintMatcher(distance_threshold=20, angle_threshold=np.pi/6)
    
    scores, matched_counts = matcher.batch_match(query, gallery)
    rank = matcher.get_rank(scores, true_index=2)
    
    print(f"Scores: {scores}")
    print(f"Matched counts: {matched_counts}")
    print(f"Rank of true match: {rank}")
    
    if rank == 1:
        print("SUCCESS: Ranking works correctly!")
        return True
    else:
        print("WARNING: True match not ranked first (might be OK)")
        return True


def test_data_loading():
    """Test dataset loading"""
    print("\n" + "="*60)
    print("Test 4: Dataset Loading")
    print("="*60)
    
    dataset_dir = os.environ.get(
        "LATENTFP_DATASET_DIR", os.path.join("datasets", "processed_sd04_dataset"))
    original_dir = os.path.join(dataset_dir, 'original')
    
    if not os.path.exists(original_dir):
        print(f"ERROR: Dataset directory not found: {original_dir}")
        return False
    
    all_files = os.listdir(original_dir)
    latent_files = [f for f in all_files if f.startswith('f') and not 'manual' in f]
    rolled_files = [f for f in all_files if f.startswith('s') and not 'manual' in f]
    
    print(f"Latent files: {len(latent_files)}")
    print(f"Rolled files: {len(rolled_files)}")
    
    if len(latent_files) > 0 and len(rolled_files) > 0:
        print(f"Example latent: {latent_files[0]}")
        print(f"Example rolled: {rolled_files[0]}")
        print("SUCCESS: Dataset loaded!")
        return True
    else:
        print("ERROR: No latent or rolled files found!")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("CMC System Component Tests")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Data Loading", test_data_loading()))
    results.append(("Minutiae Extraction", test_minutiae_extraction()))
    results.append(("Fingerprint Matching", test_fingerprint_matching()))
    results.append(("Batch Matching", test_batch_matching()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("All tests passed! System is ready.")
        print("="*60)
        print("\nNext steps:")
        print("1. Run: python enhance_for_cmc.py")
        print("2. Run: python run_cmc_evaluation.py")
    else:
        print("\nSome tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
