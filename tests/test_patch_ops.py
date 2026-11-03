"""Sliding-window patch bookkeeping used by the sparse enhancers."""

from __future__ import annotations

import unittest

import numpy as np

from latentfp.transforms.patch_ops import (
    accumulate_patch,
    count_patch_hits,
    iter_patch_positions,
    reconstruct_overlapping,
    weighted_average,
)


class TestIterPatchPositions(unittest.TestCase):
    def test_full_cover_grid(self) -> None:
        positions = iter_patch_positions(64, 64, patch_size=32, stride=16)
        self.assertEqual(
            positions,
            [(0, 0), (0, 16), (0, 32),
             (16, 0), (16, 16), (16, 32),
             (32, 0), (32, 16), (32, 32)],
        )

    def test_no_partial_patches(self) -> None:
        height, width, size = 50, 37, 12
        for top, left in iter_patch_positions(height, width, size, stride=5):
            self.assertLessEqual(top + size, height)
            self.assertLessEqual(left + size, width)

    def test_patch_larger_than_image_yields_nothing(self) -> None:
        self.assertEqual(iter_patch_positions(10, 10, patch_size=11, stride=1), [])

    def test_exhaustive_stride_covers_every_start(self) -> None:
        positions = iter_patch_positions(20, 20, patch_size=4, stride=1)
        self.assertEqual(len(positions), 17 * 17)
        self.assertEqual(positions[-1], (16, 16))

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            iter_patch_positions(10, 10, patch_size=0, stride=1)
        with self.assertRaises(ValueError):
            iter_patch_positions(10, 10, patch_size=4, stride=0)


class TestCountPatchHits(unittest.TestCase):
    def test_total_hits_equal_coverage(self) -> None:
        positions = iter_patch_positions(40, 40, patch_size=16, stride=8)
        hits = count_patch_hits(40, 40, 16, 8)
        self.assertEqual(hits.sum(), len(positions) * 16 * 16)

    def test_interior_is_covered_more_often_than_the_corner(self) -> None:
        hits = count_patch_hits(64, 64, 16, 8)
        # the top-left corner is covered once, the centre four times
        self.assertEqual(hits[0, 0], 1)
        self.assertEqual(hits[32, 32], 4)


class TestAccumulateAndAverage(unittest.TestCase):
    def test_single_patch_is_returned_unchanged(self) -> None:
        accumulator = np.zeros((8, 8))
        weight = np.zeros((8, 8))
        patch = np.arange(16, dtype=float).reshape(4, 4)
        accumulate_patch(accumulator, weight, patch, top=2, left=3)

        out = weighted_average(accumulator, weight)
        np.testing.assert_allclose(out[2:6, 3:7], patch, atol=1e-9)
        self.assertEqual(out[0, 0], 0.0)

    def test_identical_overlapping_patches_average_to_themselves(self) -> None:
        shape = (8, 8)
        patch = np.full((4, 4), 0.25)
        out = reconstruct_overlapping(
            [((0, 0), patch), ((1, 1), patch), ((2, 2), patch)], shape
        )
        covered = out > 0
        np.testing.assert_allclose(out[covered], 0.25, atol=1e-9)

    def test_average_not_sum(self) -> None:
        accumulator = np.ones((4, 4)) * 3.0
        weight = np.ones((4, 4)) * 3.0
        out = weighted_average(accumulator, weight)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_fallback_keeps_uncovered_pixels(self) -> None:
        accumulator = np.zeros((6, 6))
        weight = np.zeros((6, 6))
        accumulate_patch(accumulator, weight, np.full((2, 2), 0.5), top=0, left=0)
        fallback = np.full((6, 6), 0.9)
        out = weighted_average(accumulator, weight, fallback=fallback)
        np.testing.assert_allclose(out[0:2, 0:2], 0.5, atol=1e-9)
        np.testing.assert_allclose(out[4:, 4:], 0.9, atol=1e-9)

    def test_patch_weight_scales_the_contribution(self) -> None:
        accumulator = np.zeros((4, 4))
        weight = np.zeros((4, 4))
        accumulate_patch(accumulator, weight, np.full((2, 2), 2.0), 0, 0,
                         patch_weight=3.0)
        np.testing.assert_allclose(weight[0:2, 0:2], 3.0)
        out = weighted_average(accumulator, weight)
        np.testing.assert_allclose(out[0:2, 0:2], 2.0, atol=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            weighted_average(np.zeros((4, 4)), np.zeros((4, 5)))

    def test_patch_outside_buffer_raises(self) -> None:
        accumulator = np.zeros((4, 4))
        weight = np.zeros((4, 4))
        with self.assertRaises(ValueError):
            accumulate_patch(accumulator, weight, np.zeros((3, 3)), top=2, left=2)

    def test_non_2d_patch_raises(self) -> None:
        accumulator = np.zeros((4, 4))
        weight = np.zeros((4, 4))
        with self.assertRaises(ValueError):
            accumulate_patch(accumulator, weight, np.zeros((2, 2, 2)), 0, 0)

    def test_edge_patches_are_covered_by_construction(self) -> None:
        """A patch grid built with iter_patch_positions leaves no hole."""
        height, width, size, stride = 33, 29, 9, 4
        hits = count_patch_hits(height, width, size, stride)
        # every pixel must be inside at least one patch
        self.assertTrue(bool((hits > 0).all()))


if __name__ == "__main__":
    unittest.main()
