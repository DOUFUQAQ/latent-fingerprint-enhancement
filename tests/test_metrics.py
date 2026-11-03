"""SSIM / Dice / PSNR reference maths and loss-weight bookkeeping."""

from __future__ import annotations

import math
import unittest

import numpy as np

from latentfp.losses import metrics


def _naive_box_filter(image: np.ndarray, window: int) -> np.ndarray:
    """Straightforward loop version used as a cross-check."""
    radius = window // 2
    padded = np.pad(image, radius, mode="reflect")
    height, width = image.shape
    out = np.zeros_like(image, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            out[y, x] = padded[y : y + window, x : x + window].mean()
    return out


class TestBoxFilter(unittest.TestCase):
    def test_matches_naive_implementation(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.random((9, 11))
        for window in (3, 5, 7):
            with self.subTest(window=window):
                np.testing.assert_allclose(
                    metrics.box_filter(image, window),
                    _naive_box_filter(image, window),
                    atol=1e-10,
                )

    def test_constant_image_is_preserved(self) -> None:
        image = np.full((12, 12), 0.37)
        np.testing.assert_allclose(metrics.box_filter(image, 5), image, atol=1e-12)

    def test_shape_is_preserved(self) -> None:
        self.assertEqual(metrics.box_filter(np.zeros((7, 13)), 3).shape, (7, 13))

    def test_even_window_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrics.box_filter(np.zeros((4, 4)), 4)

    def test_non_2d_input_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrics.box_filter(np.zeros((4, 4, 1)), 3)


class TestSsim(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.image = rng.random((32, 32))

    def test_identical_images_score_one(self) -> None:
        np.testing.assert_allclose(
            metrics.ssim_map(self.image, self.image), 1.0, atol=1e-9
        )
        self.assertAlmostEqual(metrics.ssim_loss(self.image, self.image), 0.0, places=7)

    def test_constant_images_score_one(self) -> None:
        flat = np.full((16, 16), 0.5)
        # sigma_x and sigma_y are 0 up to float round-off, so the loss is
        # ~1e-9 rather than exactly 0
        self.assertLess(metrics.ssim_loss(flat, flat), 1e-6)

    def test_noise_lowers_the_score(self) -> None:
        noisy = np.clip(self.image + 0.2, 0.0, 1.0)
        loss = metrics.ssim_loss(noisy, self.image)
        self.assertGreater(loss, 0.0)
        self.assertLessEqual(loss, 1.0)

    def test_more_noise_is_worse(self) -> None:
        mild = np.clip(self.image + np.random.default_rng(1).normal(0, 0.02, self.image.shape), 0, 1)
        heavy = np.clip(self.image + np.random.default_rng(1).normal(0, 0.20, self.image.shape), 0, 1)
        self.assertLess(metrics.ssim_loss(mild, self.image),
                        metrics.ssim_loss(heavy, self.image))

    def test_map_has_input_shape(self) -> None:
        self.assertEqual(metrics.ssim_map(self.image, self.image).shape, self.image.shape)

    def test_shape_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrics.ssim_loss(np.zeros((8, 8)), np.zeros((8, 9)))


class TestDice(unittest.TestCase):
    def test_perfect_overlap(self) -> None:
        mask = (np.arange(64).reshape(8, 8) % 2).astype(float)
        self.assertAlmostEqual(metrics.dice_coefficient(mask, mask), 1.0, places=6)
        self.assertAlmostEqual(metrics.dice_loss(mask, mask), 0.0, places=6)

    def test_disjoint_masks_score_near_zero(self) -> None:
        left = np.zeros((4, 8))
        left[:, :4] = 1.0
        right = np.zeros((4, 8))
        right[:, 4:] = 1.0
        self.assertLess(metrics.dice_coefficient(left, right), 1e-5)

    def test_half_overlap(self) -> None:
        pred = np.zeros((2, 4))
        pred[:, :2] = 1.0
        target = np.zeros((2, 4))
        target[:, 1:3] = 1.0
        # intersection 2, sums 4 + 4 -> 2*2 / 8 = 0.5
        self.assertAlmostEqual(metrics.dice_coefficient(pred, target), 0.5, places=6)

    def test_all_zero_masks_do_not_divide_by_zero(self) -> None:
        value = metrics.dice_coefficient(np.zeros((4, 4)), np.zeros((4, 4)))
        self.assertTrue(math.isfinite(value))

    def test_shape_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrics.dice_coefficient(np.zeros((4, 4)), np.zeros((4, 5)))


class TestPsnr(unittest.TestCase):
    def test_identical_images_are_infinite(self) -> None:
        image = np.random.default_rng(0).random((8, 8))
        self.assertEqual(metrics.psnr(image, image), math.inf)

    def test_known_offset(self) -> None:
        pred = np.zeros((4, 4))
        target = np.full((4, 4), 0.1)
        expected = 10.0 * math.log10(1.0 / 0.01)
        self.assertAlmostEqual(metrics.psnr(pred, target), expected, places=6)

    def test_shape_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrics.psnr(np.zeros((4, 4)), np.zeros((5, 4)))


class TestNormalizeWeights(unittest.TestCase):
    def test_absolute_values_sum_to_one(self) -> None:
        weights = {"l1": 100.0, "perceptual": 10.0, "tv": 0.1}
        normalized = metrics.normalize_weights(weights)
        self.assertAlmostEqual(sum(abs(v) for v in normalized.values()), 1.0, places=12)

    def test_ratios_are_preserved(self) -> None:
        normalized = metrics.normalize_weights({"a": 3.0, "b": 1.0})
        self.assertAlmostEqual(normalized["a"] / normalized["b"], 3.0, places=12)

    def test_empty_weights_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrics.normalize_weights({})

    def test_all_zero_weights_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrics.normalize_weights({"a": 0.0, "b": 0.0})


class TestCombineLosses(unittest.TestCase):
    def test_unweighted_sum(self) -> None:
        terms = {"l1": 0.5, "dice": 0.25}
        self.assertAlmostEqual(metrics.combine_losses(terms), 0.75, places=12)

    def test_weighted_sum(self) -> None:
        terms = {"l1": 0.5, "dice": 0.25}
        weights = {"l1": 2.0, "dice": 4.0}
        self.assertAlmostEqual(metrics.combine_losses(terms, weights), 2.0, places=12)

    def test_matches_the_enhanced_gan_objective(self) -> None:
        """Reproduces the lambda set used by train_enhanced_gan.py."""
        terms = {"adv": 1.0, "l1": 1.0, "perceptual": 1.0, "style": 1.0,
                 "orientation": 1.0, "tv": 1.0, "dice": 1.0}
        weights = {"adv": 1.0, "l1": 100.0, "perceptual": 10.0, "style": 10.0,
                   "orientation": 5.0, "tv": 0.1, "dice": 10.0}
        expected = 1.0 + 100.0 + 10.0 + 10.0 + 5.0 + 0.1 + 10.0
        self.assertAlmostEqual(metrics.combine_losses(terms, weights), expected, places=12)

    def test_missing_weight_raises(self) -> None:
        with self.assertRaises(KeyError):
            metrics.combine_losses({"l1": 1.0, "dice": 1.0}, {"l1": 1.0})

    def test_unknown_weight_raises(self) -> None:
        with self.assertRaises(KeyError):
            metrics.combine_losses({"l1": 1.0}, {"l1": 1.0, "typo": 1.0})

    def test_non_finite_term_raises(self) -> None:
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    metrics.combine_losses({"l1": bad})

    def test_empty_terms_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrics.combine_losses({})


class TestTorchPorts(unittest.TestCase):
    def test_ssim_loss_torch_matches_numpy_on_identical_input(self) -> None:
        if not metrics.torch_available():
            self.skipTest("torch is not installed")
        import torch

        rng = np.random.default_rng(3)
        image = rng.random((1, 1, 32, 32))
        tensor = torch.from_numpy(image).float()
        value = float(metrics.ssim_loss_torch(tensor, tensor))
        self.assertAlmostEqual(value, 0.0, places=5)

    def test_ssim_loss_torch_agrees_with_numpy_on_noisy_input(self) -> None:
        if not metrics.torch_available():
            self.skipTest("torch is not installed")
        import torch

        rng = np.random.default_rng(11)
        image = rng.random((1, 1, 48, 48)).astype(np.float32)
        noisy = np.clip(image + rng.normal(0, 0.1, image.shape), 0, 1).astype(np.float32)

        # avg_pool2d zero-pads while the numpy reference reflects, so the two
        # differ slightly along the 5-pixel border; the tolerance covers that
        torch_value = float(metrics.ssim_loss_torch(
            torch.from_numpy(noisy), torch.from_numpy(image)))
        numpy_value = metrics.ssim_loss(noisy[0, 0], image[0, 0])
        self.assertAlmostEqual(torch_value, numpy_value, delta=0.1)

    def test_dice_loss_torch_extremes(self) -> None:
        if not metrics.torch_available():
            self.skipTest("torch is not installed")
        import torch

        logits = torch.full((1, 1, 4, 4), 20.0)
        target = torch.ones((1, 1, 4, 4))
        value = float(metrics.dice_loss_torch(logits, target))
        self.assertLess(value, 1e-3)


if __name__ == "__main__":
    unittest.main()
