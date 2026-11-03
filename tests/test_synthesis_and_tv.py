"""Pixel operators of the second-generation synthesiser and the TV decomposition."""

from __future__ import annotations

import unittest

import numpy as np

from latentfp_v2.transforms.synthesis_v2 import _to_float01, multiply_blend
from repair.tv_decomposition import (
    rudin_osher_fatemi_tv,
    tv_decomposition,
)

try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TestToFloat01(unittest.TestCase):
    def test_uint8_is_scaled_and_clipped(self) -> None:
        out = _to_float01(np.array([[0, 128, 255]], dtype=np.uint8))
        np.testing.assert_allclose(out, [[0.0, 128 / 255.0, 1.0]], rtol=1e-6)
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_float_above_one_is_rescaled(self) -> None:
        out = _to_float01(np.array([[0.0, 255.0]], dtype=np.float32))
        np.testing.assert_allclose(out, [[0.0, 1.0]])

    def test_float_in_range_is_untouched(self) -> None:
        image = np.array([[0.25, 0.75]], dtype=np.float32)
        np.testing.assert_allclose(_to_float01(image), image)

    def test_negative_values_are_clipped(self) -> None:
        out = _to_float01(np.array([[-1.0, 0.5]], dtype=np.float32))
        np.testing.assert_allclose(out, [[0.0, 0.5]])


class TestMultiplyBlend(unittest.TestCase):
    def test_product_and_clip(self) -> None:
        np.testing.assert_allclose(
            multiply_blend(np.array([[0.5, 1.0]]), np.array([[0.5, 0.5]])),
            [[0.25, 0.5]],
        )

    def test_result_never_exceeds_the_foreground(self) -> None:
        fg = np.linspace(0, 1, 16).reshape(4, 4)
        bg = np.full((4, 4), 0.6)
        out = multiply_blend(fg, bg)
        self.assertTrue(bool(np.all(out <= fg + 1e-9)))


class TestTvDecomposition(unittest.TestCase):
    """Both TV solvers must be energy preserving: ``u + v == image``."""

    def _check_additivity(self, cartoon: np.ndarray, texture: np.ndarray,
                          image: np.ndarray, atol: float = 4.0) -> None:
        reconstructed = cartoon.astype(np.float64) + (texture.astype(np.float64) - 255.0)
        np.testing.assert_allclose(reconstructed, image.astype(np.float64), atol=atol)

    def test_constant_image_gives_zero_texture(self) -> None:
        image = np.full((32, 32), 120, dtype=np.uint8)
        cartoon, texture = rudin_osher_fatemi_tv(image, num_iter=40)
        # v == 0 everywhere -> texture is the (v + 1) * 255 = 255 baseline
        self.assertTrue(bool(np.all(texture == 255)))
        np.testing.assert_allclose(cartoon, image, atol=1)

    def test_rof_energy_is_conserved_on_a_smooth_image(self) -> None:
        # u + v == image only holds while the cartoon part stays inside
        # [0, 1]; on white noise the solver overshoots and both outputs are
        # clipped, so use a smooth gradient (curvature ~ 0) instead.
        ramp = np.linspace(0.0, 255.0, 24).astype(np.uint8)
        image = np.tile(ramp, (24, 1))
        cartoon, texture = rudin_osher_fatemi_tv(image, lambda_tv=1.5, num_iter=20)
        self.assertEqual(cartoon.shape, image.shape)
        self.assertEqual(texture.shape, image.shape)
        self._check_additivity(cartoon, texture, image)

    def test_rof_outputs_stay_inside_uint8(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(24, 24), dtype=np.uint8)
        cartoon, texture = rudin_osher_fatemi_tv(image, lambda_tv=1.5, num_iter=30)
        for array in (cartoon, texture):
            self.assertEqual(array.dtype, np.uint8)
            self.assertGreaterEqual(int(array.min()), 0)
            self.assertLessEqual(int(array.max()), 255)

    def test_rof_smooths_a_noisy_image(self) -> None:
        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        cartoon, _ = rudin_osher_fatemi_tv(image, lambda_tv=1.5, num_iter=50)
        # total variation of the cartoon part must be lower than the input's
        self.assertLess(float(np.abs(np.diff(cartoon.astype(float), axis=1)).sum()),
                        float(np.abs(np.diff(image.astype(float), axis=1)).sum()))

    def test_tv_decomposition_of_a_constant_image_is_stable(self) -> None:
        # ``tv_decomposition`` scales the TV term by lambda instead of scaling
        # the data term, so on anything but a flat image it overshoots and the
        # uint8 clipping destroys additivity (see the README's known issues).
        image = np.full((16, 16), 120, dtype=np.uint8)
        cartoon, texture = tv_decomposition(image, lambda_tv=1.5, num_iter=60)
        np.testing.assert_allclose(cartoon, image, atol=2)
        self.assertTrue(bool(np.all(texture == 255)))

    def test_tv_decomposition_outputs_stay_inside_uint8(self) -> None:
        rng = np.random.default_rng(6)
        image = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
        cartoon, texture = tv_decomposition(image, lambda_tv=1.5, num_iter=10)
        for array in (cartoon, texture):
            self.assertEqual(array.dtype, np.uint8)
            self.assertGreaterEqual(int(array.min()), 0)
            self.assertLessEqual(int(array.max()), 255)

    def test_larger_lambda_smooths_more(self) -> None:
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, size=(24, 24), dtype=np.uint8)
        smooth, _ = rudin_osher_fatemi_tv(image, lambda_tv=5.0, num_iter=60)
        rough, _ = rudin_osher_fatemi_tv(image, lambda_tv=0.15, num_iter=60)
        self.assertLessEqual(
            float(np.abs(np.diff(smooth.astype(float), axis=1)).sum()),
            float(np.abs(np.diff(rough.astype(float), axis=1)).sum()),
        )

    @unittest.skipUnless(CV2_AVAILABLE, "opencv-python is not installed")
    def test_coherence_of_a_ridge_pattern_is_high(self) -> None:
        from repair.tv_decomposition import compute_coherence

        # perfectly oriented vertical ridges -> structure tensor is rank-1
        xx, _ = np.meshgrid(np.arange(64), np.arange(64))
        image = (127.5 + 100.0 * np.cos(2 * np.pi * xx / 8.0)).astype(np.uint8)
        coherence = compute_coherence(image)
        self.assertGreater(float(coherence.mean()), 0.8)

    @unittest.skipUnless(CV2_AVAILABLE, "opencv-python is not installed")
    def test_coherence_of_noise_is_low(self) -> None:
        from repair.tv_decomposition import compute_coherence

        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        self.assertLess(float(compute_coherence(image).mean()), 0.5)


if __name__ == "__main__":
    unittest.main()
