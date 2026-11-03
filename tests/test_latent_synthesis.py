"""Latent-print degradation synthesis (pure-numpy operators).

OpenCV is optional for the operators tested here; the two functions that
really need it (``gaussian_blur`` / ``motion_blur``) are skipped when it is
missing.
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from latentfp.transforms import advanced_synthesis as adv
from latentfp.transforms import latent_synthesis as lat

try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def _ramp(height: int = 16, width: int = 16) -> np.ndarray:
    return np.tile(np.linspace(0.0, 1.0, width), (height, 1)).astype(np.float32)


class TestEnsureFloat01(unittest.TestCase):
    def test_uint8_is_scaled(self) -> None:
        image = np.array([[0, 255]], dtype=np.uint8)
        out = lat._ensure_float01(image)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, [[0.0, 1.0]])

    def test_float_in_range_is_untouched(self) -> None:
        image = _ramp()
        np.testing.assert_allclose(lat._ensure_float01(image), image)

    def test_values_are_clipped(self) -> None:
        # NOTE: the 1/255 rescaling is applied to the whole array as soon as
        # max(image) > 1, so a mixed 0..1 / 0..255 array is rescaled as well.
        out = lat._ensure_float01(np.array([[-5.0, 300.0, 0.5]], dtype=np.float32))
        np.testing.assert_allclose(out, [[0.0, 1.0, 0.5 / 255.0]], rtol=1e-6)


class TestDeterministicOperators(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(1234)
        np.random.seed(1234)

    def test_gamma_of_one_is_identity_and_two_squares(self) -> None:
        image = _ramp()
        np.testing.assert_allclose(
            lat.random_gamma(image, gamma_range=(1.0, 1.0), p=1.0), image, atol=1e-6
        )
        np.testing.assert_allclose(
            lat.random_gamma(image, gamma_range=(2.0, 2.0), p=1.0), image ** 2, atol=1e-6
        )

    def test_gamma_is_skipped_when_p_is_zero(self) -> None:
        image = _ramp()
        np.testing.assert_allclose(lat.random_gamma(image, p=0.0), image)

    def test_contrast_brightness_identity(self) -> None:
        image = _ramp()
        out = lat.random_contrast_brightness(
            image, alpha_range=(1.0, 1.0), beta_range=(0.0, 0.0), p=1.0
        )
        np.testing.assert_allclose(out, image, atol=1e-6)

    def test_contrast_brightness_stays_in_range(self) -> None:
        image = _ramp()
        out = lat.random_contrast_brightness(image, p=1.0)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_gaussian_noise_is_disabled_at_p_zero(self) -> None:
        image = _ramp()
        np.testing.assert_allclose(lat.add_gaussian_noise(image, p=0.0), image)

    def test_gaussian_noise_is_bounded(self) -> None:
        image = np.full((32, 32), 0.5, dtype=np.float32)
        out = lat.add_gaussian_noise(image, sigma_range=(0.05, 0.05), p=1.0)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)
        self.assertFalse(np.allclose(out, image))

    def test_speckle_noise_is_disabled_at_p_zero(self) -> None:
        image = _ramp()
        np.testing.assert_allclose(lat.add_speckle_noise(image, p=0.0), image)

    def test_occlusions_overwrite_rectangles(self) -> None:
        image = np.full((40, 40), 0.5, dtype=np.float32)
        out = lat.random_occlusions(
            image, num_range=(1, 1), size_frac_range=(0.2, 0.2), p=1.0
        )
        self.assertFalse(np.allclose(out, image))
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_occlusions_are_disabled_at_p_zero(self) -> None:
        image = _ramp()
        np.testing.assert_allclose(lat.random_occlusions(image, p=0.0), image)

    def test_vignette_darkens_the_corners_not_the_centre(self) -> None:
        image = np.full((64, 64), 0.8, dtype=np.float32)
        out = lat.vignette(image, strength_range=(1.0, 1.0), p=1.0)
        # the mask is exactly 1 at the centre pixel and 0 in the far corner
        self.assertAlmostEqual(float(out[32, 32]), 0.8, places=5)
        self.assertLess(float(out[:8, :8].mean()), 0.6)
        self.assertGreater(float(out[32, 32]), float(out[:8, :8].mean()))

    def test_vignette_is_disabled_at_p_zero(self) -> None:
        image = _ramp()
        np.testing.assert_allclose(lat.vignette(image, p=0.0), image)


@unittest.skipUnless(CV2_AVAILABLE, "opencv-python is not installed")
class TestSynthesizeLatent(unittest.TestCase):
    """``synthesize_latent`` runs the whole chain, two steps of which need cv2."""

    def test_output_is_float01_with_input_shape(self) -> None:
        random.seed(0)
        np.random.seed(0)
        clean = _ramp(32, 32)
        latent = lat.synthesize_latent(clean)
        self.assertEqual(latent.shape, clean.shape)
        self.assertEqual(latent.dtype, np.float32)
        self.assertGreaterEqual(float(latent.min()), 0.0)
        self.assertLessEqual(float(latent.max()), 1.0)

    def test_contrast_is_reduced(self) -> None:
        random.seed(7)
        np.random.seed(7)
        clean = _ramp(64, 64)
        latent = lat.synthesize_latent(clean)
        # the tail of the pipeline rescales by alpha in [0.5, 0.9]
        self.assertLess(float(np.ptp(latent)), float(np.ptp(clean)))

    def test_seeded_rng_makes_the_call_reproducible_and_restores_state(self) -> None:
        clean = _ramp(24, 24)
        np.random.seed(1)
        first = lat.synthesize_latent(clean, rng=random.Random(99))
        np.random.seed(1)
        second = lat.synthesize_latent(clean, rng=random.Random(99))
        np.testing.assert_allclose(first, second)

        random.seed(5)
        before = random.random()
        random.seed(5)
        lat.synthesize_latent(clean, rng=random.Random(1))
        after = random.random()
        self.assertEqual(before, after)


class TestCv2Operators(unittest.TestCase):
    @unittest.skipUnless(CV2_AVAILABLE, "opencv-python is not installed")
    def test_gaussian_blur_smooths(self) -> None:
        image = np.zeros((16, 16), dtype=np.float32)
        image[8, 8] = 1.0
        out = lat.gaussian_blur(image, k_choices=(5,), p=1.0)
        self.assertLess(float(out[8, 8]), 1.0)
        self.assertGreater(float(out[8, 9]), 0.0)

    @unittest.skipUnless(CV2_AVAILABLE, "opencv-python is not installed")
    def test_motion_blur_keeps_the_range(self) -> None:
        image = _ramp()
        out = lat.motion_blur(image, p=1.0)
        self.assertEqual(out.shape, image.shape)
        self.assertGreaterEqual(float(out.min()), -1e-6)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-6)


class TestAdvancedSynthesis(unittest.TestCase):
    def test_multiply_blend_is_a_product(self) -> None:
        fg = np.array([[0.5, 1.0]], dtype=np.float32)
        bg = np.array([[0.4, 0.2]], dtype=np.float32)
        np.testing.assert_allclose(adv.multiply_blend(fg, bg), [[0.2, 0.2]])

    def test_multiply_blend_darkens(self) -> None:
        fg = _ramp()
        bg = np.full_like(fg, 0.5)
        out = adv.multiply_blend(fg, bg)
        self.assertTrue(bool(np.all(out <= fg + 1e-6)))

    def test_background_has_the_requested_shape_and_range(self) -> None:
        np.random.seed(0)
        background = adv.create_complex_background((32, 48))
        self.assertEqual(background.shape, (32, 48))
        # dtype is float32 or float64 depending on which random branches fire
        self.assertTrue(np.issubdtype(background.dtype, np.floating))
        self.assertGreaterEqual(float(background.min()), 0.0)
        self.assertLessEqual(float(background.max()), 1.0)

    def test_background_is_deterministic_under_a_seed(self) -> None:
        np.random.seed(3)
        random.seed(3)
        first = adv.create_complex_background((16, 16))
        np.random.seed(3)
        random.seed(3)
        second = adv.create_complex_background((16, 16))
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
