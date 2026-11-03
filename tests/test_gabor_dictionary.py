"""Gabor dictionary construction and sparse reconstruction bookkeeping.

The dictionary maths is pure numpy and is exercised directly.  The two
solvers (``Lasso`` / ``OrthogonalMatchingPursuit``) need scikit-learn and are
skipped when it is unavailable.
"""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np

from repair import gabor_sparse_coding as gsc
from repair import sparse_enhancement as se

try:
    import sklearn  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TestCreateGaborKernel(unittest.TestCase):
    def test_shape_is_exactly_size_by_size(self) -> None:
        for size in (4, 8, 24):
            with self.subTest(size=size):
                kernel = gsc.create_gabor_kernel(
                    size=size, theta=0.0, freq=8.0 / size,
                    sigma_x=size / 2.0, sigma_y=size / 2.0, phase=0.0,
                )
                self.assertEqual(kernel.shape, (size, size))

    def test_values_are_finite(self) -> None:
        kernel = gsc.create_gabor_kernel(8, 0.3, 0.3, 4.0, 4.0, 0.5)
        self.assertTrue(bool(np.isfinite(kernel).all()))

    def test_orientation_changes_the_kernel(self) -> None:
        first = gsc.create_gabor_kernel(16, 0.0, 0.25, 8.0, 8.0, 0.0)
        second = gsc.create_gabor_kernel(16, np.pi / 2, 0.25, 8.0, 8.0, 0.0)
        self.assertFalse(bool(np.allclose(first, second)))

    def test_phase_changes_the_kernel(self) -> None:
        first = gsc.create_gabor_kernel(16, 0.0, 0.25, 8.0, 8.0, 0.0)
        second = gsc.create_gabor_kernel(16, 0.0, 0.25, 8.0, 8.0, np.pi / 2)
        self.assertFalse(bool(np.allclose(first, second)))


def _quiet_build(patch_size: int, n_orientations: int, n_frequencies: int,
                 n_phases: int):
    """Call ``build_gabor_dictionary`` without its progress prints.

    The implementation prints a U+2713 check mark, which raises
    ``UnicodeEncodeError`` on a cp936 Windows console; capturing stdout keeps
    the tests portable and the output quiet.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return gsc.build_gabor_dictionary(
            patch_size=patch_size, n_orientations=n_orientations,
            n_frequencies=n_frequencies, n_phases=n_phases,
        )


class TestBuildGaborDictionary(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # tiny configuration: the default 24x24 / 16x9x6 dictionary is 864
        # atoms and would make this module slow for no extra information
        cls.dictionary, cls.params = _quiet_build(
            patch_size=4, n_orientations=4, n_frequencies=2, n_phases=2
        )

    def test_shape_is_patch_squared_by_atom_count(self) -> None:
        self.assertEqual(self.dictionary.shape, (16, 4 * 2 * 2))

    def test_every_atom_has_unit_norm(self) -> None:
        norms = np.linalg.norm(self.dictionary, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)

    def test_every_atom_is_zero_mean(self) -> None:
        means = self.dictionary.mean(axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-9)

    def test_params_match_the_atom_count(self) -> None:
        self.assertEqual(len(self.params), self.dictionary.shape[1])
        for theta, freq, phase in self.params:
            self.assertGreaterEqual(theta, 0.0)
            self.assertGreater(freq, 0.0)
            self.assertGreaterEqual(phase, 0.0)

    def test_construction_is_deterministic(self) -> None:
        again, params = _quiet_build(
            patch_size=4, n_orientations=4, n_frequencies=2, n_phases=2
        )
        np.testing.assert_allclose(again, self.dictionary)
        self.assertEqual(params, self.params)

    def test_default_configuration_atom_count(self) -> None:
        # 16 orientations x 9 frequencies x 6 phases, as reported in the README
        self.assertEqual(16 * 9 * 6, 864)


class TestReconstructPatch(unittest.TestCase):
    def test_round_trip(self) -> None:
        dictionary, _ = _quiet_build(
            patch_size=4, n_orientations=2, n_frequencies=1, n_phases=1
        )
        alpha = np.zeros(dictionary.shape[1])
        alpha[0] = 2.0
        mean = 0.25

        patch = gsc.reconstruct_patch(alpha, dictionary, mean=mean, norm=1.0)
        expected = (dictionary @ alpha + mean).reshape(4, 4)
        np.testing.assert_allclose(patch, expected)

    def test_norm_rescales_the_reconstruction(self) -> None:
        dictionary, _ = _quiet_build(
            patch_size=4, n_orientations=2, n_frequencies=1, n_phases=1
        )
        alpha = np.ones(dictionary.shape[1])
        plain = gsc.reconstruct_patch(alpha, dictionary, mean=0.0, norm=1.0)
        doubled = gsc.reconstruct_patch(alpha, dictionary, mean=0.0, norm=2.0)
        np.testing.assert_allclose(doubled, plain * 2.0)


class TestComputeSparsity(unittest.TestCase):
    def test_counts_only_coefficients_above_the_threshold(self) -> None:
        alpha = np.array([0.0, 0.0005, 0.01, -0.5])
        sparsity, l1_norm = gsc.compute_sparsity(alpha, threshold=0.001)
        # 0.0005 is below the threshold, so 2 of 4 coefficients count
        self.assertAlmostEqual(sparsity, 2 / 4)
        # the L1 norm itself is *not* thresholded
        self.assertAlmostEqual(l1_norm, 0.5105, places=6)

    def test_all_zero_coefficients(self) -> None:
        sparsity, l1_norm = gsc.compute_sparsity(np.zeros(5))
        self.assertEqual(sparsity, 0.0)
        self.assertEqual(l1_norm, 0.0)


class TestSparseCodingSolvers(unittest.TestCase):
    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_omp_reconstruction_reduces_residual(self) -> None:
        dictionary, _ = _quiet_build(
            patch_size=4, n_orientations=4, n_frequencies=2, n_phases=2
        )
        rng = np.random.default_rng(0)
        patch = rng.random((4, 4))
        alpha, mean, norm = gsc.sparse_coding_patch(
            patch, dictionary, method="omp", n_nonzero=4
        )
        approximation = (dictionary @ alpha + mean).reshape(4, 4)
        residual = np.linalg.norm(patch - approximation)
        baseline = np.linalg.norm(patch - patch.mean())
        self.assertLess(residual, baseline)
        self.assertEqual(norm, 1.0)

    def test_unknown_solver_raises(self) -> None:
        dictionary = np.eye(4)
        with self.assertRaises(ValueError):
            gsc.sparse_coding_patch(np.zeros((2, 2)), dictionary, method="nope")


class TestSparseEnhancementPrimitives(unittest.TestCase):
    """The second (band-guided) implementation in ``repair/sparse_enhancement.py``."""

    def test_gabor_atoms_are_orthonormalised(self) -> None:
        builder = se.GaborDictionary(patch_size=4, orientations=4,
                                     frequencies=[5.0, 7.0],
                                     phases=[0.0, np.pi / 2])
        dictionary = builder.build_dictionary()
        self.assertEqual(dictionary.shape, (16, 4 * 2 * 2))
        np.testing.assert_allclose(np.linalg.norm(dictionary, axis=0), 1.0, atol=1e-9)
        np.testing.assert_allclose(dictionary.mean(axis=0), 0.0, atol=1e-9)

    def test_gabor_atom_labels_match_the_columns(self) -> None:
        builder = se.GaborDictionary(patch_size=4, orientations=4,
                                     frequencies=[5.0, 7.0],
                                     phases=[0.0, np.pi / 2])
        dictionary = builder.build_dictionary()
        atoms = dictionary.shape[1]
        self.assertEqual(builder.atom_orientations.shape, (atoms,))
        self.assertEqual(builder.atom_wavelengths.shape, (atoms,))
        self.assertEqual(builder.atom_phases.shape, (atoms,))
        self.assertEqual(len(set(builder.atom_orientations.tolist())), 4)

    def test_gabor_kernel_has_the_requested_size_for_even_patch(self) -> None:
        builder = se.GaborDictionary(patch_size=24, orientations=1,
                                     frequencies=[9.0], phases=[0.0])
        kernel = builder._create_gabor_kernel(24, 9.0, 0.0, 0.0)
        self.assertEqual(kernel.shape, (24, 24))

    def test_tv_decomposition_is_additive(self) -> None:
        rng = np.random.default_rng(5)
        image = rng.integers(0, 256, size=(16, 16)).astype(np.uint8)
        texture, cartoon = se.TVDecomposition(lambda_tv=0.1, max_iter=5).decompose(image)
        np.testing.assert_allclose(texture + cartoon, image / 255.0, atol=1e-6)

    def test_tv_decomposition_of_a_flat_image_has_flat_texture(self) -> None:
        image = np.full((16, 16), 128, dtype=np.uint8)
        texture, cartoon = se.TVDecomposition(lambda_tv=0.1, max_iter=5).decompose(image)
        self.assertLess(float(texture.std()), 1e-6)
        self.assertGreater(float(texture.mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
