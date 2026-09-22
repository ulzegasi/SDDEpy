"""Peak-loss semantics and the real SABC/worker/save interfaces."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import SABC_SolarDynamo as inference
from process_fdist import make_process_f_dist
from spectral_peak_stats import (
    _gleissberg_loss,
    _native_losses,
    build_spectral_peak_stats,
    validate_spectral_peak_resume,
)


ROOT = Path(__file__).resolve().parents[1]


def toy_simulator(theta, y, rng):
    """Cheap, stochastic spectra for interface tests; no physical SDDE claim."""
    time = np.arange(y.shape[1])[None, :]
    phase = rng.uniform(0, 2 * np.pi, size=(len(theta), 1))
    y[:] = (
        20 + 10 * np.cos(2 * np.pi * time / (8 + theta[:, 0:1]) + phase)
        + 4 * np.cos(2 * np.pi * time / (60 + 8 * theta[:, 1:2]))
        + rng.normal(0, 0.2, size=y.shape)
    )


class SpectralPeakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observed = np.loadtxt(ROOT / 'data/silso_SN_y_202601.csv', delimiter=',')[49:-6, 1]
        cls.adapter = build_spectral_peak_stats(cls.observed)

    def test_observed_self_match_and_no_amplitude_matching(self):
        output = np.empty((2, 1))
        self.adapter.batch(np.array([self.observed, self.observed * 13]), output)
        np.testing.assert_allclose(output, 0, atol=1e-12)
        self.assertAlmostEqual(1 / self.adapter.config.gleissberg_target_frequency, 98.7521848667)
        rng = np.random.default_rng(123)
        y = rng.normal(size=(10, 271))
        np.testing.assert_allclose(
            self.adapter.feature_losses(y), self.adapter.feature_losses(7.3 * y), atol=1e-12,
        )

    def test_main_must_be_dominant_and_secondary_peaks_cannot_be_reused(self):
        config = self.adapter.config
        spectrum = np.full(136, 0.01)
        spectrum[np.array(config.native_target_bins)] = [10, 5, 1, 1, 1, 1]
        np.testing.assert_array_equal(_native_losses(spectrum, config), np.zeros(6))
        merged = spectrum.copy()
        merged[16:19] = [0.2, 1.0, 0.2]
        self.assertEqual(np.count_nonzero(_native_losses(merged, config)[2:4] < 1), 1)
        short_dominant = spectrum.copy()
        short_dominant[30] = 20  # 9-year cycle cannot be hidden behind a minor 11-year bump.
        self.assertEqual(_native_losses(short_dominant, config)[0], 1)
        displaced = spectrum.copy()
        displaced[25], displaced[26] = 0.01, 10
        self.assertAlmostEqual(_native_losses(displaced, config)[0], 1 / 3)

    def test_missing_and_weak_gleissberg_penalties_are_gradual(self):
        config = self.adapter.config
        f = config.gleissberg_target_frequency
        self.assertEqual(_gleissberg_loss([], config), 1)
        for visibility in [0, 0.25, 0.5, 0.75, 1]:
            self.assertAlmostEqual(_gleissberg_loss([(f, visibility)], config), 1 - visibility)
        for delta in [0, 0.1, 0.5, 1, 2]:
            self.assertAlmostEqual(_gleissberg_loss([(f + delta / 271, 1)], config), min(delta, 1))

    def test_one_missing_primary_outweighs_all_four_secondaries(self):
        weights = np.array(self.adapter.config.weights)
        self.assertGreater(weights[0], weights[2:].sum())
        self.assertGreater(weights[1], weights[2:].sum())
        # Verify aggregation occurs before the SABC adapter sees the distance.
        losses = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1]], dtype=float)
        output = np.empty((2, 1))
        with patch.object(type(self.adapter), 'feature_losses', return_value=losses):
            self.adapter.batch(np.ones((2, 271)), output)
        np.testing.assert_allclose(output[:, 0], [5 / 14, 4 / 14])

    def test_missing_or_invalid_records_receive_maximum_loss(self):
        y = np.array([np.zeros(271), np.ones(271), np.full(271, np.nan)])
        output = np.empty((3, 1))
        self.adapter.batch(y, output)
        np.testing.assert_array_equal(output, np.ones((3, 1)))
        self.adapter.batch(np.empty((0, 271)), np.empty((0, 1)))
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.adapter.batch(np.ones((1, 272)), np.empty((1, 1)))

    def test_pickle_and_resume_preserve_observation_and_score(self):
        restored = pickle.loads(pickle.dumps(self.adapter))
        validate_spectral_peak_resume(restored.batch, self.adapter.batch)
        for previous, current in [(None, self.adapter.batch), (self.adapter.batch, None)]:
            with self.assertRaisesRegex(ValueError, 'Start a fresh inference'):
                validate_spectral_peak_resume(previous, current)
        changed = replace(self.adapter, config=replace(self.adapter.config, weights=(1.,) * 6))
        with self.assertRaises(ValueError):
            validate_spectral_peak_resume(changed.batch, self.adapter.batch)
        changed_observation = build_spectral_peak_stats(self.observed * 2)
        with self.assertRaises(ValueError):
            validate_spectral_peak_resume(changed_observation.batch, self.adapter.batch)
        validate_spectral_peak_resume(None, None)  # Existing backends are unaffected.

    def test_cli_is_opt_in_and_rejects_incompatible_choices(self):
        with patch('sys.argv', ['prog']):
            self.assertEqual(inference._parse_args().summary_stats, 'fft')
        with patch('sys.argv', ['prog', '--summary-stats', 'spectral_peaks']):
            self.assertEqual(inference._parse_args().summary_stats, 'spectral_peaks')
        for extra in [['--dataset', 'C14'], ['--algorithm', 'multi_eps'], ['--fourier-range', '1:6:120']]:
            with patch('sys.argv', ['prog', '--summary-stats', 'spectral_peaks'] + extra):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    inference._parse_args()

    def test_spawn_workers_clone_and_pickle_preserve_scalar_distance(self):
        theta = np.random.default_rng(18).uniform(1, 10, size=(8, 6))
        original = make_process_f_dist(
            n_samples=271, ss_obs=self.adapter.ss_obs, simulator=toy_simulator,
            stats_fn=self.adapter.batch, seed=123, n_workers=2,
        )
        clone = original.clone(seed=123)
        restored = None
        try:
            expected = original(theta)
            np.testing.assert_array_equal(clone(theta), expected)
            restored = pickle.loads(pickle.dumps(original))
            # Existing ProcessFDist restores the configured seed on unpickle.
            np.testing.assert_array_equal(restored(theta), expected)
        finally:
            for distance in (original, clone, restored):
                if distance is not None and distance._pool is not None:
                    distance._pool.close()
                    distance._pool.join()
                    distance._pool = None
        self.assertEqual(expected.shape, (8, 1))
        self.assertTrue(np.all((expected >= 0) & (expected <= 1)))

    def test_driver_runs_real_sabc_and_saves_readable_and_pickled_settings(self):
        # Exercise the production driver and actual SABC, with only SDDE and budget replaced.
        api = list(inference._import_sabc_package())
        make_config, real_sabc = api[1], api[4]

        def small_config(**kwargs):
            kwargs.update(n_particles=24, show_progressbar=False, show_checkpoint=None)
            return make_config(**kwargs)

        def small_sabc(config, *, n_simulation):
            self.assertEqual(config.f_dist.ss_obs.size, 1)
            self.assertEqual(config.algorithm, 'single_eps')
            self.assertEqual(config.f_dist.stats_fn.__self__.config.weights, (5, 5, 1, 1, 1, 1))
            return real_sabc(config, n_simulation=120)

        api[1], api[4] = small_config, small_sabc
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            argv = ['prog', '--model', 'jupiter', '--summary-stats', 'spectral_peaks',
                    '--n-workers', '1', '--run-name', 'peak_test',
                    '--simulator-seed', '123', '--algorithm-seed', '18', '--proposal-seed', '22']
            with (
                patch('sys.argv', argv),
                patch.object(inference, 'init_julia'),
                patch.object(inference, 'build_simulator', return_value=toy_simulator),
                patch.object(inference, '_import_sabc_package', return_value=tuple(api)),
                patch.object(inference, 'LOCAL_OUT_DIR', outdir),
                redirect_stdout(io.StringIO()),
            ):
                inference.main()
            metadata = json.loads((outdir / 'spectral_peaks_peak_test.json').read_text())
            self.assertEqual(metadata['version'], 'spectral_peaks_v1')
            result = api[2](outdir / 'SABCresult_peak_test.pkl')
            self.assertEqual(result.population.shape, (24, 6))
            self.assertEqual(result.rho.shape, (24, 1))
            self.assertEqual(result.config.f_dist.stats_fn.__self__.config, self.adapter.config)
            self.assertTrue(np.all(np.isfinite(result.rho)))


if __name__ == '__main__':
    unittest.main()
