from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from enca_summary_stats import (
    CANONICAL_NOISEGRID_BACKEND,
    _timeseries_to_fft_log_amplitudes,
    _timeseries_to_rfft_log1p_amplitudes,
    build_enca_fft_cnn_summary_stats,
    build_mlp_summary_stats,
)


class FourierPreprocessingTests(unittest.TestCase):
    def test_mlp_transform_uses_a_hann_window(self):
        samples = np.arange(1, 9, dtype=np.float32).reshape(1, -1)

        actual = _timeseries_to_fft_log_amplitudes(
            samples,
            num_fft_components=4,
            fft_log_eps=1e-8,
        )

        windowed = samples * np.hanning(samples.shape[1]).astype(np.float32)[None, :]
        amplitudes = np.abs(np.fft.fft(windowed, axis=1)) / samples.shape[1]
        expected = np.log(amplitudes[:, :4] + 1e-8).astype(np.float32)
        np.testing.assert_allclose(actual, expected)

    def test_fourier_cnn_hann_transform_windows_before_rfft(self):
        samples = np.arange(1, 9, dtype=np.float32).reshape(1, -1)

        actual = _timeseries_to_rfft_log1p_amplitudes(
            samples,
            num_fft_components=4,
            fft_window="Hann",
        )

        windowed = samples * np.hanning(samples.shape[1]).astype(np.float32)[None, :]
        expected = np.log1p(np.abs(np.fft.rfft(windowed, axis=1))[:, :4])
        np.testing.assert_allclose(actual[:, :, 0], expected.astype(np.float32))

    def test_fourier_cnn_none_transform_keeps_legacy_behavior(self):
        samples = np.arange(1, 9, dtype=np.float32).reshape(1, -1)

        actual = _timeseries_to_rfft_log1p_amplitudes(
            samples,
            num_fft_components=4,
            fft_window="none",
        )

        expected = np.log1p(np.abs(np.fft.rfft(samples, axis=1))[:, :4])
        np.testing.assert_allclose(actual[:, :, 0], expected.astype(np.float32))


class FourierCnnWindowConfigurationTests(unittest.TestCase):
    def _make_run(self, hyper_parameters: dict) -> Path:
        run_dir = Path(self.temp_dir.name)
        (run_dir / "hyper_parameters.json").write_text(
            json.dumps(hyper_parameters),
            encoding="utf-8",
        )
        (run_dir / "model_best_ckpt-1.index").touch()
        return run_dir

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_auto_reads_uppercase_training_window_metadata(self):
        run_dir = self._make_run(
            {
                "len_timeseries": 8,
                "ndims_latent": 2,
                "num_fft_components": 4,
                "WINDOW": "Hann",
            }
        )

        stats = build_enca_fft_cnn_summary_stats(run_dir=run_dir, fft_window="auto")

        self.assertEqual(stats.config.fft_window, "hann")

    def test_auto_preserves_no_window_for_legacy_metadata(self):
        run_dir = self._make_run(
            {
                "len_timeseries": 8,
                "ndims_latent": 2,
                "num_fft_components": 4,
            }
        )

        stats = build_enca_fft_cnn_summary_stats(run_dir=run_dir, fft_window="auto")

        self.assertEqual(stats.config.fft_window, "none")

    def test_explicit_window_overrides_metadata(self):
        run_dir = self._make_run(
            {
                "len_timeseries": 8,
                "ndims_latent": 2,
                "num_fft_components": 4,
                "window": "none",
            }
        )

        stats = build_enca_fft_cnn_summary_stats(run_dir=run_dir, fft_window="hann")

        self.assertEqual(stats.config.fft_window, "hann")

    def test_jupiter_contract_keeps_all_latent_statistics(self):
        run_dir = self._make_run(
            {
                "len_timeseries": 8,
                "ndims_latent": 7,
                "num_fft_components": 4,
                "window": "Hann",
                "model": "jupiter",
                "num_model_parameters": 6,
                "simulation_backend": CANONICAL_NOISEGRID_BACKEND,
            }
        )

        stats = build_enca_fft_cnn_summary_stats(
            run_dir=run_dir,
            expected_model="jupiter",
        )

        self.assertEqual(stats.config.model, "jupiter")
        self.assertEqual(stats.config.num_model_parameters, 6)
        self.assertEqual(stats.config.ndims_latent, 7)

    def test_inference_model_must_match_encoder_model(self):
        run_dir = self._make_run(
            {
                "len_timeseries": 8,
                "ndims_latent": 6,
                "num_fft_components": 4,
                "window": "Hann",
                "model": "jupiter",
                "num_model_parameters": 6,
                "simulation_backend": CANONICAL_NOISEGRID_BACKEND,
            }
        )

        with self.assertRaisesRegex(ValueError, "Select a matching training run"):
            build_enca_fft_cnn_summary_stats(
                run_dir=run_dir,
                expected_model="original",
            )

    def test_jupiter_encoder_requires_six_regressors(self):
        run_dir = self._make_run(
            {
                "len_timeseries": 8,
                "ndims_latent": 6,
                "num_fft_components": 4,
                "window": "Hann",
                "model": "jupiter",
                "num_model_parameters": 5,
                "simulation_backend": CANONICAL_NOISEGRID_BACKEND,
            }
        )

        with self.assertRaisesRegex(ValueError, "requires 6 parameter regressors"):
            build_enca_fft_cnn_summary_stats(
                run_dir=run_dir,
                expected_model="jupiter",
            )

    def test_sabc_rejects_legacy_fourier_cnn_backend(self):
        run_dir = self._make_run(
            {
                "len_timeseries": 8,
                "ndims_latent": 5,
                "num_fft_components": 4,
                "model": "original",
                "num_model_parameters": 5,
            }
        )

        with self.assertRaisesRegex(ValueError, "Retrain in a fresh directory"):
            build_enca_fft_cnn_summary_stats(
                run_dir=run_dir,
                expected_model="original",
            )

    def test_sabc_rejects_legacy_mlp_backend(self):
        run_dir = self._make_run(
            {
                "len_timeseries": 8,
                "ndims_latent": 5,
                "num_fft_components": 4,
                "representation_mode": "fourier_amplitude",
                "model": "original",
                "num_model_parameters": 5,
            }
        )

        with self.assertRaisesRegex(ValueError, "Retrain in a fresh directory"):
            build_mlp_summary_stats(
                run_dir=run_dir,
                expected_model="original",
            )

    def test_sabc_requires_hann_for_canonical_fourier_cnn(self):
        run_dir = self._make_run(
            {
                "len_timeseries": 8,
                "ndims_latent": 5,
                "num_fft_components": 4,
                "window": "none",
                "model": "original",
                "num_model_parameters": 5,
                "simulation_backend": CANONICAL_NOISEGRID_BACKEND,
            }
        )

        with self.assertRaisesRegex(ValueError, "mandatory Hann preprocessing"):
            build_enca_fft_cnn_summary_stats(
                run_dir=run_dir,
                expected_model="original",
            )


if __name__ == "__main__":
    unittest.main()
