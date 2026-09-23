"""The hybrid must preserve CNN coordinates and use the correct FFT bin."""
import json
import io
import pickle
import shutil
import tempfile
import unittest
from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import enca_summary_stats as neural
import hybrid_summary_stats as hybrid
import importance_sampling_filter as filtering
import SABC_SolarDynamo as inference


def toy_simulator(theta, y, rng):
    time = np.arange(y.shape[1])[None, :]
    y[:] = (20 + theta[:, 4:5] * np.cos(2 * np.pi * time / (8 + theta[:, :1]))
            + rng.normal(0, 1, size=y.shape))


class HybridTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "training"
        self.run.mkdir()
        self.hp = {
            "len_timeseries": 271, "ndims_latent": 5,
            "num_fft_components": 100, "model": "original",
            "num_model_parameters": 5, "window": "Hann",
            "simulation_backend": neural.CANONICAL_NOISEGRID_BACKEND,
        }
        self.write_hp()
        self.write_checkpoint(10)

    def write_hp(self):
        (self.run / "hyper_parameters.json").write_text(json.dumps(self.hp))

    def write_checkpoint(self, step):
        (self.run / f"model_best_ckpt-{step}.index").write_bytes(b"index")
        (self.run / f"model_best_ckpt-{step}.data-00000-of-00001").write_bytes(str(step).encode())

    def build(self):
        return hybrid.build_hybrid_summary_stats(run_dir=self.run, expected_tobs=271)

    def test_matches_original_coordinates_and_independent_inverse_fft(self):
        adapter = pickle.loads(pickle.dumps(self.build()))
        self.assertEqual(adapter.config.fourier_bin, 23)
        self.assertEqual(adapter.metadata()["fourier_index_julia"], 24)
        self.assertAlmostEqual(adapter.metadata()["bin_period_years"], 271 / 23)
        rng = np.random.default_rng(12)
        records = rng.normal(size=(3, 271))
        # A strong bin-23 tone makes an accidental bin-22 selection observable.
        records[0] += 10 * np.cos(2 * np.pi * 23 * np.arange(271) / 271)
        encoder = Mock(side_effect=lambda x, training: Mock(
            numpy=lambda: np.stack([x[:, j, 0] for j in range(5)], axis=1)))
        base = neural.EncaSummaryStats(adapter.config.encoder)
        out = np.empty((3, 6))
        with patch.object(neural, "_load_encoder", return_value=encoder):
            adapter.batch(records, out)
            for i, record in enumerate(records):
                np.testing.assert_array_equal(out[i, :5], base.observed(record))
                np.testing.assert_array_equal(out[i], adapter.observed(record))
        expected = np.abs(np.fft.ifft(records * np.hanning(271), axis=1))[:, 23]
        np.testing.assert_allclose(out[:, 5], expected, rtol=1e-13)
        wrong_bin = np.abs(np.fft.ifft(records[0] * np.hanning(271)))[22]
        self.assertGreater(abs(out[0, 5] - wrong_bin), 0.5)
        with self.assertRaisesRegex(ValueError, "exactly six"):
            adapter.batch(records, np.empty((3, 5)))
        with self.assertRaisesRegex(ValueError, "time-series array"):
            adapter.observed(np.ones(270))

    def test_rejects_wrong_training_and_preserves_normal_cnn_model_guard(self):
        with self.assertRaisesRegex(ValueError, "matching training"):
            neural.build_enca_fft_cnn_summary_stats(run_dir=self.run, expected_model="jupiter")
        for changes, message in [
            ({"ndims_latent": 6}, "z=5"),
            ({"model": "jupiter", "num_model_parameters": 6}, "matching training"),
            ({"window": "none"}, "mandatory Hann"),
            ({"simulation_backend": "legacy"}, "Retrain"),
            ({"num_model_parameters": 6}, "requires 5"),
            ({"len_timeseries": 300}, "Tobs=271"),
        ]:
            old = self.hp.copy()
            self.hp.update(changes)
            self.write_hp()
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.build()
            self.hp = old
        self.write_hp()
        with self.assertRaisesRegex(ValueError, "requires --model jupiter"):
            hybrid.build_hybrid_summary_stats(run_dir=self.run, expected_model="original")

    def test_checkpoint_is_pinned_relocatable_and_checked(self):
        saved = self.build()
        self.write_checkpoint(20)
        relocated = self.root / "local-copy"
        shutil.copytree(self.run, relocated)
        rebuilt = hybrid.rebuild_hybrid_summary_stats(
            saved, run_dir=relocated, expected_tobs=271, expected_model="jupiter")
        self.assertTrue(rebuilt.config.encoder.checkpoint_basename.endswith("-10.index"))
        hybrid.validate_hybrid_resume(saved.batch, rebuilt.batch)
        with self.assertRaisesRegex(ValueError, "checkpoint or FFT settings"):
            hybrid.validate_hybrid_resume(saved.batch, self.build().batch)
        changed = hybrid.HybridSummaryStats(replace(saved.config, fourier_bin=22))
        with self.assertRaisesRegex(ValueError, "checkpoint or FFT settings"):
            hybrid.validate_hybrid_resume(saved.batch, changed.batch)
        with self.assertRaisesRegex(ValueError, "switch into or out"):
            hybrid.validate_hybrid_resume(saved.batch, None)
        with self.assertRaisesRegex(ValueError, "switch into or out"):
            hybrid.validate_hybrid_resume(None, saved.batch)
        (relocated / "model_best_ckpt-10.data-00000-of-00001").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint or FFT settings"):
            hybrid.rebuild_hybrid_summary_stats(
                saved, run_dir=relocated, expected_tobs=271, expected_model="jupiter")
        hybrid.validate_hybrid_resume(None, None)

    def test_filter_uses_saved_hybrid_with_jupiter_and_parent_statistics(self):
        saved = self.build()
        observations = np.ones(271)
        encoder = Mock(side_effect=lambda x, training: Mock(numpy=lambda: np.ones((len(x), 5))))
        with patch.object(neural, "_load_encoder", return_value=encoder):
            ss_obs = saved.observed(observations)
            result = SimpleNamespace(config=SimpleNamespace(f_dist=SimpleNamespace(
                stats_fn=saved.batch, ss_obs=ss_obs, distance="abs")))
            self.assertIs(filtering._saved_hybrid_stats(result, hybrid.HYBRID_MODE), saved)
            for mode in ("fft", "enca_fft_cnn"):
                with self.assertRaisesRegex(ValueError, "This saved run uses"):
                    filtering._saved_hybrid_stats(result, mode)
            with self.assertRaisesRegex(ValueError, "does not contain"):
                filtering._saved_hybrid_stats(SimpleNamespace(), hybrid.HYBRID_MODE)
            with (
                patch.object(filtering, "load_dataset", return_value=(np.arange(271), observations, 271)),
                patch.object(filtering, "build_simulator") as simulator,
                patch.object(filtering, "make_process_sim_then_stats_f_dist") as factory,
                patch.object(filtering, "make_process_f_dist") as other_factory,
            ):
                filtering._build_reconstruction_f_dist(
                    "obsSN", model="jupiter", n_workers=4, seed=123,
                    synthetic_data_path=None, summary_stats=hybrid.HYBRID_MODE,
                    fourier_range=None, train_run_dir=str(self.run),
                    enca_checkpoint_basename="model_best_ckpt",
                    hybrid_stats=saved, hybrid_ss_obs=ss_obs,
                )
                simulator.assert_called_once_with(Twarmup=200, Tobs=271, model="jupiter", threaded=False)
                factory.assert_called_once()
                other_factory.assert_not_called()
                np.testing.assert_array_equal(factory.call_args.kwargs["ss_obs"], ss_obs)
                hybrid.validate_hybrid_resume(saved.batch, factory.call_args.kwargs["stats_fn"])

    def test_cli_and_annual_sampling_requirements(self):
        args = ["prog", "--summary-stats", hybrid.HYBRID_MODE, "--train-run-dir", str(self.run)]
        with patch("sys.argv", args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            inference._parse_args()
        with patch("sys.argv", args + ["--model", "jupiter"]):
            self.assertEqual(inference._parse_args().summary_stats, hybrid.HYBRID_MODE)
        with patch("sys.argv", args + ["--model", "jupiter", "--fourier-range", "24"]), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            inference._parse_args()
        with patch("sys.argv", args):
            self.assertEqual(filtering._parse_args().summary_stats, hybrid.HYBRID_MODE)
        hybrid.validate_hybrid_years(1750.5 + np.arange(271))
        with self.assertRaisesRegex(ValueError, "annual"):
            hybrid.validate_hybrid_years(np.arange(271) * 2)

    def test_driver_runs_sabc_with_six_statistics_and_saves_hybrid_adapter(self):
        api = list(inference._import_sabc_package())
        make_config, real_sabc = api[1], api[4]

        def small_config(**kwargs):
            kwargs.update(n_particles=24, show_progressbar=False, show_checkpoint=None)
            return make_config(**kwargs)

        def small_sabc(config, *, n_simulation):
            self.assertEqual(config.f_dist.ss_obs.size, 6)
            self.assertEqual(config.prior.lower.size, 6)
            return real_sabc(config, n_simulation=72)

        api[1], api[4] = small_config, small_sabc
        outdir = self.root / "output"
        argv = ["prog", "--model", "jupiter", "--summary-stats", hybrid.HYBRID_MODE,
                "--train-run-dir", str(self.run), "--run-name", "hybrid_test",
                "--n-workers", "1", "--simulator-seed", "123",
                "--algorithm-seed", "18", "--proposal-seed", "22"]
        encoder = Mock(side_effect=lambda x, training: Mock(
            numpy=lambda: np.stack([x[:, j, 0] for j in range(5)], axis=1)))
        with (
            patch("sys.argv", argv),
            patch.object(inference, "init_julia"),
            patch.object(inference, "build_simulator", return_value=toy_simulator) as simulator,
            patch.object(inference, "_import_sabc_package", return_value=tuple(api)),
            patch.object(inference, "LOCAL_OUT_DIR", outdir),
            patch.object(neural, "_load_encoder", return_value=encoder),
            redirect_stdout(io.StringIO()),
        ):
            inference.main()
        self.assertEqual(simulator.call_args.kwargs["model"], "jupiter")
        result = api[2](outdir / "SABCresult_hybrid_test.pkl")
        self.assertEqual(result.population.shape, (24, 6))
        self.assertEqual(result.rho.shape, (24, 6))
        self.assertIsInstance(filtering._saved_hybrid_stats(result, hybrid.HYBRID_MODE), hybrid.HybridSummaryStats)
        self.assertTrue(np.isfinite(result.rho).all())
        self.assertEqual(len(list(outdir.iterdir())), 5)


if __name__ == "__main__":
    unittest.main()
