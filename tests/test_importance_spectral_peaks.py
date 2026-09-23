"""The filter must use the peak score saved by inference, including its weights."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import importance_sampling_filter as filtering
from process_fdist import make_process_f_dist
from spectral_peak_stats import build_spectral_peak_stats


def toy_simulator(theta, y, rng):
    time = np.arange(y.shape[1])[None, :]
    y[:] = (
        20 + 10 * np.cos(2 * np.pi * time / (8 + theta[:, 0:1]))
        + 4 * np.cos(2 * np.pi * time / (60 + 8 * theta[:, 1:2]))
        + rng.normal(0, 0.5, size=y.shape)
    )


class ImportanceSpectralPeakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        observed = np.loadtxt(root / 'data/silso_SN_y_202601.csv', delimiter=',')[49:-6, 1]
        cls.adapter = build_spectral_peak_stats(observed)

    def result(self, adapter=None):
        adapter = self.adapter if adapter is None else adapter
        distance = make_process_f_dist(
            n_samples=271, ss_obs=adapter.ss_obs, stats_fn=adapter.batch,
            simulator=toy_simulator, seed=456, n_workers=1,
        )
        population = np.random.default_rng(18).uniform(1, 10, size=(24, 6))
        return SimpleNamespace(
            config=SimpleNamespace(f_dist=distance), population=population,
            rho=distance(population),
        )

    def arguments(self, summary='spectral_peaks'):
        argv = ['filter', '--run-name', 'obsSN_single_jupiter_spectralpeaks',
                '--summary-stats', summary, '--n-repeats', '3', '--n-workers', '1', '--seed', '123']
        with patch('sys.argv', argv):
            return filtering._parse_args()

    def save_result(self, directory, result):
        run = 'obsSN_single_jupiter_spectralpeaks'
        with (directory / f'SABCresult_{run}.pkl').open('wb') as handle:
            pickle.dump(result, handle)
        np.savetxt(directory / f'post_population_{run}.csv', result.population, delimiter=',')
        return run

    def test_cli_accepts_spectral_peaks_without_training_or_fourier_range(self):
        args = self.arguments()
        self.assertEqual(args.summary_stats, 'spectral_peaks')
        self.assertIsNone(args.train_run_dir)
        with patch('sys.argv', ['filter', '--summary-stats', 'spectral_peaks', '--fourier-range', '1:6:120']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                filtering._parse_args()
        with patch('sys.argv', ['filter']):
            self.assertEqual(filtering._parse_args().summary_stats, 'fft')

    def test_saved_score_rejects_wrong_backend_in_both_directions(self):
        result = self.result()
        with self.assertRaisesRegex(ValueError, '--summary-stats spectral_peaks'):
            filtering._saved_spectral_stats(result, 'fft')
        other_result = SimpleNamespace(config=SimpleNamespace(f_dist=SimpleNamespace(stats_fn=None)))
        with self.assertRaisesRegex(ValueError, 'original spectral-peaks scoring settings'):
            filtering._saved_spectral_stats(other_result, 'spectral_peaks')
        self.assertIsNone(filtering._saved_spectral_stats(other_result, 'fft'))
        result.config.f_dist.ss_obs = np.ones(1)
        with self.assertRaisesRegex(ValueError, 'against zero'):
            filtering._saved_spectral_stats(result, 'spectral_peaks')

    def test_builder_uses_saved_settings_and_worker_stats_without_reloading_data(self):
        custom = replace(self.adapter, config=replace(
            self.adapter.config, weights=(6., 4., 1., 1., 1., 1.),
            gleissberg_target_frequency=self.adapter.config.gleissberg_target_frequency + 0.0001,
        ))
        saved = pickle.loads(pickle.dumps(self.result(custom)))
        adapter = filtering._saved_spectral_stats(saved, 'spectral_peaks')
        with (
            patch.object(filtering, 'load_dataset') as load_data,
            patch.object(filtering, 'build_simulator', return_value=toy_simulator) as simulator,
            patch.object(filtering, 'make_process_f_dist', wraps=make_process_f_dist) as worker_stats,
            patch.object(filtering, 'make_process_sim_then_stats_f_dist') as parent_stats,
        ):
            distance = filtering._build_reconstruction_f_dist(
                'obsSN', model='jupiter', n_workers=4, seed=123,
                synthetic_data_path=None, summary_stats='spectral_peaks', fourier_range=None,
                train_run_dir=None, enca_checkpoint_basename='model_best_ckpt', spectral_stats=adapter,
            )
        load_data.assert_not_called()
        parent_stats.assert_not_called()
        worker_stats.assert_called_once()
        simulator.assert_called_once_with(Twarmup=200, Tobs=271, model='jupiter', threaded=False)
        self.assertEqual(distance.stats_fn.__self__.config, custom.config)
        np.testing.assert_array_equal(distance.ss_obs, [0.])

    def test_filter_averages_scalar_losses_and_writes_the_usual_two_files(self):
        result = self.result()
        expected_reconstructed = np.zeros((len(result.population), 1))
        rng = np.random.default_rng(123)
        for _ in range(3):
            y = np.empty((len(result.population), 271))
            toy_simulator(result.population, y, rng)
            scores = np.empty((len(y), 1))
            self.adapter.batch(y, scores)
            expected_reconstructed += scores / 3

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run = self.save_result(directory, result)
            with (
                patch.object(filtering, 'OUTPUT_DIR', directory),
                patch.object(filtering, 'build_simulator', return_value=toy_simulator),
                patch.object(filtering, '_filter_from_norms', wraps=filtering._filter_from_norms) as apply_filter,
                redirect_stdout(io.StringIO()),
            ):
                filtering._process_one_run(self.arguments(), run)
            np.testing.assert_allclose(apply_filter.call_args_list[0].args[1], expected_reconstructed[:, 0])
            np.testing.assert_array_equal(apply_filter.call_args_list[1].args[1], result.rho[:, 0])
            for method in ('reconst', 'sabc'):
                indices = np.atleast_1d(np.loadtxt(directory / f'kept_ind_{method}_{run}.csv', dtype=int))
                self.assertTrue(np.all((indices >= 0) & (indices < len(result.population))))
                self.assertEqual(len(indices), len(np.unique(indices)))
            self.assertEqual(len(list(directory.glob('kept_ind_*.csv'))), 2)

    def test_wrong_backend_stops_before_resimulation_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run = self.save_result(directory, self.result())
            with (
                patch.object(filtering, 'OUTPUT_DIR', directory),
                patch.object(filtering, '_reconstruct_rho') as reconstruct,
                self.assertRaisesRegex(ValueError, '--summary-stats spectral_peaks'),
            ):
                filtering._process_one_run(self.arguments('fft'), run)
            reconstruct.assert_not_called()
            self.assertEqual(list(directory.glob('kept_ind_*.csv')), [])


if __name__ == '__main__':
    unittest.main()
