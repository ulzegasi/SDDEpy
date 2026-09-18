"""Selection must preserve encoder width and survive saved/worker adapters."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import enca_summary_stats as stats


class MlpSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'hyper_parameters.json').write_text(json.dumps({
            'len_timeseries': 8, 'ndims_latent': 6,
            'representation_mode': 'fourier_amplitude', 'num_fft_components': 4,
        }))
        (self.root / 'model_best_ckpt-1.index').touch()

    def test_observed_batch_default_and_pickle(self):
        full = stats.build_mlp_summary_stats(run_dir=self.root)
        selected = stats.build_mlp_summary_stats(run_dir=self.root, use_first_stats=5)
        selected = pickle.loads(pickle.dumps(selected))
        self.assertEqual(selected.config.ndims_latent, 6)
        self.assertEqual(selected.config.mlp_use_first_stats, 5)
        encoder = Mock(side_effect=lambda x, training: Mock(
            numpy=lambda: np.tile(np.arange(6.), (len(x), 1))))
        with patch.object(stats, '_load_encoder', return_value=encoder):
            np.testing.assert_array_equal(full.observed(np.ones(8)), np.arange(6.))
            np.testing.assert_array_equal(selected.observed(np.ones(8)), np.arange(5.))
            out = np.empty((3, 5))
            selected.batch(np.ones((3, 8)), out)
            np.testing.assert_array_equal(out, np.tile(np.arange(5.), (3, 1)))
            all_explicit = stats.build_mlp_summary_stats(run_dir=self.root, use_first_stats=6)
            np.testing.assert_array_equal(all_explicit.observed(np.ones(8)), full.observed(np.ones(8)))

    def test_invalid_selection(self):
        for value in (0, -1, 7, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'encoder width'):
                stats.build_mlp_summary_stats(run_dir=self.root, use_first_stats=value)

    def test_old_pickled_config_defaults_to_all(self):
        adapter = stats.build_mlp_summary_stats(run_dir=self.root)
        object.__delattr__(adapter.config, 'mlp_use_first_stats')
        restored = pickle.loads(pickle.dumps(adapter))
        self.assertIsNone(restored.config.mlp_use_first_stats)

    def test_cli_requires_mlp_and_positive_count(self):
        import SABC_SolarDynamo as inference
        for argv in (['prog', '--mlp-use-first-stats', '5'],
                     ['prog', '--summary-stats', 'mlp', '--train-run-dir', '.', '--mlp-use-first-stats', '0']):
            with patch('sys.argv', argv), self.assertRaises(SystemExit):
                inference._parse_args()
        with patch('sys.argv', ['prog', '--summary-stats', 'mlp', '--train-run-dir', '.', '--mlp-use-first-stats', '5']):
            self.assertEqual(inference._parse_args().mlp_use_first_stats, 5)
