from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import SABC_SolarDynamo as inference
import importance_sampling_filter as importance_filter
import solar_dynamo_sabc_setup as setup


class PriorSelectionTests(unittest.TestCase):
    def test_original_prior_has_five_parameters(self):
        lower, upper = inference._prior_bounds("original")
        self.assertEqual(lower.shape, (5,))
        self.assertEqual(upper.shape, (5,))

    def test_jupiter_prior_has_epsilon_only(self):
        lower, upper = inference._prior_bounds("jupiter")
        self.assertEqual(lower.shape, (6,))
        self.assertEqual(upper.shape, (6,))
        self.assertEqual(lower[-1], 0.0)
        self.assertEqual(upper[-1], 0.1)

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            inference._prior_bounds("unknown")


class SimulatorSelectionTests(unittest.TestCase):
    def test_original_simulator_routes_to_original_batch_function(self):
        theta = np.ones((2, 5), dtype=float)
        y = np.empty((2, 3), dtype=float)
        expected = np.full((2, 3), 1.0)
        expected_rng = np.random.default_rng(1)
        expected_noise = expected_rng.standard_normal(
            size=(theta.shape[0], int(round((2 + 3) / setup.SDDE_DT)))
        )

        with (
            patch.object(
                setup,
                "sn_from_noise_batch_original",
                return_value=expected,
            ) as original,
            patch.object(setup, "sn_from_noise_batch_jupiter") as jupiter,
        ):
            setup.simulator_batch(
                theta,
                y,
                np.random.default_rng(1),
                Twarmup=2,
                Tobs=3,
                model="original",
            )

        np.testing.assert_array_equal(y, expected)
        original.assert_called_once()
        np.testing.assert_allclose(original.call_args.args[1], expected_noise)
        jupiter.assert_not_called()

    def test_jupiter_simulator_routes_to_jupiter_batch_function(self):
        theta = np.ones((2, 6), dtype=float)
        y = np.empty((2, 3), dtype=float)
        expected = np.full((2, 3), 2.0)
        expected_rng = np.random.default_rng(1)
        expected_noise = expected_rng.standard_normal(
            size=(theta.shape[0], int(round((2 + 3) / setup.SDDE_DT)))
        )
        expected_phases = expected_rng.uniform(0.0, 2.0 * np.pi, size=theta.shape[0])

        with (
            patch.object(setup, "sn_from_noise_batch_original") as original,
            patch.object(
                setup,
                "sn_from_noise_batch_jupiter",
                return_value=expected,
            ) as jupiter,
        ):
            setup.simulator_batch(
                theta,
                y,
                np.random.default_rng(1),
                Twarmup=2,
                Tobs=3,
                model="jupiter",
            )

        np.testing.assert_array_equal(y, expected)
        original.assert_not_called()
        jupiter.assert_called_once()
        theta_simulator = jupiter.call_args.args[0]
        noise_simulator = jupiter.call_args.args[1]
        self.assertEqual(theta_simulator.shape, (2, 7))
        np.testing.assert_array_equal(theta_simulator[:, :6], theta)
        np.testing.assert_allclose(theta_simulator[:, 6], expected_phases)
        np.testing.assert_allclose(noise_simulator, expected_noise)

    def test_simulator_rejects_wrong_parameter_dimension(self):
        with self.assertRaisesRegex(ValueError, "expects 6 inference parameters"):
            setup.simulator_batch(
                np.ones((2, 7), dtype=float),
                np.empty((2, 3), dtype=float),
                np.random.default_rng(1),
                Twarmup=2,
                Tobs=3,
                model="jupiter",
            )

    def test_build_simulator_preserves_model_for_spawn_pickling(self):
        simulator = setup.build_simulator(Twarmup=200, Tobs=271, model="jupiter")
        self.assertEqual(simulator.keywords["model"], "jupiter")
        self.assertEqual(simulator.keywords["Twarmup"], 200)
        self.assertEqual(simulator.keywords["Tobs"], 271)


class RunNamingTests(unittest.TestCase):
    def test_original_default_run_name_is_unchanged(self):
        self.assertEqual(
            inference._default_run_name("obsSN", "single_eps", "fft"),
            "obsSN_single_eps",
        )

    def test_jupiter_default_run_name_is_distinct(self):
        self.assertEqual(
            inference._default_run_name("obsSN", "single_eps", "fft", "jupiter"),
            "obsSN_single_eps_jupiter",
        )


class ImportanceFilterSelectionTests(unittest.TestCase):
    def test_filter_recognizes_original_population(self):
        self.assertEqual(
            importance_filter._model_from_population(np.ones((10, 5))),
            "original",
        )

    def test_filter_recognizes_jupiter_population(self):
        self.assertEqual(
            importance_filter._model_from_population(np.ones((10, 6))),
            "jupiter",
        )


if __name__ == "__main__":
    unittest.main()
