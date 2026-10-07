"""CVRP5D contracts and reachable random probes; no LLM/real evaluator."""
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_dir
import numpy as np

from bemrs.cvrp_aco_behavior import CVRPACOBehaviorEmbedder
from bemrs.cvrp_entropy_cost_behavior import CVRPEntropyCostEmbedder, NAMES
from bemrs.predictor import build_behavior_embedder
from bemrs.runtime_config import apply_runtime_config

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'problems/cvrp_aco/dataset/train50_dataset.npy'


def make_encoder(**kwargs):
    options = dict(n_matrices=2, n_customers=50, seed=1111,
                   instance_pooling='mean', probe_mode='random_stratified', probes_per_stage=12)
    options.update(kwargs)
    return CVRPACOBehaviorEmbedder(DATASET, **options)


class CVRPProbeTests(unittest.TestCase):
    def test_balanced_unique_and_reachable_for_all_three_seeds(self):
        with patch.dict(os.environ, {}, clear=True):
            for seed in (1111, 2222, 3333):
                encoder = make_encoder(n_matrices=8, seed=seed)
                for instance, states in zip(encoder._instances, encoder._state_banks):
                    self.assertEqual(len(states), 36)
                    self.assertEqual([sum(s['probe_layer'] == k for s in states) for k in range(3)],
                                     [12, 12, 12])
                    self.assertEqual(len({s['decision_key'] for s in states}), 36)
                    demand = instance['demand']
                    for state in states:
                        served, current, used = set(), 0, 0.0
                        self.assertEqual(state['path_prefix'][0], 0)
                        for node in state['path_prefix'][1:]:
                            if node == 0:
                                self.assertNotEqual(current, 0)
                                used = 0.0
                            else:
                                self.assertNotIn(node, served)
                                self.assertLessEqual(used+demand[node], encoder.capacity+1e-12)
                                served.add(node)
                                used += demand[node]
                            current = node
                        self.assertEqual(current, state['current'])
                        self.assertAlmostEqual(used/encoder.capacity, state['used_capacity_ratio'])
                        self.assertAlmostEqual(len(served)/(len(demand)-1), state['progress'])
                        self.assertEqual(min(2, int(state['progress']*3)), state['probe_layer'])
                        self.assertEqual(set(state['unvisited']), set(range(1, len(demand)))-served)
                        feasible = tuple(int(j) for j in state['unvisited']
                                         if demand[j] <= encoder.capacity-used+1e-12)
                        self.assertGreaterEqual(len(feasible), 2)
                        actions = ((0,)+feasible) if current else feasible
                        self.assertEqual(state['decision_key'], (current, actions))

    def test_reproducible_and_different_seed_changes_bank(self):
        with patch.dict(os.environ, {}, clear=True):
            a, b, c = make_encoder(), make_encoder(), make_encoder(seed=2222)
        def signature(encoder):
            return [(s['decision_key'], s['used_capacity_ratio'], s['path_prefix'])
                    for states in encoder._state_banks for s in states]
        self.assertEqual(signature(a), signature(b))
        self.assertNotEqual(signature(a), signature(c))

    def test_shortage_fails_instead_of_duplicating_states(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'probe shortage'):
                make_encoder(n_matrices=1, n_customers=2)

    def test_legacy_bank_still_available(self):
        with patch.dict(os.environ, {}, clear=True):
            encoder = make_encoder(probe_mode='legacy', n_matrices=1)
        self.assertEqual(len(encoder._state_banks[0]), 36)
        self.assertTrue(all(s['current'] != 0 for s in encoder._state_banks[0]))


class CVRPFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {}, clear=True):
            cls.base = make_encoder()
        cls.five = CVRPEntropyCostEmbedder(cls.base)

    def test_exact_preference_formula_and_depot_conditioning(self):
        distance = np.array([[1, 4, 7, 8, 9], [4, 1, 2, 5, 8],
                             [7, 2, 1, 3, 6], [8, 5, 3, 1, 3], [9, 8, 6, 3, 1]], float)
        instance = dict(distance=distance, demand=np.array([0, 4, 8, 16, 24], float))
        heuristic = np.ones((5, 5))
        heuristic[1] = [.4, 1, .7, .2, .1]
        state = dict(current=1, unvisited=np.array([2, 3, 4]), used_capacity_ratio=.5)
        row = self.base._state_core_features(heuristic, instance, state)
        np.testing.assert_allclose(row[:3], [.2, .8, .2], atol=1e-12)
        self.assertAlmostEqual(row[4], .4/1.4)
        state['used_capacity_ratio'] = .6
        row = self.base._state_core_features(heuristic, instance, state)
        np.testing.assert_allclose(row[:3], [2/9, 7/9, 2/9], atol=1e-12)

    def test_no_depot_self_loop_at_depot(self):
        instance = self.base._instances[0]
        state = dict(current=0, unvisited=np.array([1, 2, 3]), used_capacity_ratio=0.)
        row = self.base._state_core_features(np.ones_like(instance['distance']), instance, state)
        self.assertEqual(row[4], 0)
        self.assertAlmostEqual(row[6], 1)

    def test_five_dimensions_and_original_preferences_unchanged(self):
        code = 'def heuristics(distance, demand):\n    return (1+demand[None,:])/(distance+1e-9)'
        old = self.base.encode_single(code)
        with patch.object(self.base, '_trend_features', side_effect=AssertionError('unused slopes')):
            feature = self.five.encode_single(code)
        self.assertEqual(feature.shape, (5,))
        self.assertEqual(self.five.feature_names, NAMES)
        np.testing.assert_allclose(feature[[0, 2, 3, 4]], old[[6, 0, 1, 2]], atol=1e-6)
        self.assertGreater(feature[1], 0)
        self.assertEqual(self.five.last_diagnostics['probes'], 72)

    def test_one_call_per_instance_and_uniform_entropy(self):
        function = Mock(side_effect=lambda distance, coordinates, demand, capacity: np.ones_like(distance))
        with patch.object(self.base, '_compile_heuristic', return_value=function):
            feature = self.five.encode_single('uniform')
        self.assertEqual(function.call_count, self.base.n_matrices)
        self.assertAlmostEqual(feature[0], 1, places=6)

    def test_invalid_and_timeout_fallback(self):
        for code in ('invalid !', 'def heuristics(distance):\n    return np.zeros(2)'):
            np.testing.assert_array_equal(self.five.encode_single(code), [-1]*5)
        with patch.object(self.base, 'heuristic_timeout', .01):
            np.testing.assert_array_equal(self.five.encode_single(
                'def heuristics(distance):\n    while True: pass'), [-1]*5)
        self.assertIn('CVRPBehaviorTimeout', self.five.last_diagnostics['error'])

    def test_empty_batch_and_two_dimension_protocol(self):
        self.assertEqual(self.five.encode([]).shape, (0, 5))
        two = CVRPEntropyCostEmbedder(self.base, 'entropy_time2')
        code = 'def heuristics(distance):\n    return np.ones_like(distance)'
        self.assertEqual(two.encode([code, code]).shape, (2, 2))

    def test_current_hydra_config_reaches_factory(self):
        with initialize_config_dir(config_dir=str(ROOT/'cfg'), version_base=None):
            cfg = compose(config_name='config', overrides=['problem=cvrp_aco', 'seed=3333'])
        with patch.dict(os.environ, {}, clear=True):
            runtime = apply_runtime_config(cfg)
            os.environ['BEMRS_FEATURE_GROUP'] = cfg.problem.behavior.feature_group
            encoder = build_behavior_embedder('cvrp_aco', 50, ROOT,
                                             behavior_matrices=cfg.problem.behavior.matrices)
        self.assertEqual(runtime['BEMRS_CVRP_PROBE_MODE'], 'random_stratified')
        self.assertEqual(runtime['BEMRS_CVRP_PROBES_PER_STAGE'], '12')
        self.assertEqual(encoder.output_dim, 5)
        self.assertEqual(encoder.probe_mode, 'random_stratified')
        self.assertEqual(encoder.n_matrices, 8)
        self.assertEqual(sum(map(len, encoder._state_banks)), 288)


if __name__ == '__main__':
    unittest.main()
