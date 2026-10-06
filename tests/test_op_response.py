"""Offline tests: execute synthetic/seed heuristics, never the real evaluator."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from bemrs.op_aco_behavior import OPACOBehaviorEmbedder
from bemrs.op_response_behavior import OPResponseBehaviorEmbedder, perturb_state
from bemrs.predictor import build_behavior_embedder

ROOT = Path(__file__).resolve().parents[1]


class ResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = OPACOBehaviorEmbedder(ROOT/'problems/op_aco/dataset/train50_dataset.npz',
                    n_matrices=2, n_nodes=50, probes_per_stage=4, seed=1111,
                    heuristic_timeout=.5, encode_timeout=5, instance_pooling='mean')
        cls.encoder = OPResponseBehaviorEmbedder(cls.base)

    def test_uniform_is_max_entropy_and_stable(self):
        feature = self.encoder.encode_single('def heuristics(prize, distance, max_len):\n    return np.ones_like(distance)')
        self.assertEqual(feature.shape,(3,))
        self.assertAlmostEqual(feature[0],1,places=6)
        self.assertGreater(feature[1],0)
        self.assertAlmostEqual(feature[2],1,places=6)
        self.assertGreater(self.encoder.last_diagnostics['valid_pairs'],0)

    def test_joint_response_reproducible_except_clock(self):
        code = 'def heuristics(prize, distance, max_len):\n    return prize[None,:] / (distance + 1e-6)'
        a, b = self.encoder.encode_single(code), self.encoder.encode_single(code)
        np.testing.assert_array_equal(a[[0,2]],b[[0,2]])
        self.assertTrue(0 <= a[0] <= 1)
        self.assertTrue(0 <= a[2] < 1)

    def test_randomized_candidate_uses_paired_common_rng(self):
        code = 'def heuristics(prize, distance, max_len):\n    return np.random.random(distance.shape)'
        a, b = self.encoder.encode_single(code), self.encoder.encode_single(code)
        np.testing.assert_array_equal(a[[0,2]],b[[0,2]])
        self.assertAlmostEqual(a[2],1,places=6)

    def test_geometry_and_all_continuous_inputs(self):
        instance = self.base._instances[0]
        state = self.base._state_banks[0][0]
        trace = self.base._reference_trace(instance,state['reference_mode'],self.base.seed)
        # Use the initial route state so path length is exactly known.
        state = trace[0]
        noise = dict(prize=np.full(50,.5),coordinates=np.full((50,2),.5),budget=.5)
        changed, cs = perturb_state(instance,state,[0],noise)
        np.testing.assert_allclose(changed['distance'],changed['distance'].T)
        np.testing.assert_array_equal(changed['coordinate'][0],instance['coordinate'][0])
        np.testing.assert_allclose(changed['prize'],instance['prize']*1.025)
        self.assertAlmostEqual(changed['max_len'],instance['max_len']*1.025)
        np.testing.assert_array_equal(cs['visited'],state['visited'])
        same, _ = perturb_state(instance,state,[0],noise,fraction=0)
        np.testing.assert_allclose(same['distance'],instance['distance'])

    def test_failures_are_not_successful_features(self):
        np.testing.assert_array_equal(self.encoder.encode_single('def heuristics(p,d,b):\n    return np.zeros(2)'),[-1]*3)
        old = self.base.heuristic_timeout
        try:
            self.base.heuristic_timeout = .01
            np.testing.assert_array_equal(self.encoder.encode_single('def heuristics(p,d,b):\n    while True: pass'),[-1]*3)
        finally:
            self.base.heuristic_timeout = old

    def test_factory_selects_actual_three_features(self):
        with patch.dict(os.environ, {'BEMRS_FEATURE_GROUP':'response3'}):
            encoder = build_behavior_embedder('op_aco',50,ROOT,behavior_matrices=1)
        self.assertIsInstance(encoder,OPResponseBehaviorEmbedder)
        self.assertEqual(encoder.output_dim,3)
        with patch.dict(os.environ, {'BEMRS_FEATURE_GROUP':'mean'}):
            encoder = build_behavior_embedder('op_aco',50,ROOT,behavior_matrices=1)
        self.assertEqual(encoder.output_dim,6)


if __name__ == '__main__':
    unittest.main()
