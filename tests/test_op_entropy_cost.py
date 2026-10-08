"""Feature contract and old preference equivalence, no search/evaluator."""
from pathlib import Path
import os
import unittest
from unittest.mock import patch
import numpy as np
from bemrs.op_aco_behavior import OPACOBehaviorEmbedder
from bemrs.op_entropy_cost_behavior import OPEntropyCostEmbedder
from bemrs.predictor import build_behavior_embedder

ROOT=Path(__file__).resolve().parents[1]


class EntropyCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=OPACOBehaviorEmbedder(ROOT/'problems/op_aco/dataset/train50_dataset.npz',
                                      n_matrices=2,n_nodes=50,seed=1111,instance_pooling='mean')
        cls.two=OPEntropyCostEmbedder(cls.base,'entropy_time2')
        cls.five=OPEntropyCostEmbedder(cls.base,'entropy_time_preferences5')

    def test_old_means_preserved_and_two_has_no_preferences(self):
        code='def heuristics(prize,distance,maxlen):\n    return prize[None,:]/(distance+1e-9)'
        old=self.base.encode_single(code)
        two,five=self.two.encode_single(code),self.five.encode_single(code)
        self.assertEqual(two.shape,(2,))
        self.assertEqual(five.shape,(5,))
        self.assertAlmostEqual(two[0],five[0],places=6)
        np.testing.assert_allclose(five[[0,2,3,4]],old[[10,0,2,4]],atol=1e-6)
        self.assertGreater(two[1],0)
        self.assertGreater(five[1],0)

    def test_time_ratio_and_raw_seconds_are_separate(self):
        code='def heuristics(prize,distance,maxlen):\n    return prize[None,:]/(distance+1e-9)'
        for encoder in (self.two,self.five):
            feature=encoder.encode_single(code)
            diagnostics=encoder.last_diagnostics
            self.assertEqual(diagnostics['status'],'ok')
            seconds=diagnostics['feature_extraction_time_s']
            self.assertGreater(seconds,0)
            self.assertAlmostEqual(float(feature[1]),seconds/self.base.encode_timeout,places=7)
            self.assertEqual(encoder.feature_names[1],'feature_extraction_time_ratio')
            self.assertEqual(diagnostics['time_normalization_budget_s'],self.base.encode_timeout)
            self.assertIn('time_ratio_v2',encoder.extractor_version)
            self.assertNotIn('feature_extraction_time_s',diagnostics['features'])

    def test_time_ratio_boundaries_reuse_existing_deadline(self):
        for budget in (1.0,5.0,10.0):
            with patch.object(self.base,'encode_timeout',budget):
                encoder=OPEntropyCostEmbedder(self.base,'entropy_time_preferences5')
            for elapsed,expected in ((0.0,0.0),(budget/2,0.5),(budget,1.0),
                                     (budget*2,1.0),(-budget,0.0)):
                self.assertEqual(encoder._normalized_extraction_time(elapsed),expected)

    def test_invalid_normalization_deadline_is_rejected(self):
        for budget in (0.0,-1.0,float('nan'),float('inf')):
            with patch.object(self.base,'encode_timeout',budget):
                with self.assertRaisesRegex(ValueError,'positive finite encode_timeout'):
                    OPEntropyCostEmbedder(self.base,'entropy_time_preferences5')

    def test_all_successful_coordinates_are_in_unit_interval(self):
        codes=(
            'def heuristics(p,d,b):\n    return np.ones_like(d)',
            'def heuristics(p,d,b):\n    return p[None,:]/(d+1e-9)',
            'def heuristics(p,d,b):\n    return np.exp(-100*d)',
            'def heuristics(p,d,b):\n    return -np.ones_like(d)',
        )
        for encoder in (self.two,self.five):
            for code in codes:
                feature=encoder.encode_single(code)
                self.assertEqual(encoder.last_diagnostics['status'],'ok')
                self.assertTrue(np.isfinite(feature).all())
                self.assertTrue(((feature>=0)&(feature<=1)).all(),feature)

    def test_no_perturbation_and_same_call_count(self):
        for encoder in (self.two,self.five):
            fn=lambda p,d,b: np.ones_like(d)
            from unittest.mock import Mock
            mock=Mock(side_effect=fn)
            with patch.object(self.base,'_compile_heuristic',return_value=mock):
                result=encoder.encode_single('uniform')
            self.assertEqual(mock.call_count,self.base.n_matrices)
            self.assertAlmostEqual(result[0],1,places=6)

    def test_invalid_and_timeout_fallback(self):
        for encoder in (self.two,self.five):
            np.testing.assert_array_equal(encoder.encode_single('def heuristics(p,d,b):\n    return np.zeros(2)'),
                                          [-1]*encoder.output_dim)
            self.assertEqual(encoder.last_diagnostics['status'],'failed')
        with patch.object(self.base,'heuristic_timeout',.01):
            np.testing.assert_array_equal(self.two.encode_single('def heuristics(p,d,b):\n    while True: pass'),[-1]*2)

    def test_factory_and_batch_shape(self):
        for group,dim in [('entropy_time2',2),('entropy_time_preferences5',5)]:
            with patch.dict(os.environ,{'BEMRS_FEATURE_GROUP':group}):
                encoder=build_behavior_embedder('op_aco',50,ROOT,behavior_matrices=1)
            self.assertEqual(encoder.output_dim,dim)
            self.assertEqual(encoder.encode([]).shape,(0,dim))


if __name__=='__main__':
    unittest.main()
