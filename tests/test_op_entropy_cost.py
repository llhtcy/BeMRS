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
