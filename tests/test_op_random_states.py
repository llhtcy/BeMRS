import unittest
from pathlib import Path
import importlib.util
import numpy as np
from bemrs.op_aco_behavior import OPACOBehaviorEmbedder

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('random_op_states',ROOT/'experiments/op_entropy_cost/random_states.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class RandomStatesTests(unittest.TestCase):
    def test_stratified_coverage_and_decision_deduplication(self):
        for seed in (1111,2222,3333):
            encoder=OPACOBehaviorEmbedder(ROOT/'problems/op_aco/dataset/train50_dataset.npz',n_matrices=5,seed=seed)
            for i,instance in enumerate(encoder._instances):
                a=module.make_stratified_states(encoder,instance,seed+i*1009)
                b=module.make_stratified_states(encoder,instance,seed+i*1009)
                self.assertEqual([s['decision_key'] for s in a],[s['decision_key'] for s in b])
                self.assertEqual(len({s['decision_key'] for s in a}),12)
                self.assertEqual([sum(s['probe_layer']==j for s in a) for j in range(3)],[4,4,4])
                for s in a:
                    self.assertEqual(min(2,int(s['committed_ratio']*3)),s['probe_layer'])
                    self.assertGreaterEqual(len(s['decision_key'][1]),2)

    def test_reproducible_unique_reachable_and_seed_changes(self):
        encoder=OPACOBehaviorEmbedder(ROOT/'problems/op_aco/dataset/train50_dataset.npz',n_matrices=1,seed=1111)
        instance=encoder._instances[0]
        a=module.make_random_states(encoder,instance,1111)
        b=module.make_random_states(encoder,instance,1111)
        c=module.make_random_states(encoder,instance,2222)
        def signature(states):
            return [(s['current'],s['visited'].tobytes(),s['travel_distance']) for s in states]
        self.assertEqual(len(a),12)
        self.assertEqual(len(set(signature(a))),12)
        self.assertEqual(signature(a),signature(b))
        self.assertNotEqual(signature(a),signature(c))
        for s in a:
            self.assertTrue(s['visited'][0] and s['visited'][s['current']])
            self.assertGreaterEqual(len(encoder._feasible_candidates(instance,s['current'],s['visited'],s['travel_distance'])),2)
            self.assertLessEqual(s['committed_ratio'],1+1e-12)
            self.assertGreaterEqual(s['travel_distance'],0)


if __name__=='__main__':
    unittest.main()
