import unittest
from unittest.mock import Mock
from collections import Counter
import numpy as np
from bemrs.core.engine import SearchEngine
from bemrs.offspring_plan import evaluation_slots


class OffspringPlanTests(unittest.TestCase):
    def engine(self, sizes, seed=1111):
        engine = SearchEngine.__new__(SearchEngine)
        pools = {}
        index = 0
        for rid, size in enumerate(sizes):
            entries = []
            for i in range(size):
                entries.append(dict(parent=dict(code=f'code_{index}', objective=float(index)),
                                    objective=float(index), point=np.array([i*i], float)))
                index += 1
            pools[rid] = entries
        engine._region_parent_pools = Mock(return_value=pools)
        engine._active_region_operator_schedule = {}
        engine.behavior_expand_parent_counts = (1, 2)
        engine.bx_parent_selection_tau = 1.5
        engine._region_rng = np.random.default_rng(seed)
        return engine

    def test_quota_and_legacy_selection(self):
        for sizes in ([4,4,4], [2,3,7], [5,5,6], [1], [0,1,2], []):
            engine = self.engine(sizes)
            for turn in range(2):
                rows = engine._build_mixed_parent_plan([])
                for rid,n in enumerate(sizes):
                    for op in ('bx','br'):
                        self.assertLessEqual(sum(r['region_id']==rid and r['operator']==op for r in rows), n)
                sigs = [(r['operator'], engine._parent_batch_signature(r['parents'])) for r in rows]
                self.assertEqual(len(sigs), len(set(sigs)))
                for row in rows:
                    if row['operator']=='br':
                        self.assertEqual(len(row['parents']),2)
                        pool=engine._region_parent_pools.return_value[row['region_id']]
                        self.assertEqual(row['parents'][0],pool[0]['parent'])
                if rows:
                    self.assertEqual({r['kind'] for r in rows if r['operator']=='bx'},
                                     {'intra_bx' if turn==0 else 'inter_bx'})
            self.assertEqual(engine._region_parent_pools.call_count,2)

    def test_balanced_local_counts(self):
        rows=self.engine([4,4,4])._build_mixed_parent_plan([])
        self.assertEqual(Counter(r['operator'] for r in rows),dict(bx=12,br=9))

    def test_rng_is_reproducible_and_used(self):
        def signature(seed):
            return [(r['operator'],tuple(p['code'] for p in r['parents']))
                    for r in self.engine([8,8,8],seed)._build_mixed_parent_plan([])]
        self.assertEqual(signature(1111), signature(1111))
        self.assertNotEqual(signature(1111), signature(2222))

    def test_evaluation_ratio(self):
        for n,b in [(0,0),(1,1),(21,5),(32,7),(48,10)]:
            self.assertEqual(evaluation_slots(n,.2,210),b)
        self.assertEqual(evaluation_slots(48,.2,3),3)
        self.assertEqual(evaluation_slots(48,.2,210,2),2)
        for ratio in (0,-1,1.1,float('nan')):
            with self.assertRaises(ValueError): evaluation_slots(10,ratio,100)
