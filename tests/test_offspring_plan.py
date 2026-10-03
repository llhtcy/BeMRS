import unittest
from collections import Counter
from unittest.mock import Mock
import numpy as np
from bemrs.offspring_plan import build_parent_plan, evaluation_slots
from bemrs.core.engine import SearchEngine


class OffspringPlanTests(unittest.TestCase):
    def pools(self, sizes):
        pools = {}
        count = 0
        for rid, size in enumerate(sizes):
            pools[rid] = []
            for i in range(size):
                parent = dict(code=f'code_{count}', objective=float(count))
                pools[rid].append(dict(parent=parent, objective=float(count),
                                       point=np.array([rid * 100 + i * i], float)))
                count += 1
        return pools

    def plan(self, sizes):
        return build_parent_plan(self.pools(sizes), lambda code: code)

    def test_expected_counts(self):
        for sizes, expected in [([4,4,4],24), ([5,5,6],32), ([6,5,5],32),
                                ([1],2), ([2],4), ([1,1,1],6), ([0,2],4)]:
            with self.subTest(sizes=sizes):
                rows = self.plan(sizes)
                self.assertEqual(len(rows), expected)
                signatures = [(r['operator'], tuple(p['code'] for p in r['parents'])) for r in rows]
                self.assertEqual(len(signatures), len(set(signatures)))
        self.assertEqual(Counter(r['kind'] for r in self.plan([4,4,4])),
                         dict(single_br=3,intra_bx=12,intra_br=9))

    def test_distances_order_and_cross_anchor(self):
        pools = self.pools([4,4,4])
        rows = build_parent_plan(pools, lambda code: code)
        for row in rows:
            if row['kind'] in ('intra_bx','intra_br'):
                entries = pools[row['region_id']]
                first, second = row['parents']
                if row['operator'] == 'bx':
                    anchor = next(e for e in entries if e['parent'] == first)
                    distances = {e['parent']['code']: float(np.linalg.norm(e['point']-anchor['point']))
                                 for e in entries if e['parent'] != first}
                    self.assertEqual(distances[second['code']], max(distances.values()))
                else:
                    self.assertEqual(first['code'], entries[0]['parent']['code'])
                    self.assertIn(second['code'], {e['parent']['code'] for e in entries[1:]})
        ordered = {tuple(p['code'] for p in r['parents']) for r in self.plan([2]) if r['kind']=='intra_bx'}
        self.assertEqual(ordered, {('code_0','code_1'),('code_1','code_0')})

    def test_duplicate_membership_and_empty(self):
        pools = self.pools([2,2])
        pools[1].append(pools[0][0])
        self.assertEqual(build_parent_plan(pools, lambda c:c), self.plan([2,2]))
        self.assertEqual(self.plan([]), [])

    def test_evaluation_ratio(self):
        for n, expected in [(0,0),(1,1),(9,1),(10,1),(11,2),(38,4),(49,5),(80,8)]:
            self.assertEqual(evaluation_slots(n,.1,210), expected)
        self.assertEqual(evaluation_slots(80,.1,3),3)
        self.assertEqual(evaluation_slots(80,.1,210,2),2)
        self.assertEqual(evaluation_slots(80,.1,0),0)
        for ratio in (0,-.1,1.1,float('nan')):
            with self.assertRaises(ValueError): evaluation_slots(10,ratio,100)

    def test_mixed_generation_preserves_operators_and_failure(self):
        engine = SearchEngine.__new__(SearchEngine)
        engine.llm_batch_generation_enabled = True
        engine._claim_generated_algorithm_slots = Mock()
        engine._register_lineage_event = Mock()
        def generate(op, parents):
            return [dict(code=None if i==0 else f'{op}{i}', algorithm='mock', llm_attempts=1)
                    for i in range(len(parents))]
        engine.evol = Mock()
        engine.evol.generate_batch.side_effect = generate
        rows = engine._generate_mixed_plan(self.plan([4,4,4]))
        self.assertEqual(len(rows),24)
        self.assertEqual(Counter(o['operator'] for _,o in rows),dict(bx=12,br=12))
        self.assertEqual(sum(bool(o.get('skip_real_eval')) for _,o in rows),2)
        self.assertEqual(engine.evol.generate_batch.call_count,2)
        self.assertEqual(engine._register_lineage_event.call_count,24)

if __name__ == '__main__':
    unittest.main()
