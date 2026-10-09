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
                    if row['kind'] == 'inter_bx':
                        self.assertGreaterEqual(len(row['parents']), 1)
                        self.assertEqual(len(row['parents']), row['actual_parent_count'])
                        self.assertEqual(len(row['parents']), row['requested_parent_count'])
                    else:
                        self.assertEqual(len(row['parents']), 2)
                    if row['operator']=='br':
                        self.assertEqual(len(row['parents']),2)
                        pool=engine._region_parent_pools.return_value[row['region_id']]
                        self.assertEqual(row['parents'][0],pool[0]['parent'])
                if sum(n > 0 for n in sizes) == 1 and max(sizes) >= 2:
                    self.assertEqual({r['kind'] for r in rows if r['operator']=='bx'},
                                     {'intra_bx'})
                if sum(sizes) < 2:
                    self.assertEqual(rows, [])
            self.assertEqual(engine._region_parent_pools.call_count,2)

    def test_balanced_local_counts(self):
        rows=self.engine([4,4,4])._build_mixed_parent_plan([])
        self.assertEqual(Counter(r['operator'] for r in rows),dict(bx=12,br=9))
        self.assertEqual(Counter(r['kind'] for r in rows),
                         dict(intra_bx=6, inter_bx=6, intra_br=9))

    def test_odd_split_rotates(self):
        engine = self.engine([5,5,5])
        for turn in range(2):
            rows = engine._build_mixed_parent_plan([])
            for rid in range(3):
                self.assertEqual(sum(r['region_id']==rid and r['kind']=='intra_bx'
                                     for r in rows), 3 if turn == 0 else 2)
                self.assertEqual(sum(r['region_id']==rid and r['kind']=='inter_bx'
                                     for r in rows), 2 if turn == 0 else 3)

    def test_unused_slots_transfer(self):
        for unavailable, expected in (('intra_bx','inter_bx'), ('inter_bx','intra_bx')):
            engine = self.engine([4,4,4])
            original = engine._build_region_parent_batches
            def build(pop, op, size, pools=None, quotas=None):
                mode = 'inter_bx' if engine._bx_use_inter_region_this_round else 'intra_bx'
                if op == 'bx' and mode == unavailable:
                    return []
                return original(pop, op, size, pools=pools, quotas=quotas)
            engine._build_region_parent_batches = build
            rows = engine._build_mixed_parent_plan([])
            self.assertEqual(sum(r['operator']=='bx' for r in rows), 12)
            self.assertEqual({r['kind'] for r in rows if r['operator']=='bx'}, {expected})

    def test_rng_is_reproducible_and_used(self):
        def signature(seed):
            return [(r['operator'],tuple(p['code'] for p in r['parents']))
                    for r in self.engine([8,8,8],seed)._build_mixed_parent_plan([])]
        self.assertEqual(signature(1111), signature(1111))
        self.assertNotEqual(signature(1111), signature(2222))

    def test_cross_region_bx_cycles_one_to_active_K(self):
        for k in (2, 3, 5):
            engine = self.engine([4] * k)
            pools = engine._region_parent_pools.return_value
            region_by_code = {entry['parent']['code']: rid
                              for rid, entries in pools.items() for entry in entries}
            engine._active_parent_batch_signatures = set()
            rows = engine._build_inter_region_bx_parent_batches(
                [], k * 4, pools=pools, quotas={rid: 4 for rid in pools})
            self.assertTrue(rows)
            self.assertEqual([row['requested_parent_count'] for row in rows[:k]], list(range(1, k + 1)))
            for row in rows:
                self.assertGreaterEqual(len(row['parents']), 1)
                self.assertLessEqual(len(row['parents']), k)
                self.assertEqual(row['requested_parent_count'], len(row['parents']))
                first = row['parents'][0]
                self.assertEqual(first, pools[row['region_id']][0]['parent'])
                self.assertEqual(len({region_by_code[parent['code']] for parent in row['parents']}), len(row['parents']))

    def test_cross_region_bx_tries_next_region_after_pair_exhaustion(self):
        engine = self.engine([1, 1, 1])
        pools = engine._region_parent_pools.return_value
        engine._active_parent_batch_signatures = set()
        rows = engine._build_inter_region_bx_parent_batches(
            [], 2, pools=pools, quotas={0: 2})
        self.assertEqual([len(r['parents']) for r in rows], [1, 2])
        self.assertEqual(rows[1]['parents'][1]['code'], 'code_1')

    def test_single_parent_cross_bx_is_allowed_but_local_paths_stay_two_parent(self):
        engine = self.engine([1])
        engine._active_parent_batch_signatures = set()
        rows = engine._build_inter_region_bx_parent_batches([], 3)
        self.assertEqual([len(row['parents']) for row in rows], [1])
        self.assertEqual(engine._build_mixed_parent_plan([]), [])
        rows = self.engine([1, 4])._build_mixed_parent_plan([])
        self.assertTrue(rows)
        self.assertTrue(all(len(r['parents']) == 2 for r in rows if r['kind'] != 'inter_bx'))

    def test_archive_fallback_is_also_strictly_two_parent(self):
        for size in (1, 2, 6):
            engine = self.engine([size])
            pop = [entry['parent'] for entry in engine._region_parent_pools.return_value[0]]
            for operator in ('bx', 'br'):
                engine._active_parent_batch_signatures = set()
                rows = engine._build_archive_parent_batches(pop, operator, size)
                if size == 1:
                    self.assertEqual(rows, [])
                else:
                    self.assertTrue(rows)
                for parents in rows:
                    self.assertEqual(len(parents), 2)
                    self.assertNotEqual(parents[0]['code'], parents[1]['code'])

    def test_pair_signature_rejects_wrong_sizes_and_reversed_duplicate(self):
        engine = self.engine([3])
        engine._active_parent_batch_signatures = set()
        parents = [e['parent'] for e in engine._region_parent_pools.return_value[0]]
        self.assertFalse(engine._register_parent_batch(parents[:1]))
        self.assertFalse(engine._register_parent_batch(parents))
        self.assertTrue(engine._register_parent_batch(parents[:2]))
        self.assertFalse(engine._register_parent_batch(parents[1::-1]))

    def test_evaluation_ratio(self):
        for n,b in [(0,0),(1,1),(21,5),(32,7),(48,10)]:
            self.assertEqual(evaluation_slots(n,.2,210),b)
        self.assertEqual(evaluation_slots(48,.2,3),3)
        self.assertEqual(evaluation_slots(48,.2,210,2),2)
        for ratio in (0,-1,1.1,float('nan')):
            with self.assertRaises(ValueError): evaluation_slots(10,ratio,100)
