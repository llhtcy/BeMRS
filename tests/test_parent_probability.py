"""Exact local BX/BR probability and cross-region 1..K checks; no external calls."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bemrs.core.engine import SearchEngine


class RecordingRNG:
    def __init__(self, position=0):
        self.position = position
        self.calls = []

    def choice(self, count, p):
        self.calls.append((count, np.array(p, copy=True)))
        return min(self.position, count - 1)


class ParentProbabilityTests(unittest.TestCase):
    def distances(self, size=4):
        points = np.arange(size, dtype=float)
        return np.abs(points[:, None] - points[None, :])

    def engine(self, sizes):
        engine = SearchEngine.__new__(SearchEngine)
        engine._region_rng = RecordingRNG()
        engine._active_region_operator_schedule = {}
        engine._active_parent_batch_signatures = set()
        pools = {}
        index = 0
        for rid, size in enumerate(sizes):
            pools[rid] = []
            for item in range(size):
                pools[rid].append({
                    'parent': {'code': f'parent_{index}', 'objective': float(index)},
                    'objective': float(index),
                    'point': np.array([item * item], dtype=float),
                })
                index += 1
        return engine, pools

    def test_exact_reciprocal_formula_and_opposite_rank_directions(self):
        for near, preferred in ((False, 3), (True, 1)):
            for position in range(3):
                rng = RecordingRNG(position)
                selected = SearchEngine._reciprocal_rank_parent_indices(
                    self.distances(), 2, 0, rng, prefer_near=near)
                count, probabilities = rng.calls[0]
                self.assertEqual(count, 3)
                np.testing.assert_allclose(probabilities, [6/11, 3/11, 2/11])
                self.assertAlmostEqual(float(probabilities.sum()), 1.0)
                self.assertTrue(np.all(probabilities > 0))
                self.assertEqual(selected, [0, preferred + position * (1 if near else -1)])

    def test_candidate_count_is_not_fixed(self):
        for count in (1, 2, 3, 5, 12):
            rng = RecordingRNG()
            selected = SearchEngine._reciprocal_rank_parent_indices(
                self.distances(count + 1), 2, 0, rng)
            self.assertEqual(selected, [0, count])
            if count == 1:
                self.assertEqual(rng.calls, [])
            else:
                weights = 1 / np.arange(1, count + 1, dtype=float)
                np.testing.assert_allclose(rng.calls[0][1], weights / weights.sum())

    def test_anchor_distinct_objectives_and_no_replacement_preserved(self):
        values = [1, 1, 2, 2, 3]
        rng = RecordingRNG()
        selected = SearchEngine._reciprocal_rank_parent_indices(
            self.distances(5), 5, 0, rng, objective_values=values)
        self.assertEqual(selected[0], 0)
        self.assertNotIn(1, selected)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(len(selected), len({values[i] for i in selected}))

    def test_empty_and_singleton_pools(self):
        rng = RecordingRNG()
        self.assertEqual(SearchEngine._reciprocal_rank_parent_indices(
            np.empty((0, 0)), 2, 0, rng), [])
        self.assertEqual(SearchEngine._reciprocal_rank_parent_indices(
            np.zeros((1, 1)), 2, 0, rng), [0])
        self.assertEqual(rng.calls, [])

    def test_distance_ties_keep_stable_order(self):
        selected = SearchEngine._reciprocal_rank_parent_indices(
            np.zeros((4, 4)), 2, 0, RecordingRNG(), prefer_near=False)
        self.assertEqual(selected, [0, 1])

    def test_existing_seed_controls_sampling(self):
        def samples(seed, near):
            rng = np.random.default_rng(seed)
            return [SearchEngine._reciprocal_rank_parent_indices(
                self.distances(8), 2, 0, rng, prefer_near=near)[1]
                for _ in range(100)]
        for near in (False, True):
            self.assertEqual(samples(1111, near), samples(1111, near))
            self.assertNotEqual(samples(1111, near), samples(2222, near))

    def test_local_builders_use_new_formula_and_preserve_first_parents(self):
        for operator in ('bx', 'br'):
            engine, pools = self.engine([4])
            with self.assertLogs(level='INFO') as captured:
                rows = engine._build_region_parent_batches(
                    [], operator, 1, pools=pools, quotas={0: 1})
            self.assertEqual(rows[0]['parents'][0], pools[0][0]['parent'])
            expected = 3 if operator == 'bx' else 1
            self.assertEqual(rows[0]['parents'][1], pools[0][expected]['parent'])
            np.testing.assert_allclose(engine._region_rng.calls[0][1], [6/11, 3/11, 2/11])
            self.assertIn('behavior_reciprocal_rank_', '\n'.join(captured.output))
        engine, pools = self.engine([4])
        rows = engine._build_region_parent_batches([], 'bx', 2, pools=pools, quotas={0: 2})
        self.assertEqual([row['parents'][0] for row in rows],
                         [pools[0][0]['parent'], pools[0][1]['parent']])

    def test_cross_region_bx_cycles_parent_count_without_probability_sampling(self):
        engine, pools = self.engine([2, 2, 2])
        engine._bx_use_inter_region_this_round = True
        engine._region_rng.choice = Mock(side_effect=AssertionError('Cross-BX must not sample'))
        with patch.object(engine, '_reciprocal_rank_parent_indices',
                          side_effect=AssertionError('Cross-BX must not use new helper')):
            with self.assertLogs(level='INFO') as captured:
                rows = engine._build_region_parent_batches(
                [], 'bx', 3, pools=pools, quotas={0: 3})
        self.assertEqual([row['requested_parent_count'] for row in rows], [1, 2, 3])
        self.assertEqual([len(row['parents']) for row in rows], [1, 2, 3])
        self.assertTrue(all(row['parents'][0] == pools[0][0]['parent'] for row in rows))
        self.assertEqual([row['parents'][1]['code'] for row in rows[1:]], ['parent_2', 'parent_2'])
        self.assertEqual(rows[2]['parents'][2]['code'], 'parent_4')
        self.assertIn('selection=legacy_cross_top_priority_cycle_1_to_K', '\n'.join(captured.output))
        engine._region_rng.choice.assert_not_called()


if __name__ == '__main__':
    unittest.main()
