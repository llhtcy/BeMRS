"""Regression tests for regions, without LLM or expensive objective calls."""
import unittest
from unittest.mock import Mock
import numpy as np
from bemrs.core.engine import SearchEngine


class RegionTests(unittest.TestCase):
    def test_archive_strict_distinct_score_cap(self):
        engine = SearchEngine.__new__(SearchEngine)
        engine.region_count = 3
        engine.region_archive_target_distinct = 24
        engine._evaluated_algorithm_history = [
            dict(code=f'code_{score}_{copy}', objective=float(score),
                 behavior_embedding=np.array([score, copy], float))
            for score in reversed(range(30)) for copy in range(3)]
        records, matrix, objectives = engine._region_advantage_archive_rows()
        self.assertEqual(len(records), 24)
        np.testing.assert_array_equal(objectives, np.arange(24))
        self.assertEqual(matrix.shape, (24, 2))
        engine._evaluated_algorithm_history = engine._evaluated_algorithm_history[-6:]
        records, _, objectives = engine._region_advantage_archive_rows()
        self.assertEqual(len(records), 2)
        self.assertEqual(len(set(objectives)), 2)
        engine.region_archive_target_distinct = 1
        self.assertEqual(len(engine._region_advantage_archive_rows()[0]), 1)

    def engine(self):
        engine = SearchEngine.__new__(SearchEngine)
        engine.current_generation = 1
        engine.region_rebuild_tolerance = 5
        engine.region_relative_improvement = 1e-4
        engine.preserve_global_best_anchor = True
        engine._region_states = [
            dict(region_id=r, best_objective=value, center_point=np.array([r * 10.]),
                 center_feature=np.array([r * 10.]), center_code=f'code{r}',
                 selected_count=0, evaluated_count=0, recent_events=[],
                 rebuild_stagnation_count=0)
            for r, value in [(1, 10.), (2, 20.)]]
        engine._sync_region_snapshot = Mock()
        engine._register_region_centers = Mock()
        engine._region_objective_scale = Mock(return_value=1.)
        engine._transform_region_features = lambda x: np.asarray(x, dtype=float)
        engine._rebuild_kmeans_regions = Mock()
        return engine

    def test_best_updates_from_real_evaluation_and_centroid_stays_fixed(self):
        engine = self.engine()
        rows = [dict(region_allocation_id=2, objective=15., real_eval_attempted=True,
                     behavior_embedding=np.array([1000.]), code='better', algorithm_id='better'),
                dict(region_allocation_id=1, objective=-100., real_eval_attempted=False,
                     behavior_pred_score=-100., code='only_prediction')]
        engine._update_regions_after_evaluation(rows, {1: 1, 2: 1}, {1, 2})
        self.assertEqual(engine._region_states[1]['best_objective'], 15.)
        self.assertEqual(engine._region_states[0]['best_objective'], 10.)
        np.testing.assert_array_equal(engine._region_states[1]['center_point'], [20.])
        np.testing.assert_array_equal(engine._region_states[1]['center_feature'], [1000.])
        self.assertEqual(engine._region_states[1]['rebuild_stagnation_count'], 0)
        self.assertFalse(any('length' in row or 'failure_count' in row for row in engine._region_states))

    def test_original_improvement_threshold_is_preserved(self):
        engine = self.engine()
        engine._update_regions_after_evaluation([
            dict(region_allocation_id=2, objective=19.9999, real_eval_attempted=True)], {2: 1}, {2})
        self.assertEqual(engine._region_states[1]['best_objective'], 20.)
        self.assertEqual(engine._region_states[1]['rebuild_stagnation_count'], 1)

    def test_stagnation_recluster_and_best_anchor_protection(self):
        engine = self.engine()
        for generation in range(1, 6):
            engine.current_generation = generation
            engine._update_regions_after_evaluation([], {1: 0, 2: 0}, {1, 2})
        engine._rebuild_kmeans_regions.assert_called_once_with(['R2_no_fresh_candidate'])
        self.assertEqual(engine._region_states[0]['rebuild_stagnation_count'], 0)

    def test_fixed_operator_cycle_without_radius(self):
        engine = self.engine()
        engine._active_region_operator_schedule_generation = None
        engine._active_region_operator_schedule = {}
        engine.region_candidates_per_region = 8
        engine.should_use_regions = Mock(return_value=True)
        engine._initialize_regions = Mock(return_value=True)
        observed = []
        for generation in range(1, 9):
            engine.current_generation = generation
            schedule = engine.prepare_region_operator_schedule(['bx', 'br'])
            observed.append(schedule[1]['operator'])
            self.assertEqual(schedule[1]['generation_quota'], 8)
            self.assertEqual(schedule, engine.prepare_region_operator_schedule(['bx', 'br']))
            self.assertNotIn('normalized_radius', schedule[1])
        self.assertEqual(observed, ['bx', 'bx', 'bx', 'br'] * 2)

    def test_assignment_has_no_box_boundary(self):
        engine = self.engine()
        rows = [dict(behavior_embedding=np.array([1000.]), region_source_id=1)]
        engine._annotate_region_candidates(rows)
        self.assertEqual(rows[0]['region_allocation_id'], 2)
        self.assertEqual(rows[0]['region_distance'], 980.)
        self.assertNotIn('region_inside', rows[0])

    def test_kmeans_any_region_count(self):
        engine = self.engine()
        engine.region_seed = 1111
        points = np.arange(24., dtype=float).reshape(-1, 1)
        values = np.arange(24., dtype=float)[::-1]
        for k in (1, 2, 3, 6):
            engine.region_count = k
            anchors, centers, assignments, stats = engine._kmeans_region_layout(points, values)
            self.assertEqual(len(anchors), k)
            for region, anchor in enumerate(anchors):
                self.assertEqual(values[anchor], min(values[assignments == region]))


if __name__ == '__main__':
    unittest.main()
