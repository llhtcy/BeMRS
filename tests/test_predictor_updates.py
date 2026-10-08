"""Round-based update policy tests; no LLM or real objective evaluation."""
import os
import unittest
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_dir

from bemrs.core.engine import SearchEngine
from bemrs.runtime_config import apply_runtime_config
from main import ROOT, SUPPORTED_SEARCH


class PredictorUpdateTests(unittest.TestCase):
    def engine(self, samples=50):
        engine = SearchEngine.__new__(SearchEngine)
        engine.current_generation = 1
        engine._behavior_trained_samples = 0
        engine._behavior_trained_samples_generation = None
        engine.surrogate_selection_enabled = True
        engine.behavior_filter_start_with_predictor = True
        engine.behavior_filter_xgb_rescue_enabled = True
        engine._behavior_filter_predictor_ready_for_batch = None
        engine.get_successful_real_eval_count = Mock(return_value=samples)
        engine.behavior_predictor = Mock(min_samples=50, surrogate_model='xgboost_direct')
        predictor = engine.behavior_predictor
        predictor.get_sample_count.return_value = samples
        predictor.is_ready.return_value = False

        def fit():
            predictor.is_ready.return_value = True
            return True

        predictor.train.side_effect = fit
        return engine

    def update(self, engine):
        return engine._train_predictor_if_needed(
            engine.behavior_predictor, '_behavior_trained_samples', 'test')

    def test_first_fit_then_one_new_sample_is_enough_next_round(self):
        engine = self.engine()
        self.assertTrue(self.update(engine))
        self.assertEqual(engine._behavior_trained_samples, 50)
        engine.current_generation = 2
        engine.behavior_predictor.get_sample_count.return_value = 51
        self.assertTrue(self.update(engine))
        self.assertEqual(engine.behavior_predictor.train.call_count, 2)
        self.assertEqual(engine._behavior_trained_samples, 51)

    def test_filter_and_ranking_share_one_fit_per_round(self):
        engine = self.engine()
        self.assertTrue(engine._behavior_filter_predictor_ready())
        self.assertTrue(engine._ensure_surrogate())
        self.assertTrue(engine._behavior_filter_predictor_ready())
        engine.behavior_predictor.train.assert_called_once()

    def test_same_round_new_samples_are_deferred_until_next_round(self):
        engine = self.engine()
        self.update(engine)
        engine.behavior_predictor.get_sample_count.return_value = 55
        self.assertTrue(self.update(engine))
        self.assertEqual(engine._behavior_trained_samples, 50)
        engine.behavior_predictor.train.assert_called_once()
        engine.current_generation = 2
        self.assertTrue(self.update(engine))
        self.assertEqual(engine._behavior_trained_samples, 55)
        self.assertEqual(engine.behavior_predictor.train.call_count, 2)

    def test_no_new_samples_does_not_retrain_even_in_later_rounds(self):
        engine = self.engine()
        self.update(engine)
        for generation in (2, 3, 4):
            engine.current_generation = generation
            self.assertTrue(self.update(engine))
        engine.behavior_predictor.train.assert_called_once()

    def test_below_gate_does_not_consume_round_fit_opportunity(self):
        engine = self.engine(49)
        self.assertFalse(self.update(engine))
        engine.behavior_predictor.train.assert_not_called()
        engine.behavior_predictor.get_sample_count.return_value = 50
        self.assertTrue(self.update(engine))
        engine.behavior_predictor.train.assert_called_once()

    def test_failed_fit_is_attempted_once_and_retried_next_round(self):
        engine = self.engine()
        engine.behavior_predictor.train.side_effect = None
        engine.behavior_predictor.train.return_value = False
        self.assertFalse(self.update(engine))
        self.assertFalse(self.update(engine))
        self.assertEqual(engine._behavior_trained_samples, 0)
        engine.behavior_predictor.train.assert_called_once()
        engine.current_generation = 2
        self.assertFalse(self.update(engine))
        self.assertEqual(engine.behavior_predictor.train.call_count, 2)

    def test_failed_refresh_does_not_claim_new_samples_were_trained(self):
        engine = self.engine()
        self.update(engine)
        engine.current_generation = 2
        engine.behavior_predictor.get_sample_count.return_value = 51
        engine.behavior_predictor.train.side_effect = None
        engine.behavior_predictor.train.return_value = False
        self.assertTrue(self.update(engine))  # A retained old model remains usable.
        self.assertEqual(engine._behavior_trained_samples, 50)
        self.update(engine)
        self.assertEqual(engine.behavior_predictor.train.call_count, 2)

    def test_config_and_environment_no_longer_expose_retired_knobs(self):
        for problem in SUPPORTED_SEARCH:
            with initialize_config_dir(config_dir=str(ROOT / 'cfg'), version_base=None):
                cfg = compose(config_name='config', overrides=[f'problem={problem}'])
            self.assertNotIn('retrain_interval', cfg.method.predictor)
            self.assertNotIn('operators', cfg.problem)
            self.assertEqual(cfg.method.predictor.min_samples, 50)
            self.assertEqual(cfg.method.predictor.max_samples, 200)
            self.assertEqual(cfg.method.region.bx_parent_selection.tau_bx, 1.5)
            with patch.dict(os.environ, {
                'BEMRS_PREDICTOR_RETRAIN_INTERVAL': '10',
                'BEMRS_BX_PARENT_COUNTS': '1,2,3',
            }, clear=True):
                applied = apply_runtime_config(cfg)
                self.assertNotIn('BEMRS_PREDICTOR_RETRAIN_INTERVAL', applied)
                self.assertNotIn('BEMRS_BX_PARENT_COUNTS', applied)
                self.assertNotIn('BEMRS_PREDICTOR_RETRAIN_INTERVAL', os.environ)
                self.assertNotIn('BEMRS_BX_PARENT_COUNTS', os.environ)


if __name__ == '__main__':
    unittest.main()
