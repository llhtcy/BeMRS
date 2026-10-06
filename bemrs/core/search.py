import json
import logging
import os
import time
import numpy as np
from .engine import SearchEngine


class BeMRS:
    """Archive-driven search without a separate global population."""

    def __init__(self, cfg, problem):
        self.cfg = cfg
        self.prob = problem
        self.output_path = './'

    def _save_archive(self, engine, generation):
        archive, _, _ = engine._region_advantage_archive_rows()
        if not archive:
            raise RuntimeError('No evaluated algorithm with valid behavior features in the advantage archive.')
        # Retain historical filenames so existing result readers still work.
        with open(self.output_path + f'population_generation_{generation}.json', 'w') as file:
            json.dump(archive, file, indent=2)
        best = dict(archive[0])
        best['best_algorithm_source'] = best.get('generation_parent_scope', 'unknown')
        best['best_algorithm_region_id'] = best.get('region_source_id')
        path = self.output_path + f'best_population_generation_{generation}.json'
        with open(path, 'w') as file:
            json.dump(best, file, indent=2)
        logging.info('[AdvantageArchiveSnapshot] generation=%s size=%s target=%s scores_unique=True',
                     generation, len(archive), engine.region_archive_target_distinct)
        logging.info('[BestAlgorithm] generation=%s objective=%s source=%s region=%s algorithm_id=%s',
                     generation, best.get('objective'), best.get('best_algorithm_source'),
                     best.get('best_algorithm_region_id'), best.get('algorithm_id'))
        return archive, path

    def run(self):
        print('- Evolution ready for', self.prob.problem, '-')
        started = time.time()
        engine = SearchEngine(self.cfg, self.prob)
        engine.set_timing_output(self.output_path)
        engine.set_current_generation(0)
        initial_archive_start = engine.predictor.get_sample_count()
        engine.population_generation_with_prompt_seed()
        records, matrix, _ = engine._region_advantage_archive_rows()
        if not records:
            raise RuntimeError('Initialization produced no evaluated algorithm with valid behavior features.')
        engine._initialization_complete = True
        engine.region_count = min(engine.region_count, len(records), len(np.unique(matrix, axis=0)))
        if engine.region_enabled and not engine.is_real_eval_budget_exhausted():
            if not engine._initialize_regions(reason='initialization_complete'):
                raise RuntimeError('Failed to partition valid initialization algorithms.')
        archive, best_path = self._save_archive(engine, 0)
        logging.info('[InitializationComplete] evaluated=%s regions=%s archive=%s next=mixed refill=False',
                     engine.get_successful_real_eval_count(), len(engine._region_states), len(archive))
        visualize = os.environ.get('BEMRS_VISUALIZE_FEATURES', '1').lower() not in {'0', 'false', 'no'}
        method = os.environ.get('BEMRS_VISUALIZE_METHOD', 'pca')
        if visualize:
            engine.predictor.visualize_behavior_features(self.output_path, iteration=0,
                                                        method=method, recent_start_index=initial_archive_start)
        generation = 0
        stalled = 0
        max_stalled = max(1, int(os.environ.get('BEMRS_MAX_STALLED_GENERATIONS', 100)))
        while not engine.is_real_eval_budget_exhausted():
            engine.raise_if_generated_algorithm_limit_reached(context='formal search')
            generation += 1
            engine.set_current_generation(generation)
            eval_start = engine.get_successful_real_eval_count()
            sample_start = engine.predictor.get_sample_count()
            use_regions = engine.should_use_regions() and engine._initialize_regions()
            operators = ('mixed',) if use_regions else ('bx', 'br')
            for operator in operators:
                if engine.is_real_eval_budget_exhausted():
                    break
                # Every call uses the current archive, including after fallback BX.
                archive = engine._region_advantage_archive_rows()[0]
                print(f' OP: {operator} |', end='')
                engine.get_algorithm(archive, operator)
            archive, best_path = self._save_archive(engine, generation)
            if visualize:
                engine.predictor.visualize_behavior_features(self.output_path, iteration=generation,
                                                            method=method, recent_start_index=sample_start)
            eval_end = engine.get_successful_real_eval_count()
            engine.raise_if_generated_algorithm_limit_reached(context=f'generation {generation}')
            stalled = stalled + 1 if eval_end == eval_start else 0
            if stalled >= max_stalled:
                raise RuntimeError(f'Unable to reach the real-evaluation budget: {stalled} stalled generations '
                                   f'at {eval_end}/{engine.real_eval_budget}.')
            print(f'--- generation {generation} finished | successful evaluations {eval_end}/{engine.real_eval_budget}'
                  f' | generated algorithms {engine.get_generated_algorithm_count()}/{engine.generated_algorithm_limit}'
                  f' | Time Cost: {(time.time() - started) / 60:.1f} m')
        logging.info('[RealEvalBudget] completed exactly %s/%s successful evaluations | attempted=%s invalid=%s generated_algorithms=%s/%s',
                     engine.get_successful_real_eval_count(), engine.real_eval_budget, engine.get_real_eval_count(),
                     engine.get_invalid_real_eval_count(), engine.get_generated_algorithm_count(), engine.generated_algorithm_limit)
        return archive[0]['code'], best_path
