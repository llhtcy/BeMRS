import numpy as np
import json
import random
import time
import os
import re
import logging
from .engine import SearchEngine

class BeMRS:

    def __init__(self, cfg, problem):
        from . import population
        self.cfg = cfg
        self.prob = problem
        self.manage = population
        self.pop_size = int(cfg.pop_size)
        self.operators = ['bx', 'br']
        self.operator_weights = [1, 1]
        self.debug_mode = False
        self.use_seed = False
        self.load_pop = False
        self.output_path = './'
        self.timeout = cfg.timeout
        random.seed(2024)

    @staticmethod
    def _canonical_code(code):
        code = str(code or '')
        code = re.sub('```(?:python)?', '', code)
        code = code.replace('```', '')
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        return '\n'.join((line.rstrip() for line in code.strip().splitlines()))

    def add2pop(self, population, offspring):
        existing_codes = {self._canonical_code(ind.get('code', '')) for ind in population}
        for off in offspring:
            code_key = self._canonical_code(off.get('code', ''))
            if not code_key or code_key in existing_codes:
                if self.debug_mode:
                    print('duplicated code, skip ... ')
                continue
            population.append(off)
            existing_codes.add(code_key)

    @staticmethod
    def _initialization_score_key(individual):
        """Use the same five-decimal resolution stored by real evaluation."""
        try:
            value = float(individual.get('objective', float('inf')))
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return round(value, 5)

    def _count_distinct_initialization_scores(self, population):
        return len({key for key in (self._initialization_score_key(individual) for individual in population) if key is not None})

    def run(self):
        print('- Evolution ready for', self.prob.problem, '-')
        time_start = time.time()
        interface_prob = self.prob
        interface_ec = SearchEngine(self.cfg, interface_prob)
        if hasattr(interface_ec, 'set_timing_output'):
            interface_ec.set_timing_output(self.output_path)
        if hasattr(interface_ec, 'set_current_generation'):
            interface_ec.set_current_generation(0)
        initial_archive_start = interface_ec.predictor.get_sample_count() if hasattr(interface_ec, 'predictor') else 0
        population = []
        loaded_initial_population = False
        population = interface_ec.population_generation_with_prompt_seed()
        n_start = 0
        population = [p for p in population if np.isfinite(interface_ec._safe_objective(p.get('objective')))]
        if not population or not interface_ec._behavior_explore_parent_archive:
            raise RuntimeError('Initialization produced no evaluated algorithm with valid behavior features.')
        interface_ec._initialization_complete = True
        records, matrix, _ = interface_ec._region_advantage_archive_rows()
        interface_ec.region_count = min(interface_ec.region_count, len(records),
                                        len(np.unique(matrix, axis=0)))
        if interface_ec.region_count < 1:
            raise RuntimeError('Initialization produced no usable behavior points for partitioning.')
        if interface_ec.region_enabled and not interface_ec._initialize_regions(reason='initialization_complete'):
            raise RuntimeError('Failed to partition valid initialization algorithms.')
        population = self.manage.population_management(population, min(len(population), self.pop_size))
        logging.info('[InitializationComplete] evaluated=%s regions=%s population=%s next=mixed refill=False',
                     interface_ec.get_successful_real_eval_count(), len(interface_ec._region_states), len(population))
        print('Pop initial: ')
        for off in population:
            print(' Obj: ', off['objective'], end='|')
        print(f'\ninitial population has been created with {self._count_distinct_initialization_scores(population)} distinct scores!')
        if not loaded_initial_population:
            filename = self.output_path + 'population_generation_0.json'
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
        else:
            filename = self.load_pop_path
        visualize_features = os.environ.get('BEMRS_VISUALIZE_FEATURES', '1').lower() not in {'0', 'false', 'no'}
        visualize_method = os.environ.get('BEMRS_VISUALIZE_METHOD', 'pca')
        if visualize_features and hasattr(interface_ec, 'predictor'):
            interface_ec.predictor.visualize_behavior_features(self.output_path, iteration=0, method=visualize_method, recent_start_index=initial_archive_start)
        n_op = len(self.operators)
        pop = n_start
        stalled_generations = 0
        max_stalled_generations = max(1, int(os.environ.get('BEMRS_MAX_STALLED_GENERATIONS', 100)))
        while not interface_ec.is_real_eval_budget_exhausted():
            interface_ec.raise_if_generated_algorithm_limit_reached(context='formal search')
            if hasattr(interface_ec, 'set_current_generation'):
                interface_ec.set_current_generation(pop + 1)
            generation_eval_start = interface_ec.get_successful_real_eval_count()
            generation_archive_start = interface_ec.predictor.get_sample_count() if hasattr(interface_ec, 'predictor') else 0
            if interface_ec.should_run_behavior_exploration():
                op = 'be'
                round_info = interface_ec.get_behavior_exploration_round()
                print(f' OP: be, [post-init behavior escape {round_info[0]}/{round_info[1]}, no refill] ', end='|')
                parents, offsprings = interface_ec.get_algorithm(population, op)
                interface_ec.complete_behavior_exploration_round()
                self.add2pop(population, offsprings)
                for off in offsprings:
                    print(' Obj: ', off['objective'], end='|')
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print()
                if not interface_ec.should_run_behavior_exploration() and hasattr(interface_ec, 'rebuild_regions_after_behavior_exploration'):
                    interface_ec.rebuild_regions_after_behavior_exploration()
            else:
                use_mixed = interface_ec.should_use_regions() and interface_ec._initialize_regions()
                round_operators = ['mixed'] if use_mixed else self.operators
                for i, op in enumerate(round_operators):
                    if interface_ec.is_real_eval_budget_exhausted():
                        break
                    print(f' OP: {op}, [{i + 1} / {len(round_operators)}] ', end='|')
                    op_w = 1 if use_mixed else self.operator_weights[i]
                    if np.random.rand() >= op_w:
                        continue
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                    self.add2pop(population, offsprings)
                    for off in offsprings:
                        print(' Obj: ', off['objective'], end='|')
                    size_act = min(len(population), self.pop_size)
                    population = self.manage.population_management(population, size_act)
                    print()
            pop += 1
            filename = self.output_path + 'population_generation_' + str(pop) + '.json'
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            filename = self.output_path + 'best_population_generation_' + str(pop) + '.json'
            best_record = dict(population[0])
            best_record['best_algorithm_source'] = best_record.get('generation_parent_scope', 'legacy_unknown')
            best_record['best_algorithm_region_id'] = best_record.get('region_source_id')
            with open(filename, 'w') as f:
                json.dump(best_record, f, indent=5)
            logging.info('[BestAlgorithm] generation=%s objective=%s source=%s region=%s algorithm_id=%s', pop, best_record.get('objective'), best_record.get('best_algorithm_source'), best_record.get('best_algorithm_region_id'), best_record.get('algorithm_id'))
            if visualize_features and hasattr(interface_ec, 'predictor'):
                interface_ec.predictor.visualize_behavior_features(self.output_path, iteration=pop, method=visualize_method, recent_start_index=generation_archive_start)
            generation_eval_end = interface_ec.get_successful_real_eval_count()
            interface_ec.raise_if_generated_algorithm_limit_reached(context=f'generation {pop}')
            if generation_eval_end == generation_eval_start:
                stalled_generations += 1
            else:
                stalled_generations = 0
            if stalled_generations >= max_stalled_generations:
                raise RuntimeError(f'Unable to reach the exact real-evaluation budget: stalled for {stalled_generations} generations at {generation_eval_end}/{interface_ec.real_eval_budget} successful evaluations. All generated candidates may be invalid, duplicated, or filtered.')
            print(f'--- generation {pop} finished | successful evaluations {generation_eval_end}/{interface_ec.real_eval_budget} | generated algorithms {interface_ec.get_generated_algorithm_count()}/{interface_ec.generated_algorithm_limit} | attempts {interface_ec.get_real_eval_count()} | Time Cost: {(time.time() - time_start) / 60:.1f} m')
            print('Pop Objs: ', end=' ')
            for i in range(len(population)):
                print(str(population[i]['objective']) + ' ', end='')
            print()
        logging.info('[RealEvalBudget] completed exactly %s/%s successful evaluations | attempted=%s invalid=%s generated_algorithms=%s/%s', interface_ec.get_successful_real_eval_count(), interface_ec.real_eval_budget, interface_ec.get_real_eval_count(), interface_ec.get_invalid_real_eval_count(), interface_ec.get_generated_algorithm_count(), interface_ec.generated_algorithm_limit)
        return (population[0]['code'], filename)
