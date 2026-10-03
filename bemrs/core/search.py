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

    def _manage_initial_population_with_score_diversity(self, population, required_distinct_scores):
        """Retain the best representatives of at least five score levels."""
        required_distinct_scores = max(1, int(required_distinct_scores))
        if self.pop_size < required_distinct_scores:
            raise RuntimeError(f'Initialization score-diversity requirement is impossible: population size={self.pop_size}, required distinct scores={required_distinct_scores}.')
        finite_population = [individual for individual in population if self._initialization_score_key(individual) is not None]
        finite_population.sort(key=lambda individual: float(individual['objective']))
        score_representatives = []
        seen_scores = set()
        for individual in finite_population:
            score_key = self._initialization_score_key(individual)
            if score_key in seen_scores:
                continue
            seen_scores.add(score_key)
            score_representatives.append(individual)
            if len(score_representatives) >= required_distinct_scores:
                break
        if len(score_representatives) < required_distinct_scores:
            raise RuntimeError(f'Initialization ended without enough distinct finite scores: found={len(score_representatives)}, required={required_distinct_scores}.')
        target_size = min(len(population), self.pop_size)
        managed = self.manage.population_management(population, target_size)
        retained = []
        retained_codes = set()
        for individual in score_representatives + list(managed) + finite_population:
            code_key = self._canonical_code(individual.get('code', ''))
            if not code_key or code_key in retained_codes:
                continue
            retained.append(individual)
            retained_codes.add(code_key)
            if len(retained) >= target_size:
                break
        retained.sort(key=lambda individual: float(individual.get('objective', float('inf'))))
        return retained

    def _complete_initialization_before_be(self, interface_ec, population):
        """Run additional I1 batches until five distinct scores are retained."""
        required = max(1, int(interface_ec.init_min_distinct_scores))
        if interface_ec.real_eval_budget < required:
            raise RuntimeError(f'Initialization requires at least {required} distinct scores before BE, but max_fe is only {interface_ec.real_eval_budget}.')
        extra_round = 0
        while self._count_distinct_initialization_scores(population) < required:
            found = self._count_distinct_initialization_scores(population)
            if interface_ec.is_real_eval_budget_exhausted():
                raise RuntimeError(f'Real-evaluation budget was exhausted during I1 initialization with only {found}/{required} distinct finite scores; BE and formal evolution were not started.')
            interface_ec.raise_if_generated_algorithm_limit_reached(context='I1 initialization before BE')
            extra_round += 1
            logging.info('[InitializationScoreDiversity] round=%s distinct_scores=%s/%s population=%s generated=%s/%s successful_evals=%s/%s action=continue_i1', extra_round, found, required, len(population), interface_ec.get_generated_algorithm_count(), interface_ec.generated_algorithm_limit, interface_ec.get_successful_real_eval_count(), interface_ec.real_eval_budget)
            extra_population = interface_ec.population_generation(prior_population=population)
            self.add2pop(population, extra_population)
        population = self._manage_initial_population_with_score_diversity(population, required)
        retained_scores = self._count_distinct_initialization_scores(population)
        if retained_scores < required:
            raise RuntimeError(f'Initial population management removed required score diversity: retained={retained_scores}, required={required}.')
        logging.info('[InitializationScoreDiversity] complete=True distinct_scores=%s/%s population=%s extra_i1_rounds=%s generated=%s/%s successful_evals=%s/%s next=BE', retained_scores, required, len(population), extra_round, interface_ec.get_generated_algorithm_count(), interface_ec.generated_algorithm_limit, interface_ec.get_successful_real_eval_count(), interface_ec.real_eval_budget)
        return population

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
        population = self._complete_initialization_before_be(interface_ec, population)
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
