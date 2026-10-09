import math
import os
import re
import csv
import hashlib
import itertools
import json
import time
import logging
import uuid
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
try:
    from ..predictor import SurrogatePredictor, build_behavior_embedder
    from ..runtime_config import config_get
except ImportError as import_error:
    if 'relative import' not in str(import_error):
        raise
    from bemrs.predictor import SurrogatePredictor, build_behavior_embedder
    from bemrs.runtime_config import config_get
from .operators import Evolution
BEHAVIOR_EXPAND_OPERATORS = frozenset({'bx'})
BEHAVIOR_REFINE_OPERATORS = frozenset({'br'})
STANDARD_BEHAVIOR_OPERATORS = BEHAVIOR_EXPAND_OPERATORS | BEHAVIOR_REFINE_OPERATORS
_FIXED_OPERATOR_CYCLE = ('bx', 'bx', 'bx', 'br')
_KMEANS_N_INIT = 10
_BEHAVIOR_PARENT_COUNT = 2

class SearchEngine:

    def __init__(self, cfg, interface_prob):
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.prompt_seed_source = str(getattr(prompts, 'seed_func', '') or '')
        self.use_prompt_seed = os.environ.get('BEMRS_INIT_USE_PROMPT_SEED', '1').lower() not in {'0', 'false', 'no', 'off'}
        self.evol = Evolution(cfg.base_url, cfg.api_key, cfg.model, False, prompts, use_local_llm=False, url=None)
        self.problem_name = str(getattr(interface_prob, 'problem', '')).strip().lower()
        self.problem_size = int(getattr(interface_prob, 'problem_size', 50))
        self.problem_root_dir = getattr(interface_prob, 'root_dir', None)
        problem_config = getattr(interface_prob, 'config', None)
        task_problem_config = getattr(problem_config, 'problem', None)
        task_behavior_config = config_get(task_problem_config, 'behavior')
        self.debug = False
        cfg_max_fe = getattr(problem_config, 'max_fe', None)
        self.real_eval_budget = max(1, int(os.environ.get('BEMRS_REAL_EVAL_BUDGET', cfg_max_fe if cfg_max_fe is not None else 510)))
        self.real_eval_count = 0
        self.successful_real_eval_count = 0
        self.generated_algorithm_limit_multiplier = max(1, int(os.environ.get('BEMRS_GENERATED_ALGORITHM_LIMIT_MULTIPLIER', 10)))
        self.generated_algorithm_limit = int(self.generated_algorithm_limit_multiplier * self.real_eval_budget)
        self.generated_algorithm_count = 0
        self._evaluation_objective_cache = {}
        self._evaluation_algorithm_id_cache = {}
        self._evaluation_attempted_code_cache = set()
        self._last_n_evaluated = 0
        self._last_n_eval_skipped = 0
        self._last_existing_eval_duplicate_count = 0
        self._last_real_eval_batch_time = 0.0
        behavior_factory_kwargs = dict(problem_name=self.problem_name, problem_size=self.problem_size, root_dir=self.problem_root_dir, behavior_problem_size=getattr(task_behavior_config, 'problem_size', None), behavior_dataset=getattr(task_behavior_config, 'dataset', None), behavior_representation=getattr(task_behavior_config, 'representation', None))
        configured_behavior_matrices = getattr(task_behavior_config, 'matrices', None)
        configured_behavior_probes = getattr(task_behavior_config, 'probes', None)
        if configured_behavior_matrices is not None:
            behavior_factory_kwargs['behavior_matrices'] = configured_behavior_matrices
        if configured_behavior_probes is not None:
            behavior_factory_kwargs['behavior_probes'] = configured_behavior_probes
        behavior_embedder = build_behavior_embedder(**behavior_factory_kwargs)
        common_predictor_kwargs = dict(min_samples=int(os.environ.get('BEMRS_PREDICTOR_MIN_SAMPLES', 50)), max_samples=int(os.environ.get('BEMRS_PREDICTOR_MAX_SAMPLES', 100)), random_state=int(os.environ.get('BEMRS_PREDICTOR_SEED', 42)), surrogate_model='xgboost_direct', xgb_n_estimators=int(os.environ.get('BEMRS_XGB_N_ESTIMATORS', 250)), xgb_max_depth=int(os.environ.get('BEMRS_XGB_MAX_DEPTH', 3)), xgb_learning_rate=float(os.environ.get('BEMRS_XGB_LEARNING_RATE', 0.03)), xgb_subsample=float(os.environ.get('BEMRS_XGB_SUBSAMPLE', 0.8)), xgb_colsample_bytree=float(os.environ.get('BEMRS_XGB_COLSAMPLE_BYTREE', 0.8)), xgb_reg_alpha=float(os.environ.get('BEMRS_XGB_REG_ALPHA', 0.1)), xgb_reg_lambda=float(os.environ.get('BEMRS_XGB_REG_LAMBDA', 2.0)), xgb_min_child_weight=float(os.environ.get('BEMRS_XGB_MIN_CHILD_WEIGHT', 2.0)), xgb_tree_method=os.environ.get('BEMRS_XGB_TREE_METHOD', 'hist'), xgb_n_jobs=int(os.environ.get('BEMRS_XGB_N_JOBS', 1)))
        self.surrogate_selection_enabled = os.environ.get('BEMRS_SURROGATE_SELECTION_ENABLED', '1').lower() not in {'0', 'false', 'no', 'off'}
        self._behavior_predictor_kwargs = dict(common_predictor_kwargs)
        self.behavior_predictor = SurrogatePredictor(embedder=behavior_embedder, **common_predictor_kwargs)
        self.predictor = self.behavior_predictor
        self.embedder = self.behavior_predictor.embedder
        self.behavior_filter_enabled = os.environ.get('BEMRS_BEHAVIOR_FILTER_ENABLED', '1').lower() not in {'0', 'false', 'no', 'off'}
        self.behavior_filter_start_with_predictor = os.environ.get('BEMRS_BEHAVIOR_FILTER_START_WITH_PREDICTOR', '0').lower() not in {'0', 'false', 'no', 'off'}
        self.behavior_filter_atol = max(0.0, float(os.environ.get('BEMRS_BEHAVIOR_FILTER_ATOL', 1e-08)))
        self.behavior_filter_rtol = max(0.0, float(os.environ.get('BEMRS_BEHAVIOR_FILTER_RTOL', 1e-06)))
        self.behavior_filter_xgb_rescue_enabled = os.environ.get('BEMRS_BEHAVIOR_FILTER_XGB_RESCUE', '0').lower() not in {'0', 'false', 'no', 'off'}
        self._behavior_filter_active_logged = False
        self._behavior_filter_predictor_ready_for_batch = None
        self.behavior_novelty_slot_enabled = os.environ.get('BEMRS_BEHAVIOR_NOVELTY_SLOT_ENABLED', '1').lower() not in {'0', 'false', 'no', 'off'}
        self._behavior_trained_samples = 0
        self._behavior_trained_samples_generation = None
        self._evaluated_algorithm_history = []
        self._evaluated_algorithm_codes = set()
        self.region_enabled = os.environ.get('BEMRS_REGION_ENABLED', '1').lower() not in {'0', 'false', 'no', 'off'}
        self.region_count = max(1, int(os.environ.get('BEMRS_REGION_COUNT', 3)))
        archive_target_raw = os.environ.get('BEMRS_REGION_ARCHIVE_TARGET_DISTINCT')
        if archive_target_raw is None or str(archive_target_raw).strip() == '':
            raise ValueError('BeMRS requires region.archive_target_distinct.')
        self.region_archive_target_distinct = max(1, int(archive_target_raw))
        self.region_candidates_per_region = max(1, int(os.environ.get('BEMRS_REGION_CANDIDATES_PER_REGION', 4)))
        total_candidates_raw = os.environ.get('BEMRS_REGION_TOTAL_CANDIDATES')
        self.region_total_candidates = max(1, int(total_candidates_raw)) if total_candidates_raw else None
        self.region_relative_improvement = max(0.0, float(os.environ.get('BEMRS_REGION_RELATIVE_IMPROVEMENT', 0.0001)))
        self.region_rebuild_tolerance = max(1, int(os.environ.get('BEMRS_REGION_REBUILD_TOLERANCE', 10)))
        self.preserve_global_best_anchor = os.environ.get('BEMRS_PRESERVE_GLOBAL_BEST_ANCHOR', '0').lower() not in {'0', 'false', 'no', 'off'}
        self.region_builder = 'behavior_kmeans'
        self.bx_balanced_parent_schedule = False
        self.region_scale_window = max(4, int(os.environ.get('BEMRS_REGION_SCALE_WINDOW', 100)))
        from bemrs.offspring_plan import evaluation_slots
        self.evaluation_ratio = float(os.environ.get('BEMRS_EVALUATION_RATIO', '0.1'))
        evaluation_slots(0, self.evaluation_ratio, 0)
        self.regional_allocated_count = {}
        self._regional_allocation_cursor = None
        self.region_seed = int(os.environ.get('BEMRS_REGION_SEED', os.environ.get('BEMRS_PREDICTOR_SEED', 42)))
        self._region_rng = np.random.default_rng(self.region_seed)
        self._active_region_operator_schedule = {}
        self._active_region_operator_schedule_generation = None
        self._fixed_operator_cycle_index = 0
        self._region_scaler = None
        self._region_feature_dim = None
        self._region_space_dim = None
        self._region_states = []
        self._region_fit_count = 0
        self._region_center_history_ids = set()
        self._region_center_history = []
        self._kmeans_region_stats = {}
        self._kmeans_full_rebuild_count = 0
        self._active_parent_batch_signatures = set()
        self._bx_region_round_count = 0
        self._bx_use_inter_region_this_round = False
        # Cross-region BX cycles through 1..K parents, where K is the number
        # of currently non-empty region parent pools.  The cursor persists
        # across batches and rounds; rejected duplicate attempts also advance
        # it so the schedule cannot repeatedly favor one parent count.
        self._bx_inter_region_parent_count_cursor = 0
        self.llm_batch_generation_enabled = os.environ.get('BEMRS_LLM_BATCH_GENERATION', '1').lower() not in {'0', 'false', 'no'}
        self.llm_max_workers = max(1, int(os.environ.get('BEMRS_LLM_MAX_WORKERS', 6)))
        self.run_output_path = None
        self.init_unique_candidate_target = self.region_archive_target_distinct
        self._initialization_complete = False
        self.timing_csv_path = os.environ.get('BEMRS_TIMING_CSV', 'bemrs_timing_records.csv')
        self.lineage_jsonl_path = os.environ.get('BEMRS_LINEAGE_JSONL')
        self.lineage_edges_csv_path = os.environ.get('BEMRS_LINEAGE_EDGES_CSV')
        self.lineage_save_features = os.environ.get('BEMRS_LINEAGE_SAVE_FEATURES', '1').lower() not in {'0', 'false', 'no'}
        self.current_generation = 0
        self.lineage_generated_count = 0
        self._lineage_pending = {}
        self.selection_log_top = max(0, int(os.environ.get('BEMRS_SELECTION_LOG_TOP', 10)))
        if not self.debug:
            warnings.filterwarnings('ignore')
        logging.info('[Predictor] global_xgboost_direct enabled=%s input_dim=%s min_samples=%s window=%s update=once_per_generation_with_new_samples', self.surrogate_selection_enabled, self.behavior_predictor.emb_dim, self.behavior_predictor.min_samples, self.behavior_predictor.max_samples)
        logging.info('[RealEvalBudget] exact successful-evaluation target=%s', self.real_eval_budget)
        logging.info('[InitConfig] fixed_slots=%s seed_counts_as_slot=True sequential=True context_recent_evaluated=10 refill=False next=regions', self.init_unique_candidate_target)
        logging.info('[GeneratedAlgorithmBudget] multiplier=%s limit=%s real_eval_target=%s seed_counted=False', self.generated_algorithm_limit_multiplier, self.generated_algorithm_limit, self.real_eval_budget)
        logging.info('[GenerationConfig] llm_batch=%s llm_workers=%s', self.llm_batch_generation_enabled, self.llm_max_workers)
        logging.info('[BehaviorOperatorConfig] intra_BX_parents=2 BR_parents=2 inter_BX_parents=cycle_1_to_K intra_BX=reciprocal_rank_farthest inter_BX=cycle_1_to_K_top_priority BR=reciprocal_rank_nearest insufficient_parents=skip rank_probability=(1/rank)/sum(1/rank)')
        logging.info('[RegionConfig] regions=%s archive_target_distinct=%s evaluation_ratio=%s start=initialization_complete rebuild_tolerance=%s', self.region_count, self.region_archive_target_distinct, self.evaluation_ratio, self.region_rebuild_tolerance)
        logging.info('[CandidateFilterConfig] code_dedupe=enabled already_evaluated_filter=enabled behavior_dedupe=removed parent_pool_order=true_score parent_pool_dedupe=(code,score)')
        logging.info('[RegionOperatorConfig] policy=mixed_parent_coverage kinds=intra_bx,intra_br,inter_bx dedupe=operator_and_unordered_codes')

    def set_timing_output(self, output_path):
        self.run_output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        if not os.environ.get('BEMRS_TIMING_CSV'):
            self.timing_csv_path = os.path.join(output_path, 'timing_records.csv')
        if not os.environ.get('BEMRS_LINEAGE_JSONL'):
            self.lineage_jsonl_path = os.path.join(output_path, 'algorithm_lineage.jsonl')
        if not os.environ.get('BEMRS_LINEAGE_EDGES_CSV'):
            self.lineage_edges_csv_path = os.path.join(output_path, 'algorithm_lineage_edges.csv')
        logging.info('[Lineage] outputs | records=%s edges=%s save_features=%s', self.lineage_jsonl_path, self.lineage_edges_csv_path, self.lineage_save_features)

    def set_current_generation(self, generation):
        self.current_generation = max(0, int(generation))

    def get_real_eval_count(self):
        return int(self.real_eval_count)

    def get_remaining_real_evals(self):
        return self.get_remaining_successful_evals()

    def get_successful_real_eval_count(self):
        return int(self.successful_real_eval_count)




    def should_use_regions(self):
        return bool(self.region_enabled and getattr(self, '_initialization_complete', False) and self._evaluated_algorithm_history and (not self.is_real_eval_budget_exhausted()))

    def _region_generation_profiles(self, states=None):
        """Describe regions while assigning an equal generation share."""
        states = list(states if states is not None else self._region_states)
        if not states:
            return {}
        finite_bests = [self._safe_objective(state.get('best_objective')) for state in states if np.isfinite(self._safe_objective(state.get('best_objective')))]
        global_best = min(finite_bests) if finite_bests else 0.0
        equal_weight = 1.0 / len(states)
        rows = []
        for state in states:
            region_id = int(state['region_id'])
            best = self._safe_objective(state.get('best_objective'))
            gap = self._region_relative_global_gap(best, global_best) if np.isfinite(best) else float('inf')
            rows.append({'region_id': region_id, 'best': float(best), 'gap': float(gap), 'generation_weight': float(equal_weight)})
        return {row['region_id']: row for row in rows}

    @staticmethod
    def _equal_integer_quotas(region_ids, total_count):
        """Split an integer generation budget equally and deterministically."""
        region_ids = sorted((int(region_id) for region_id in region_ids))
        total_count = max(0, int(total_count))
        quotas = {region_id: 0 for region_id in region_ids}
        if not region_ids or total_count <= 0:
            return quotas
        base, remainder = divmod(total_count, len(region_ids))
        for index, region_id in enumerate(region_ids):
            quotas[region_id] = int(base + (index < remainder))
        return quotas

    def _region_generation_quotas(self, states=None, total_count=None):
        states = list(states if states is not None else self._region_states)
        profiles = self._region_generation_profiles(states)
        region_ids = sorted(profiles)
        if total_count is None:
            total_count = len(region_ids) * int(self.region_candidates_per_region)
        quotas = self._equal_integer_quotas(region_ids, total_count)
        return (quotas, profiles)

    def prepare_region_operator_schedule(self, operators):
        """Apply the deterministic BX, BX, BX, BR cycle to every region."""
        generation = int(getattr(self, 'current_generation', 0))
        if self._active_region_operator_schedule_generation == generation and self._active_region_operator_schedule:
            return dict(self._active_region_operator_schedule)
        self._active_region_operator_schedule = {}
        self._active_region_operator_schedule_generation = generation
        if not self.should_use_regions() or not self._initialize_regions():
            return {}
        available = tuple(dict.fromkeys((str(name) for name in operators or [])))
        if not available:
            return {}
        cycle_index = int(getattr(self, '_fixed_operator_cycle_index', 0))
        preferred_operator = _FIXED_OPERATOR_CYCLE[cycle_index % len(_FIXED_OPERATOR_CYCLE)]
        if preferred_operator in available:
            active_operator = preferred_operator
        else:
            active_operator = next((name for name in _FIXED_OPERATOR_CYCLE if name in available), available[0])
        self._fixed_operator_cycle_index = cycle_index + 1
        schedule = {}
        ordered_states = sorted(self._region_states, key=lambda row: int(row.get('region_id', 0)))
        generation_quotas, generation_profiles = self._region_generation_quotas(ordered_states)
        for state in ordered_states:
            region_id = int(state['region_id'])
            schedule[region_id] = {'operator': active_operator, 'mode': 'fixed_3_to_1', 'generation_quota': int(generation_quotas.get(region_id, 0)), 'generation_weight': float(generation_profiles[region_id]['generation_weight']), 'global_gap': float(generation_profiles[region_id]['gap'])}
        self._active_region_operator_schedule = schedule
        logging.info('[RegionOperatorSchedule] generation=%s operator=%s fixed_ratio=3:1 assignments=%s', generation, active_operator, schedule)
        return dict(schedule)

    def _scheduled_region_ids(self, operator):
        schedule = getattr(self, '_active_region_operator_schedule', {})
        if not schedule:
            return None
        return {int(region_id) for region_id, row in schedule.items() if row.get('operator') == operator}

    def _region_operator_metadata(self, region_id, operator=None):
        if region_id is None:
            return {}
        row = getattr(self, '_active_region_operator_schedule', {}).get(int(region_id), {})
        if operator is not None and row.get('operator') not in {None, operator}:
            return {}
        return dict(row)

    def _operator_region_generation_quotas(self, operator, region_ids, total_count):
        """Return this operator's share of the generation-level TR quotas."""
        region_ids = sorted((int(region_id) for region_id in region_ids))
        total_count = max(0, int(total_count))
        schedule = getattr(self, '_active_region_operator_schedule', {})
        scheduled_rows = {region_id: schedule.get(region_id, {}) for region_id in region_ids if schedule.get(region_id, {}).get('operator') in {None, operator}}
        if schedule and scheduled_rows:
            planned = {region_id: int(row.get('generation_quota', 0)) for region_id, row in scheduled_rows.items()}
            if sum(planned.values()) == total_count:
                return planned
            return self._equal_integer_quotas(sorted(scheduled_rows), total_count)
        states = [state for state in self._region_states if int(state['region_id']) in region_ids]
        quotas, _ = self._region_generation_quotas(states, total_count=total_count)
        return quotas

    def get_remaining_successful_evals(self):
        return max(0, int(self.real_eval_budget - self.successful_real_eval_count))

    def get_generated_algorithm_count(self):
        return int(self.generated_algorithm_count)

    def get_remaining_generated_algorithm_slots(self):
        return max(0, int(self.generated_algorithm_limit - self.generated_algorithm_count))

    def raise_if_generated_algorithm_limit_reached(self, context='search'):
        """Fail once the LLM candidate limit is spent before max_fe completes."""
        if self.get_remaining_generated_algorithm_slots() <= 0 and (not self.is_real_eval_budget_exhausted()):
            raise RuntimeError(f'Generated-algorithm safety limit reached during {context}: generated {self.get_generated_algorithm_count()}/{self.generated_algorithm_limit} candidate slots ({self.generated_algorithm_limit_multiplier}x the real-evaluation target), but only {self.get_successful_real_eval_count()}/{self.real_eval_budget} successful real evaluations completed.')

    def _claim_generated_algorithm_slots(self, count, context):
        """Reserve LLM candidate slots before issuing generation requests."""
        count = max(0, int(count))
        if count <= 0:
            return
        remaining = self.get_remaining_generated_algorithm_slots()
        if count > remaining:
            raise RuntimeError(f'Generated-algorithm request exceeds the remaining safety limit during {context}: requested={count}, remaining={remaining}, generated={self.get_generated_algorithm_count()}/{self.generated_algorithm_limit}.')
        self.generated_algorithm_count += count
        logging.info('[GeneratedAlgorithmBudget] context=%s claimed=%s generated=%s/%s successful_evals=%s/%s', context, count, self.get_generated_algorithm_count(), self.generated_algorithm_limit, self.get_successful_real_eval_count(), self.real_eval_budget)

    def get_invalid_real_eval_count(self):
        return max(0, self.get_real_eval_count() - self.get_successful_real_eval_count())

    def is_real_eval_budget_exhausted(self):
        return self.get_remaining_successful_evals() <= 0

    def _record_real_evaluations(self, n_evaluated, n_successful=None):
        n_evaluated = max(0, int(n_evaluated))
        if n_successful is None:
            n_successful = n_evaluated
        n_successful = max(0, min(n_evaluated, int(n_successful)))
        self.real_eval_count += n_evaluated
        self.successful_real_eval_count += n_successful
        if self.successful_real_eval_count > self.real_eval_budget:
            raise RuntimeError(f'Successful evaluation target exceeded: {self.successful_real_eval_count}/{self.real_eval_budget}.')
        logging.info('[RealEvalBudget] attempted_batch=%s successful_batch=%s invalid_batch=%s attempted=%s successful=%s/%s invalid=%s remaining_successful=%s', n_evaluated, n_successful, n_evaluated - n_successful, self.real_eval_count, self.successful_real_eval_count, self.real_eval_budget, self.get_invalid_real_eval_count(), self.get_remaining_successful_evals())

    def _append_timing_record(self, record):
        if not record or not self.timing_csv_path:
            return
        fieldnames = ['event', 'operator', 'n_generated', 'n_candidates', 'n_after_feature_filter', 'n_selected', 'n_evaluated', 'n_skipped', 'generation_batch_time', 'behavior_feature_batch_time', 'embedding_batch_time', 'predictor_selection_time', 'real_eval_batch_time']
        path = self.timing_csv_path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({name: record.get(name) for name in fieldnames})

    @staticmethod
    def _fmt_float(value, width=9, precision=4):
        try:
            value = float(value)
            if not np.isfinite(value):
                return f"{'nan':>{width}}"
            return f'{value:>{width}.{precision}g}'
        except Exception:
            return f"{'-':>{width}}"

    def _log_selection_ranking(self, operator, ranked_candidates, selected_ids, cur_fe, top_k):
        if not ranked_candidates or self.selection_log_top <= 0:
            return
        display_n = min(len(ranked_candidates), max(self.selection_log_top, int(top_k)))
        logging.info('[BehaviorRank] operator=%s top_k=%s selected=%s/%s predictor=behavior_only extractor=%s', operator, top_k, len(selected_ids), len(ranked_candidates), getattr(self.embedder, 'extractor_version', 'unknown'))
        logging.info('[BehaviorRank] showing top %s/%s by global behavior prediction; stars are selected candidates', display_n, len(ranked_candidates))
        logging.info('[BehaviorRank] %4s %3s %3s %6s %9s %12s', 'rank', 'sel', 'region', 'source', 'prediction', 'code_hash')
        for rank, cand in enumerate(ranked_candidates[:display_n], start=1):
            off = self._candidate_to_offspring(cand)
            code = off.get('code', '')
            code_hash = str(abs(hash(self._canonical_code(code))))[-12:]
            source = {'region': 'region', 'advantage_archive': 'archive'}.get(off.get('generation_parent_scope'), '-')
            logging.debug('[BehaviorRank] %4d %3s %3s %6s %9s %12s', rank, '*' if id(cand) in selected_ids else '', off.get('region_allocation_id', '-'), source, self._fmt_float(off.get('behavior_pred_score')), code_hash)

    def _log_post_eval_rankings(self, evaluated_candidates):
        if not evaluated_candidates:
            return
        rows = []
        for cand in evaluated_candidates:
            off = self._candidate_to_offspring(cand)
            pred = self._finite_or_none(off.get('ensemble_score'))
            obj = self._finite_or_none(off.get('objective'))
            if pred is None or obj is None:
                continue
            code = off.get('code', '')
            rows.append({'cand': cand, 'pred': pred, 'obj': obj, 'behavior_pred': self._finite_or_none(off.get('behavior_pred_score')), 'embedding_pred': self._finite_or_none(off.get('embedding_pred_score')), 'code_hash': str(abs(hash(self._canonical_code(code))))[-12:], 'shared': bool(off.get('eval_shared', False))})
        if not rows:
            return
        pred_sorted = sorted(rows, key=lambda row: row['pred'])
        true_sorted = sorted(rows, key=lambda row: row['obj'])
        pred_rank_by_id = {id(row['cand']): rank for rank, row in enumerate(pred_sorted, start=1)}
        true_rank_by_id = {id(row['cand']): rank for rank, row in enumerate(true_sorted, start=1)}
        display_n = min(len(rows), max(self.selection_log_top, len(rows)))

        def log_table(title, ordered_rows):
            logging.debug('[PostEvalRank] %s | showing %s/%s evaluated candidates (lower is better)', title, display_n, len(rows))
            logging.debug('[PostEvalRank] %4s %4s %4s %9s %9s %9s %6s %12s', 'row', 'pr', 'tr', 'true', 'prediction', 'b_pred', 'shared', 'code_hash')
            for row_idx, row in enumerate(ordered_rows[:display_n], start=1):
                logging.debug('[PostEvalRank] %4d %4d %4d %s %s %s %6s %12s', row_idx, pred_rank_by_id[id(row['cand'])], true_rank_by_id[id(row['cand'])], self._fmt_float(row['obj']), self._fmt_float(row['pred']), self._fmt_float(row['behavior_pred']), 'Y' if row['shared'] else '', row['code_hash'])
        log_table('prediction order', pred_sorted)
        log_table('true objective order', true_sorted)

    def add2pop(self, population, offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print('duplicated result, retrying ... ')
                return False
        population.append(offspring)
        return True

    def check_duplicate(self, population, code):
        code_key = self._canonical_code(code)
        for ind in population:
            if code_key == self._canonical_code(ind.get('code', '')):
                return True
        return False

    @staticmethod
    def _candidate_to_offspring(candidate):
        return candidate[1] if isinstance(candidate, tuple) else candidate

    @staticmethod
    def _canonical_code(code):
        code = str(code or '')
        code = re.sub('```(?:python)?', '', code)
        code = code.replace('```', '')
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        return '\n'.join((line.rstrip() for line in code.strip().splitlines()))

    @classmethod
    def _algorithm_id(cls, code):
        canonical = cls._canonical_code(code)
        if not canonical:
            return None
        digest = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
        return f'alg_{digest[:16]}'

    @classmethod
    def _lineage_json_safe(cls, value):
        if isinstance(value, np.ndarray):
            return [cls._lineage_json_safe(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return cls._lineage_json_safe(value.item())
        if isinstance(value, dict):
            return {str(key): cls._lineage_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._lineage_json_safe(item) for item in value]
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return str(value)

    def _parent_snapshot(self, parent):
        if not isinstance(parent, dict):
            return None
        code = parent.get('code', '')
        algorithm_id = parent.get('algorithm_id') or self._algorithm_id(code)
        if algorithm_id is not None:
            parent['algorithm_id'] = algorithm_id
        return {'algorithm_id': algorithm_id, 'lineage_event_id': parent.get('lineage_event_id'), 'generation': parent.get('generation'), 'operator': parent.get('operator'), 'generation_parent_scope': parent.get('generation_parent_scope'), 'region_source_id': parent.get('region_source_id'), 'algorithm': parent.get('algorithm'), 'code': code, 'objective': self._finite_or_none(parent.get('objective'))}

    def _register_lineage_event(self, parents, offspring, operator):
        if not isinstance(offspring, dict):
            return None
        if parents is None:
            parent_list = []
        elif isinstance(parents, dict):
            parent_list = [parents]
        else:
            parent_list = list(parents)
        parent_rows = [row for row in (self._parent_snapshot(parent) for parent in parent_list) if row is not None]
        self.lineage_generated_count += 1
        generated_order = int(self.lineage_generated_count)
        event_id = f'lin_{uuid.uuid4().hex}'
        algorithm_id = self._algorithm_id(offspring.get('code', ''))
        offspring['lineage_event_id'] = event_id
        offspring['algorithm_id'] = algorithm_id
        offspring['generated_order'] = generated_order
        offspring['parent_algorithm_ids'] = [row['algorithm_id'] for row in parent_rows if row.get('algorithm_id') is not None]
        offspring['generation'] = int(self.current_generation)
        offspring['operator'] = str(operator)
        offspring.setdefault('lineage_status', 'generated')
        self._lineage_pending[event_id] = {'event_id': event_id, 'generated_order': generated_order, 'created_at_unix': time.time(), 'generation': int(self.current_generation), 'operator': str(operator), 'parents': parent_rows, 'child': {'algorithm_id': algorithm_id, 'algorithm': offspring.get('algorithm'), 'code': offspring.get('code'), 'generation_parent_scope': offspring.get('generation_parent_scope'), 'region_source_id': offspring.get('region_source_id')}, 'llm': {'prompt': offspring.pop('_lineage_prompt', None), 'response': offspring.pop('_lineage_response', None), 'attempts': offspring.pop('_lineage_llm_attempts', None)}}
        return event_id

    def _set_lineage_status(self, candidates, status, **fields):
        for candidate in candidates or []:
            offspring = self._candidate_to_offspring(candidate)
            if not isinstance(offspring, dict):
                continue
            offspring['lineage_status'] = str(status)
            offspring.update(fields)

    def _append_lineage_edge_rows(self, record):
        if not self.lineage_edges_csv_path:
            return
        parents = record.get('parents', [])
        if not parents:
            return
        path = self.lineage_edges_csv_path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fieldnames = ['event_id', 'generated_order', 'generation', 'operator', 'generation_parent_scope', 'region_source_id', 'region_membership_id', 'region_allocation_id', 'parent_index', 'parent_algorithm_id', 'child_algorithm_id', 'parent_objective', 'child_objective', 'lineage_status', 'selected_for_evaluation', 'real_eval_attempted', 'real_eval_success']
        write_header = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for parent_index, parent in enumerate(parents):
                writer.writerow({'event_id': record.get('event_id'), 'generated_order': record.get('generated_order'), 'generation': record.get('generation'), 'operator': record.get('operator'), 'generation_parent_scope': record['child'].get('generation_parent_scope'), 'region_source_id': record['child'].get('region_source_id'), 'region_membership_id': record['child'].get('region_membership_id'), 'region_allocation_id': record['child'].get('region_allocation_id'), 'parent_index': parent_index, 'parent_algorithm_id': parent.get('algorithm_id'), 'child_algorithm_id': record['child'].get('algorithm_id'), 'parent_objective': parent.get('objective'), 'child_objective': record['child'].get('objective'), 'lineage_status': record['pipeline'].get('status'), 'selected_for_evaluation': record['pipeline'].get('selected_for_evaluation'), 'real_eval_attempted': record['evaluation'].get('attempted'), 'real_eval_success': record['evaluation'].get('success')})

    def _finalize_lineage_candidates(self, candidates):
        if not candidates:
            return
        written = 0
        status_counts = {}
        for candidate in candidates:
            offspring = self._candidate_to_offspring(candidate)
            if not isinstance(offspring, dict):
                continue
            event_id = offspring.get('lineage_event_id')
            record = self._lineage_pending.pop(event_id, None)
            if record is None:
                continue
            record['child'].update({'algorithm_id': offspring.get('algorithm_id'), 'algorithm': offspring.get('algorithm'), 'code': offspring.get('code'), 'objective': self._finite_or_none(offspring.get('objective')), 'generation_parent_scope': offspring.get('generation_parent_scope'), 'region_source_id': offspring.get('region_source_id'), 'region_membership_id': offspring.get('region_membership_id'), 'region_allocation_id': offspring.get('region_allocation_id')})
            record['pipeline'] = {'status': offspring.get('lineage_status', 'generated'), 'generation_attempts': offspring.get('generation_attempts'), 'generation_parent_scope': offspring.get('generation_parent_scope'), 'region_source_id': offspring.get('region_source_id'), 'region_membership_id': offspring.get('region_membership_id'), 'region_allocation_id': offspring.get('region_allocation_id'), 'behavior_feature_failed': bool(offspring.get('behavior_feature_failed', False)), 'skip_real_eval': bool(offspring.get('skip_real_eval', False)), 'skip_reason': offspring.get('skip_reason'), 'selected_by_predictor': offspring.get('selected_by_predictor'), 'selected_for_evaluation': offspring.get('selected_for_evaluation'), 'initialization_context': offspring.get('initialization_context'), 'duplicate_of_algorithm_id': offspring.get('duplicate_of_algorithm_id'), 'behavior_duplicate_scope': offspring.get('behavior_duplicate_scope'), 'selection_method': offspring.get('selection_method'), 'behavior_novelty_slot': bool(offspring.get('behavior_novelty_slot', False)), 'behavior_archive_novelty_distance': self._finite_or_none(offspring.get('behavior_archive_novelty_distance'))}
            record['pipeline']['generation_kind'] = offspring.get('generation_kind')
            record['prediction'] = {'prediction': self._finite_or_none(offspring.get('behavior_pred_score'))}
            record['evaluation'] = {'attempted': bool(offspring.get('real_eval_attempted', False)), 'success': bool(offspring.get('real_eval_success', False)), 'shared': bool(offspring.get('eval_shared', False)), 'status': offspring.get('evaluation_status')}
            if self.lineage_save_features:
                record['features'] = {'behavior': offspring.get('behavior_embedding')}
            record = self._lineage_json_safe(record)
            if self.lineage_jsonl_path:
                path = self.lineage_jsonl_path
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                with open(path, 'a', encoding='utf-8') as file:
                    file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            self._append_lineage_edge_rows(record)
            written += 1
            status = record.get('pipeline', {}).get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        if written:
            logging.info('[Lineage] saved=%s generation=%s statuses=%s', written, self.current_generation, status_counts)

    @staticmethod
    def _safe_objective(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else float('inf')
        except Exception:
            return float('inf')

    def _cache_evaluated_objective(self, code, objective, algorithm_id=None):
        """Remember the first finite real-evaluation result for canonical code."""
        code_key = self._canonical_code(code)
        value = self._safe_objective(objective)
        if not code_key or not np.isfinite(value):
            return False
        cache = getattr(self, '_evaluation_objective_cache', None)
        if cache is None:
            cache = {}
            self._evaluation_objective_cache = cache
        id_cache = getattr(self, '_evaluation_algorithm_id_cache', None)
        if id_cache is None:
            id_cache = {}
            self._evaluation_algorithm_id_cache = id_cache
        if code_key not in cache:
            cache[code_key] = float(value)
            id_cache[code_key] = algorithm_id or self._algorithm_id(code_key)
            return True
        return False

    def _cached_evaluated_objective(self, code):
        code_key = self._canonical_code(code)
        cache = getattr(self, '_evaluation_objective_cache', {})
        if not code_key or code_key not in cache:
            return None
        value = self._safe_objective(cache[code_key])
        return float(value) if np.isfinite(value) else None

    @staticmethod
    def _finite_or_none(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except Exception:
            return None

    def _train_predictor_if_needed(self, predictor, attr_name, label):
        """Refresh once per generation, before use, only with new real samples.

        Filtering and ranking share the model trained on the first eligible
        call of a generation. Samples added by a later fallback operator are
        incorporated in the next generation. Failed fits are retried only in
        a later generation, without marking those samples as trained.
        """
        sample_count = predictor.get_sample_count()
        trained_count = int(getattr(self, attr_name, 0))
        generation = int(getattr(self, 'current_generation', 0))
        generation_attr = attr_name + '_generation'
        last_attempt_generation = getattr(self, generation_attr, None)
        new_samples = sample_count > trained_count
        if sample_count < predictor.min_samples:
            reason = 'below_min_samples'
        elif not new_samples:
            reason = 'no_new_samples'
        elif last_attempt_generation == generation:
            reason = 'already_attempted_this_generation'
        else:
            # Record attempts as well as successes: no repeated fits by the
            # filter/ranking callers, even if the fit fails.
            setattr(self, generation_attr, generation)
            train_start = time.perf_counter()
            fit_succeeded = bool(predictor.train())
            train_time = time.perf_counter() - train_start
            if fit_succeeded and predictor.is_ready():
                setattr(self, attr_name, sample_count)
            logging.info('[PredictorTrain] %s fit | generation=%s samples=%s previous_trained=%s added=%s update=once_per_generation_with_new_samples time=%.3fs fit_success=%s ready=%s', label, generation, sample_count, trained_count, sample_count - trained_count, train_time, fit_succeeded, predictor.is_ready())
            return predictor.is_ready()
        logging.info('[PredictorTrain] %s reuse | generation=%s samples=%s trained_samples=%s reason=%s ready=%s', label, generation, sample_count, trained_count, reason, predictor.is_ready())
        return predictor.is_ready()

    def _ensure_surrogate(self):
        if not self.surrogate_selection_enabled or self.behavior_predictor is None:
            return False
        if self.behavior_filter_start_with_predictor and self._behavior_filter_predictor_ready_for_batch is False:
            return False
        if self.get_successful_real_eval_count() < self.behavior_predictor.min_samples:
            return False
        ready = False
        if self.behavior_predictor.get_sample_count() >= self.behavior_predictor.min_samples:
            ready = self._train_predictor_if_needed(self.behavior_predictor, '_behavior_trained_samples', 'behavior_global_xgboost_direct')
        return ready

    def _predict_behavior(self, candidate, global_ready):
        offspring = self._candidate_to_offspring(candidate)
        feature = offspring.get('behavior_embedding', offspring.get('embedding'))
        if feature is None:
            return None
        global_mu = None
        if global_ready:
            global_mu, _ = self.behavior_predictor.predict(feature)
        if global_mu is not None and (not np.isfinite(global_mu)):
            global_mu = None
        if global_mu is not None:
            predicted = float(global_mu)
            model_source = 'global_xgboost_direct'
        else:
            predicted = None
            model_source = 'unavailable'
        offspring['global_behavior_pred_score'] = float(global_mu) if global_mu is not None else None
        offspring['behavior_pred_score'] = predicted
        offspring['pred_score'] = predicted
        offspring['ensemble_score'] = predicted
        offspring['surrogate_model_source'] = model_source
        return predicted

    def _dedupe_candidates_by_code(self, candidates):
        unique = []
        duplicates = []
        seen = {}
        for cand in candidates:
            off = self._candidate_to_offspring(cand)
            code_key = self._canonical_code(off.get('code', ''))
            if not code_key:
                off['lineage_status'] = 'invalid_or_empty_code'
                duplicates.append(cand)
                continue
            if code_key in seen:
                off['lineage_status'] = 'duplicate_code'
                off['duplicate_of_algorithm_id'] = seen[code_key]
                duplicates.append(cand)
                continue
            seen[code_key] = off.get('algorithm_id') or self._algorithm_id(off.get('code', ''))
            unique.append(cand)
        return (unique, duplicates)

    def _behavior_vectors_match(self, left, right):
        """Return True only for essentially identical finite raw behavior."""
        left = np.asarray(left, dtype=np.float32).reshape(-1)
        right = np.asarray(right, dtype=np.float32).reshape(-1)
        if left.size == 0 or left.shape != right.shape or (not np.all(np.isfinite(left))) or (not np.all(np.isfinite(right))):
            return False
        return bool(np.allclose(left, right, atol=float(getattr(self, 'behavior_filter_atol', 1e-08)), rtol=float(getattr(self, 'behavior_filter_rtol', 1e-06))))

    def _behavior_filter_predictor_ready(self):
        """Return whether a predictor-gated behavior filter may run now.

        BeMRS intentionally shares the XGBoost warm-up boundary.  Training
        is triggered here because behavior filtering precedes the ordinary
        surrogate-ranking call in the candidate pipeline.
        """
        requires_predictor = bool(getattr(self, 'behavior_filter_start_with_predictor', False) or getattr(self, 'behavior_filter_xgb_rescue_enabled', False))
        if not requires_predictor:
            return True
        predictor = getattr(self, 'behavior_predictor', None)
        if not getattr(self, 'surrogate_selection_enabled', False) or predictor is None or getattr(predictor, 'surrogate_model', None) != 'xgboost_direct':
            return False
        if self.get_successful_real_eval_count() < int(self.behavior_predictor.min_samples):
            return False
        if predictor.get_sample_count() < int(predictor.min_samples):
            return False
        ready = self._train_predictor_if_needed(predictor, '_behavior_trained_samples', 'behavior_filter_xgboost_direct')
        if ready and (not getattr(self, '_behavior_filter_active_logged', False)):
            logging.info('[BehaviorDuplicateFilter] activated with XGBoost | successful_evals=%s samples=%s atol=%g rtol=%g rescue=%s', self.get_successful_real_eval_count(), predictor.get_sample_count(), float(getattr(self, 'behavior_filter_atol', 1e-08)), float(getattr(self, 'behavior_filter_rtol', 1e-06)), bool(getattr(self, 'behavior_filter_xgb_rescue_enabled', False)))
            self._behavior_filter_active_logged = True
        return bool(ready and predictor.is_ready())

    def _predict_behavior_filter_rescue(self, feature, anchor_objective):
        """Return the XGBoost rescue decision without using candidate truth."""
        if not getattr(self, 'behavior_filter_xgb_rescue_enabled', False):
            return (False, None)
        anchor_objective = self._finite_or_none(anchor_objective)
        if anchor_objective is None:
            return (False, None)
        prediction, _ = self.behavior_predictor.predict(feature)
        prediction = self._finite_or_none(prediction)
        return (bool(prediction is not None and prediction < anchor_objective), prediction)

    def _filter_candidates_by_raw_behavior(self, candidates):
        """Filter raw-behavior duplicates against history and this batch.

        The first representative is retained. Historical features come from
        every successful real evaluation in ``behavior_predictor._X`` and are
        not restricted by the surrogate's recent-sample training cap.
        """
        candidates = list(candidates or [])
        if not self.behavior_filter_enabled or not candidates:
            return (candidates, [])
        predictor_ready = self._behavior_filter_predictor_ready()
        self._behavior_filter_predictor_ready_for_batch = bool(predictor_ready)
        if not predictor_ready:
            predictor = getattr(self, 'behavior_predictor', None)
            logging.info('[BehaviorDuplicateFilter] waiting for XGBoost; pass through input=%s successful_evals=%s/%s samples=%s/%s', len(candidates), self.get_successful_real_eval_count(), int(self.behavior_predictor.min_samples), predictor.get_sample_count() if predictor is not None else 0, getattr(predictor, 'min_samples', 0))
            return (candidates, [])
        history_features = list(getattr(self.behavior_predictor, '_X', []))
        history_codes = list(getattr(self.behavior_predictor, '_codes', []))
        history_objectives = list(getattr(self.behavior_predictor, '_y', []))
        history = []
        for index, feature in enumerate(history_features):
            code = history_codes[index] if index < len(history_codes) else ''
            code_key = self._canonical_code(code)
            history.append((feature, getattr(self, '_evaluation_algorithm_id_cache', {}).get(code_key), history_objectives[index] if index < len(history_objectives) else None))
        retained = []
        filtered = []
        batch_representatives = []
        history_filtered = 0
        batch_filtered = 0
        rescued = 0
        for candidate in candidates:
            offspring = self._candidate_to_offspring(candidate)
            feature = offspring.get('behavior_embedding', offspring.get('embedding'))
            match_scope = None
            match_algorithm_id = None
            match_anchor_objective = None
            if not self._behavior_feature_failed(feature):
                if getattr(self, 'behavior_filter_xgb_rescue_enabled', False):
                    for batch_feature, batch_algorithm_id in batch_representatives:
                        if self._behavior_vectors_match(feature, batch_feature):
                            match_scope = 'same_batch'
                            match_algorithm_id = batch_algorithm_id
                            break
                if match_scope is None:
                    for archived_feature, archived_algorithm_id, archived_objective in history:
                        if self._behavior_vectors_match(feature, archived_feature):
                            match_scope = 'evaluated_history'
                            match_algorithm_id = archived_algorithm_id
                            match_anchor_objective = archived_objective
                            break
                if match_scope is None and (not getattr(self, 'behavior_filter_xgb_rescue_enabled', False)):
                    for batch_feature, batch_algorithm_id in batch_representatives:
                        if self._behavior_vectors_match(feature, batch_feature):
                            match_scope = 'same_batch'
                            match_algorithm_id = batch_algorithm_id
                            break
            if match_scope is None:
                retained.append(candidate)
                batch_representatives.append((feature, offspring.get('algorithm_id')))
                continue
            rescue_prediction = None
            if match_scope == 'evaluated_history':
                rescue, rescue_prediction = self._predict_behavior_filter_rescue(feature, match_anchor_objective)
                if rescue:
                    offspring['lineage_status'] = 'behavior_duplicate_rescued'
                    offspring['behavior_duplicate_rescued'] = True
                    offspring['behavior_duplicate_scope'] = match_scope
                    offspring['duplicate_of_algorithm_id'] = match_algorithm_id
                    offspring['behavior_rescue_prediction'] = float(rescue_prediction)
                    offspring['behavior_rescue_anchor_objective'] = float(match_anchor_objective)
                    offspring['behavior_rescue_model_source'] = 'global_xgboost_direct'
                    retained.append(candidate)
                    batch_representatives.append((feature, offspring.get('algorithm_id')))
                    rescued += 1
                    continue
            offspring['lineage_status'] = 'behavior_duplicate_filtered'
            offspring['selected_for_evaluation'] = False
            offspring['selected_by_predictor'] = False
            offspring['skip_reason'] = 'raw_behavior_duplicate'
            offspring['behavior_duplicate_scope'] = match_scope
            offspring['duplicate_of_algorithm_id'] = match_algorithm_id
            if match_scope == 'evaluated_history':
                offspring['behavior_rescue_prediction'] = rescue_prediction
                offspring['behavior_rescue_anchor_objective'] = self._finite_or_none(match_anchor_objective)
                offspring['behavior_rescue_model_source'] = 'global_xgboost_direct' if getattr(self, 'behavior_filter_xgb_rescue_enabled', False) else None
            filtered.append(candidate)
            if match_scope == 'evaluated_history':
                history_filtered += 1
            else:
                batch_filtered += 1
        if filtered or rescued:
            logging.info('[BehaviorDuplicateFilter] input=%s retained=%s filtered=%s history=%s same_batch=%s rescued=%s raw_features=True atol=%g rtol=%g predictor_gate=%s xgb_rescue=%s', len(candidates), len(retained), len(filtered), history_filtered, batch_filtered, rescued, float(getattr(self, 'behavior_filter_atol', 1e-08)), float(getattr(self, 'behavior_filter_rtol', 1e-06)), bool(getattr(self, 'behavior_filter_start_with_predictor', False)), bool(getattr(self, 'behavior_filter_xgb_rescue_enabled', False)))
        return (retained, filtered)

    def _select_behavior_archive_novelty_candidate(self, candidates, excluded_candidates=None):
        """Select the raw-behavior candidate farthest from evaluated history.

        Distance is Euclidean after per-dimension standardization fitted only
        on the complete successful real-evaluation archive. The score is the
        distance to the nearest archived algorithm, not distance to a region
        center. No distance threshold or PCA dimension is involved.
        """
        candidates = list(candidates or [])
        excluded_ids = {id(row) for row in excluded_candidates or []}
        available = [row for row in candidates if id(row) not in excluded_ids]
        if not available:
            return None
        history_rows = []
        feature_dim = None
        for feature in list(getattr(self.behavior_predictor, '_X', [])):
            row = np.asarray(feature, dtype=np.float64).reshape(-1)
            if row.size == 0 or not np.all(np.isfinite(row)):
                continue
            if feature_dim is None:
                feature_dim = int(row.size)
            if row.size == feature_dim:
                history_rows.append(row)
        if not history_rows:
            return available[0]
        history = np.vstack(history_rows)
        mean = np.mean(history, axis=0)
        scale = np.std(history, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        history_scaled = (history - mean) / scale
        ranked = []
        for position, candidate in enumerate(available):
            offspring = self._candidate_to_offspring(candidate)
            feature = offspring.get('behavior_embedding', offspring.get('embedding'))
            row = np.asarray(feature, dtype=np.float64).reshape(-1)
            if row.size != history.shape[1] or not np.all(np.isfinite(row)):
                novelty = float('-inf')
            else:
                point = (row - mean) / scale
                novelty = float(np.min(np.linalg.norm(history_scaled - point, axis=1)))
            offspring['behavior_archive_novelty_distance'] = novelty if np.isfinite(novelty) else None
            ranked.append((novelty, -position, candidate))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        novelty, _, selected = ranked[0]
        if not np.isfinite(novelty):
            return available[0]
        offspring = self._candidate_to_offspring(selected)
        offspring['selected_by_predictor'] = False
        offspring['selection_method'] = 'behavior_archive_maximin'
        offspring['behavior_novelty_slot'] = True
        logging.info('[BehaviorNoveltySlot] selected=%s distance=%.6g archive=%s candidates=%s excluded_by_surrogate=%s raw_features=True', offspring.get('algorithm_id'), novelty, len(history), len(candidates), len(excluded_ids))
        return selected

    @staticmethod
    def _behavior_feature_failed(feat):
        if feat is None:
            return True
        arr = np.asarray(feat, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return True
        return bool(np.allclose(arr, -1.0, atol=1e-08))

    def _attach_behavior_features(self, candidates):
        self._last_behavior_feature_batch_time = 0.0
        self._last_embedding_batch_time = 0.0
        if not candidates:
            return candidates
        valid_pairs = []
        codes = []
        for cand in candidates:
            off = self._candidate_to_offspring(cand)
            code = off.get('code') if isinstance(off, dict) else None
            if code:
                valid_pairs.append((cand, off, self._canonical_code(code)))
                codes.append(code)
        if not codes:
            return candidates
        behavior_start = time.perf_counter()
        try:
            print(f'[Behavior] Extracting features for {len(codes)} candidates...')
            persistent_cache = getattr(self, '_behavior_feature_cache', None)
            if persistent_cache is None:
                persistent_cache = {}
                self._behavior_feature_cache = persistent_cache
            behavior_cache = {}
            missing_codes = {}
            cache_hits = 0
            for _, off, code_key in valid_pairs:
                if code_key in persistent_cache:
                    behavior_cache[code_key] = persistent_cache[code_key]
                    cache_hits += 1
                elif code_key not in missing_codes:
                    missing_codes[code_key] = off.get('code', '')
            if missing_codes:
                missing_keys = list(missing_codes)
                encoded = np.asarray(self.embedder.encode([missing_codes[key] for key in missing_keys], batch_size=getattr(self.embedder, 'batch_size', None)), dtype=np.float32)
                if len(encoded) != len(missing_keys):
                    raise ValueError(f'Behavior embedder returned an unexpected batch length: {len(encoded)} != {len(missing_keys)}')
                for code_key, feat in zip(missing_keys, encoded):
                    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
                    behavior_cache[code_key] = feat
                    if not self._behavior_feature_failed(feat):
                        persistent_cache[code_key] = feat.copy()
            for _, off, code_key in valid_pairs:
                feat = behavior_cache[code_key]
                failed = self._behavior_feature_failed(feat)
                off['behavior_embedding'] = feat.tolist()
                off['embedding'] = feat.tolist()
                off['feature_source'] = 'behavior'
                off['behavior_feature_failed'] = bool(failed)
                off['skip_real_eval'] = bool(failed)
                if failed:
                    off['skip_reason'] = 'behavior_feature_failed_or_timeout'
                    off['lineage_status'] = 'behavior_feature_failed'
                else:
                    off['lineage_status'] = 'features_ready'
            behavior_elapsed = time.perf_counter() - behavior_start
            print(f'[Behavior] Done | n={len(valid_pairs)} unique_codes={len({row[2] for row in valid_pairs})} computed={len(missing_codes)} cache_hits={cache_hits} wall={behavior_elapsed:.3f}s')
        except Exception as err:
            print(f'[Behavior] Warning: feature extraction failed: {err}')
            for _, off, _ in valid_pairs:
                off['behavior_feature_failed'] = True
                off['skip_real_eval'] = True
                off['skip_reason'] = 'behavior_feature_batch_failed'
                off['lineage_status'] = 'behavior_feature_batch_failed'
        self._last_behavior_feature_batch_time = time.perf_counter() - behavior_start
        self._last_embedding_batch_time = 0.0
        return candidates

    @staticmethod
    def _robust_standardize_features(matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        center = np.median(matrix, axis=0)
        q25, q75 = np.percentile(matrix, [25.0, 75.0], axis=0)
        scale = q75 - q25
        std = np.std(matrix, axis=0)
        scale = np.where(scale > 1e-12, scale, std)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (matrix - center) / scale
        return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _pairwise_euclidean_distances(matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        diff = matrix[:, None, :] - matrix[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=-1))
        if matrix.shape[1] > 0:
            distances /= math.sqrt(matrix.shape[1])
        return distances


    @staticmethod
    def _maxmin_select_indices(distance_matrix, target_size, candidate_indices=None, selected_indices=None):
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        n_items = int(distance_matrix.shape[0])
        target_size = max(0, int(target_size))
        if target_size == 0 or n_items == 0:
            return []
        if candidate_indices is None:
            pool = list(range(n_items))
        else:
            pool = [int(idx) for idx in candidate_indices if 0 <= int(idx) < n_items]
        references = []
        if selected_indices is not None:
            references = [int(idx) for idx in selected_indices if 0 <= int(idx) < n_items]
        reference_set = set(references)
        pool = list(dict.fromkeys((idx for idx in pool if idx not in reference_set)))
        chosen = []
        if not references and pool:
            if target_size == 1 or len(pool) == 1:
                submatrix = distance_matrix[np.ix_(pool, pool)]
                mean_distances = np.mean(submatrix, axis=1)
                first = pool[int(np.argmax(mean_distances))]
                chosen.append(first)
                references.append(first)
                pool.remove(first)
            else:
                submatrix = distance_matrix[np.ix_(pool, pool)].copy()
                np.fill_diagonal(submatrix, -np.inf)
                first_pos, second_pos = np.unravel_index(int(np.argmax(submatrix)), submatrix.shape)
                first = pool[int(first_pos)]
                second = pool[int(second_pos)]
                chosen.extend([first, second])
                references.extend([first, second])
                pool = [idx for idx in pool if idx not in {first, second}]
        while pool and len(chosen) < target_size:
            min_distances = [float(np.min(distance_matrix[idx, references])) if references else 0.0 for idx in pool]
            best_position = int(np.argmax(min_distances))
            best_idx = pool.pop(best_position)
            chosen.append(best_idx)
            references.append(best_idx)
        return chosen[:target_size]

    def _region_archive_rows(self):
        records = []
        rows = []
        objectives = []
        feature_dim = None
        for record in self._evaluated_algorithm_history:
            if not isinstance(record, dict):
                continue
            feature = record.get('behavior_embedding', record.get('embedding'))
            objective = self._safe_objective(record.get('objective'))
            if feature is None or not np.isfinite(objective):
                continue
            row = np.asarray(feature, dtype=np.float64).reshape(-1)
            if row.size == 0 or not np.all(np.isfinite(row)):
                continue
            if feature_dim is None:
                feature_dim = int(row.size)
            if row.size != feature_dim:
                continue
            records.append(record)
            rows.append(row)
            objectives.append(float(objective))
        if not rows:
            return ([], None, None)
        return (records, np.vstack(rows), np.asarray(objectives, dtype=np.float64))

    def _region_advantage_archive_rows(self):
        merged = []
        seen_codes = set()
        for record in self._evaluated_algorithm_history:
            if not isinstance(record, dict):
                continue
            code_key = self._canonical_code(record.get('code'))
            feature = record.get('behavior_embedding', record.get('embedding'))
            objective = self._safe_objective(record.get('objective'))
            if not code_key or code_key in seen_codes or feature is None or (not np.isfinite(objective)):
                continue
            row = np.asarray(feature, dtype=np.float64).reshape(-1)
            if row.size == 0 or not np.all(np.isfinite(row)):
                continue
            seen_codes.add(code_key)
            merged.append((record, row, float(objective)))
        if not merged:
            return ([], None, None)
        merged.sort(key=lambda item: item[2])
        target_distinct = int(self.region_archive_target_distinct)
        selected = []
        distinct_scores = set()
        cutoff_score = None
        feature_dim = int(merged[0][1].size)
        for record, row, objective in merged:
            if row.size != feature_dim:
                continue
            if objective in distinct_scores:
                continue
            selected.append((record, row, objective))
            distinct_scores.add(objective)
            if len(distinct_scores) >= target_distinct:
                cutoff_score = objective
                break
        records = [item[0] for item in selected]
        matrix = np.vstack([item[1] for item in selected])
        objectives = np.asarray([item[2] for item in selected], dtype=np.float64)
        logging.info('[RegionAdvantageArchive] records=%s distinct_scores=%s target_distinct=%s cutoff=%s total_archive=%s policy=strict_unique_score_cap', len(records), len(distinct_scores), target_distinct, f'{cutoff_score:.6g}' if cutoff_score is not None else 'all', len(merged))
        return (records, matrix, objectives)

    def _transform_region_features(self, features):
        if self._region_scaler is None:
            raise RuntimeError('Behavior-region behavior map is not fitted')
        matrix = np.asarray(features, dtype=np.float64)
        was_1d = matrix.ndim == 1
        if was_1d:
            matrix = matrix.reshape(1, -1)
        transformed = self._region_scaler.transform(matrix)
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        return transformed[0] if was_1d else transformed

    def _inverse_transform_region_features(self, points):
        """Map region-space centroids back to raw behavior-feature space."""
        if self._region_scaler is None:
            raise RuntimeError('Behavior-region behavior map is not fitted')
        matrix = np.asarray(points, dtype=np.float64)
        was_1d = matrix.ndim == 1
        if was_1d:
            matrix = matrix.reshape(1, -1)
        raw = self._region_scaler.inverse_transform(matrix)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return raw[0] if was_1d else raw

    def _region_objective_scale(self):
        """Return a robust recent objective scale shared by all regions."""
        _, _, objectives = self._region_archive_rows()
        if objectives is None or len(objectives) == 0:
            finite_best = [float(state['best_objective']) for state in self._region_states if np.isfinite(float(state['best_objective']))]
            objectives = np.asarray(finite_best, dtype=np.float64)
        else:
            objectives = np.asarray(objectives[-self.region_scale_window:], dtype=np.float64)
        objectives = objectives[np.isfinite(objectives)]
        if len(objectives) == 0:
            return 1.0
        q25, q75 = np.percentile(objectives, [25.0, 75.0])
        iqr = float(q75 - q25)
        reference = max(1.0, abs(float(np.min(objectives))))
        return max(iqr, 1e-06 * reference, 1e-12)

    @staticmethod
    def _region_relative_global_gap(value, global_best):
        """Return the minimization gap as a proportion of the global best."""
        value = float(value)
        global_best = float(global_best)
        if not np.isfinite(value) or not np.isfinite(global_best):
            return float('inf')
        denominator = max(abs(global_best), 1e-12)
        return max(0.0, (value - global_best) / denominator)

    def _region_record_algorithm_id(self, record):
        if not isinstance(record, dict):
            return None
        algorithm_id = record.get('algorithm_id')
        if algorithm_id:
            return str(algorithm_id)
        return self._algorithm_id(record.get('code'))

    def _persist_region_center_history(self):
        output_path = getattr(self, 'run_output_path', None)
        if not output_path:
            return
        try:
            os.makedirs(output_path, exist_ok=True)
            history_path = os.path.join(output_path, 'region_center_history.json')
            with open(history_path, 'w', encoding='utf-8') as file:
                json.dump(getattr(self, '_region_center_history', []), file, indent=2, ensure_ascii=False)
        except OSError as exc:
            logging.warning('[RegionCenterHistory] failed to save history: %s', exc)

    def _register_region_centers(self, records, objectives, selected_indices, reason, region_ids=None):
        if not hasattr(self, '_region_center_history_ids'):
            self._region_center_history_ids = set()
        if not hasattr(self, '_region_center_history'):
            self._region_center_history = []
        region_ids = list(region_ids or [])
        registered = []
        for offset, index in enumerate(selected_indices):
            record = records[int(index)]
            algorithm_id = self._region_record_algorithm_id(record)
            if algorithm_id is None or algorithm_id in self._region_center_history_ids:
                continue
            entry = {'selection_order': len(self._region_center_history) + 1, 'reason': str(reason), 'region_id': int(region_ids[offset]) if offset < len(region_ids) else None, 'algorithm_id': algorithm_id, 'objective': float(objectives[int(index)]), 'generated_order': record.get('generated_order'), 'operator': record.get('operator')}
            self._region_center_history_ids.add(algorithm_id)
            self._region_center_history.append(entry)
            registered.append(entry)
        if registered:
            self._persist_region_center_history()
            logging.info('[RegionCenterHistory] reason=%s added=%s total=%s centers=%s', reason, len(registered), len(self._region_center_history), [(entry['algorithm_id'], round(entry['objective'], 6)) for entry in registered])
        return registered

    def _kmeans_region_layout(self, space, objectives):
        """Cluster behavior points and select each cluster's best real anchor."""
        points = np.asarray(space, dtype=np.float64)
        values = np.asarray(objectives, dtype=np.float64).reshape(-1)
        region_count = int(self.region_count)
        if points.ndim != 2 or len(points) != len(values) or len(points) < region_count or (region_count < 1):
            raise ValueError('K-means region input is incomplete')
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
            raise ValueError('K-means region input contains non-finite values')
        unique_point_count = int(len(np.unique(points, axis=0)))
        if unique_point_count < region_count:
            raise ValueError(f'K-means needs at least {region_count} distinct behavior points, got {unique_point_count}')
        model = KMeans(n_clusters=region_count, random_state=int(self.region_seed), n_init=_KMEANS_N_INIT)
        raw_assignments = model.fit_predict(points).astype(np.int64)
        raw_centroids = np.asarray(model.cluster_centers_, dtype=np.float64)
        selected_anchor_indices = {}
        cluster_members = {}
        for cluster_id in range(region_count):
            member_indices = np.flatnonzero(raw_assignments == cluster_id)
            if len(member_indices) == 0:
                raise ValueError(f'K-means produced empty cluster {cluster_id}')
            cluster_members[int(cluster_id)] = member_indices.astype(np.int64)
        ordered_candidates = sorted(range(len(values)), key=lambda index: (float(values[index]), int(index)))
        for candidate_index in ordered_candidates:
            cluster_id = int(raw_assignments[candidate_index])
            if cluster_id in selected_anchor_indices:
                continue
            candidate_objective = float(values[candidate_index])
            selected_anchor_indices[cluster_id] = int(candidate_index)
        if len(selected_anchor_indices) < region_count:
            missing_clusters = sorted(set(range(region_count)) - set(selected_anchor_indices))
            raise ValueError(f'K-means clusters {missing_clusters} have no valid anchor')
        cluster_rows = []
        for cluster_id in range(region_count):
            member_indices = cluster_members[cluster_id]
            anchor_index = int(selected_anchor_indices[cluster_id])
            cluster_rows.append({'raw_cluster_id': int(cluster_id), 'anchor_index': anchor_index, 'anchor_objective': float(values[anchor_index]), 'member_indices': member_indices.astype(np.int64), 'centroid': raw_centroids[cluster_id].copy()})
        cluster_rows.sort(key=lambda row: (row['anchor_objective'], row['anchor_index'], row['raw_cluster_id']))
        raw_to_region = {row['raw_cluster_id']: region_index for region_index, row in enumerate(cluster_rows)}
        assignments = np.asarray([raw_to_region[int(cluster_id)] for cluster_id in raw_assignments], dtype=np.int64)
        centroids = np.vstack([row['centroid'] for row in cluster_rows])
        anchor_indices = [int(row['anchor_index']) for row in cluster_rows]
        member_counts = [int(len(row['member_indices'])) for row in cluster_rows]
        if region_count > 1:
            center_distance = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
            minimum_center_distance = float(np.min(center_distance[np.triu_indices(region_count, k=1)]))
        else:
            minimum_center_distance = 0.0
        stats = {'builder': 'behavior_kmeans', 'n_init': int(_KMEANS_N_INIT), 'inertia': float(model.inertia_), 'member_counts': member_counts, 'minimum_center_distance': minimum_center_distance, 'anchor_objectives': [float(values[index]) for index in anchor_indices]}
        return (anchor_indices, centroids, assignments, stats)

    def _initialize_regions(self, reason='warmup', retained_center_state=None):
        if self._region_states or not self.should_use_regions():
            return bool(self._region_states)
        records, matrix, objectives = self._region_advantage_archive_rows()
        if matrix is None or len(records) < self.region_count:
            return False
        records = list(records)
        matrix = np.asarray(matrix, dtype=np.float64)
        objectives = np.asarray(objectives, dtype=np.float64).reshape(-1)
        retained_seed_index = None
        if retained_center_state is not None:
            retained_algorithm_id = retained_center_state.get('center_algorithm_id')
            retained_code = self._canonical_code(retained_center_state.get('center_code'))
            for index, record in enumerate(records):
                record_id = self._region_record_algorithm_id(record)
                record_code = self._canonical_code(record.get('code'))
                if retained_algorithm_id is not None and record_id == str(retained_algorithm_id) or (retained_code and record_code == retained_code):
                    retained_seed_index = int(index)
                    break
            if retained_seed_index is None:
                retained_feature = np.asarray(retained_center_state.get('center_feature'), dtype=np.float64).reshape(-1)
                if retained_feature.size != matrix.shape[1] or not np.all(np.isfinite(retained_feature)):
                    logging.warning('[RegionInit] cannot retain R1 center: its behavior feature is unavailable or incompatible')
                    return False
                retained_seed_index = len(records)
                records.append({'code': retained_center_state.get('center_code'), 'algorithm_id': retained_algorithm_id, 'objective': retained_center_state.get('best_objective'), 'operator': 'retained_r1_center'})
                matrix = np.vstack([matrix, retained_feature])
                objectives = np.append(objectives, float(retained_center_state.get('best_objective', np.inf)))
        fit_count = len(records)
        fit_matrix = matrix[:fit_count]
        self._region_scaler = StandardScaler()
        fit_scaled = self._region_scaler.fit_transform(fit_matrix)
        fit_space = fit_scaled
        self._region_feature_dim = int(matrix.shape[1])
        self._region_space_dim = int(fit_space.shape[1])
        self._region_fit_count = int(fit_count)
        all_space = self._transform_region_features(matrix)
        try:
            seed_indices, region_points, assignments, center_stats = self._kmeans_region_layout(all_space[:fit_count], objectives[:fit_count])
        except (ValueError, FloatingPointError) as exc:
            logging.warning('[BehaviorRegionKMeans] initialization canceled | reason=%s archive=%s error=%s', reason, fit_count, exc)
            return False
        centroid_features = self._inverse_transform_region_features(region_points)
        states = []
        for region_offset, seed_index in enumerate(seed_indices):
            member_indices = np.flatnonzero(assignments == region_offset)
            if len(member_indices) == 0:
                member_indices = np.asarray([seed_index], dtype=np.int64)
            center_index = int(seed_index)
            center_point = np.asarray(region_points[region_offset], dtype=np.float64).copy()
            center_feature = matrix[center_index].copy()
            state = {'region_id': int(region_offset + 1), 'region_builder': 'behavior_kmeans', 'center_feature': center_feature, 'center_point': center_point, 'center_code': self._canonical_code(records[center_index].get('code')), 'center_algorithm_id': self._region_record_algorithm_id(records[center_index]), 'best_objective': float(objectives[center_index]), 'rebuild_count': 0, 'restart_count': 0, 'member_count': int(len(member_indices)), 'selected_count': 0, 'evaluated_count': 0, 'rebuild_stagnation_count': 0, 'last_local_improvement_generation': None, 'global_gap': 0.0, 'recent_events': []}
            state['center_algorithm'] = records[center_index].get('algorithm')
            state['centroid_feature'] = np.asarray(centroid_features[region_offset], dtype=np.float64).copy()
            state['anchor_distance_to_centroid'] = float(np.linalg.norm(all_space[center_index] - center_point))
            states.append(state)
            for member_index in member_indices:
                records[int(member_index)]['region_membership_id'] = int(region_offset + 1)
        self._region_states = states
        self._kmeans_region_stats = dict(center_stats)
        self._register_region_centers(records, objectives, seed_indices, reason=reason, region_ids=[state['region_id'] for state in states])
        global_best = min((float(state['best_objective']) for state in states))
        for state in states:
            state['global_gap'] = self._region_relative_global_gap(state['best_objective'], global_best)
        self._sync_region_snapshot()
        logging.info('[RegionInit] reason=%s regions=%s archive=%s dimension=%s stats=%s', reason, len(states), len(records), self._region_space_dim, center_stats)
        return True

    def _global_best_rebuild_center_state(self, records, matrix, objectives):
        """Return a R1 seed that represents the current global incumbent.

        A full region rebuild must not replace the best algorithm found so far
        with the previous R1 center.  The advantage archive is ordered by
        objective in normal execution, but selecting the minimum explicitly
        keeps this invariant independent of archive ordering.
        """
        if matrix is None or objectives is None or (not records):
            return None
        features = np.asarray(matrix, dtype=np.float64)
        values = np.asarray(objectives, dtype=np.float64).reshape(-1)
        if features.ndim != 2 or len(records) != len(values):
            return None
        finite_indices = np.flatnonzero(np.isfinite(values))
        if len(finite_indices) == 0:
            return None
        best_index = int(min(finite_indices.tolist(), key=lambda index: (float(values[index]), int(index))))
        feature = np.asarray(features[best_index], dtype=np.float64).reshape(-1)
        if feature.size == 0 or not np.all(np.isfinite(feature)):
            return None
        record = records[best_index]
        algorithm_id = self._region_record_algorithm_id(record)
        code = self._canonical_code(record.get('code'))
        if algorithm_id is None and (not code):
            return None
        return {'center_algorithm_id': algorithm_id, 'center_code': code, 'center_feature': feature.copy(), 'best_objective': float(values[best_index]), 'generated_order': record.get('generated_order'), 'operator': record.get('operator')}

    def _rebuild_regions_from_archive(self, reason):
        """Recluster the current evaluated advantage archive."""
        reason = str(reason)
        if not self.region_enabled or self.is_real_eval_budget_exhausted():
            return False
        records, matrix, objectives = self._region_advantage_archive_rows()
        if matrix is None or len(records) < self.region_count:
            logging.warning('[RegionRebuild] deferred | reason=%s archive=%s required=%s', reason, len(records), self.region_count)
            return False
        retained_center_state = self._global_best_rebuild_center_state(records, matrix, objectives)
        if retained_center_state is None:
            logging.warning('[RegionRebuild] canceled | reason=%s current global-best algorithm has no valid behavior feature; previous regions retained', reason)
            return False
        previous_region_count = len(self._region_states)
        previous_map = (self._region_scaler, self._region_feature_dim, self._region_space_dim, self._region_states, self._region_fit_count)
        self._region_scaler = None
        self._region_feature_dim = None
        self._region_space_dim = None
        self._region_states = []
        self._region_fit_count = 0
        rebuilt = self._initialize_regions(reason=reason, retained_center_state=retained_center_state)
        if not rebuilt:
            self._region_scaler, self._region_feature_dim, self._region_space_dim, self._region_states, self._region_fit_count = previous_map
            self._sync_region_snapshot()
            logging.warning('[BehaviorRegionKMeansRebuild] retained previous regions: the latest archive could not form K valid behavior clusters')
        if rebuilt:
            logging.info('[RegionRebuild] completed | reason=%s successful_evals=%s archive=%s previous_regions=%s new_regions=%s fit_samples=%s retained_global_best_center=%s retained_global_best_objective=%s', reason, self.get_successful_real_eval_count(), len(records), previous_region_count, len(self._region_states), self._region_fit_count, retained_center_state.get('center_algorithm_id') if retained_center_state else None, retained_center_state.get('best_objective') if retained_center_state else None)
        return bool(rebuilt)

    def _rebuild_kmeans_regions(self, reasons):
        """Coalesce regional stagnation into one joint K-means reclustering."""
        reason_rows = [str(value) for value in reasons or [] if value]
        if not reason_rows:
            return False
        reason = 'kmeans_stagnation:' + ','.join(sorted(set(reason_rows)))
        rebuilt = self._rebuild_regions_from_archive(reason=reason)
        if rebuilt:
            self._kmeans_full_rebuild_count += 1
            for state in self._region_states:
                state['rebuild_count'] = int(self._kmeans_full_rebuild_count)
                state['restart_count'] = int(self._kmeans_full_rebuild_count)
            self._sync_region_snapshot()
            logging.info('[BehaviorRegionKMeansRebuild] count=%s reasons=%s scope=all_regions', self._kmeans_full_rebuild_count, sorted(set(reason_rows)))
        return bool(rebuilt)

    def _sync_region_snapshot(self):
        if self.behavior_predictor is None:
            return
        finite_region_bests = [float(state['best_objective']) for state in self._region_states if np.isfinite(float(state['best_objective']))]
        snapshot = {'active': bool(self._region_states), 'region_builder': 'behavior_kmeans', 'region_builder_stats': dict(getattr(self, '_kmeans_region_stats', {})), 'kmeans_full_rebuild_count': int(getattr(self, '_kmeans_full_rebuild_count', 0)), 'fit_samples': int(self._region_fit_count), 'space_dim': self._region_space_dim, 'global_best': float(min(finite_region_bests)) if finite_region_bests else None, 'objective_scale': float(self._region_objective_scale()) if self._region_states else None, 'center_history': list(getattr(self, '_region_center_history', [])), 'parent_source': 'advantage_archive', 'archive_region_ids': [], 'regions': []}
        predictor_codes = [self._canonical_code(code) for code in getattr(self.behavior_predictor, '_codes', [])]
        archive_matrix = np.asarray(getattr(self.behavior_predictor, '_X', []), dtype=np.float64)
        if self._region_states and archive_matrix.ndim == 2 and (len(archive_matrix) > 0) and (archive_matrix.shape[1] == self._region_feature_dim):
            try:
                region_ids = [int(state['region_id']) for state in self._region_states]
                archive_objectives = np.asarray(getattr(self.behavior_predictor, '_y', []), dtype=np.float64).reshape(-1)
                archive_space = self._transform_region_features(archive_matrix)
                region_centers = np.vstack([np.asarray(state['center_point'], dtype=np.float64) for state in self._region_states])
                nearest = np.argmin(np.sum((archive_space[:, None, :] - region_centers[None, :, :]) ** 2, axis=2), axis=1)
                snapshot['archive_region_ids'] = [region_ids[int(index)] for index in nearest]
            except Exception as err:
                logging.warning('[BehaviorRegionCSV] failed to snapshot archive membership: %s', err)
        for state in self._region_states:
            center_code = state.get('center_code')
            center_archive_index = None
            if center_code in predictor_codes:
                center_archive_index = int(predictor_codes.index(center_code))
            region_snapshot = {key: value for key, value in state.items() if key not in {'center_feature', 'center_point', 'centroid_feature'}} | {'center_archive_index': center_archive_index, 'center_point': np.asarray(state['center_point'], dtype=float).tolist()}
            region_snapshot['centroid_feature'] = np.asarray(state.get('centroid_feature', state['center_feature']), dtype=float).tolist()
            snapshot['regions'].append(region_snapshot)
        self.behavior_predictor.online_region_snapshot = snapshot

    def _advance_region_rebuild_state(self, state, *, scheduled, fresh_count, attempted_count, valid_count, local_improvement):
        """Update the one scale-independent no-improvement counter.

        A scheduled region receives at most one failure mark per operator
        batch. The three failure reasons are mutually exclusive and share the
        same tolerance. An improvement in any batch of the current generation
        prevents another operator batch from immediately counting a failure.
        """
        generation = int(self.current_generation)
        if local_improvement:
            state['rebuild_stagnation_count'] = 0
            state['last_local_improvement_generation'] = generation
            return ('local_improvement', False)
        if state.get('last_local_improvement_generation') == generation:
            return ('improved_this_generation', False)
        if not scheduled:
            return ('not_scheduled', False)
        if int(fresh_count) <= 0:
            reason = 'no_fresh_candidate'
        elif int(attempted_count) <= 0:
            return ('fresh_candidate_not_evaluated', False)
        elif int(valid_count) <= 0:
            reason = 'no_successful_evaluation'
        else:
            reason = 'evaluated_without_improvement'
        state['rebuild_stagnation_count'] = int(state.get('rebuild_stagnation_count', 0)) + 1
        should_rebuild = bool(state['rebuild_stagnation_count'] >= self.region_rebuild_tolerance)
        if should_rebuild and self._is_protected_global_best_anchor(state):
            state['rebuild_stagnation_count'] = 0
            logging.info('[RegionAnchorProtection] region=R%s retained global_best=%.6g trigger=%s generation=%s', state.get('region_id'), float(state.get('best_objective')), reason, int(self.current_generation))
            return ('global_best_anchor_protected', False)
        return (reason, should_rebuild)

    def _is_protected_global_best_anchor(self, state):
        """Return whether ``state`` is the one dynamic elite anchor.

        Ties are broken by region id so protection never freezes more than one
        region.  When another region finds a strictly better algorithm, that
        region automatically becomes the protected anchor on the next update.
        """
        if not getattr(self, 'preserve_global_best_anchor', False):
            return False
        candidates = [candidate for candidate in self._region_states if np.isfinite(self._safe_objective(candidate.get('best_objective')))]
        if not candidates:
            return False
        protected = min(candidates, key=lambda candidate: (self._safe_objective(candidate.get('best_objective')), int(candidate.get('region_id', 0))))
        return protected is state

    def _update_regions_after_evaluation(self, evaluated_candidates, fresh_candidate_counts=None, scheduled_region_ids=None):
        """Maintain real-performance anchors and the existing KMeans stagnation rule."""
        if not self._region_states:
            return
        fresh_candidate_counts = {int(k): max(0, int(v)) for k, v in dict(fresh_candidate_counts or {}).items()}
        scheduled_region_ids = {int(k) for k in scheduled_region_ids or []}
        grouped = {}
        for candidate in evaluated_candidates or []:
            offspring = self._candidate_to_offspring(candidate)
            region_id = self._region_accounting_id(offspring)
            if region_id is not None:
                grouped.setdefault(int(region_id), []).append(offspring)
        objective_scale = float(self._region_objective_scale())
        pending_rebuilds = []
        for state in self._region_states:
            region_id = int(state['region_id'])
            rows = grouped.get(region_id, [])
            attempted = [row for row in rows if row.get('real_eval_attempted')]
            valid = [row for row in attempted if np.isfinite(self._safe_objective(row.get('objective')))]
            best_row = min(valid, key=lambda row: self._safe_objective(row.get('objective')), default=None)
            best_value = self._safe_objective(best_row.get('objective')) if best_row is not None else float('inf')
            previous_best = float(state['best_objective'])
            threshold = float(self.region_relative_improvement * max(abs(previous_best), objective_scale, 1e-12))
            local_improvement = bool(best_value < previous_best - threshold)
            state['selected_count'] += len(rows) if attempted else 0
            state['evaluated_count'] += len(attempted)
            if local_improvement:
                feature = np.asarray(best_row.get('behavior_embedding', best_row.get('embedding')), dtype=np.float64).reshape(-1)
                point = self._transform_region_features(feature)
                state['best_objective'] = float(best_value)
                state['center_feature'] = feature
                state['anchor_distance_to_centroid'] = float(np.linalg.norm(point - state['center_point']))
                state['center_code'] = self._canonical_code(best_row.get('code'))
                state['center_algorithm'] = best_row.get('algorithm')
                state['center_algorithm_id'] = self._region_record_algorithm_id(best_row)
                self._register_region_centers([best_row], np.asarray([best_value]), [0], reason=f'improvement_R{region_id}', region_ids=[region_id])
            reason, should_rebuild = self._advance_region_rebuild_state(state, scheduled=region_id in scheduled_region_ids, fresh_count=fresh_candidate_counts.get(region_id, 0), attempted_count=len(attempted), valid_count=len(valid), local_improvement=local_improvement)
            event = {'generation': int(self.current_generation), 'attempted': len(attempted), 'valid': len(valid), 'region_best': float(state['best_objective']), 'local_improvement': local_improvement, 'local_improvement_threshold': threshold, 'rebuild_reason': reason, 'rebuild_stagnation_count': int(state.get('rebuild_stagnation_count', 0))}
            state['recent_events'].append(event)
            state['recent_events'] = state['recent_events'][-20:]
            logging.info('[RegionUpdate] region=R%s %s action=%s', region_id, event, 'recluster' if should_rebuild else 'retain')
            if should_rebuild:
                pending_rebuilds.append(f'R{region_id}_{reason}')
        global_best = min((float(state['best_objective']) for state in self._region_states))
        for state in self._region_states:
            state['global_gap'] = self._region_relative_global_gap(state['best_objective'], global_best)
        if pending_rebuilds:
            self._rebuild_kmeans_regions(pending_rebuilds)
        else:
            self._sync_region_snapshot()

    def _archive_predictor_samples(self, evaluated_candidates, pop=None):
        if self.behavior_predictor is None:
            return
        existing_codes = {self._canonical_code(ind.get('code', '')) for ind in pop} if pop else set()
        samples_to_add = []
        for cand in evaluated_candidates:
            off = self._candidate_to_offspring(cand)
            code = off.get('code', '')
            code_key = self._canonical_code(code)
            obj = self._safe_objective(off.get('objective', float('inf')))
            if np.isfinite(obj) and code_key and (code_key not in existing_codes):
                behavior_feat = off.get('behavior_embedding', off.get('embedding'))
                if behavior_feat is None:
                    logging.warning('[BehaviorArchive] Skip sample with missing behavior feature: %s', str(abs(hash(code_key)))[-12:])
                    continue
                samples_to_add.append((code, obj, behavior_feat))
                existing_codes.add(code_key)
        if samples_to_add:
            for code, obj, behavior_feat in samples_to_add:
                self.behavior_predictor.add_sample(behavior_feat, obj, code, reencode_code=False)
            if self.surrogate_selection_enabled:
                logging.info('[PredictorArchive] added=%s total=%s model=xgboost_direct', len(samples_to_add), self.behavior_predictor.get_sample_count())
            else:
                logging.info('[PredictorArchive] added=%s total=%s model=disabled', len(samples_to_add), self.behavior_predictor.get_sample_count())

    def _archive_evaluated_algorithms(self, evaluated_candidates):
        added = 0
        for candidate in evaluated_candidates or []:
            offspring = self._candidate_to_offspring(candidate)
            if not isinstance(offspring, dict):
                continue
            code_key = self._canonical_code(offspring.get('code'))
            objective = self._safe_objective(offspring.get('objective'))
            feature = offspring.get('behavior_embedding', offspring.get('embedding'))
            if not code_key or code_key in self._evaluated_algorithm_codes or (not np.isfinite(objective)) or self._behavior_feature_failed(feature):
                continue
            self._evaluated_algorithm_history.append(dict(offspring))
            self._evaluated_algorithm_codes.add(code_key)
            added += 1
        if added:
            logging.info('[EvaluatedAlgorithmHistory] added=%s total=%s', added, len(self._evaluated_algorithm_history))

    def _evaluate_candidates(self, candidates, pop=None, iteration=0, region_fresh_counts=None, region_scheduled_ids=None):
        if not candidates:
            self._last_n_evaluated = 0
            self._last_n_eval_skipped = 0
            self._last_existing_eval_duplicate_count = 0
            self._last_real_eval_batch_time = 0.0
            if self._region_states and region_scheduled_ids:
                self._update_regions_after_evaluation([], fresh_candidate_counts=region_fresh_counts, scheduled_region_ids=region_scheduled_ids)
            return []
        for candidate in candidates:
            offspring = self._candidate_to_offspring(candidate)
            offspring['selected_for_evaluation'] = True
            offspring.setdefault('real_eval_attempted', False)
            offspring.setdefault('real_eval_success', False)
        before_filter = len(candidates)
        skipped_candidates = [cand for cand in candidates if self._candidate_to_offspring(cand).get('skip_real_eval') or self._candidate_to_offspring(cand).get('behavior_feature_failed')]
        candidates = [cand for cand in candidates if not self._candidate_to_offspring(cand).get('skip_real_eval') and (not self._candidate_to_offspring(cand).get('behavior_feature_failed'))]
        skipped_feature_failed = before_filter - len(candidates)
        self._set_lineage_status(skipped_candidates, 'evaluation_skipped_feature_failure', selected_for_evaluation=False, evaluation_status='feature_failure')
        if skipped_feature_failed > 0:
            print(f'[EvalSkip] skipped {skipped_feature_failed} candidates due to behavior feature timeout/failure')
        if not candidates:
            self._last_n_evaluated = 0
            self._last_n_eval_skipped = skipped_feature_failed
            self._last_existing_eval_duplicate_count = 0
            self._last_real_eval_batch_time = 0.0
            if self._region_states and region_scheduled_ids:
                self._update_regions_after_evaluation([], fresh_candidate_counts=region_fresh_counts, scheduled_region_ids=region_scheduled_ids)
            return []
        pop_objectives = {}
        if pop:
            for ind in pop:
                code_key = self._canonical_code(ind.get('code', ''))
                obj = self._safe_objective(ind.get('objective', float('inf')))
                if code_key and np.isfinite(obj):
                    pop_objectives[code_key] = obj
                    self._cache_evaluated_objective(ind.get('code', ''), obj, algorithm_id=ind.get('algorithm_id'))
        groups = {}
        unique_candidates = []
        shared_from_population = {}
        shared_from_archive = {}
        for cand in candidates:
            off = self._candidate_to_offspring(cand)
            code_key = self._canonical_code(off.get('code', ''))
            if not code_key:
                continue
            groups.setdefault(code_key, []).append(cand)
            if code_key in pop_objectives:
                shared_from_population[code_key] = pop_objectives[code_key]
            elif self._cached_evaluated_objective(code_key) is not None:
                shared_from_archive[code_key] = self._cached_evaluated_objective(code_key)
            elif len(groups[code_key]) == 1:
                unique_candidates.append(cand)
        shared_objectives = dict(shared_from_archive)
        shared_objectives.update(shared_from_population)
        eval_candidates = [cand for cand in unique_candidates if self._canonical_code(self._candidate_to_offspring(cand).get('code', '')) not in shared_objectives]
        remaining_budget = self.get_remaining_successful_evals()
        deferred_by_budget = eval_candidates[remaining_budget:]
        eval_candidates = eval_candidates[:remaining_budget]
        self._set_lineage_status(deferred_by_budget, 'evaluation_deferred_budget', selected_for_evaluation=False, evaluation_status='budget_deferred')
        if deferred_by_budget:
            logging.info('[RealEvalBudget] deferred %s candidates; only %s successful slots remain', len(deferred_by_budget), remaining_budget)
        skipped_same_batch = len(candidates) - len(groups)
        skipped_existing = len(shared_objectives)
        self._last_existing_eval_duplicate_count = skipped_existing
        if skipped_same_batch > 0 or skipped_existing > 0:
            print(f'[EvalDedupe] input={len(candidates)}, unique={len(groups)}, eval={len(eval_candidates)}, same_batch_dupes={skipped_same_batch}, shared_from_population={len(shared_from_population)}, shared_from_archive={len(shared_from_archive)}')

        def _count_candidates_by_region(rows):
            counts = {}
            for row in rows:
                offspring = self._candidate_to_offspring(row)
                region_id = offspring.get('region_allocation_id')
                key = int(region_id) if region_id is not None else 'none'
                counts[key] = counts.get(key, 0) + 1
            return counts
        shared_representatives = [groups[code_key][0] for code_key in shared_objectives if groups.get(code_key)]
        logging.info('[RegionBudgetActual] planned=%s real_eval=%s shared=%s deferred=%s', _count_candidates_by_region(candidates), _count_candidates_by_region(eval_candidates), _count_candidates_by_region(shared_representatives), _count_candidates_by_region(deferred_by_budget))
        codes = [self._candidate_to_offspring(cand).get('code') for cand in eval_candidates]
        for candidate in eval_candidates:
            offspring = self._candidate_to_offspring(candidate)
            offspring['real_eval_attempted'] = True
            offspring['evaluation_status'] = 'evaluation_started'
            offspring['lineage_status'] = 'evaluation_started'
        for code_key, objective in shared_from_population.items():
            for candidate in groups.get(code_key, []):
                offspring = self._candidate_to_offspring(candidate)
                offspring['real_eval_attempted'] = False
                offspring['real_eval_success'] = True
                offspring['eval_shared'] = True
                offspring['evaluation_status'] = 'shared_from_population'
                offspring['lineage_status'] = 'evaluation_shared'
        for code_key, objective in shared_from_archive.items():
            for candidate in groups.get(code_key, []):
                offspring = self._candidate_to_offspring(candidate)
                offspring['real_eval_attempted'] = False
                offspring['real_eval_success'] = True
                offspring['eval_shared'] = True
                offspring['evaluation_status'] = 'shared_from_archive'
                offspring['lineage_status'] = 'evaluation_shared_archive'
                offspring['duplicate_of_algorithm_id'] = getattr(self, '_evaluation_algorithm_id_cache', {}).get(code_key)
        real_eval_start = time.perf_counter()
        self._evaluation_attempted_code_cache.update((self._canonical_code(code) for code in codes if self._canonical_code(code)))
        objs = self.interface_eval.batch_evaluate(codes, iteration) if codes else []
        self._last_real_eval_batch_time = time.perf_counter() - real_eval_start if codes else 0.0
        n_successful = sum((np.isfinite(self._safe_objective(obj)) for obj in objs))
        self._record_real_evaluations(len(codes), n_successful=n_successful)
        successful_by_region = {}
        invalid_by_region = {}
        for candidate, objective in zip(eval_candidates, objs):
            offspring = self._candidate_to_offspring(candidate)
            region_id = offspring.get('region_allocation_id')
            key = int(region_id) if region_id is not None else 'none'
            target = successful_by_region if np.isfinite(self._safe_objective(objective)) else invalid_by_region
            target[key] = target.get(key, 0) + 1
        logging.info('[RegionBudgetResult] attempted=%s successful=%s invalid=%s', _count_candidates_by_region(eval_candidates), successful_by_region, invalid_by_region)
        objective_by_key = dict(shared_objectives)
        for cand, obj in zip(eval_candidates, objs):
            offspring = self._candidate_to_offspring(cand)
            code_key = self._canonical_code(offspring.get('code', ''))
            objective = self._safe_objective(obj)
            objective_by_key[code_key] = objective
            success = bool(np.isfinite(objective))
            if success:
                self._cache_evaluated_objective(offspring.get('code', ''), objective, algorithm_id=offspring.get('algorithm_id'))
            offspring['real_eval_success'] = success
            offspring['evaluation_status'] = 'evaluated_success' if success else 'evaluated_invalid'
            offspring['lineage_status'] = offspring['evaluation_status']
        completed_groups = []
        for code_key, same_code_candidates in groups.items():
            if code_key not in objective_by_key:
                continue
            obj = objective_by_key.get(code_key, float('inf'))
            for cand in same_code_candidates:
                off = self._candidate_to_offspring(cand)
                off['objective'] = np.round(self._safe_objective(obj), 5)
                off['eval_shared'] = bool(len(same_code_candidates) > 1 or code_key in shared_objectives)
                if off.get('eval_shared') and (not off.get('real_eval_attempted')):
                    off['real_eval_success'] = bool(np.isfinite(self._safe_objective(obj)))
                    if code_key in shared_from_archive:
                        off['evaluation_status'] = 'shared_from_archive'
                        off['lineage_status'] = 'evaluation_shared_archive'
                    else:
                        off['evaluation_status'] = 'shared_from_population'
                        off['lineage_status'] = 'evaluation_shared'
            completed_groups.append(same_code_candidates)
        representatives = [same_code_candidates[0] for same_code_candidates in completed_groups]
        self._log_post_eval_rankings(representatives)
        regions_were_active = bool(self._region_states)
        self._archive_predictor_samples(representatives, pop=pop)
        self._archive_evaluated_algorithms(representatives)
        if regions_were_active:
            self._update_regions_after_evaluation(representatives, fresh_candidate_counts=region_fresh_counts, scheduled_region_ids=region_scheduled_ids)
        else:
            self._initialize_regions()
        self._last_n_evaluated = len(eval_candidates)
        self._last_n_eval_skipped = skipped_feature_failed + skipped_same_batch + skipped_existing + len(deferred_by_budget)
        return representatives

    def _filter_candidates_already_evaluated(self, candidates, pop=None):
        """Remove exact-code matches with any successful evaluation in this run."""
        evaluated_codes = set(getattr(self, '_evaluation_objective_cache', {}))
        evaluated_codes.update(getattr(self, '_evaluation_attempted_code_cache', set()))
        if pop:
            for individual in pop:
                code_key = self._canonical_code(individual.get('code', ''))
                objective = self._safe_objective(individual.get('objective', float('inf')))
                if code_key and np.isfinite(objective):
                    evaluated_codes.add(code_key)
        fresh = []
        already_evaluated = []
        for candidate in candidates:
            offspring = self._candidate_to_offspring(candidate)
            code_key = self._canonical_code(offspring.get('code', ''))
            if code_key and code_key in evaluated_codes:
                already_evaluated.append(candidate)
            else:
                fresh.append(candidate)
        return (fresh, already_evaluated)

    def _rank_by_ensemble(self, candidates, cur_fe, top_k, minimize=True):
        selection_start = time.perf_counter()
        if not candidates:
            self._last_predictor_selection_time = 0.0
            return ([], [])
        if getattr(self, 'behavior_filter_start_with_predictor', False) and getattr(self, '_behavior_filter_predictor_ready_for_batch', None) is False:
            self._last_predictor_selection_time = 0.0
            top_k = max(1, min(int(top_k), len(candidates)))
            return (candidates[:top_k], candidates)
        behavior_ready = bool(self.behavior_predictor.get_sample_count() >= self.behavior_predictor.min_samples)
        if not behavior_ready:
            self._last_predictor_selection_time = 0.0
            top_k = max(1, min(int(top_k), len(candidates)))
            return (candidates[:top_k], candidates)
        behavior_ready = self._train_predictor_if_needed(self.behavior_predictor, '_behavior_trained_samples', 'behavior')
        if not behavior_ready:
            self._last_predictor_selection_time = time.perf_counter() - selection_start
            top_k = max(1, min(int(top_k), len(candidates)))
            return (candidates[:top_k], candidates)
        scored = []
        for cand in candidates:
            off = self._candidate_to_offspring(cand)
            b_feat = off.get('behavior_embedding', off.get('embedding'))
            b_mu, b_std = (None, 0.0)
            if b_feat is not None:
                b_mu, b_std = self.behavior_predictor.predict(b_feat)
            predicted = float(b_mu) if b_mu is not None and np.isfinite(b_mu) else None
            rank_score = predicted if predicted is not None and minimize else -predicted if predicted is not None else float('inf')
            off['behavior_pred_score'] = predicted
            off['pred_score'] = rank_score
            off['ensemble_score'] = rank_score
            off['ranking_score'] = -rank_score
            scored.append(cand)
        ranked = sorted(scored, key=lambda cand: self._candidate_to_offspring(cand).get('pred_score', float('inf')))
        top_k = max(1, min(int(top_k), len(ranked)))
        selected = ranked[:top_k]
        self._last_predictor_selection_time = time.perf_counter() - selection_start
        logging.info('[BehaviorPredictor] selected=%s/%s ready=%s selection_mode=predicted_score_topk', len(selected), len(ranked), behavior_ready)
        return (selected, ranked)

    @staticmethod
    def _summarize_initialization_idea(offspring, max_length=180):
        text = str(offspring.get('algorithm') or '').strip()
        if not text:
            text = str(offspring.get('code') or '').strip().split('\n', 1)[0]
        text = re.sub('\\s+', ' ', text).strip(' {}')
        if len(text) > max_length:
            text = text[:max_length - 3].rstrip() + '...'
        return text

    def _build_initialization_context(self, attempt_index, previous_ideas):
        return {'candidate_index': int(attempt_index), 'previous_ideas': list(previous_ideas[-10:])}

    def population_generation_with_prompt_seed(self):
        """Build a fixed-size initialization with one unchanged prompt seed."""
        if not self.use_prompt_seed:
            return self._sequential_initialization(self.init_unique_candidate_target)
        seed_source = str(self.prompt_seed_source or '').strip()
        if not seed_source:
            logging.warning('[InitializationSeed] prompt seed is empty; falling back to I1-only initialization')
            return self._sequential_initialization(self.init_unique_candidate_target)
        seed_code = seed_source
        if not re.search('^\\s*(?:import\\s+numpy\\s+as\\s+np|from\\s+numpy\\s+import)', seed_code, re.M):
            seed_code = 'import numpy as np\n\n' + seed_code
        seed_input = {'algorithm': 'Fixed problem seed from prompts/{}/seed_func.txt'.format(self.problem_name), 'code': seed_code}
        seed_start = time.perf_counter()
        seed_population = self.population_generation_seed([seed_input])
        seed_total_time = time.perf_counter() - seed_start
        seed_feature_valid = sum((individual.get('behavior_embedding') is not None and (not individual.get('behavior_feature_failed', False)) for individual in seed_population))
        self._append_timing_record({'event': 'initialization_seed', 'operator': 'seed', 'n_generated': 1, 'n_candidates': 1, 'n_after_feature_filter': int(seed_feature_valid), 'n_selected': 1, 'n_evaluated': int(getattr(self, '_last_n_evaluated', 0)), 'n_skipped': int(getattr(self, '_last_n_eval_skipped', 0)), 'generation_batch_time': 0.0, 'behavior_feature_batch_time': float(getattr(self, '_last_behavior_feature_batch_time', 0.0)), 'embedding_batch_time': float(getattr(self, '_last_embedding_batch_time', 0.0)), 'predictor_selection_time': 0.0, 'real_eval_batch_time': float(getattr(self, '_last_real_eval_batch_time', 0.0))})
        generated_target = max(0, int(self.init_unique_candidate_target) - 1)
        generated_population = self._sequential_initialization(generated_target, seed_population, start_index=2)
        population = list(seed_population) + list(generated_population)
        logging.info('[InitializationSeed] retained=%s feature_valid=%s i1_target=%s initial_population=%s seconds=%.3f', len(seed_population), seed_feature_valid, generated_target, len(population), seed_total_time)
        return population

    def _sequential_initialization(self, slots, prior_population=None, start_index=1):
        """One attempt per slot; only evaluated successful ideas enter context."""
        population = [p for p in list(prior_population or [])
                      if np.isfinite(self._safe_objective(p.get('objective')))]
        prior_count = len(population)
        seen = {self._canonical_code(p.get('code')) for p in list(prior_population or [])}
        for slot in range(int(slots)):
            if self.is_real_eval_budget_exhausted() or self.get_remaining_generated_algorithm_slots() <= 0:
                break
            ideas = [self._summarize_initialization_idea(p) for p in population[-10:]]
            context = self._build_initialization_context(start_index + slot, ideas)
            started = time.perf_counter()
            candidate = self.get_offspring([], 'i1', generation_context=context)
            generation_time = time.perf_counter() - started
            off = self._candidate_to_offspring(candidate)
            code = self._canonical_code(off.get('code'))
            selected = []
            feature_time = 0.0
            eval_time = 0.0
            eval_count = 0
            if not code:
                off['lineage_status'] = 'initial_invalid_code'
            elif code in seen:
                off['lineage_status'] = 'initial_duplicate_code'
            else:
                seen.add(code)
                self._attach_behavior_features([candidate])
                feature_time = getattr(self, '_last_behavior_feature_batch_time', 0.0)
                if not off.get('skip_real_eval') and not off.get('behavior_feature_failed'):
                    selected = self._evaluate_candidates([candidate], pop=[], iteration=0)
                    eval_time = getattr(self, '_last_real_eval_batch_time', 0.0)
                    eval_count = getattr(self, '_last_n_evaluated', 0)
                    population.extend(self._candidate_to_offspring(c) for c in selected
                                      if np.isfinite(self._safe_objective(self._candidate_to_offspring(c).get('objective'))))
            self._finalize_lineage_candidates([candidate])
            self._append_timing_record({'event': 'initialization_sequential', 'operator': 'i1',
                'n_generated': 1, 'n_candidates': 1, 'n_after_feature_filter': len(selected),
                'n_selected': eval_count, 'n_evaluated': eval_count,
                'n_skipped': int(not selected), 'generation_batch_time': generation_time,
                'behavior_feature_batch_time': feature_time,
                'embedding_batch_time': 0.0, 'predictor_selection_time': 0.0,
                'real_eval_batch_time': eval_time})
            logging.info('[InitSequential] slot=%s context_ideas=%s successful=%s refill=False',
                         start_index + slot, len(ideas), len(population))
        return population[prior_count:]

    def population_generation_seed(self, seeds):
        population = []
        remaining_budget = self.get_remaining_successful_evals()
        seeds = list(seeds)[:remaining_budget]
        codes = [seed['code'] for seed in seeds]
        real_eval_start = time.perf_counter()
        fitness = self.interface_eval.batch_evaluate(codes, 0) if codes else []
        self._last_real_eval_batch_time = time.perf_counter() - real_eval_start if codes else 0.0
        self._last_n_evaluated = len(codes)
        self._last_n_eval_skipped = 0
        n_successful = sum((np.isfinite(self._safe_objective(obj)) for obj in fitness))
        self._record_real_evaluations(len(codes), n_successful=n_successful)
        for seed, objective in zip(seeds, fitness):
            self._cache_evaluated_objective(seed.get('code', ''), objective, algorithm_id=seed.get('algorithm_id'))
        for i in range(len(seeds)):
            try:
                seed_alg = {'algorithm': seeds[i]['algorithm'], 'code': seeds[i]['code'], 'objective': np.round(self._safe_objective(fitness[i]), 5), 'other_inf': None, 'generation_parent_scope': 'seed', 'real_eval_attempted': True, 'real_eval_success': bool(np.isfinite(self._safe_objective(fitness[i]))), 'evaluation_status': 'evaluated_success' if np.isfinite(self._safe_objective(fitness[i])) else 'evaluated_invalid'}
                seed_alg['lineage_status'] = seed_alg['evaluation_status']
                self._register_lineage_event(None, seed_alg, 'seed')
                population.append(seed_alg)
            except Exception:
                print('Error in seed algorithm')
                exit()
        self._attach_behavior_features([(None, ind) for ind in population])
        self._archive_predictor_samples([(None, ind) for ind in population], pop=[])
        self._archive_evaluated_algorithms([(None, ind) for ind in population])
        for individual in population:
            individual['lineage_status'] = individual.get('evaluation_status', 'seed_loaded')
        self._finalize_lineage_candidates([(None, ind) for ind in population])
        print('Initiliazation finished! Get ' + str(len(seeds)) + ' seed algorithms')
        return population

    def _get_alg(self, pop, operator, generation_context=None):
        offspring = {'algorithm': None, 'code': None, 'objective': None, 'other_inf': None}
        if operator == 'i1':
            parents = None
            offspring['generation_parent_scope'] = 'initialization'
            self._claim_generated_algorithm_slots(1, context='serial_i1')
            [offspring['code'], offspring['algorithm']] = self.evol.i1(initialization_context=generation_context)
        elif operator in BEHAVIOR_EXPAND_OPERATORS:
            parents = self._sample_operator_parents(pop, operator)
            self._claim_generated_algorithm_slots(1, context=f'serial_{operator}')
            [offspring['code'], offspring['algorithm']] = self.evol.bx(parents)
        elif operator in BEHAVIOR_REFINE_OPERATORS:
            parents = self._sample_operator_parents(pop, operator)
            self._claim_generated_algorithm_slots(1, context=f'serial_{operator}')
            [offspring['code'], offspring['algorithm']] = self.evol.br(parents)
        else:
            print(f'Evolution operator [{operator}] has not been implemented ! \n')
            parents = None
        offspring['_lineage_prompt'] = getattr(self.evol, 'last_prompt_content', None)
        offspring['_lineage_response'] = getattr(self.evol, 'last_response', None)
        offspring['_lineage_llm_attempts'] = getattr(self.evol, 'last_llm_attempts', None)
        if generation_context:
            offspring['initialization_context'] = dict(generation_context)
        offspring.setdefault('generation_parent_scope', 'advantage_archive')
        return (parents, offspring)

    def _sample_operator_parents(self, pop, operator):
        if operator == 'i1':
            return None
        if operator in STANDARD_BEHAVIOR_OPERATORS:
            batches = self._build_archive_parent_batches(pop, operator, batch_size=1)
            if batches:
                return batches[0]
            raise RuntimeError(f'No unused behavior-guided parent set remains for operator={operator}')
        raise ValueError(f'Unsupported evolution operator: {operator}')

    def _parent_batch_signature(self, parents):
        code_keys = [self._canonical_code(parent.get('code')) for parent in list(parents or []) if isinstance(parent, dict)]
        if not code_keys or any((not code_key for code_key in code_keys)):
            return None
        if len(set(code_keys)) != len(code_keys):
            return None
        return tuple(sorted(code_keys))

    def _register_parent_batch(self, parents, expected_count=_BEHAVIOR_PARENT_COUNT):
        """Register one unordered, distinct-code parent set.

        Region-local BX/BR and archive fallback retain the strict two-parent
        default.  Cross-region BX passes its scheduled 1..K count explicitly
        so the shared unordered-code dedupe remains in effect without
        weakening the other parent builders.
        """
        if len(parents or []) != int(expected_count):
            return False
        signature = self._parent_batch_signature(parents)
        if signature is None or signature in self._active_parent_batch_signatures:
            return False
        self._active_parent_batch_signatures.add(signature)
        return True

    @staticmethod
    def _diverse_parent_indices(distance_matrix, parent_count, anchor, variant=0, objective_values=None):
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        n_items = int(distance_matrix.shape[0])
        if n_items == 0:
            return []
        selected = [int(anchor) % n_items]
        values = None
        selected_objectives = set()
        if objective_values is not None:
            values = np.asarray(objective_values, dtype=np.float64).reshape(-1)
            if len(values) != n_items:
                raise ValueError('objective_values must match the parent distance matrix')
            anchor_value = float(values[selected[0]])
            if np.isfinite(anchor_value):
                selected_objectives.add(anchor_value)
        parent_count = min(max(1, int(parent_count)), n_items)
        while len(selected) < parent_count:
            remaining = [index for index in range(n_items) if index not in selected and (values is None or not np.isfinite(float(values[index])) or float(values[index]) not in selected_objectives)]
            if not remaining:
                break
            ranked = sorted(remaining, key=lambda index: float(np.min(distance_matrix[index, selected])), reverse=True)
            top_width = min(4, len(ranked))
            choice = ranked[(int(variant) + len(selected) - 1) % top_width]
            selected.append(int(choice))
            if values is not None and np.isfinite(float(values[choice])):
                selected_objectives.add(float(values[choice]))
        return selected

    @staticmethod
    def _reciprocal_rank_parent_indices(distance_matrix, parent_count, anchor, rng, objective_values=None, prefer_near=False):
        """Sample distance ranks: BX prefers far, BR prefers near.

        The anchor is fixed as the first parent. Every additional parent is
        sampled without replacement after recomputing its minimum behavior
        distance to the current selected set. Ranking is descending by that
        distance for BX, ascending for BR (prefer_near=True).
        Rank 1 is preferred, with probability ``(1 / rank) / sum(1 / rank)``.
        There is no temperature or additional selection parameter. The caller
        supplies the already-seeded experiment RNG so this method does not
        create an independent random stream.
        """
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        n_items = int(distance_matrix.shape[0])
        if n_items == 0:
            return []
        if distance_matrix.ndim != 2 or distance_matrix.shape[1] != n_items:
            raise ValueError('distance_matrix must be a square matrix')
        if rng is None or not hasattr(rng, 'choice'):
            raise ValueError('rng with a choice method is required')
        selected = [int(anchor) % n_items]
        values = None
        selected_objectives = set()
        if objective_values is not None:
            values = np.asarray(objective_values, dtype=np.float64).reshape(-1)
            if len(values) != n_items:
                raise ValueError('objective_values must match the parent distance matrix')
            anchor_value = float(values[selected[0]])
            if np.isfinite(anchor_value):
                selected_objectives.add(anchor_value)
        parent_count = min(max(1, int(parent_count)), n_items)
        while len(selected) < parent_count:
            remaining = [index for index in range(n_items) if index not in selected and (values is None or not np.isfinite(float(values[index])) or float(values[index]) not in selected_objectives)]
            if not remaining:
                break
            if len(remaining) == 1:
                choice = int(remaining[0])
                selected.append(choice)
                if values is not None and np.isfinite(float(values[choice])):
                    selected_objectives.add(float(values[choice]))
                continue
            ranked = sorted(remaining, key=lambda index: float(np.min(distance_matrix[index, selected])), reverse=not prefer_near)
            weights = 1.0 / np.arange(1, len(ranked) + 1, dtype=np.float64)
            probabilities = weights / float(np.sum(weights))
            choice_position = int(rng.choice(len(ranked), p=probabilities))
            choice = int(ranked[choice_position])
            selected.append(choice)
            if values is not None and np.isfinite(float(values[choice])):
                selected_objectives.add(float(values[choice]))
        return selected

    @staticmethod
    def _refinement_parent_indices(distance_matrix, anchor, objective_values=None, variant=0):
        """Pair one quality anchor with a nearby, different-score contrast."""
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        n_items = int(distance_matrix.shape[0])
        if n_items == 0:
            return []
        anchor = int(anchor) % n_items
        remaining = [index for index in range(n_items) if index != anchor]
        if not remaining:
            return [anchor]
        if objective_values is not None:
            values = np.asarray(objective_values, dtype=np.float64).reshape(-1)
            if len(values) != n_items:
                raise ValueError('objective_values must match the parent distance matrix')
            anchor_value = float(values[anchor])
            if np.isfinite(anchor_value):
                scale = max(1.0, abs(anchor_value))
                tolerance = 1e-09 * scale
                different = [index for index in remaining if not np.isfinite(float(values[index])) or abs(float(values[index]) - anchor_value) > tolerance]
                if different:
                    remaining = different
        ranked = sorted(remaining, key=lambda index: float(distance_matrix[anchor, index]))
        contrast = ranked[int(variant) % len(ranked)]
        return [anchor, int(contrast)]

    def _region_parent_pools(self, pop):
        if not self._initialize_regions():
            return {}
        parent_pool, matrix, objectives = self._region_advantage_archive_rows()
        if matrix is None or not parent_pool:
            return {}
        if matrix.shape[1] != self._region_feature_dim:
            logging.warning('[RegionAdvantageParents] feature dimension changed: archive=%s fitted=%s', matrix.shape[1], self._region_feature_dim)
            return {}
        parent_pool = list(parent_pool)
        matrix = np.asarray(matrix, dtype=np.float64)
        objectives = np.asarray(objectives, dtype=np.float64).reshape(-1)
        forced_anchor_region_by_index = {}
        code_to_index = {self._canonical_code(parent.get('code')): index for index, parent in enumerate(parent_pool) if self._canonical_code(parent.get('code'))}
        for state_index, state in enumerate(self._region_states):
            anchor_code = self._canonical_code(state.get('center_code'))
            anchor_feature = np.asarray(state.get('center_feature', []), dtype=np.float64).reshape(-1)
            anchor_objective = self._safe_objective(state.get('best_objective'))
            if not anchor_code or anchor_feature.size != matrix.shape[1] or (not np.all(np.isfinite(anchor_feature))) or (not np.isfinite(anchor_objective)):
                continue
            anchor_index = code_to_index.get(anchor_code)
            if anchor_index is None:
                anchor_index = len(parent_pool)
                parent_pool.append({'code': anchor_code, 'algorithm': state.get('center_algorithm'), 'algorithm_id': state.get('center_algorithm_id'), 'objective': float(anchor_objective), 'behavior_embedding': anchor_feature.copy(), 'operator': 'kmeans_region_anchor'})
                matrix = np.vstack([matrix, anchor_feature])
                objectives = np.append(objectives, float(anchor_objective))
                code_to_index[anchor_code] = anchor_index
            forced_anchor_region_by_index[int(anchor_index)] = int(state_index)
        space = self._transform_region_features(matrix)
        centers = np.vstack([state['center_point'] for state in self._region_states])
        distance2 = np.sum((space[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(distance2, axis=1)
        for anchor_index, state_index in forced_anchor_region_by_index.items():
            assignments[int(anchor_index)] = int(state_index)
        pools = {}
        pool_stats = {}
        for state_index, state in enumerate(self._region_states):
            assigned = np.flatnonzero(assignments == state_index).astype(int).tolist()
            candidate_indices = assigned
            if not candidate_indices:
                candidate_indices = np.argsort(distance2[:, state_index]).astype(int).tolist()
            ordered = sorted(candidate_indices, key=lambda index: (float(objectives[index]), float(distance2[index, state_index]), index))
            entries = []
            seen_codes = set()
            selected_scores = []
            for index in ordered:
                parent = parent_pool[index]
                code_key = self._canonical_code(parent.get('code'))
                objective = float(objectives[index])
                if not code_key or code_key in seen_codes or (not np.isfinite(objective)):
                    continue
                duplicate_score = any((abs(objective - previous) <= 1e-09 * max(1.0, abs(objective), abs(previous)) for previous in selected_scores))
                if duplicate_score:
                    continue
                seen_codes.add(code_key)
                selected_scores.append(objective)
                entries.append({'parent': parent, 'point': space[index], 'objective': objective})
            pools[int(state['region_id'])] = entries
            state['member_count'] = int(len(assigned))
            finite_objectives = [self._safe_objective(entry['parent'].get('objective')) for entry in entries if np.isfinite(self._safe_objective(entry['parent'].get('objective')))]
            pool_stats[int(state['region_id'])] = {'assigned': int(len(assigned)), 'parents': int(len(entries)), 'dedupe': 'code+score', 'best': round(float(min(finite_objectives)), 6) if finite_objectives else None}
        logging.info('[RegionAdvantageParents] source=advantage_archive target=%s records=%s distinct_scores=%s regions=%s parent_order=true_score parent_dedupe=code+score behavior_dedupe=removed assignment=%s parent_source=advantage_archive', self.region_archive_target_distinct, len(parent_pool), len(set((float(value) for value in objectives))), pool_stats, 'nearest_behavior_center')
        return pools

    def _build_inter_region_bx_parent_batches(self, pop, batch_size, pools=None, quotas=None):
        """Build cross-region BX batches with a persistent 1..K parent cycle.

        The scheduled region owns each generation slot and supplies parent 1.
        Let K be the number of currently non-empty region parent pools.  The
        requested parent count is ``1 + cursor % K``; at count one the BX
        prompt receives only the owning region's elite anchor.  For larger
        counts, other regions are ordered by their current Top-1 true
        objective and contribute their Top-1 entries first.  Lower-ranked
        entries are used only when a set would repeat an already registered
        unordered code combination.

        This is deliberately scoped to cross-region BX.  Region-local BX,
        BR, and archive fallback continue to use the strict two-parent
        registrar and reciprocal distance-rank probability.
        """
        pools = self._region_parent_pools(pop) if pools is None else pools
        if not pools:
            return []
        ranked_entries = {}
        for region_id, entries in pools.items():
            if not entries:
                continue
            ranked_entries[int(region_id)] = sorted(entries, key=lambda entry: (self._safe_objective(entry['parent'].get('objective')), self._canonical_code(entry['parent'].get('code'))))
        all_region_ids = sorted(ranked_entries)
        if not all_region_ids:
            return []
        scheduled_region_ids = self._scheduled_region_ids('bx') if quotas is None else set(quotas)
        anchor_region_ids = [region_id for region_id in all_region_ids if scheduled_region_ids is None or region_id in scheduled_region_ids]
        if not anchor_region_ids:
            logging.info('[RegionParents] operator=bx has no scheduled regions')
            return []
        batch_size = max(0, int(batch_size))
        target_quotas = self._operator_region_generation_quotas('bx', anchor_region_ids, batch_size) if quotas is None else quotas
        accepted_by_region = {region_id: 0 for region_id in anchor_region_ids}
        rows = []
        attempts = 0
        max_attempts = max(batch_size * 32, 64)
        region_cursor = 0
        parent_cursor = int(getattr(self, '_bx_inter_region_parent_count_cursor', 0))
        cursor_start = parent_cursor
        active_region_count = len(all_region_ids)
        while len(rows) < batch_size and attempts < max_attempts:
            active_anchor_ids = [region_id for region_id in anchor_region_ids if accepted_by_region[region_id] < target_quotas.get(region_id, 0)]
            if not active_anchor_ids:
                break
            parent_cycle = parent_cursor // active_region_count
            anchor_region_id = active_anchor_ids[(region_cursor + parent_cycle) % len(active_anchor_ids)]
            region_cursor += 1
            requested_parent_count = 1 + parent_cursor % active_region_count
            parent_cursor += 1
            other_region_ids = sorted((region_id for region_id in all_region_ids if region_id != anchor_region_id), key=lambda region_id: (self._safe_objective(ranked_entries[region_id][0]['parent'].get('objective')), region_id))
            selected_region_ids = [anchor_region_id] + other_region_ids[:requested_parent_count - 1]
            selected_entries = None
            replaceable_slots = list(range(1, len(selected_region_ids)))
            max_depth = max((len(ranked_entries[selected_region_ids[slot]]) for slot in replaceable_slots), default=1)
            max_variants = 1 + len(replaceable_slots) * max_depth
            for variant in range(max_variants):
                candidate_entries = []
                valid_variant = True
                changed_slot = None
                changed_depth = 0
                if variant > 0 and replaceable_slots:
                    changed_slot = replaceable_slots[(variant - 1) % len(replaceable_slots)]
                    changed_depth = 1 + (variant - 1) // len(replaceable_slots)
                for slot, region_id in enumerate(selected_region_ids):
                    entry_index = changed_depth if slot == changed_slot else 0
                    entries = ranked_entries[region_id]
                    if entry_index >= len(entries):
                        valid_variant = False
                        break
                    candidate_entries.append(entries[entry_index])
                if not valid_variant:
                    continue
                parents = [entry['parent'] for entry in candidate_entries]
                keys = [self._canonical_code(parent.get('code')) for parent in parents]
                if any(not key for key in keys) or len(set(keys)) != len(keys):
                    continue
                if self._register_parent_batch(parents, expected_count=len(parents)):
                    selected_entries = candidate_entries
                    break
            attempts += 1
            if selected_entries is None:
                continue
            accepted_by_region[anchor_region_id] += 1
            rows.append({'region_id': int(anchor_region_id),
                         'parents': [entry['parent'] for entry in selected_entries],
                         'requested_parent_count': int(requested_parent_count),
                         'actual_parent_count': int(len(selected_entries)),
                         'operator_schedule': self._region_operator_metadata(anchor_region_id, operator='bx')})
        self._bx_inter_region_parent_count_cursor = parent_cursor
        if len(rows) < batch_size:
            logging.warning('[RegionParents] only %s/%s non-repeating parent sets available for operator=bx', len(rows), batch_size)
        logging.info('[RegionParents] operator=bx generated_parent_batches=%s planned_quotas=%s allocations=%s parent_count_histogram=%s selection=legacy_cross_top_priority_cycle_1_to_K parent_source=advantage_archive K_active=%s cursor_start=%s cursor_end=%s',
                     len(rows), target_quotas,
                     {region_id: sum((row['region_id'] == region_id for row in rows)) for region_id in anchor_region_ids},
                     {count: sum((row['actual_parent_count'] == count for row in rows)) for count in sorted({row['actual_parent_count'] for row in rows})},
                     active_region_count, cursor_start, parent_cursor)
        return rows

    def _build_region_parent_batches(self, pop, operator, batch_size, pools=None, quotas=None):
        if operator == 'bx' and False:
            return self._build_balanced_bx_parent_batches(pop, batch_size)
        if operator == 'bx' and getattr(self, '_bx_use_inter_region_this_round', False):
            return self._build_inter_region_bx_parent_batches(pop, batch_size, pools, quotas)
        pools = self._region_parent_pools(pop) if pools is None else pools
        if not pools:
            return []
        scheduled_region_ids = self._scheduled_region_ids(operator) if quotas is None else set(quotas)
        if scheduled_region_ids is not None:
            pools = {region_id: entries for region_id, entries in pools.items() if int(region_id) in scheduled_region_ids}
            if not pools:
                logging.info('[RegionParents] operator=%s has no scheduled regions', operator)
                return []
        region_ids = sorted(pools)
        batch_size = max(0, int(batch_size))
        target_quotas = self._operator_region_generation_quotas(operator, region_ids, batch_size) if quotas is None else quotas
        rows = []
        attempts = 0
        max_attempts = max(batch_size * 32, 64)
        accepted_by_region = {region_id: 0 for region_id in region_ids}
        attempts_by_region = {region_id: 0 for region_id in region_ids}
        region_geometry = {}
        for region_id in region_ids:
            entries = pools[region_id]
            if not entries:
                continue
            quality_order = sorted(range(len(entries)), key=lambda index: self._safe_objective(entries[index]['parent'].get('objective')))
            points = np.vstack([entry['point'] for entry in entries])
            region_geometry[region_id] = (quality_order, self._pairwise_euclidean_distances(points))
        region_cursor = 0
        while len(rows) < batch_size and attempts < max_attempts:
            active_region_ids = [region_id for region_id in region_ids if accepted_by_region[region_id] < target_quotas.get(region_id, 0)]
            if not active_region_ids:
                break
            region_id = active_region_ids[region_cursor % len(active_region_ids)]
            region_cursor += 1
            entries = pools[region_id]
            if not entries:
                attempts += 1
                continue
            if len(entries) < _BEHAVIOR_PARENT_COUNT:
                attempts += 1
                continue
            quality_order, distances = region_geometry[region_id]
            region_round = attempts_by_region[region_id]
            attempts_by_region[region_id] += 1
            if operator in BEHAVIOR_REFINE_OPERATORS:
                anchor_index = quality_order[0]
            else:
                anchor_index = quality_order[region_round % len(quality_order)]
            objective_values = [self._safe_objective(entry['parent'].get('objective')) for entry in entries]
            distinct_objectives = {float(value) for value in objective_values if np.isfinite(value)}
            max_parent_count = min(len(entries), len(distinct_objectives) if distinct_objectives else len(entries))
            if max_parent_count < _BEHAVIOR_PARENT_COUNT:
                attempts += 1
                continue
            parent_count = _BEHAVIOR_PARENT_COUNT
            if operator in BEHAVIOR_EXPAND_OPERATORS | BEHAVIOR_REFINE_OPERATORS:
                selected_indices = self._reciprocal_rank_parent_indices(distances, parent_count=parent_count, anchor=anchor_index, rng=self._region_rng, objective_values=objective_values, prefer_near=operator in BEHAVIOR_REFINE_OPERATORS)
            elif operator in BEHAVIOR_REFINE_OPERATORS and parent_count > 1:
                selected_indices = self._refinement_parent_indices(distances, anchor=anchor_index, objective_values=objective_values, variant=region_round)
            else:
                selected_indices = self._diverse_parent_indices(distances, parent_count=parent_count, anchor=anchor_index, variant=region_round // max(1, len(quality_order)), objective_values=objective_values)
            attempts += 1
            if len(selected_indices) < parent_count:
                continue
            parents = [entries[index]['parent'] for index in selected_indices]
            if not self._register_parent_batch(parents):
                continue
            accepted_by_region[region_id] += 1
            rows.append({'region_id': int(region_id), 'parents': parents, 'requested_parent_count': _BEHAVIOR_PARENT_COUNT, 'actual_parent_count': int(len(parents)), 'operator_schedule': self._region_operator_metadata(region_id, operator=operator)})
        if len(rows) < batch_size:
            logging.warning('[RegionParents] only %s/%s non-repeating parent sets available', len(rows), batch_size)
        logging.info('[RegionParents] operator=%s generated_parent_batches=%s planned_quotas=%s allocations=%s parent_count_histogram=%s selection=%s parent_source=advantage_archive', operator, len(rows), target_quotas, {region_id: sum((row['region_id'] == region_id for row in rows)) for region_id in region_ids}, {count: sum((row['actual_parent_count'] == count for row in rows)) for count in sorted({row['actual_parent_count'] for row in rows})}, 'behavior_reciprocal_rank_nearest' if operator in BEHAVIOR_REFINE_OPERATORS else 'behavior_reciprocal_rank_farthest' if operator in BEHAVIOR_EXPAND_OPERATORS else 'behavior_max_min')
        return rows

    @staticmethod
    def _region_accounting_id(offspring):
        """Return the region that receives budget and outcome credit."""
        for key in ('region_allocation_id', 'region_source_id', 'region_membership_id'):
            value = offspring.get(key)
            if value is not None:
                return int(value)
        return None

    def _annotate_region_candidates(self, candidates):
        if not self._region_states or not candidates:
            return
        centers = np.vstack([state['center_point'] for state in self._region_states])
        region_ids = [int(state['region_id']) for state in self._region_states]
        state_by_id = {int(state['region_id']): state for state in self._region_states}
        for candidate in candidates:
            offspring = self._candidate_to_offspring(candidate)
            feature = offspring.get('behavior_embedding', offspring.get('embedding'))
            if feature is None:
                continue
            point = self._transform_region_features(feature)
            distance2 = np.sum((centers - point[None, :]) ** 2, axis=1)
            membership_id = region_ids[int(np.argmin(distance2))]
            raw_source_id = offspring.get('region_source_id')
            source_id = int(raw_source_id) if raw_source_id is not None else None
            if source_id not in state_by_id:
                source_id = None
            allocation_id = membership_id
            allocation_state = state_by_id[allocation_id]
            offspring['region_membership_id'] = membership_id
            offspring['region_allocation_id'] = int(allocation_id)
            offspring['region_source_matches_membership'] = bool(source_id is not None and source_id == membership_id)
            offspring['region_distance'] = float(np.linalg.norm(point - allocation_state['center_point']))

    def _select_by_advantage_regions(self, candidates, top_k):
        """Allocate slots by real-performance rank, then rank locally."""
        if not candidates or top_k <= 0:
            return []
        groups = {}
        for candidate in candidates:
            offspring = self._candidate_to_offspring(candidate)
            region_id = offspring.get('region_allocation_id')
            if region_id is None:
                continue
            groups.setdefault(int(region_id), []).append(candidate)
        if not groups:
            shuffled = list(candidates)
            self._region_rng.shuffle(shuffled)
            return shuffled[:min(int(top_k), len(shuffled))]
        top_k = min(int(top_k), sum((len(rows) for rows in groups.values())))
        state_by_id = {int(state['region_id']): state for state in self._region_states}
        finite_region_best = [self._safe_objective(state.get('best_objective')) for state in self._region_states if np.isfinite(self._safe_objective(state.get('best_objective')))]
        global_best = min(finite_region_best) if finite_region_best else 0.0
        region_ids = sorted(groups)
        gap_by_region = {}
        for region_id in region_ids:
            state = state_by_id.get(region_id, {})
            region_best = self._safe_objective(state.get('best_objective'))
            gap = self._region_relative_global_gap(region_best, global_best)
            gap_by_region[region_id] = float(gap)
        ranked_region_ids = sorted(region_ids, key=lambda region_id: (self._safe_objective(state_by_id.get(region_id, {}).get('best_objective')), int(region_id)))
        rank_by_region = {region_id: rank for rank, region_id in enumerate(ranked_region_ids, start=1)}
        capacities = {region_id: len(groups[region_id]) for region_id in region_ids}
        from ..regional_allocation import annealed_winner_take_most_quotas
        quotas, details, self._regional_allocation_cursor = annealed_winner_take_most_quotas(ranked_region_ids, capacities, top_k, self.get_successful_real_eval_count(), self.real_eval_budget, self.regional_allocated_count, self._regional_allocation_cursor)
        self.regional_allocated_count = details['regional_allocated_count']
        details['regional_best'] = {r: self._safe_objective(state_by_id.get(r, {}).get('best_objective')) for r in ranked_region_ids}
        logging.info('[RegionalAllocation] %s', details)
        global_ready = self._ensure_surrogate()
        prediction_count = 0
        if global_ready:
            for candidate in candidates:
                predicted = self._predict_behavior(candidate, global_ready)
                if predicted is not None and np.isfinite(predicted):
                    prediction_count += 1
        selected = []
        selection_slot = 0
        for region_id in region_ids:
            rows = groups[region_id]
            quota = min(quotas[region_id], len(rows))
            if quota <= 0:
                continue
            ranked_rows = sorted(enumerate(rows), key=lambda item: (0 if np.isfinite(self._safe_objective(self._candidate_to_offspring(item[1]).get('behavior_pred_score'))) else 1, self._safe_objective(self._candidate_to_offspring(item[1]).get('behavior_pred_score')), float(self._candidate_to_offspring(item[1]).get('region_distance', float('inf'))), item[0]))
            chosen_indices = [index for index, _ in ranked_rows[:quota]]
            for local_slot, index in enumerate(chosen_indices, start=1):
                candidate = rows[int(index)]
                offspring = self._candidate_to_offspring(candidate)
                selection_slot += 1
                offspring['region_selection_slot'] = selection_slot
                offspring['region_local_distance_rank'] = local_slot
                offspring['region_budget_allocation'] = quota
                offspring['advantage_region_gap'] = gap_by_region[region_id]
                offspring['advantage_region_rank'] = rank_by_region[region_id]
                used_prediction = np.isfinite(self._safe_objective(offspring.get('behavior_pred_score')))
                offspring['selected_by_predictor'] = bool(used_prediction)
                offspring['selection_method'] = 'advantage_region_xgboost_direct' if used_prediction else 'advantage_region_nearest_fallback'
                selected.append(candidate)
        self._region_rng.shuffle(selected)
        selected_ids = {id(candidate) for candidate in selected}
        ranked_for_log = sorted(candidates, key=lambda candidate: (self._safe_objective(self._candidate_to_offspring(candidate).get('behavior_pred_score')), self._safe_objective(self._candidate_to_offspring(candidate).get('region_distance'))))
        selection_log_top = int(getattr(self, 'selection_log_top', 10))
        if selection_log_top > 0 and ranked_for_log:
            shown = ranked_for_log[:selection_log_top]
            rank_log_label = 'BehaviorRank'
            logging.debug('[%s] rank sel region source prediction code_hash', rank_log_label)
            for rank, candidate in enumerate(shown, start=1):
                offspring = self._candidate_to_offspring(candidate)
                logging.debug('[%s] %4s %3s %6s %9s %12s', rank_log_label, rank, '*' if id(candidate) in selected_ids else '', offspring.get('region_allocation_id', '-'), self._fmt_float(offspring.get('behavior_pred_score')).strip(), str(abs(hash(offspring.get('code', ''))))[-12:])
        selection_mode = 'global_xgboost_prediction' if prediction_count else 'nearest_region_fallback'
        logging.info('[AdvantageRegion] candidates=%s selected=%s/%s available=%s priority=%s quotas=%s selection_mode=%s predicted=%s/%s', len(candidates), len(selected), top_k, {region_id: len(groups[region_id]) for region_id in region_ids}, ranked_region_ids, quotas, selection_mode, prediction_count, len(candidates))
        logging.info('[RegionBudget] total_slots=%s global_best=%.6g represented=%s/%s allocation=annealed_winner_take_most', top_k, global_best, len(region_ids), len(state_by_id))
        for region_id in sorted(set(state_by_id) | set(region_ids)):
            state = state_by_id.get(region_id, {})
            region_best = self._safe_objective(state.get('best_objective'))
            gap = self._region_relative_global_gap(region_best, global_best)
            available = len(groups.get(region_id, []))
            if region_id not in groups:
                logging.info('[RegionBudget] R%s available=0 best=%.6g gap=%.4f quota=0 reason=no_candidates', region_id, region_best, gap)
                continue
            logging.info('[RegionBudget] R%s available=%s best=%.6g gap=%.4f advantage_rank=%s quota=%s', region_id, available, region_best, gap_by_region[region_id], rank_by_region[region_id], quotas[region_id])
        return selected

    def _get_offspring_from_region_parents(self, parents, operator, region_id):
        offspring = {'algorithm': None, 'code': None, 'objective': None, 'other_inf': None, 'operator': operator, 'generation_parent_scope': 'region' if region_id is not None else 'advantage_archive', 'generation_attempts': 1}
        if region_id is not None:
            offspring['region_source_id'] = int(region_id)
            schedule = self._region_operator_metadata(region_id, operator=operator)
            if schedule:
                offspring['region_operator_mode'] = schedule.get('mode')
        try:
            if operator in BEHAVIOR_EXPAND_OPERATORS:
                self._claim_generated_algorithm_slots(1, context=f'region_fallback_{operator}')
                offspring['code'], offspring['algorithm'] = self.evol.bx(parents)
            elif operator in BEHAVIOR_REFINE_OPERATORS:
                self._claim_generated_algorithm_slots(1, context=f'region_fallback_{operator}')
                offspring['code'], offspring['algorithm'] = self.evol.br(parents)
            else:
                raise ValueError(f'Unsupported behavior-region operator: {operator}')
            offspring['_lineage_prompt'] = getattr(self.evol, 'last_prompt_content', None)
            offspring['_lineage_response'] = getattr(self.evol, 'last_response', None)
            offspring['_lineage_llm_attempts'] = getattr(self.evol, 'last_llm_attempts', 1)
        except Exception as err:
            offspring['behavior_feature_failed'] = True
            offspring['skip_real_eval'] = True
            failure_scope = offspring['generation_parent_scope']
            offspring['skip_reason'] = f'{failure_scope}_generation_failed: {err}'
            offspring['lineage_status'] = f'{failure_scope}_generation_failed'
        self._register_lineage_event(parents, offspring, operator)
        return (parents, offspring)

    def _build_archive_parent_batches(self, pop, operator, batch_size):
        """Build behavior-diverse, non-repeating archive parent sets."""
        parent_pool = []
        seen_codes = set()
        for parent in list(pop or []):
            if not isinstance(parent, dict):
                continue
            code_key = self._canonical_code(parent.get('code'))
            if not code_key or code_key in seen_codes:
                continue
            seen_codes.add(code_key)
            parent_pool.append(parent)
        if not parent_pool:
            raise ValueError('Candidate generation requires a non-empty advantage archive')
        if len(parent_pool) < _BEHAVIOR_PARENT_COUNT:
            logging.info('[ArchiveParents] operator=%s skipped | reason=fewer_than_two_distinct_parents', operator)
            return []
        finite_parents = [parent for parent in parent_pool if np.isfinite(self._safe_objective(parent.get('objective')))]
        elite_parent = min(finite_parents or parent_pool, key=lambda parent: self._safe_objective(parent.get('objective')))
        elite_code = self._canonical_code(elite_parent.get('code'))
        behavior_rows = []
        valid_geometry = True
        feature_dim = None
        for parent in parent_pool:
            feature = parent.get('behavior_embedding', parent.get('embedding'))
            row = np.asarray(feature, dtype=np.float64).reshape(-1) if feature is not None else np.asarray([])
            if row.size == 0 or not np.all(np.isfinite(row)):
                valid_geometry = False
                break
            feature_dim = feature_dim or int(row.size)
            if row.size != feature_dim:
                valid_geometry = False
                break
            behavior_rows.append(row)
        if valid_geometry:
            geometry = self._robust_standardize_features(np.vstack(behavior_rows))
            distances = self._pairwise_euclidean_distances(geometry)
        else:
            indices = np.arange(len(parent_pool), dtype=np.float64).reshape(-1, 1)
            distances = self._pairwise_euclidean_distances(indices)
        elite_index = parent_pool.index(elite_parent)
        objective_values = [self._safe_objective(parent.get('objective')) for parent in parent_pool]
        batches = []
        attempts = 0
        max_attempts = max(int(batch_size) * 32, 64)
        while len(batches) < max(0, int(batch_size)) and attempts < max_attempts:
            anchor = elite_index if operator in BEHAVIOR_REFINE_OPERATORS or attempts == 0 else attempts % len(parent_pool)
            parent_count = _BEHAVIOR_PARENT_COUNT
            if operator in BEHAVIOR_REFINE_OPERATORS and parent_count > 1:
                selected_indices = self._refinement_parent_indices(distances, anchor=anchor, objective_values=objective_values, variant=attempts)
            else:
                selected_indices = self._diverse_parent_indices(distances, parent_count=parent_count, anchor=anchor, variant=attempts // max(1, len(parent_pool)), objective_values=objective_values)
            parents = [parent_pool[index] for index in selected_indices]
            attempts += 1
            if len(parents) != _BEHAVIOR_PARENT_COUNT:
                continue
            if not self._register_parent_batch(parents):
                continue
            batches.append(parents)
        if len(batches) < int(batch_size):
            logging.warning('[ArchiveParents] only %s/%s non-repeating parent sets available', len(batches), batch_size)
        logging.info('[ArchiveParents] operator=%s candidates=%s unique_parent_sets=%s parent_count_histogram=%s elite_in_first_batch=%s elite_objective=%.6g behavior_diverse=%s selection=%s', operator, len(batches), len(batches), {count: sum((len(parents) == count for parents in batches)) for count in sorted({len(parents) for parents in batches})}, bool(batches and elite_code in self._parent_batch_signature(batches[0])), self._safe_objective(elite_parent.get('objective')), valid_geometry, 'best_anchor_plus_nearby_contrast' if operator in BEHAVIOR_REFINE_OPERATORS else 'behavior_max_min')
        return batches


    def get_offspring_batch(self, pop, operator, batch_size, parent_scope='auto'):
        """Generate one operator batch concurrently and register lineage serially."""
        batch_size = max(0, int(batch_size))
        if batch_size == 0:
            return []
        parent_scope = str(parent_scope or 'auto').strip().lower()
        region_parent_rows = None
        if parent_scope != 'advantage_archive' and operator in STANDARD_BEHAVIOR_OPERATORS and self.should_use_regions():
            region_parent_rows = self._build_region_parent_batches(pop, operator, batch_size)
            parent_batches = [row['parents'] for row in region_parent_rows]
            generation_parent_scope = 'region'
        elif parent_scope == 'advantage_archive' or operator in STANDARD_BEHAVIOR_OPERATORS:
            parent_batches = self._build_archive_parent_batches(pop, operator, batch_size)
            generation_parent_scope = 'advantage_archive'
        elif operator == 'i1':
            parent_batches = [None for _ in range(batch_size)]
            generation_parent_scope = 'initialization'
        else:
            parent_batches = [self._sample_operator_parents(pop, operator) for _ in range(batch_size)]
            generation_parent_scope = 'advantage_archive'
        self._claim_generated_algorithm_slots(len(parent_batches), context=f'batch_{operator}')
        generated_rows = self.evol.generate_batch(operator=operator, parent_batches=parent_batches)
        candidates = []
        for row_index, (parents, generated) in enumerate(zip(parent_batches, generated_rows)):
            offspring = {'algorithm': generated.get('algorithm'), 'code': generated.get('code'), 'objective': None, 'other_inf': None, '_lineage_prompt': generated.get('prompt'), '_lineage_response': generated.get('response'), '_lineage_llm_attempts': generated.get('llm_attempts', 1), 'operator': operator, 'generation_parent_scope': generation_parent_scope, 'generation_attempts': int(generated.get('llm_attempts', 1))}
            if region_parent_rows is not None:
                region_row = region_parent_rows[row_index]
                offspring['region_source_id'] = int(region_row['region_id'])
                schedule = region_row.get('operator_schedule') or {}
                if schedule:
                    offspring['region_operator_mode'] = schedule.get('mode')
                for key in ('bx_parent_schedule_slot', 'bx_parent_schedule_kind'):
                    if key in region_row:
                        offspring[key] = region_row[key]
            if not offspring['code']:
                offspring['behavior_feature_failed'] = True
                offspring['skip_real_eval'] = True
                offspring['skip_reason'] = 'llm_response_unparsable'
                offspring['lineage_status'] = 'llm_response_unparsable'
            self._register_lineage_event(parents, offspring, operator)
            candidates.append((parents, offspring))
        return candidates

    def get_offspring(self, pop, operator, generation_context=None):
        parents, offspring = (None, {'algorithm': None, 'code': None, 'objective': None, 'other_inf': None})
        generation_attempts = 0
        try:
            parents, offspring = self._get_alg(pop, operator, generation_context=generation_context)
            generation_attempts += 1
            self._register_lineage_event(parents, offspring, operator)
            n_retry = 1
            while self.check_duplicate(pop, offspring['code']):
                offspring['generation_attempts'] = int(generation_attempts)
                offspring['lineage_status'] = 'generation_duplicate_retry'
                self._finalize_lineage_candidates([(parents, offspring)])
                n_retry += 1
                if self.debug:
                    print('duplicated code, wait 1 second and retrying ... ')
                parents, offspring = self._get_alg(pop, operator, generation_context=generation_context)
                generation_attempts += 1
                self._register_lineage_event(parents, offspring, operator)
                if n_retry > 1:
                    break
        except Exception as err:
            print(err)
        if not offspring.get('lineage_event_id'):
            self._register_lineage_event(parents, offspring, operator)
        offspring['operator'] = operator
        offspring['generation_attempts'] = int(generation_attempts)
        return (parents, offspring)

    def _build_mixed_parent_plan(self, pop):
        pools = self._region_parent_pools(pop)
        quotas = {rid: len(entries) for rid, entries in pools.items() if entries}
        bx_round = int(getattr(self, '_bx_region_round_count', 0))
        self._bx_region_round_count = bx_round + 1
        rows = []
        has_other_region = len(quotas) > 1
        intra_quotas = {rid: (n // 2 + int(n % 2 and bx_round % 2 == 0))
                        if has_other_region else n for rid, n in quotas.items()}
        inter_quotas = {rid: n - intra_quotas[rid] for rid, n in quotas.items()}
        self._active_parent_batch_signatures = set()

        def add_bx(mode, requested):
            requested = {rid: n for rid, n in requested.items() if n > 0}
            if not requested:
                return
            self._bx_use_inter_region_this_round = mode == 'inter_bx'
            batches = self._build_region_parent_batches(
                pop, 'bx', sum(requested.values()), pools=pools, quotas=requested)
            for row in batches:
                row['operator'] = 'bx'
                row['kind'] = mode
                rows.append(row)

        add_bx('intra_bx', intra_quotas)
        add_bx('inter_bx', inter_quotas)
        # Transfer unfilled slots to the other mode, sharing BX duplicate keys.
        for target, source, planned in (
                ('inter_bx', 'intra_bx', intra_quotas),
                ('intra_bx', 'inter_bx', inter_quotas)):
            deficits = {rid: max(0, n - sum(r['region_id'] == rid and r['kind'] == source
                                          for r in rows)) for rid, n in planned.items()}
            remaining = {rid: min(deficits[rid], quotas[rid] - sum(
                r['region_id'] == rid for r in rows)) for rid in quotas}
            if target != 'inter_bx' or has_other_region:
                add_bx(target, remaining)
        self._bx_use_inter_region_this_round = False
        self._active_parent_batch_signatures = set()
        for row in self._build_region_parent_batches(
                pop, 'br', sum(quotas.values()), pools=pools, quotas=quotas):
            row['operator'] = 'br'
            row['kind'] = 'intra_br'
            rows.append(row)
        logging.info('[MixedParentQuotas] parent_counts=%s intra_BX_quotas=%s inter_BX_quotas=%s actual=%s',
                     quotas, intra_quotas, inter_quotas,
                     {rid: {kind: sum(r['region_id'] == rid and r['kind'] == kind for r in rows)
                            for kind in ('intra_bx', 'inter_bx', 'intra_br')} for rid in quotas})
        return rows

    def _generate_mixed_plan(self, plan):
        """Generate frozen parent plans, keeping actual operators in lineage.

        BX and BR requests share one downstream filtering/evaluation batch.
        Never refill a failed request with another parent pair.
        """
        candidates = []
        for operator in ('bx', 'br'):
            rows = [row for row in plan if row['operator'] == operator]
            if not rows:
                continue
            if not self.llm_batch_generation_enabled:
                for row in rows:
                    candidate = self._get_offspring_from_region_parents(
                        row['parents'], operator, row['region_id'])
                    candidate[1]['generation_kind'] = row['kind']
                    candidates.append(candidate)
                continue
            self._claim_generated_algorithm_slots(len(rows), context='mixed_' + operator)
            generated = self.evol.generate_batch(operator, [r['parents'] for r in rows])
            if len(generated) != len(rows):
                raise ValueError('Mixed generation results must align with parent plan')
            for row, result in zip(rows, generated):
                offspring = dict(algorithm=result.get('algorithm'), code=result.get('code'),
                                 objective=None, other_inf=None, operator=operator,
                                 generation_parent_scope='region', region_source_id=row['region_id'],
                                 generation_kind=row['kind'], region_operator_mode='mixed_parent_coverage',
                                 generation_attempts=int(result.get('llm_attempts', 1)),
                                 _lineage_prompt=result.get('prompt'),
                                 _lineage_response=result.get('response'),
                                 _lineage_llm_attempts=result.get('llm_attempts', 1))
                if not offspring['code']:
                    offspring.update(behavior_feature_failed=True, skip_real_eval=True,
                                     skip_reason='llm_response_unparsable',
                                     lineage_status='llm_response_unparsable')
                self._register_lineage_event(row['parents'], offspring, operator)
                candidates.append((row['parents'], offspring))
        return candidates

    def get_algorithm(self, pop, operator):
        self._last_requested_candidate_count = 0
        if self.is_real_eval_budget_exhausted():
            logging.info('[RealEvalBudget] budget exhausted; skip operator=%s', operator)
            return ([], [])
        offspring_list = []
        self._active_parent_batch_signatures = set()
        predictor_has_enough_samples = self.surrogate_selection_enabled and self.get_successful_real_eval_count() >= self.behavior_predictor.min_samples and (self.behavior_predictor.get_sample_count() >= self.behavior_predictor.min_samples)
        cur_fe = self.get_successful_real_eval_count()
        mixed_round = operator == 'mixed'
        region_active = bool((mixed_round or operator in STANDARD_BEHAVIOR_OPERATORS) and self.should_use_regions() and self._initialize_regions())
        mixed_plan = None
        if mixed_round:
            if not region_active:
                raise ValueError('Mixed regional generation requires initialized regions')
            mixed_plan = self._build_mixed_parent_plan(pop)
        self._bx_use_inter_region_this_round = False
        n_generate = len(self._region_advantage_archive_rows()[0])
        regional_generate_count = 0
        region_scheduled_ids = set()
        region_fresh_counts = {}
        if mixed_round:
            n_generate = len(mixed_plan)
            regional_generate_count = n_generate
            region_scheduled_ids = {int(s['region_id']) for s in self._region_states}
            logging.info('[MixedParentPlan] total=%s counts=%s dedupe=per_operator_unordered', n_generate,
                         {kind: sum(r['kind'] == kind for r in mixed_plan) for kind in
                          ('intra_bx', 'intra_br', 'inter_bx')})
        elif region_active:
            scheduled_region_ids = self._scheduled_region_ids(operator)
            eligible_states = [state for state in self._region_states if scheduled_region_ids is None or int(state['region_id']) in scheduled_region_ids]
            region_scheduled_ids = {int(state['region_id']) for state in eligible_states}
            active_schedule = getattr(self, '_active_region_operator_schedule', {})
            if active_schedule and scheduled_region_ids is not None:
                generation_quotas = {int(state['region_id']): int(active_schedule.get(int(state['region_id']), {}).get('generation_quota', 0)) for state in eligible_states}
            else:
                generation_quotas, _ = self._region_generation_quotas(eligible_states, total_count=self.region_total_candidates)
            regional_generate_count = int(sum(generation_quotas.values()))
            n_generate = regional_generate_count
            if operator == 'bx' and regional_generate_count > 0:
                bx_round = int(getattr(self, '_bx_region_round_count', 0))
                self._bx_use_inter_region_this_round = bool(bx_round % 2)
                self._bx_region_round_count = bx_round + 1
            logging.info('[RegionGenerationBudget] operator=%s total=%s regional=%s quotas=%s', operator, n_generate, regional_generate_count, generation_quotas)
        remaining_generation_slots = self.get_remaining_generated_algorithm_slots()
        if remaining_generation_slots <= 0:
            self.raise_if_generated_algorithm_limit_reached(context=f'operator_{operator}')
        if n_generate > remaining_generation_slots:
            logging.warning('[GeneratedAlgorithmBudget] cap operator batch | operator=%s requested=%s allowed=%s generated=%s/%s', operator, n_generate, remaining_generation_slots, self.get_generated_algorithm_count(), self.generated_algorithm_limit)
            n_generate = int(remaining_generation_slots)
            if region_active:
                regional_generate_count = int(n_generate)
        self._last_requested_candidate_count = max(0, int(n_generate))
        if n_generate <= 0:
            logging.info('[OperatorBatchSkip] operator=%s reason=zero_generation_quota', operator)
            return ([], [])
        generation_start = time.perf_counter()
        regional_offspring = []
        if mixed_round:
            offspring_list = self._generate_mixed_plan(mixed_plan[:n_generate])
            regional_offspring = list(offspring_list)
        elif self.llm_batch_generation_enabled and n_generate > 1:
            logging.info('[LLMBatch] operator=%s offspring=%s workers=%s regional=%s', operator, n_generate, min(self.llm_max_workers, n_generate), regional_generate_count)
            if region_active:
                try:
                    regional_offspring = self.get_offspring_batch(pop, operator, regional_generate_count, parent_scope='region')
                except Exception as err:
                    logging.warning('[LLMBatch] regional generation failed for operator=%s; fall back to serial generation: %s', operator, err)
                offspring_list = list(regional_offspring)
            else:
                try:
                    offspring_list = self.get_offspring_batch(pop, operator, n_generate)
                except Exception as err:
                    logging.warning('[LLMBatch] batch generation failed for operator=%s; fall back to serial generation: %s', operator, err)
                    offspring_list = []
        if region_active and not mixed_round:
            missing_regional = max(0, regional_generate_count - len(regional_offspring))
            if missing_regional:
                logging.warning('[RegionParents] using serial generation for %s missing regional candidates', missing_regional)
                region_rows = self._build_region_parent_batches(pop, operator, missing_regional)
                regional_offspring.extend((self._get_offspring_from_region_parents(row['parents'], operator, row['region_id']) for row in region_rows))
            offspring_list = list(regional_offspring)
            logging.info('[RegionalCandidatePool] operator=%s regional=%s real_eval_top_k=unchanged', operator, len(regional_offspring))
        elif not mixed_round and not offspring_list:
            for _ in range(n_generate):
                print(operator)
                offspring_list.append(self.get_offspring(pop, operator))
        generation_batch_time = time.perf_counter() - generation_start
        all_generated_candidates = list(offspring_list)
        before_filter = len(offspring_list)
        offspring_list, duplicate_candidates = self._dedupe_candidates_by_code(offspring_list)
        for candidate in duplicate_candidates:
            self._candidate_to_offspring(candidate)['selected_for_evaluation'] = False
        n_pre_selection_deduped = len(duplicate_candidates)
        if n_pre_selection_deduped > 0:
            logging.info('[CandidateDedupe] removed %s duplicate-code candidates before behavior extraction; unique=%s/%s', n_pre_selection_deduped, len(offspring_list), before_filter)
        offspring_list, already_evaluated_candidates = self._filter_candidates_already_evaluated(offspring_list, pop=pop)
        n_already_evaluated_filtered = len(already_evaluated_candidates)
        if n_already_evaluated_filtered > 0:
            self._set_lineage_status(already_evaluated_candidates, 'candidate_filtered_already_evaluated', selected_for_evaluation=False, selected_by_predictor=False, evaluation_status='already_evaluated_filtered', selection_method='evaluation_history_filter')
            logging.info('[EvaluatedCodeFilter] removed %s already-evaluated candidates before behavior extraction; fresh=%s/%s', n_already_evaluated_filtered, len(offspring_list), len(offspring_list) + n_already_evaluated_filtered)
        self._attach_behavior_features(offspring_list)
        skipped_candidates = [cand for cand in offspring_list if self._candidate_to_offspring(cand).get('skip_real_eval') or self._candidate_to_offspring(cand).get('behavior_feature_failed')]
        for candidate in skipped_candidates:
            self._candidate_to_offspring(candidate)['selected_for_evaluation'] = False
        offspring_list = [cand for cand in offspring_list if not self._candidate_to_offspring(cand).get('skip_real_eval') and (not self._candidate_to_offspring(cand).get('behavior_feature_failed'))]
        if skipped_candidates:
            logging.info('[CandidateSkip] skipped %s behavior-timeout candidates before behavior-region allocation', len(skipped_candidates))
        ratio_candidate_count = len(offspring_list)
        offspring_list, behavior_duplicate_candidates = self._filter_candidates_by_raw_behavior(offspring_list)
        n_after_feature_filter = len(offspring_list)
        if region_active:
            self._annotate_region_candidates(offspring_list)
            for candidate in offspring_list:
                offspring = self._candidate_to_offspring(candidate)
                region_id = offspring.get('region_allocation_id')
                if region_id is None:
                    continue
                region_id = int(region_id)
                region_fresh_counts[region_id] = region_fresh_counts.get(region_id, 0) + 1
        n_unique_before_selection = len(offspring_list)
        n_selected = len(offspring_list)
        predictor_selection_time = 0.0
        remaining_budget = self.get_remaining_real_evals()
        from bemrs.offspring_plan import evaluation_slots
        ratio_slots = evaluation_slots(ratio_candidate_count, self.evaluation_ratio,
                                       remaining_budget, len(offspring_list))
        fallback_eval_count = ratio_slots
        logging.info('[EvaluationRatio] ratio=%s denominator=valid_unique_fresh_before_behavior_filter N=%s available=%s B_total=%s remaining=%s',
                     self.evaluation_ratio, ratio_candidate_count, len(offspring_list), ratio_slots, remaining_budget)
        if region_active:
            top_k = ratio_slots
            allocation_start = time.perf_counter()
            reserve_novelty_slot = bool(self.behavior_novelty_slot_enabled and top_k >= 2 and (len(offspring_list) > top_k))
            surrogate_top_k = top_k - 1 if reserve_novelty_slot else top_k
            logging.info('[RegionalBatchBudget] B_total=%s novelty_slots=%s B_region=%s', top_k, int(reserve_novelty_slot), surrogate_top_k)
            surrogate_selected_candidates = self._select_by_advantage_regions(offspring_list, top_k=surrogate_top_k)
            selected_candidates = list(surrogate_selected_candidates)
            novelty_candidate = None
            if reserve_novelty_slot:
                novelty_candidate = self._select_behavior_archive_novelty_candidate(offspring_list, excluded_candidates=surrogate_selected_candidates)
                if novelty_candidate is not None:
                    selected_candidates.append(novelty_candidate)
            predictor_selection_time = time.perf_counter() - allocation_start
            selected_ids = {id(candidate) for candidate in selected_candidates}
            not_selected = [candidate for candidate in offspring_list if id(candidate) not in selected_ids]
            self._set_lineage_status(surrogate_selected_candidates, 'advantage_region_xgb_selected', selected_for_evaluation=True)
            if novelty_candidate is not None:
                self._set_lineage_status([novelty_candidate], 'behavior_novelty_selected', selected_for_evaluation=True, selected_by_predictor=False)
            self._set_lineage_status(not_selected, 'advantage_region_xgb_not_selected', selected_for_evaluation=False, selected_by_predictor=False)
            for candidate in not_selected:
                offspring = self._candidate_to_offspring(candidate)
                offspring['selection_method'] = 'advantage_region_xgboost_direct' if np.isfinite(self._safe_objective(offspring.get('behavior_pred_score'))) else 'advantage_region_nearest_fallback'
            offspring_list = selected_candidates
            n_selected = len(offspring_list)
            if reserve_novelty_slot:
                stage_label = 'BeMRSSelection'
                logging.info('[%s] total_slots=%s predictor_slots=%s novelty_slots=%s selected=%s', stage_label, top_k, len(surrogate_selected_candidates), 1 if novelty_candidate is not None else 0, n_selected)
        elif predictor_has_enough_samples:
            top_k = ratio_slots
            logging.info('[Predictor] cur_fe=%s candidate_pool=%s unique_pool=%s top_k=%s selection=predicted_score', cur_fe, n_after_feature_filter, len(offspring_list), top_k)
            selected_candidates, ranked_candidates = self._rank_by_ensemble(candidates=offspring_list, cur_fe=cur_fe, top_k=top_k, minimize=True)
            selected_ids = {id(cand) for cand in selected_candidates}
            self._log_selection_ranking(operator=operator, ranked_candidates=ranked_candidates, selected_ids=selected_ids, cur_fe=cur_fe, top_k=top_k)
            not_selected = [cand for cand in offspring_list if id(cand) not in selected_ids]
            self._set_lineage_status(selected_candidates, 'surrogate_selected', selected_for_evaluation=True, selected_by_predictor=True)
            self._set_lineage_status(not_selected, 'surrogate_not_selected', selected_for_evaluation=False, selected_by_predictor=False)
            offspring_list = selected_candidates
            n_selected = len(offspring_list)
            predictor_selection_time = getattr(self, '_last_predictor_selection_time', 0.0)
            logging.info(f'[Predictor] selected {len(offspring_list)} candidates')
        elif len(offspring_list) > fallback_eval_count:
            logging.info('[RandomSelection] behavior regions not active; sample %s/%s candidates', fallback_eval_count, len(offspring_list))
            chosen_indices = set(np.atleast_1d(self._region_rng.choice(len(offspring_list), size=fallback_eval_count, replace=False)).astype(int).tolist())
            fallback_not_selected = [candidate for index, candidate in enumerate(offspring_list) if index not in chosen_indices]
            offspring_list = [candidate for index, candidate in enumerate(offspring_list) if index in chosen_indices]
            self._set_lineage_status(offspring_list, 'random_selected', selected_for_evaluation=True, selected_by_predictor=False)
            self._set_lineage_status(fallback_not_selected, 'random_not_selected', selected_for_evaluation=False, selected_by_predictor=False)
            n_selected = len(offspring_list)
        else:
            self._set_lineage_status(offspring_list, 'random_selected', selected_for_evaluation=True, selected_by_predictor=False)
        evaluated = self._evaluate_candidates(offspring_list, pop=pop, iteration=0, region_fresh_counts=region_fresh_counts, region_scheduled_ids=region_scheduled_ids)
        self._finalize_lineage_candidates(all_generated_candidates)
        self._append_timing_record({'event': 'operator_batch', 'operator': operator, 'n_generated': n_generate, 'n_candidates': before_filter, 'n_after_feature_filter': n_after_feature_filter, 'n_selected': n_selected, 'n_evaluated': getattr(self, '_last_n_evaluated', 0), 'n_skipped': n_pre_selection_deduped + n_already_evaluated_filtered + len(skipped_candidates) + len(behavior_duplicate_candidates) + (n_unique_before_selection - n_selected) + getattr(self, '_last_n_eval_skipped', 0), 'generation_batch_time': generation_batch_time, 'behavior_feature_batch_time': getattr(self, '_last_behavior_feature_batch_time', 0.0), 'embedding_batch_time': getattr(self, '_last_embedding_batch_time', 0.0), 'predictor_selection_time': predictor_selection_time, 'real_eval_batch_time': getattr(self, '_last_real_eval_batch_time', 0.0)})
        out_p = []
        out_off = []
        for p, off in evaluated:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f'>>> check offsprings: \n {off}')
        return (out_p, out_off)
