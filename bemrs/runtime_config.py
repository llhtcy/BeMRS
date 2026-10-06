"""Hydra-to-runtime configuration bridge for behavior-guided BeMRS.

The search implementation historically read many ``BEMRS_*`` environment
variables.  This module keeps that stable internal contract while making the
Hydra method/problem YAML files the single user-facing source of parameters.
"""
from __future__ import annotations
import logging
import os
from collections.abc import Iterable, Mapping
_MISSING = object()

def config_get(config, path, default=None):
    """Read a dotted path from DictConfig, dictionaries, or plain objects."""
    current = config
    for part in str(path).split('.'):
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(part, _MISSING)
        else:
            current = getattr(current, part, _MISSING)
        if current is _MISSING:
            return default
    return current

def _serialize(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, Iterable) and (not isinstance(value, (str, bytes, Mapping))):
        return ','.join((str(item) for item in value))
    return str(value)
_METHOD_ENV = {'budget.real_evaluations': 'BEMRS_REAL_EVAL_BUDGET', 'initialization.use_prompt_seed': 'BEMRS_INIT_USE_PROMPT_SEED', 'initialization.prompt_seed': 'BEMRS_INIT_PROMPT_SEED', 'generation.llm_batch': 'BEMRS_LLM_BATCH_GENERATION', 'generation.llm_max_workers': 'BEMRS_LLM_MAX_WORKERS', 'predictor.enabled': 'BEMRS_SURROGATE_SELECTION_ENABLED', 'predictor.min_samples': 'BEMRS_PREDICTOR_MIN_SAMPLES', 'predictor.max_samples': 'BEMRS_PREDICTOR_MAX_SAMPLES', 'predictor.seed': 'BEMRS_PREDICTOR_SEED', 'predictor.retrain_interval': 'BEMRS_PREDICTOR_RETRAIN_INTERVAL', 'predictor.xgboost.n_estimators': 'BEMRS_XGB_N_ESTIMATORS', 'predictor.xgboost.max_depth': 'BEMRS_XGB_MAX_DEPTH', 'predictor.xgboost.learning_rate': 'BEMRS_XGB_LEARNING_RATE', 'predictor.xgboost.subsample': 'BEMRS_XGB_SUBSAMPLE', 'predictor.xgboost.colsample_bytree': 'BEMRS_XGB_COLSAMPLE_BYTREE', 'predictor.xgboost.reg_alpha': 'BEMRS_XGB_REG_ALPHA', 'predictor.xgboost.reg_lambda': 'BEMRS_XGB_REG_LAMBDA', 'predictor.xgboost.min_child_weight': 'BEMRS_XGB_MIN_CHILD_WEIGHT', 'predictor.xgboost.tree_method': 'BEMRS_XGB_TREE_METHOD', 'predictor.xgboost.n_jobs': 'BEMRS_XGB_N_JOBS', 'behavior_space.instance_pooling': 'BEMRS_BEHAVIOR_INSTANCE_POOLING', 'behavior_filter.enabled': 'BEMRS_BEHAVIOR_FILTER_ENABLED', 'behavior_filter.start_with_predictor': 'BEMRS_BEHAVIOR_FILTER_START_WITH_PREDICTOR', 'behavior_filter.atol': 'BEMRS_BEHAVIOR_FILTER_ATOL', 'behavior_filter.rtol': 'BEMRS_BEHAVIOR_FILTER_RTOL', 'behavior_filter.xgb_rescue.enabled': 'BEMRS_BEHAVIOR_FILTER_XGB_RESCUE', 'novelty_selection.enabled': 'BEMRS_BEHAVIOR_NOVELTY_SLOT_ENABLED', 'region.enabled': 'BEMRS_REGION_ENABLED', 'region.count': 'BEMRS_REGION_COUNT', 'region.archive_target_distinct': 'BEMRS_REGION_ARCHIVE_TARGET_DISTINCT', 'region.relative_improvement': 'BEMRS_REGION_RELATIVE_IMPROVEMENT', 'region.rebuild_tolerance': 'BEMRS_REGION_REBUILD_TOLERANCE', 'region.preserve_global_best_anchor': 'BEMRS_PRESERVE_GLOBAL_BEST_ANCHOR', 'region.bx_parent_selection.tau_bx': 'BEMRS_BX_PARENT_SELECTION_TAU', 'region.scale_window': 'BEMRS_REGION_SCALE_WINDOW', 'generation.evaluation_ratio': 'BEMRS_EVALUATION_RATIO', 'region.seed': 'BEMRS_REGION_SEED', 'visualization.enabled': 'BEMRS_VISUALIZE_FEATURES', 'visualization.method': 'BEMRS_VISUALIZE_METHOD', 'visualization.pca_freeze_samples': 'BEMRS_VISUAL_PCA_FREEZE_SAMPLES', 'logging.timing_csv': 'BEMRS_TIMING_CSV', 'logging.lineage_jsonl': 'BEMRS_LINEAGE_JSONL', 'logging.lineage_edges_csv': 'BEMRS_LINEAGE_EDGES_CSV', 'logging.lineage_save_features': 'BEMRS_LINEAGE_SAVE_FEATURES', 'logging.selection_top': 'BEMRS_SELECTION_LOG_TOP', 'safety.max_stalled_generations': 'BEMRS_MAX_STALLED_GENERATIONS', 'safety.generated_algorithm_multiplier': 'BEMRS_GENERATED_ALGORITHM_LIMIT_MULTIPLIER'}
_TASK_ENV = {'behavior.dataset': 'BEMRS_BEHAVIOR_DATASET', 'behavior.batch_size': 'BEMRS_BEHAVIOR_BATCH_SIZE', 'behavior.seed': 'BEMRS_BEHAVIOR_SEED', 'operators.expand_parent_counts': 'BEMRS_BX_PARENT_COUNTS'}
_TASK_SPECIFIC_ENV = {'tsp_gls': {'behavior.dataset': 'BEMRS_TSP_GLS_BEHAVIOR_DATASET', 'behavior.matrices': 'BEMRS_TSP_GLS_BEHAVIOR_MATRICES', 'behavior.problem_size': 'BEMRS_TSP_GLS_BEHAVIOR_NODES', 'behavior.probes_per_state': 'BEMRS_TSP_GLS_PROBES_PER_STATE', 'behavior.seed': 'BEMRS_TSP_GLS_BEHAVIOR_SEED', 'behavior.heuristic_timeout': 'BEMRS_TSP_GLS_HEURISTIC_TIMEOUT', 'behavior.encode_timeout': 'BEMRS_TSP_GLS_ENCODE_TIMEOUT', 'behavior.representation': 'BEMRS_TSP_GLS_BEHAVIOR_REPRESENTATION'}, 'tsp_constructive': {'behavior.dataset': 'BEMRS_TSP_BEHAVIOR_DATASET', 'behavior.matrices': 'BEMRS_TSP_BEHAVIOR_MATRICES', 'behavior.problem_size': 'BEMRS_TSP_BEHAVIOR_NODES', 'behavior.probes': 'BEMRS_TSP_BEHAVIOR_PROBES', 'behavior.stage_ratios': 'BEMRS_TSP_BEHAVIOR_STAGE_RATIOS', 'behavior.seed': 'BEMRS_TSP_BEHAVIOR_SEED', 'behavior.probe_timeout': 'BEMRS_TSP_BEHAVIOR_PROBE_TIMEOUT', 'behavior.encode_timeout': 'BEMRS_TSP_BEHAVIOR_ENCODE_TIMEOUT', }, 'cvrp_aco': {'behavior.dataset': 'BEMRS_CVRP_BEHAVIOR_DATASET', 'behavior.matrices': 'BEMRS_CVRP_BEHAVIOR_MATRICES', 'behavior.problem_size': 'BEMRS_CVRP_BEHAVIOR_NODES', 'behavior.capacity': 'BEMRS_CVRP_CAPACITY', 'behavior.stage_ratios': 'BEMRS_CVRP_STAGE_RATIOS', 'behavior.load_ratios': 'BEMRS_CVRP_LOAD_RATIOS', 'behavior.probes_per_cell': 'BEMRS_CVRP_PROBES_PER_CELL', 'behavior.seed': 'BEMRS_CVRP_BEHAVIOR_SEED', 'behavior.heuristic_timeout': 'BEMRS_CVRP_HEURISTIC_TIMEOUT', 'behavior.encode_timeout': 'BEMRS_CVRP_ENCODE_TIMEOUT', }, 'mkp_aco': {'behavior.dataset': 'BEMRS_MKP_BEHAVIOR_DATASET', 'behavior.matrices': 'BEMRS_MKP_BEHAVIOR_MATRICES', 'behavior.problem_size': 'BEMRS_MKP_BEHAVIOR_ITEMS', 'behavior.stage_ratios': 'BEMRS_MKP_STAGE_RATIOS', 'behavior.probes_per_stage': 'BEMRS_MKP_PROBES_PER_STAGE', 'behavior.seed': 'BEMRS_MKP_BEHAVIOR_SEED', 'behavior.heuristic_timeout': 'BEMRS_MKP_HEURISTIC_TIMEOUT', 'behavior.encode_timeout': 'BEMRS_MKP_ENCODE_TIMEOUT', }, 'op_aco': {'behavior.dataset': 'BEMRS_OP_BEHAVIOR_DATASET', 'behavior.matrices': 'BEMRS_OP_BEHAVIOR_MATRICES', 'behavior.problem_size': 'BEMRS_OP_BEHAVIOR_NODES', 'behavior.stage_ratios': 'BEMRS_OP_STAGE_RATIOS', 'behavior.probes_per_stage': 'BEMRS_OP_PROBES_PER_STAGE', 'behavior.seed': 'BEMRS_OP_BEHAVIOR_SEED', 'behavior.heuristic_timeout': 'BEMRS_OP_HEURISTIC_TIMEOUT', 'behavior.encode_timeout': 'BEMRS_OP_ENCODE_TIMEOUT', }, 'bpp_online': {'behavior.dataset': 'BEMRS_BPP_ONLINE_BEHAVIOR_DATASET', 'behavior.probes': 'BEMRS_BPP_ONLINE_BEHAVIOR_PROBES', 'behavior.seed': 'BEMRS_BPP_ONLINE_BEHAVIOR_SEED', 'behavior.scorer_timeout': 'BEMRS_BPP_ONLINE_SCORER_TIMEOUT', 'behavior.encode_timeout': 'BEMRS_BPP_ONLINE_ENCODE_TIMEOUT', }}

def _effective_method_value(cfg, path):
    override = config_get(cfg, f'problem.method_overrides.{path}', _MISSING)
    if override is not _MISSING:
        return override
    return config_get(cfg, f'method.{path}', _MISSING)

def apply_runtime_config(cfg):
    """Publish resolved Hydra settings for the unchanged search internals."""
    for env_name in set(_METHOD_ENV.values()) | set(_TASK_ENV.values()) | {v for mapping in _TASK_SPECIFIC_ENV.values() for v in mapping.values()}:
        os.environ.pop(env_name, None)
    applied = {}
    for path, env_name in _METHOD_ENV.items():
        value = _effective_method_value(cfg, path)
        if value is _MISSING:
            continue
        serialized = _serialize(value)
        if serialized is None:
            os.environ.pop(env_name, None)
            continue
        os.environ[env_name] = serialized
        applied[env_name] = serialized
    problem_cfg = config_get(cfg, 'problem')
    problem_name = str(config_get(problem_cfg, 'problem_name', '')).lower()
    for path, env_name in _TASK_ENV.items():
        value = config_get(problem_cfg, path, _MISSING)
        if value is _MISSING:
            continue
        serialized = _serialize(value)
        if serialized is None:
            os.environ.pop(env_name, None)
            continue
        os.environ[env_name] = serialized
        applied[env_name] = serialized
    for path, env_name in _TASK_SPECIFIC_ENV.get(problem_name, {}).items():
        value = config_get(problem_cfg, path, _MISSING)
        if value is _MISSING:
            continue
        serialized = _serialize(value)
        if serialized is None:
            os.environ.pop(env_name, None)
            continue
        os.environ[env_name] = serialized
        applied[env_name] = serialized
    logging.info('[MethodConfig] source=hydra method=%s task=%s applied_runtime_values=%d', config_get(cfg, 'method.name', 'bemrs'), problem_name, len(applied))
    logging.info('[InitializationPromptSeed] seed=%s', applied.get('BEMRS_INIT_PROMPT_SEED', 'disabled'))
    return applied
