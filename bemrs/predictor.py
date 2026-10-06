import os
import re
import ast
import csv
import hashlib
import math
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Optional, List, Union, Tuple
import numpy as np
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from .tsp_construct_behavior import TSPBehaviorEmbedder, TSPConstructiveBehaviorEmbedder

def build_behavior_embedder(problem_name: str, problem_size: Optional[int]=None, root_dir: Optional[Union[str, os.PathLike]]=None, behavior_problem_size: Optional[int]=None, behavior_dataset: Optional[Union[str, os.PathLike]]=None, behavior_matrices: Optional[int]=None, behavior_probes: Optional[int]=None, behavior_representation: Optional[str]=None, behavior_probes_per_state: Optional[int]=None, behavior_probes_per_cell: Optional[int]=None, behavior_probes_per_stage: Optional[int]=None, behavior_instance_pooling: Optional[str]=None, behavior_heuristic_timeout: Optional[float]=None, behavior_encode_timeout: Optional[float]=None, behavior_probe_timeout: Optional[float]=None):
    """Build the task-specific behavior encoder selected by Hydra problem config."""
    task = str(problem_name or '').strip().lower()
    size = int(problem_size or 50)
    root = Path(root_dir).expanduser().resolve() if root_dir else None
    if task == 'tsp_gls':
        try:
            from .tsp_gls_behavior import TSPGLSBehaviorEmbedder
        except ImportError:
            from tsp_gls_behavior import TSPGLSBehaviorEmbedder
        representation = str(behavior_representation if behavior_representation is not None else os.environ.get('BEMRS_TSP_GLS_BEHAVIOR_REPRESENTATION', 'full')).strip().lower()
        if representation in {'12', '12d', 'compact12', 'compact_preference12', 'compact_preference_12'}:
            try:
                from .tsp_gls_compact_preference12 import TSPGLSCompactPreference12Embedder
            except ImportError:
                from tsp_gls_compact_preference12 import TSPGLSCompactPreference12Embedder
            embedder_class = TSPGLSCompactPreference12Embedder
            representation = 'compact_preference12'
        elif representation in {'full', '1008', '1008d', 'raw'}:
            embedder_class = TSPGLSBehaviorEmbedder
            representation = 'full'
        else:
            raise ValueError(f"Unsupported TSP-GLS behavior representation {representation!r}; expected 'full' or 'compact_preference12'.")
        behavior_nodes = int(behavior_problem_size if behavior_problem_size is not None else os.environ.get('BEMRS_TSP_GLS_BEHAVIOR_NODES', 50))
        configured = behavior_dataset or os.environ.get('BEMRS_TSP_GLS_BEHAVIOR_DATASET') or os.environ.get('BEMRS_BEHAVIOR_DATASET')
        if configured:
            dataset_path = Path(str(configured)).expanduser()
            if not dataset_path.is_absolute() and root is not None:
                dataset_path = root / dataset_path
            dataset_path = dataset_path.resolve()
        elif root is not None:
            dataset_path = root / 'problems' / 'tsp_gls' / 'dataset' / 'behavior50_kmeans16.npy'
        else:
            raise ValueError('TSP-GLS behavior extraction requires root_dir or BEMRS_TSP_GLS_BEHAVIOR_DATASET.')
        embedder = embedder_class(dataset_path=dataset_path, n_matrices=behavior_matrices, n_nodes=behavior_nodes, probes_per_state=behavior_probes_per_state, heuristic_timeout=behavior_heuristic_timeout, encode_timeout=behavior_encode_timeout)
        logging.info('[BehaviorFactory] TSP-GLS evaluation_nodes=%d behavior_nodes=%d representation=%s', size, behavior_nodes, representation)
    elif task == 'tsp_constructive':
        behavior_nodes = int(behavior_problem_size if behavior_problem_size is not None else os.environ.get('BEMRS_TSP_BEHAVIOR_NODES', size))
        configured = behavior_dataset or os.environ.get('BEMRS_TSP_BEHAVIOR_DATASET') or os.environ.get('BEMRS_BEHAVIOR_DATASET')
        if configured:
            dataset_path = Path(str(configured)).expanduser()
            if not dataset_path.is_absolute() and root is not None:
                dataset_path = root / dataset_path
            dataset_path = dataset_path.resolve()
        elif root is not None:
            dataset_path = root / 'problems' / 'tsp_constructive' / 'dataset' / f'train{behavior_nodes}_dataset.npy'
        else:
            dataset_path = None
        embedder = TSPConstructiveBehaviorEmbedder(n_nodes=behavior_nodes, dataset_path=dataset_path, n_matrices=behavior_matrices, n_probes=behavior_probes)
        logging.info('[BehaviorFactory] TSP evaluation_nodes=%d behavior_nodes=%d', size, behavior_nodes)
    elif task == 'cvrp_aco':
        try:
            from .cvrp_aco_behavior import CVRPACOBehaviorEmbedder
        except ImportError:
            from cvrp_aco_behavior import CVRPACOBehaviorEmbedder
        behavior_nodes = int(behavior_problem_size if behavior_problem_size is not None else os.environ.get('BEMRS_CVRP_BEHAVIOR_NODES', size))
        configured = behavior_dataset or os.environ.get('BEMRS_CVRP_BEHAVIOR_DATASET') or os.environ.get('BEMRS_BEHAVIOR_DATASET')
        if configured:
            dataset_path = Path(str(configured)).expanduser()
            if not dataset_path.is_absolute() and root is not None:
                dataset_path = root / dataset_path
            dataset_path = dataset_path.resolve()
        elif root is not None:
            dataset_path = root / 'problems' / 'cvrp_aco' / 'dataset' / f'train{behavior_nodes}_dataset.npy'
        else:
            raise ValueError('CVRP behavior extraction requires root_dir or BEMRS_CVRP_BEHAVIOR_DATASET.')
        embedder = CVRPACOBehaviorEmbedder(dataset_path=dataset_path, n_matrices=behavior_matrices, n_customers=behavior_nodes, probes_per_cell=behavior_probes_per_cell, instance_pooling=behavior_instance_pooling, heuristic_timeout=behavior_heuristic_timeout, encode_timeout=behavior_encode_timeout)
        logging.info('[BehaviorFactory] CVRP evaluation_nodes=%d behavior_nodes=%d', size, behavior_nodes)
    elif task == 'mkp_aco':
        try:
            from .mkp_aco_behavior import MKPACOBehaviorEmbedder
        except ImportError:
            from mkp_aco_behavior import MKPACOBehaviorEmbedder
        behavior_items = int(behavior_problem_size if behavior_problem_size is not None else os.environ.get('BEMRS_MKP_BEHAVIOR_ITEMS', size))
        configured = behavior_dataset or os.environ.get('BEMRS_MKP_BEHAVIOR_DATASET') or os.environ.get('BEMRS_BEHAVIOR_DATASET')
        if configured:
            dataset_path = Path(str(configured)).expanduser()
            if not dataset_path.is_absolute() and root is not None:
                dataset_path = root / dataset_path
            dataset_path = dataset_path.resolve()
        elif root is not None:
            dataset_path = root / 'problems' / 'mkp_aco' / 'dataset' / f'train{behavior_items}_dataset.npz'
        else:
            raise ValueError('MKP behavior extraction requires root_dir or BEMRS_MKP_BEHAVIOR_DATASET.')
        embedder = MKPACOBehaviorEmbedder(dataset_path=dataset_path, n_matrices=behavior_matrices, n_items=behavior_items, probes_per_stage=behavior_probes_per_stage, instance_pooling=behavior_instance_pooling, heuristic_timeout=behavior_heuristic_timeout, encode_timeout=behavior_encode_timeout)
        logging.info('[BehaviorFactory] MKP evaluation_items=%d behavior_items=%d', size, behavior_items)
    elif task == 'op_aco':
        try:
            from .op_aco_behavior import OPACOBehaviorEmbedder
        except ImportError:
            from op_aco_behavior import OPACOBehaviorEmbedder
        behavior_nodes = int(behavior_problem_size if behavior_problem_size is not None else os.environ.get('BEMRS_OP_BEHAVIOR_NODES', size))
        configured = behavior_dataset or os.environ.get('BEMRS_OP_BEHAVIOR_DATASET') or os.environ.get('BEMRS_BEHAVIOR_DATASET')
        if configured:
            dataset_path = Path(str(configured)).expanduser()
            if not dataset_path.is_absolute() and root is not None:
                dataset_path = root / dataset_path
            dataset_path = dataset_path.resolve()
        elif root is not None:
            dataset_path = root / 'problems' / 'op_aco' / 'dataset' / f'train{behavior_nodes}_dataset.npz'
        else:
            raise ValueError('OP behavior extraction requires root_dir or BEMRS_OP_BEHAVIOR_DATASET.')
        embedder = OPACOBehaviorEmbedder(dataset_path=dataset_path, n_matrices=behavior_matrices, n_nodes=behavior_nodes, probes_per_stage=behavior_probes_per_stage, instance_pooling=behavior_instance_pooling, heuristic_timeout=behavior_heuristic_timeout, encode_timeout=behavior_encode_timeout)
        logging.info('[BehaviorFactory] OP evaluation_nodes=%d behavior_nodes=%d', size, behavior_nodes)
    elif task == 'bpp_online':
        try:
            from .bpp_online_behavior import BPPOnlineBehaviorEmbedder
        except ImportError:
            from bpp_online_behavior import BPPOnlineBehaviorEmbedder
        configured = behavior_dataset or os.environ.get('BEMRS_BPP_ONLINE_BEHAVIOR_DATASET') or os.environ.get('BEMRS_BEHAVIOR_DATASET')
        if configured:
            dataset_path = Path(str(configured)).expanduser()
            if not dataset_path.is_absolute() and root is not None:
                dataset_path = root / dataset_path
            dataset_path = dataset_path.resolve()
        elif root is not None:
            dataset_path = root / 'problems' / 'bpp_online' / 'dataset' / 'bpp_online_behavior_probe_bank_v1.npz'
        else:
            raise ValueError('BPP-online behavior extraction requires root_dir or BEMRS_BPP_ONLINE_BEHAVIOR_DATASET.')
        embedder = BPPOnlineBehaviorEmbedder(dataset_path=dataset_path, n_probes=behavior_probes, scorer_timeout=behavior_heuristic_timeout, encode_timeout=behavior_encode_timeout)
        logging.info('[BehaviorFactory] BPP-online evaluation_items=%d probes=%d', size, len(embedder._states))
    else:
        raise NotImplementedError(f'No task-specific behavior extractor is registered for {task!r}. Supported cfg.problem.problem_name values are: tsp_gls, tsp_constructive, cvrp_aco, mkp_aco, op_aco, bpp_online.')
    group = os.environ.get('BEMRS_FEATURE_GROUP', 'full')
    if group == 'response3':
        if task != 'op_aco':
            raise ValueError('response3 currently implements OP only')
        from .op_response_behavior import OPResponseBehaviorEmbedder
        embedder = OPResponseBehaviorEmbedder(embedder)
    elif group != 'full':
        if task not in ('op_aco', 'mkp_aco', 'cvrp_aco'):
            raise ValueError(f'Feature subsets are unavailable for {task}')
        from importlib import import_module
        from .features import FeatureSubset
        names = import_module(f'bemrs.{task}_behavior').TREND_FEATURE_NAMES
        embedder = FeatureSubset(embedder, names, group)
    logging.info('[BehaviorFactory] task=%s feature_group=%s extractor=%s output_dim=%d dataset=%s', task, group, getattr(embedder, 'extractor_version', type(embedder).__name__), int(embedder.output_dim), getattr(embedder, 'dataset_path', 'n/a'))
    return embedder
_DEFAULT_BEHAVIOR_EMBEDDER = None

class DeterministicXGBoostRegressor:
    """XGBoost regression over complete raw behavior descriptors."""

    def __init__(self, n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=2.0, min_child_weight=2.0, tree_method='hist', n_jobs=1, random_state=42):
        self.parameters = {'objective': 'reg:squarederror', 'n_estimators': int(n_estimators), 'max_depth': int(max_depth), 'learning_rate': float(learning_rate), 'subsample': float(subsample), 'colsample_bytree': float(colsample_bytree), 'reg_alpha': float(reg_alpha), 'reg_lambda': float(reg_lambda), 'min_child_weight': float(min_child_weight), 'tree_method': str(tree_method), 'n_jobs': int(n_jobs), 'random_state': int(random_state), 'verbosity': 0}
        self.model = None

    def fit(self, X, y):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError('BeMRS XGBoost Direct requires xgboost>=2.0,<4. Install the project requirements before running it.') from exc
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array.')
        if len(X) != len(y):
            raise ValueError('X and y must have the same length.')
        if len(y) < 2:
            raise ValueError('Need at least 2 samples to fit DeterministicXGBoostRegressor.')
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError('XGBoost training data must be finite after imputation.')
        self.model = xgb.XGBRegressor(**self.parameters)
        self.model.fit(X, y)
        return self

    def predict(self, emb):
        if self.model is None:
            return (None, None)
        x = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        value = float(np.asarray(self.model.predict(x), dtype=np.float64)[0])
        return (value, 0.0)

class SurrogatePredictor:

    def __init__(self, min_samples: int=100, max_samples: int=200, emb_dim: int=16, embedder=None, store_codes: bool=True, random_state: int=42, surrogate_model: str='xgboost_direct', xgb_n_estimators: int=250, xgb_max_depth: int=3, xgb_learning_rate: float=0.03, xgb_subsample: float=0.8, xgb_colsample_bytree: float=0.8, xgb_reg_alpha: float=0.1, xgb_reg_lambda: float=2.0, xgb_min_child_weight: float=2.0, xgb_tree_method: str='hist', xgb_n_jobs: int=1):
        self.min_samples = int(min_samples)
        self.max_samples = int(max_samples)
        if embedder is None:
            embedder = TSPBehaviorEmbedder()
        self.embedder = embedder
        self.emb_dim = int(getattr(self.embedder, 'output_dim', emb_dim))
        self.store_codes = bool(store_codes)
        self.random_state = int(random_state)
        self.surrogate_model = 'xgboost_direct'
        self.xgb_parameters = {'n_estimators': int(xgb_n_estimators), 'max_depth': int(xgb_max_depth), 'learning_rate': float(xgb_learning_rate), 'subsample': float(xgb_subsample), 'colsample_bytree': float(xgb_colsample_bytree), 'reg_alpha': float(xgb_reg_alpha), 'reg_lambda': float(xgb_reg_lambda), 'min_child_weight': float(xgb_min_child_weight), 'tree_method': str(xgb_tree_method), 'n_jobs': int(xgb_n_jobs)}
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._codes: List[str] = []
        self._errors: List[float] = []
        self._surrogate: Optional[Any] = None
        self._feature_imputer = None
        self._feature_scaler = None
        self._feature_pca = None
        self._feature_output_scaler = None
        self._feature_output_dim = None
        self._predict_calls = 0
        self._train_calls = 0
        self._visual_map_scaler = None
        self._visual_map_pca = None
        self._visual_map_fit_samples = 0
        self._visual_map_feature_dim = None
        self._visual_map_bounds = None

    @staticmethod
    def _is_finite_number(x) -> bool:
        try:
            return bool(np.isfinite(float(x)))
        except Exception:
            return False

    def _extract_code_holder(self, candidate):
        """
        兼容 BeMRS 中候选可能是 dict 或 tuple/list 的情况。
        返回: original_candidate, dict_like_holder, code
        """
        if isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
            if isinstance(candidate[1], dict):
                holder = candidate[1]
            elif isinstance(candidate[0], dict):
                holder = candidate[0]
            else:
                holder = None
            orig = candidate
        elif isinstance(candidate, dict):
            holder = candidate
            orig = candidate
        else:
            holder = None
            orig = candidate
        code = holder.get('code') if isinstance(holder, dict) else None
        return (orig, holder, code)

    def add_sample(self, embedding: Union[np.ndarray, List[float]], objective: float, code: Optional[str]=None, reencode_code: bool=True):
        """添加训练样本。invalid / inf objective 会被跳过，避免污染 surrogate。"""
        if not self._is_finite_number(objective):
            logging.debug(f'[Predictor] Skip non-finite objective sample: {objective}')
            return
        if reencode_code and code is not None and (self.embedder is not None):
            try:
                embedding = self.embedder.encode_single(code)
            except Exception as e:
                logging.warning(f'[Predictor] Behavior feature extraction failed, fallback to provided feature: {e}')
        emb = np.array(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self.emb_dim:
            if not self._X:
                logging.info(f'[Predictor] Feature dim auto-set from {self.emb_dim} to {emb.shape[0]}.')
                self.emb_dim = int(emb.shape[0])
            else:
                logging.warning(f'[Predictor] Feature dim mismatch: expected {self.emb_dim}, got {emb.shape[0]}. Please keep one fixed feature extractor/dimension during a run.')
                return
        self._X.append(emb)
        self._y.append(float(objective))
        if self._surrogate is not None:
            pred_mu, _ = self.predict(emb)
            if pred_mu is not None and np.isfinite(pred_mu):
                err = np.log1p(abs(float(objective) - pred_mu))
            else:
                err = 1.0
        else:
            err = 1.0
        self._errors.append(float(err))
        if self.store_codes and code is not None:
            self._codes.append(code)

    def get_sample_count(self) -> int:
        return len(self._y)

    def is_ready(self) -> bool:
        return len(self._y) >= self.min_samples and self._surrogate is not None

    def _fit_behavior_preprocessor(self, X_train: np.ndarray) -> np.ndarray:
        X_train = np.asarray(X_train, dtype=np.float32)
        self._feature_imputer = SimpleImputer(strategy='median')
        train_i = self._feature_imputer.fit_transform(X_train)
        self._feature_scaler = None
        self._feature_pca = None
        self._feature_output_scaler = None
        self._feature_output_dim = int(train_i.shape[1])
        return np.asarray(train_i, dtype=np.float32)

    def _transform_behavior_features(self, X: Union[np.ndarray, List[float]]) -> np.ndarray:
        if self._feature_imputer is None:
            raise RuntimeError('Behavior feature preprocessor is not fitted.')
        X = np.asarray(X, dtype=np.float32)
        was_1d = X.ndim == 1
        if was_1d:
            X = X.reshape(1, -1)
        X_i = self._feature_imputer.transform(X)
        X_i = np.asarray(X_i, dtype=np.float32)
        return X_i[0] if was_1d else X_i

    def _build_training_set(self):
        n_total = len(self._y)
        if n_total == 0:
            return (None, None)
        X_all = np.asarray(self._X, dtype=np.float32)
        y_all = np.asarray(self._y, dtype=np.float32)
        finite_mask = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
        X_all = X_all[finite_mask]
        y_all = y_all[finite_mask]
        if len(y_all) == 0:
            return (None, None)
        if len(y_all) > self.max_samples:
            X_train = X_all[-self.max_samples:]
            y_train = y_all[-self.max_samples:]
            logging.info(f'[Predictor] XGBoost: Using latest {self.max_samples}/{len(y_all)} finite samples')
        else:
            X_train = X_all
            y_train = y_all
            logging.info(f'[Predictor] XGBoost: Using all {len(y_all)} finite samples')
        return (X_train, y_train)

    def train(self) -> bool:
        """Train or retrain the configured deterministic behavior surrogate."""
        n_total = len(self._y)
        if n_total < self.min_samples:
            logging.info(f'[Predictor] Not enough samples: {n_total}/{self.min_samples}')
            return False
        X_train_raw, y_train = self._build_training_set()
        if X_train_raw is None or len(y_train) < 2:
            logging.warning('[Predictor] Not enough finite samples to train surrogate.')
            self._surrogate = None
            return False
        X_train = self._fit_behavior_preprocessor(X_train_raw)
        self._surrogate = DeterministicXGBoostRegressor(random_state=self.random_state, **self.xgb_parameters)
        model_label = 'xgboost_direct'
        try:
            self._surrogate.fit(X_train, y_train)
            self._train_calls += 1
            logging.info('[Predictor] trained | model=%s samples=%s feature_dim=%s', model_label, len(y_train), X_train_raw.shape[1])
            return True
        except Exception as e:
            logging.error('[Predictor] %s training failed: %s', 'xgboost_direct', e)
            self._surrogate = None
            return False

    def predict(self, embedding: Union[np.ndarray, List[float]]) -> Tuple[Optional[float], Optional[float]]:
        if self._surrogate is None:
            return (None, None)
        try:
            feature = self._transform_behavior_features(embedding)
            mu, std = self._surrogate.predict(feature)
            self._predict_calls += 1
            return (mu, std)
        except Exception as e:
            logging.warning(f'[Predictor] Prediction failed: {e}')
            return (None, None)

    @staticmethod
    def _clean_code_for_visualization(code: str) -> str:
        code = str(code or '')
        code = re.sub('```(?:python)?', '', code)
        code = code.replace('```', '')
        return code

    @staticmethod
    def _ensure_2d_projection(X: np.ndarray, method: str='pca', random_state: int=42) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError('Need at least two feature vectors for visualization.')
        if X.shape[1] == 1:
            return np.column_stack([X[:, 0], np.zeros(X.shape[0], dtype=np.float32)])
        method = str(method or 'pca').lower()
        if method == 'tsne':
            from sklearn.manifold import TSNE
            perplexity = max(1, min(30, X.shape[0] - 1))
            return TSNE(n_components=2, random_state=random_state, perplexity=perplexity, init='pca', learning_rate='auto').fit_transform(X)
        return PCA(n_components=2, random_state=random_state).fit_transform(X)

    def _project_behavior_visual_map(self, reference_X: np.ndarray, extra_X: Optional[np.ndarray]=None, method: str='pca'):
        """Project behavior vectors without changing any search-side model state."""
        reference_X = np.asarray(reference_X, dtype=np.float64)
        if reference_X.ndim != 2 or len(reference_X) < 2:
            raise ValueError('Need at least two reference behavior vectors.')
        extra = None
        if extra_X is not None:
            extra = np.asarray(extra_X, dtype=np.float64)
            if extra.ndim != 2 or extra.shape[1] != reference_X.shape[1]:
                raise ValueError('Extra behavior vectors must match the reference dimension.')
        method = str(method or 'pca').lower()
        freeze_samples = max(3, int(os.environ.get('BEMRS_VISUAL_PCA_FREEZE_SAMPLES', '30')))
        metadata = {'method': method, 'freeze_samples': freeze_samples, 'reference_samples': int(len(reference_X)), 'frozen': False, 'phase': 'provisional'}
        if method != 'pca':
            combined = reference_X if extra is None else np.vstack([reference_X, extra])
            projected = self._ensure_2d_projection(combined, method=method, random_state=self.random_state)
            reference_2d = projected[:len(reference_X)]
            extra_2d = None if extra is None else projected[len(reference_X):]
            metadata['phase'] = 'dynamic_nonlinear'
            return (reference_2d, extra_2d, reference_X, None, metadata)
        feature_dim = int(reference_X.shape[1])
        can_freeze = feature_dim >= 2 and len(reference_X) >= freeze_samples
        if self._visual_map_pca is None and can_freeze:
            fit_X = reference_X[:freeze_samples]
            scaler = StandardScaler()
            fit_scaled = scaler.fit_transform(fit_X)
            pca = PCA(n_components=2, random_state=self.random_state)
            pca.fit(fit_scaled)
            self._visual_map_scaler = scaler
            self._visual_map_pca = pca
            self._visual_map_fit_samples = int(freeze_samples)
            self._visual_map_feature_dim = feature_dim
            logging.info('[BehaviorMap] Frozen visualization PCA | samples=%s raw_dim=%s variance=%.4f diagnostic_only=True', freeze_samples, feature_dim, float(np.sum(pca.explained_variance_ratio_)))
        frozen = self._visual_map_pca is not None and self._visual_map_scaler is not None and (self._visual_map_feature_dim == feature_dim)
        if frozen:
            standardized_space = self._visual_map_scaler.transform(reference_X)
            reference_2d = self._visual_map_pca.transform(standardized_space)
            extra_2d = None
            if extra is not None:
                extra_2d = self._visual_map_pca.transform(self._visual_map_scaler.transform(extra))
            components = np.asarray(self._visual_map_pca.components_, dtype=np.float64)
            metadata.update({'frozen': True, 'phase': 'fixed', 'fit_samples': int(self._visual_map_fit_samples), 'explained_variance': float(np.sum(self._visual_map_pca.explained_variance_ratio_))})
            return (reference_2d, extra_2d, standardized_space, components, metadata)
        fit_source = reference_X
        scaler = StandardScaler()
        standardized_space = scaler.fit_transform(fit_source)
        if feature_dim == 1:
            reference_2d = np.column_stack([standardized_space[:, 0], np.zeros(len(standardized_space))])
            components = np.asarray([[1.0], [0.0]], dtype=np.float64)
            extra_2d = None
            if extra is not None:
                extra_scaled = scaler.transform(extra)
                extra_2d = np.column_stack([extra_scaled[:, 0], np.zeros(len(extra_scaled))])
        else:
            pca = PCA(n_components=2, random_state=self.random_state)
            reference_2d = pca.fit_transform(standardized_space)
            components = np.asarray(pca.components_, dtype=np.float64)
            extra_2d = None
            if extra is not None:
                extra_2d = pca.transform(scaler.transform(extra))
            metadata['explained_variance'] = float(np.sum(pca.explained_variance_ratio_))
        return (reference_2d, extra_2d, standardized_space, components, metadata)

    def _save_feature_csv(self, X: np.ndarray, y: np.ndarray, save_dir: str, subdir: str, prefix: str, title: str, iteration=None, method: str='pca', recent_start_index: Optional[int]=None, fixed_behavior_map: bool=False, include_region_metadata: bool=False):
        """Save a compact 2-D projection for later offline rendering.

        The formal BeMRS release deliberately performs no online plotting.
        For behavior features, region membership comes from the live
        nearest-center partition rather than being reconstructed in 2-D.
        """
        try:
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32).reshape(-1)
            archive_indices = np.arange(len(y), dtype=np.int64)
            finite_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            X = X[finite_mask]
            y = y[finite_mask]
            archive_indices = archive_indices[finite_mask]
            if X.ndim != 2 or X.shape[0] < 2:
                print(f'[BehaviorCSV] Not enough finite samples for {title}.')
                return None
            if fixed_behavior_map:
                X2, _, _, _, projection_metadata = self._project_behavior_visual_map(X, method=method)
            else:
                X2 = self._ensure_2d_projection(X, method=method, random_state=self.random_state)
                projection_metadata = {'method': str(method).lower(), 'phase': 'independent', 'frozen': False, 'reference_samples': int(len(X))}
            best_so_far_indices = []
            incumbent = float('inf')
            for index, objective in enumerate(y):
                if float(objective) < incumbent:
                    incumbent = float(objective)
                    best_so_far_indices.append(index)
            best_so_far_indices = np.asarray(best_so_far_indices, dtype=np.int64)
            best_flags = np.zeros(len(y), dtype=np.int64)
            best_orders = np.zeros(len(y), dtype=np.int64)
            best_flags[best_so_far_indices] = 1
            best_orders[best_so_far_indices] = np.arange(1, len(best_so_far_indices) + 1, dtype=np.int64)
            recent_start = max(0, int(recent_start_index)) if recent_start_index is not None else len(finite_mask)
            recent_flags = (archive_indices >= recent_start).astype(np.int64)
            region_ids = np.zeros(len(y), dtype=np.int64)
            center_flags = np.zeros(len(y), dtype=np.int64)
            if include_region_metadata:
                snapshot = getattr(self, 'online_region_snapshot', None) or {}
                snapshot_region_ids = snapshot.get('archive_region_ids', [])
                if len(snapshot_region_ids) == len(finite_mask):
                    region_ids = np.asarray(snapshot_region_ids, dtype=np.int64)[finite_mask]
                for state in snapshot.get('regions', []):
                    center_archive_index = state.get('center_archive_index')
                    if center_archive_index is None:
                        continue
                    matches = np.flatnonzero(archive_indices == int(center_archive_index))
                    if len(matches):
                        center_flags[int(matches[0])] = 1
            output_dir = os.path.join(save_dir, subdir)
            os.makedirs(output_dir, exist_ok=True)
            suffix = f'evals{iteration}' if iteration is not None else time.strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(output_dir, f'{prefix}_{suffix}.csv')
            fieldnames = ['archive_index', 'component_1', 'component_2', 'objective', 'is_best_so_far', 'best_so_far_order', 'is_new_real_evaluation']
            if include_region_metadata:
                fieldnames.extend(['region_id', 'is_region_center'])
            with open(csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row_index, archive_index in enumerate(archive_indices):
                    row = {'archive_index': int(archive_index), 'component_1': float(X2[row_index, 0]), 'component_2': float(X2[row_index, 1]), 'objective': float(y[row_index]), 'is_best_so_far': int(best_flags[row_index]), 'best_so_far_order': int(best_orders[row_index]), 'is_new_real_evaluation': int(recent_flags[row_index])}
                    if include_region_metadata:
                        row['region_id'] = int(region_ids[row_index])
                        row['is_region_center'] = int(center_flags[row_index])
                    writer.writerow(row)
            logging.info('[BehaviorCSV] saved title=%s samples=%s best_updates=%s projection=%s phase=%s regions=%s csv=%s', title, len(y), len(best_so_far_indices), str(method).lower(), projection_metadata.get('phase'), include_region_metadata, csv_path)
            return csv_path
        except Exception as err:
            logging.warning('[BehaviorCSV] failed to save %s: %s', title, err)
            return None

    def visualize_behavior_features(self, save_dir: str, iteration=None, method: str='pca', include_landscape: bool=True, recent_start_index: Optional[int]=None):
        """Export deterministic behavior coordinates for offline rendering."""
        if len(self._X) < 2:
            print('[BehaviorCSV] Not enough behavior samples.')
            return None
        X = np.asarray(self._X, dtype=np.float32)
        y = np.asarray(self._y, dtype=np.float32)
        csv_path = self._save_feature_csv(X=X, y=y, save_dir=save_dir, subdir='behavior_csv', prefix='behavior_features', title='Behavior Features', iteration=iteration, method=method, recent_start_index=recent_start_index, fixed_behavior_map=True, include_region_metadata=True)
        return csv_path

    def visualize_candidate_selection_batch(self, candidates, save_dir: str, generation: int, operator: str, method: str='pca'):
        """Export one generated batch as CSV for offline rendering."""
        rows = []
        records = []
        feature_dim = None
        for candidate_index, candidate in enumerate(candidates):
            _, holder, code = self._extract_code_holder(candidate)
            if not isinstance(holder, dict):
                continue
            feature = holder.get('behavior_embedding', holder.get('embedding'))
            if feature is None:
                continue
            row = np.asarray(feature, dtype=np.float32).reshape(-1)
            if row.size == 0 or not np.all(np.isfinite(row)):
                continue
            if feature_dim is None:
                feature_dim = int(row.size)
            if row.size != feature_dim:
                continue
            objective = holder.get('objective')
            try:
                objective = float(objective)
            except Exception:
                objective = float('nan')
            attempted = bool(holder.get('real_eval_attempted', False))
            success = bool(holder.get('real_eval_success', False))
            evaluated = attempted and success and np.isfinite(objective)
            rows.append(row)
            records.append({'candidate_index': int(candidate_index), 'algorithm_id': holder.get('algorithm_id', ''), 'generated_order': holder.get('generated_order', ''), 'operator': holder.get('operator', operator), 'parent_algorithm_ids': ';'.join((str(parent_id) for parent_id in holder.get('parent_algorithm_ids', []))), 'code_hash': hashlib.sha1(self._clean_code_for_visualization(code or '').encode('utf-8', errors='ignore')).hexdigest()[:12], 'selected_for_evaluation': bool(holder.get('selected_for_evaluation', False)), 'real_eval_attempted': attempted, 'real_eval_success': success, 'evaluated': evaluated, 'objective': objective, 'status': holder.get('evaluation_status') or holder.get('lineage_status') or ''})
        if len(rows) < 2:
            logging.info('[CandidateVisualize] not enough valid behavior vectors for generation=%s operator=%s', generation, operator)
            return None
        try:
            matrix = np.asarray(rows, dtype=np.float32)
            history_matrix = np.asarray(self._X, dtype=np.float32)
            history_objectives = np.asarray(self._y, dtype=np.float64)
            history_projection = None
            projection_metadata = {'phase': 'candidate_batch_only', 'frozen': False, 'reference_samples': 0}
            if history_matrix.ndim == 2 and len(history_matrix) >= 2 and (history_matrix.shape[1] == matrix.shape[1]):
                history_mask = np.all(np.isfinite(history_matrix), axis=1) & np.isfinite(history_objectives)
                history_matrix = history_matrix[history_mask]
                history_objectives = history_objectives[history_mask]
            else:
                history_matrix = np.empty((0, matrix.shape[1]), dtype=np.float32)
                history_objectives = np.empty(0, dtype=np.float64)
            if len(history_matrix) >= 2:
                history_projection, projection, _, _, projection_metadata = self._project_behavior_visual_map(history_matrix, extra_X=matrix, method=method)
            else:
                projection = self._ensure_2d_projection(matrix, method=method, random_state=self.random_state)
            evaluated_mask = np.asarray([record['evaluated'] for record in records], dtype=bool)
            selected_mask = np.asarray([record['selected_for_evaluation'] for record in records], dtype=bool)
            objectives = np.asarray([record['objective'] for record in records], dtype=np.float64)
            attempted_mask = np.asarray([record['real_eval_attempted'] for record in records], dtype=bool)
            output_dir = os.path.join(save_dir, 'candidate_selection_csv')
            os.makedirs(output_dir, exist_ok=True)
            safe_operator = re.sub('[^A-Za-z0-9_.-]+', '_', str(operator))
            stem = f'behavior_candidates_gen{int(generation):04d}_{safe_operator}'
            csv_path = os.path.join(output_dir, f'{stem}.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as file:
                fieldnames = ['candidate_index', 'algorithm_id', 'generated_order', 'operator', 'parent_algorithm_ids', 'code_hash', 'component_1', 'component_2', 'selected_for_evaluation', 'real_eval_attempted', 'real_eval_success', 'evaluated', 'objective', 'status']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for record, point in zip(records, projection):
                    output_row = dict(record)
                    output_row['component_1'] = float(point[0])
                    output_row['component_2'] = float(point[1])
                    if not np.isfinite(output_row['objective']):
                        output_row['objective'] = ''
                    writer.writerow(output_row)
            logging.info('[CandidateCSV] saved generation=%s operator=%s valid=%s selected=%s evaluated=%s csv=%s', generation, operator, len(records), int(np.sum(selected_mask)), int(np.sum(evaluated_mask)), csv_path)
            return csv_path
            png_path = os.path.join(output_dir, f'{stem}.png')
            fig, ax = plt.subplots(figsize=(10, 7))
            from matplotlib.colors import Normalize
            finite_candidate_objectives = objectives[evaluated_mask & np.isfinite(objectives)]
            color_values = np.concatenate([history_objectives, finite_candidate_objectives])
            color_norm = None
            if len(color_values):
                color_min = float(np.min(color_values))
                color_max = float(np.max(color_values))
                if color_max <= color_min:
                    color_max = color_min + 1e-09
                color_norm = Normalize(vmin=color_min, vmax=color_max, clip=True)
            history_scatter = None
            if history_projection is not None and len(history_projection):
                history_scatter = ax.scatter(history_projection[:, 0], history_projection[:, 1], c=history_objectives, cmap=plt.cm.RdYlGn_r, norm=color_norm, alpha=0.35, s=25, label='Evaluated history', zorder=0)
            not_evaluated = ~attempted_mask
            ax.scatter(projection[not_evaluated, 0], projection[not_evaluated, 1], color='#b8bec8', alpha=0.7, s=42, label='Not evaluated', zorder=1)
            scatter = None
            if np.any(evaluated_mask):
                scatter = ax.scatter(projection[evaluated_mask, 0], projection[evaluated_mask, 1], c=objectives[evaluated_mask], cmap=plt.cm.RdYlGn_r, norm=color_norm, alpha=0.9, s=70, edgecolors='white', linewidths=0.6, label='Evaluated', zorder=3)
            invalid_attempt_mask = attempted_mask & ~evaluated_mask
            if np.any(invalid_attempt_mask):
                ax.scatter(projection[invalid_attempt_mask, 0], projection[invalid_attempt_mask, 1], color='#b42318', marker='x', s=72, linewidths=1.6, label='Evaluated but invalid', zorder=3)
            if np.any(selected_mask):
                ax.scatter(projection[selected_mask, 0], projection[selected_mask, 1], facecolors='none', edgecolors='purple', s=135, linewidths=1.6, label='Selected for evaluation', zorder=4)
            if projection_metadata.get('frozen', False):
                visible_points = projection
                if history_projection is not None:
                    visible_points = np.vstack([history_projection, projection])
                current_min = np.min(visible_points, axis=0)
                current_max = np.max(visible_points, axis=0)
                if self._visual_map_bounds is None:
                    self._visual_map_bounds = np.vstack([current_min, current_max])
                else:
                    self._visual_map_bounds[0] = np.minimum(self._visual_map_bounds[0], current_min)
                    self._visual_map_bounds[1] = np.maximum(self._visual_map_bounds[1], current_max)
                spans = np.maximum(self._visual_map_bounds[1] - self._visual_map_bounds[0], 1e-06)
                padding = 0.08 * spans
                ax.set_xlim(self._visual_map_bounds[0, 0] - padding[0], self._visual_map_bounds[1, 0] + padding[0])
                ax.set_ylim(self._visual_map_bounds[0, 1] - padding[1], self._visual_map_bounds[1, 1] + padding[1])
            map_label = 'fixed map' if projection_metadata.get('frozen', False) else 'warm-up map'
            ax.set_title(f'Behavior Candidate Decision Map (generation {generation}, operator {operator}, {map_label})')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.legend()
            colorbar_source = scatter if scatter is not None else history_scatter
            if colorbar_source is not None:
                fig.colorbar(colorbar_source, ax=ax, label='Objective Value (lower is better)')
            ax.text(0.015, 0.985, f'History: {len(history_objectives)} | Generated: {len(records)}\nSelected: {int(np.sum(selected_mask))} | Evaluated: {int(np.sum(evaluated_mask))}\nVisualization only; selection unchanged', transform=ax.transAxes, ha='left', va='top', fontsize=8, bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#90a4ae', 'alpha': 0.9}, zorder=6)
            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            with open(csv_path, 'w', newline='', encoding='utf-8') as file:
                fieldnames = ['candidate_index', 'algorithm_id', 'generated_order', 'operator', 'parent_algorithm_ids', 'code_hash', 'component_1', 'component_2', 'selected_for_evaluation', 'real_eval_attempted', 'real_eval_success', 'evaluated', 'objective', 'status']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for record, point in zip(records, projection):
                    row = dict(record)
                    row['component_1'] = float(point[0])
                    row['component_2'] = float(point[1])
                    if not np.isfinite(row['objective']):
                        row['objective'] = ''
                    writer.writerow(row)
            logging.info('[CandidateVisualize] saved generation=%s operator=%s valid=%s selected=%s evaluated=%s plot=%s', generation, operator, len(records), int(np.sum(selected_mask)), int(np.sum(evaluated_mask)), png_path)
            return png_path
        except Exception as err:
            logging.warning('[CandidateVisualize] failed generation=%s operator=%s: %s', generation, operator, err)
            return None
