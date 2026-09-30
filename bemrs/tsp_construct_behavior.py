"""Task-specific online behavior encoder for constructive TSP heuristics."""

import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    from .deterministic_behavior_rng import fixed_candidate_random_state
except ImportError:
    from bemrs.deterministic_behavior_rng import (
        fixed_candidate_random_state,
    )

try:
    from .behavior_feature_runtime import (
        BehaviorProbeTimeout,
        _call_selector_with_timeout,
    )
    from .tsp_constructive_feature_library import (
        STAGE_TREND_FEATURE_NAMES,
        decision_features,
        feature_vector,
        stage_trend_features,
    )
except ImportError:
    from behavior_feature_runtime import (
        BehaviorProbeTimeout,
        _call_selector_with_timeout,
    )
    from tsp_constructive_feature_library import (
        STAGE_TREND_FEATURE_NAMES,
        decision_features,
        feature_vector,
        stage_trend_features,
    )


class TSPConstructiveBehaviorEmbedder:
    """Encode TSP policies from aligned fixed early, middle, and late states."""

    def __init__(
        self,
        n_matrices: Optional[int] = None,
        n_nodes: Optional[int] = None,
        n_probes: Optional[int] = None,
        rollout_steps: Optional[int] = None,
        seed: Optional[int] = None,
        dataset_path: Optional[Union[str, os.PathLike]] = None,
        stage_ratios: Optional[Tuple[float, ...]] = None,
    ):
        configured_n_matrices = (
            n_matrices
            if n_matrices is not None
            else os.environ.get(
                "BEMRS_TSP_BEHAVIOR_MATRICES",
                os.environ.get("BEMRS_BEHAVIOR_MATRICES", 16),
            )
        )
        self.n_matrices = (
            int(configured_n_matrices)
            if configured_n_matrices is not None
            else None
        )
        self.n_nodes = int(
            n_nodes
            if n_nodes is not None
            else os.environ.get(
                "BEMRS_TSP_BEHAVIOR_NODES",
                os.environ.get("BEMRS_BEHAVIOR_NODES", 50),
            )
        )
        self.n_probes = int(
            n_probes
            if n_probes is not None
            else os.environ.get(
                "BEMRS_TSP_BEHAVIOR_PROBES",
                os.environ.get("BEMRS_BEHAVIOR_PROBES", 8),
            )
        )
        # Retained only for constructor compatibility with older trajectory-based
        # configurations. Fixed stage states do not perform candidate rollouts.
        self.rollout_steps = rollout_steps
        configured_stage_ratios = (
            stage_ratios
            if stage_ratios is not None
            else os.environ.get("BEMRS_TSP_BEHAVIOR_STAGE_RATIOS", "0.8,0.5,0.2")
        )
        if isinstance(configured_stage_ratios, str):
            configured_stage_ratios = tuple(
                float(value.strip())
                for value in configured_stage_ratios.split(",")
                if value.strip()
            )
        self.stage_ratios = tuple(
            float(value) for value in configured_stage_ratios
        )
        self.seed = int(
            seed
            if seed is not None
            else os.environ.get(
                "BEMRS_TSP_BEHAVIOR_SEED",
                os.environ.get("BEMRS_BEHAVIOR_SEED", 42),
            )
        )
        self.batch_size = int(os.environ.get("BEMRS_BEHAVIOR_BATCH_SIZE", 32))
        self.probe_timeout = float(
            os.environ.get(
                "BEMRS_TSP_BEHAVIOR_PROBE_TIMEOUT",
                os.environ.get("BEMRS_BEHAVIOR_PROBE_TIMEOUT", 0.05),
            )
        )
        self.encode_timeout = float(
            os.environ.get(
                "BEMRS_TSP_BEHAVIOR_ENCODE_TIMEOUT",
                os.environ.get("BEMRS_BEHAVIOR_ENCODE_TIMEOUT", 2.0),
            )
        )
        self._validate_configuration()
        configured_dataset = (
            dataset_path
            if dataset_path is not None
            else os.environ.get("BEMRS_TSP_BEHAVIOR_DATASET")
        )
        self.dataset_path = self._resolve_training_dataset_path(configured_dataset)
        self._distance_matrices = self._make_distance_matrices()
        self.n_matrices = len(self._distance_matrices)
        self._fixed_stage_banks = [
            self._make_fixed_stage_states(
                matrix,
                self.n_probes,
                self.stage_ratios,
                self.seed + matrix_index * 1009 + 104729,
            )
            for matrix_index, matrix in enumerate(self._distance_matrices)
        ]
        self.output_dim = self.n_matrices * len(STAGE_TREND_FEATURE_NAMES)
        self.extractor_version = "tsp_constructive_stage_trend12_v1"
        self.extractor_description = (
            f"Twelve policy-trend summaries from {len(self.stage_ratios)} fixed "
            f"stages with {self.n_probes} aligned states per stage."
        )
        logging.info(
            "[BehaviorEmbedder] Ready | task=tsp_constructive extractor=%s "
            "dataset=%s matrices=%d nodes=%d probes_per_stage=%d stage_ratios=%s "
            "output_dim=%d probe_timeout=%.3fs encode_timeout=%.3fs",
            self.extractor_version,
            self.dataset_path,
            self.n_matrices,
            self.n_nodes,
            self.n_probes,
            ",".join(f"{value:g}" for value in self.stage_ratios),
            self.output_dim,
            self.probe_timeout,
            self.encode_timeout,
        )

    def _validate_configuration(self):
        if self.n_matrices is not None and self.n_matrices < 1:
            raise ValueError("BEMRS_TSP_BEHAVIOR_MATRICES must be positive.")
        if self.n_probes < 1:
            raise ValueError("BEMRS_TSP_BEHAVIOR_PROBES must be positive.")
        if self.n_nodes < 4:
            raise ValueError("BEMRS_TSP_BEHAVIOR_NODES must be at least 4.")
        if len(self.stage_ratios) < 2:
            raise ValueError(
                "BEMRS_TSP_BEHAVIOR_STAGE_RATIOS requires at least two stages."
            )
        if any(not 0.0 < ratio <= 1.0 for ratio in self.stage_ratios):
            raise ValueError(
                "Every BEMRS_TSP_BEHAVIOR_STAGE_RATIOS value must be in (0, 1]."
            )
        if any(
            earlier <= later
            for earlier, later in zip(self.stage_ratios, self.stage_ratios[1:])
        ):
            raise ValueError(
                "BEMRS_TSP_BEHAVIOR_STAGE_RATIOS must be strictly descending "
                "from early to late."
            )

    @staticmethod
    def _clean_code(text):
        text = re.sub(r"```(?:python)?", "", str(text or ""))
        return text.replace("```", "").strip()

    def _resolve_training_dataset_path(self, configured_path=None):
        if configured_path:
            path = Path(configured_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"TSP behavior dataset does not exist: {path}")
            return path

        filename = f"train{self.n_nodes}_dataset.npy"
        module_path = Path(__file__).resolve()
        candidates = [
            Path.cwd() / "problems" / "tsp_constructive" / "dataset" / filename,
            module_path.parent.parent.parent
            / "problems"
            / "tsp_constructive"
            / "dataset"
            / filename,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        checked = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Could not locate the TSP behavior dataset. Set "
            "BEMRS_TSP_BEHAVIOR_DATASET. Checked:\n  - " + checked
        )

    def _make_distance_matrices(self):
        dataset = np.load(self.dataset_path, allow_pickle=False)
        if dataset.ndim == 2:
            dataset = dataset[None, ...]
        if dataset.ndim != 3:
            raise ValueError(
                "TSP dataset must have shape (instances, nodes, dimensions) or "
                f"(instances, nodes, nodes), got {dataset.shape}."
            )
        matrix_count = len(dataset) if self.n_matrices is None else self.n_matrices
        if matrix_count > len(dataset):
            raise ValueError(
                f"Requested {matrix_count} TSP instances from a dataset of {len(dataset)}."
            )

        source_is_distance_matrix = dataset.shape[1] == dataset.shape[2]
        matrices = []
        for raw in dataset[:matrix_count]:
            instance = np.asarray(raw, dtype=np.float64)
            if source_is_distance_matrix:
                matrix = instance.copy()
            else:
                delta = instance[:, None, :] - instance[None, :, :]
                matrix = np.sqrt(np.sum(delta * delta, axis=-1))
            if matrix.shape != (self.n_nodes, self.n_nodes):
                raise ValueError(
                    f"TSP matrix shape {matrix.shape} does not match "
                    f"BEMRS_TSP_BEHAVIOR_NODES={self.n_nodes}."
                )
            np.fill_diagonal(matrix, 0.0)
            matrices.append(matrix)
        return matrices

    @staticmethod
    def _make_fixed_stage_states(matrix, probes_per_stage, stage_ratios, seed):
        """Build independent aligned states without candidate-induced transitions."""
        matrix = np.asarray(matrix, dtype=np.float64)
        n_nodes = int(matrix.shape[0])
        rng = np.random.default_rng(seed)
        stages = []
        for ratio in stage_ratios:
            states = []
            for _ in range(int(probes_per_stage)):
                current = int(rng.integers(0, n_nodes))
                destination_pool = [
                    node for node in range(n_nodes) if node != current
                ]
                destination = int(rng.choice(destination_pool))
                available = np.asarray(
                    [
                        node
                        for node in range(n_nodes)
                        if node not in {current, destination}
                    ],
                    dtype=int,
                )
                count = int(round(float(ratio) * len(available)))
                count = min(max(count, 2), len(available))
                unvisited = set(
                    int(node)
                    for node in rng.choice(available, size=count, replace=False)
                )
                states.append((current, destination, unvisited))
            stages.append(tuple(states))
        return tuple(stages)

    def _compile_selector(self, code):
        namespace = {
            "np": np,
            "numpy": np,
            "math": math,
            "random": random,
            "List": List,
            "Optional": Optional,
            "Tuple": Tuple,
            "__builtins__": __builtins__,
        }
        exec(self._clean_code(code), namespace)
        for name in (
            "select_next_node_v2",
            "select_next_node_v1",
            "select_next_node",
        ):
            function = namespace.get(name)
            if callable(function):
                return function
        raise ValueError("No callable TSP select_next_node function found.")

    def _fallback_feature(self):
        return np.full(self.output_dim, -1.0, dtype=np.float32)

    def encode_single(self, text):
        started = time.monotonic()
        try:
            with fixed_candidate_random_state(text, self.seed):
                function = self._compile_selector(text)
                parts = []
                failures = 0
                total_states = 0
                for matrix, stage_bank in zip(
                    self._distance_matrices, self._fixed_stage_banks
                ):
                    stage_records = []
                    for states in stage_bank:
                        stage_rows = []
                        for current, destination, initial_unvisited in states:
                            total_states += 1
                            if (
                                self.encode_timeout > 0
                                and time.monotonic() - started > self.encode_timeout
                            ):
                                raise BehaviorProbeTimeout(
                                    "TSP behavior feature encoding timeout"
                                )
                            unvisited = set(int(node) for node in initial_unvisited)
                            try:
                                chosen = int(
                                    _call_selector_with_timeout(
                                        function,
                                        current,
                                        destination,
                                        unvisited,
                                        matrix,
                                        self.probe_timeout,
                                    )
                                )
                                if chosen not in unvisited:
                                    raise ValueError(
                                        "Selector returned a visited or invalid node."
                                    )
                                stage_rows.append(
                                    decision_features(
                                        current,
                                        destination,
                                        unvisited,
                                        matrix,
                                        chosen,
                                    )
                                )
                            except BehaviorProbeTimeout:
                                failures += 1
                                stage_rows.append(None)
                            except Exception:
                                failures += 1
                                stage_rows.append(None)
                        stage_records.append(stage_rows)
                    parts.append(
                        feature_vector(
                            stage_trend_features(stage_records),
                            STAGE_TREND_FEATURE_NAMES,
                        )
                    )

            feature = np.concatenate(parts).astype(np.float32)
            if feature.shape != (self.output_dim,):
                raise ValueError(
                    f"TSP feature shape {feature.shape} does not match {(self.output_dim,)}."
                )
            feature = np.nan_to_num(
                feature, nan=-1.0, posinf=1e6, neginf=-1e6
            ).astype(np.float32)
            if failures:
                logging.debug(
                    "[BehaviorEmbedder] %d/%d TSP fixed stage states failed.",
                    failures,
                    total_states,
                )
            return feature
        except BehaviorProbeTimeout as error:
            logging.info("[BehaviorEmbedder] TSP timeout, using fallback: %s", error)
            return self._fallback_feature()
        except Exception as error:
            logging.debug("[BehaviorEmbedder] TSP encode failed: %s", error)
            return self._fallback_feature()

    def encode(self, texts: Union[str, List[str]], batch_size: int = None):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.output_dim), dtype=np.float32)
        return np.vstack([self.encode_single(text) for text in texts]).astype(np.float32)

    def finetune(self, *args, **kwargs):
        logging.info(
            "[BehaviorEmbedder] finetune skipped: TSP stage-trend features are deterministic."
        )


TSPBehaviorEmbedder = TSPConstructiveBehaviorEmbedder
