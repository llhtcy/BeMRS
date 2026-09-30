"""Deterministic decision fingerprints for TSP guided local search.

The candidate returns a guide matrix.  GLS uses that matrix only through the
utility of the edges in its current tour::

    guide[u, v] / (1 + penalty[u, v])

This encoder therefore probes fixed locally-optimal tours under three fixed
penalty states.  It records sampled utility ranks, the edge selected for the
next penalty, its margin, and its relation to geometric edge length.  Small
raw-guide diagnostics are retained as well: positive global rescaling is
usually decision-equivalent, but the official evaluator casts the guide to
float32 and long trajectories can make near-identical scales diverge slightly.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

try:
    from .deterministic_behavior_rng import fixed_candidate_random_state
except ImportError:
    from bemrs.deterministic_behavior_rng import (
        fixed_candidate_random_state,
    )


FUNCTION_NAMES = ("heuristics_v2", "heuristics", "heuristics_v1", "heuristics_v3")
PENALTY_STATE_NAMES = ("zero", "patterned", "length_conditioned")
STATE_FEATURE_NAMES = tuple(
    [f"tour_utility_rank_{index:02d}" for index in range(16)]
    + ["selected_edge_sin", "selected_edge_cos", "selection_margin", "length_correlation"]
)
GUIDE_FEATURE_NAMES = ("log1p_max_abs", "positive_fraction", "asymmetry")


class TSPGLSBehaviorTimeout(TimeoutError):
    pass


def _strip_code_fence(code):
    code = str(code or "").strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", code, flags=re.IGNORECASE | re.DOTALL)
    if match:
        code = match.group(1)
    return code.strip()


@contextlib.contextmanager
def _hard_timeout(seconds):
    """Interrupt Python loops in candidate code and check elapsed NumPy calls."""
    seconds = float(seconds)
    if seconds <= 0.0:
        yield
        return

    deadline = time.monotonic() + seconds
    previous_trace = sys.gettrace()

    def trace_candidate(frame, event, arg):
        if (
            frame.f_code.co_filename == "<tsp_gls_candidate_heuristic>"
            and time.monotonic() >= deadline
        ):
            raise TSPGLSBehaviorTimeout(f"heuristic exceeded {seconds:g} seconds")
        return trace_candidate

    sys.settrace(trace_candidate)
    try:
        yield
    finally:
        sys.settrace(previous_trace)

    if time.monotonic() >= deadline:
        raise TSPGLSBehaviorTimeout(f"heuristic exceeded {seconds:g} seconds")


def _distance_matrix(positions):
    positions = np.asarray(positions, dtype=np.float64)
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    # Match the official TSP-GLS datasets/evaluator.
    distance += np.eye(len(distance), dtype=np.float64) * 1e-5
    return distance


def _percentile_ranks(values):
    """Stable average ranks scaled to [0, 1], without a SciPy dependency."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) / (len(values) - 1)
        start = end
    return ranks


def _safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


class TSPGLSBehaviorEmbedder:
    """Map TSP-GLS heuristic code to fixed penalty-decision features.

    The deployed default uses 16 selected TSP50 instances.  Each instance has
    3 penalty states x 20 decision features plus 3 raw-guide diagnostics, for
    63 features per instance and 1008 dimensions in total.
    """

    decision_features_per_state = len(STATE_FEATURE_NAMES)
    guide_features_per_instance = len(GUIDE_FEATURE_NAMES)

    def __init__(
        self,
        dataset_path: Union[str, os.PathLike],
        n_matrices: Optional[int] = None,
        n_nodes: Optional[int] = None,
        probes_per_state: Optional[int] = None,
        seed: Optional[int] = None,
        heuristic_timeout: Optional[float] = None,
        encode_timeout: Optional[float] = None,
    ):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f"TSP-GLS behavior dataset does not exist: {self.dataset_path}"
            )
        self.n_matrices = int(
            n_matrices
            if n_matrices is not None
            else os.environ.get("BEMRS_TSP_GLS_BEHAVIOR_MATRICES", 16)
        )
        self.n_nodes = int(
            n_nodes
            if n_nodes is not None
            else os.environ.get("BEMRS_TSP_GLS_BEHAVIOR_NODES", 50)
        )
        self.probes_per_state = int(
            probes_per_state
            if probes_per_state is not None
            else os.environ.get("BEMRS_TSP_GLS_PROBES_PER_STATE", 16)
        )
        self.seed = int(
            seed
            if seed is not None
            else os.environ.get(
                "BEMRS_TSP_GLS_BEHAVIOR_SEED",
                os.environ.get("BEMRS_BEHAVIOR_SEED", 42),
            )
        )
        self.heuristic_timeout = float(
            heuristic_timeout
            if heuristic_timeout is not None
            else os.environ.get("BEMRS_TSP_GLS_HEURISTIC_TIMEOUT", 0.25)
        )
        self.encode_timeout = float(
            encode_timeout
            if encode_timeout is not None
            else os.environ.get("BEMRS_TSP_GLS_ENCODE_TIMEOUT", 5.0)
        )
        self.batch_size = int(os.environ.get("BEMRS_BEHAVIOR_BATCH_SIZE", 16))
        self._validate_configuration()

        self._distances = self._load_distances()
        gls = self._load_official_gls()
        self._state_banks = tuple(
            self._make_fixed_states(gls, distance) for distance in self._distances
        )
        self.features_per_instance = (
            len(PENALTY_STATE_NAMES) * self.probes_per_state
            + len(PENALTY_STATE_NAMES) * 4
            + self.guide_features_per_instance
        )
        self.output_dim = self.n_matrices * self.features_per_instance
        self.extractor_version = "tsp_gls_penalty_rank63_v2"
        self.extractor_description = (
            "Exact open-tour GLS penalty decisions on 16 selected TSP50 probes, "
            "including the evaluator's positive-utility threshold, with float32 "
            "guide scale, support, and asymmetry diagnostics."
        )
        logging.info(
            "[BehaviorEmbedder] Ready | task=tsp_gls extractor=%s dataset=%s "
            "matrices=%d nodes=%d states_per_matrix=%d probes_per_state=%d "
            "features_per_matrix=%d output_dim=%d heuristic_timeout=%.3fs "
            "encode_timeout=%.3fs",
            self.extractor_version,
            self.dataset_path,
            self.n_matrices,
            self.n_nodes,
            len(PENALTY_STATE_NAMES),
            self.probes_per_state,
            self.features_per_instance,
            self.output_dim,
            self.heuristic_timeout,
            self.encode_timeout,
        )

    def _validate_configuration(self):
        if self.n_matrices <= 0:
            raise ValueError("TSP-GLS behavior matrices must be positive.")
        if self.n_nodes < 4:
            raise ValueError("TSP-GLS behavior node count must be at least four.")
        if self.probes_per_state <= 0:
            raise ValueError("TSP-GLS probes per state must be positive.")

    def _load_distances(self):
        data = np.load(self.dataset_path, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            try:
                key = "positions" if "positions" in data.files else data.files[0]
                values = np.asarray(data[key], dtype=np.float64)
            finally:
                data.close()
        else:
            values = np.asarray(data, dtype=np.float64)

        if values.ndim != 3:
            raise ValueError(
                "TSP-GLS behavior dataset must be (instances,nodes,2) coordinates "
                f"or (instances,nodes,nodes) distances, got {values.shape}."
            )
        if self.n_matrices > len(values):
            raise ValueError(
                f"Requested {self.n_matrices} probes from a dataset of {len(values)}."
            )
        if self.n_nodes > values.shape[1]:
            raise ValueError(
                f"Requested {self.n_nodes} nodes from instances with {values.shape[1]}."
            )

        subset = values[: self.n_matrices, : self.n_nodes]
        if subset.shape[2] == 2:
            distances = [_distance_matrix(instance) for instance in subset]
        elif values.shape[1] == values.shape[2]:
            distances = [
                np.asarray(instance[: self.n_nodes, : self.n_nodes], dtype=np.float64).copy()
                for instance in values[: self.n_matrices]
            ]
        else:
            raise ValueError(
                "TSP-GLS behavior dataset's final dimension must be 2 or its node count."
            )
        if any(
            distance.shape != (self.n_nodes, self.n_nodes)
            or not np.all(np.isfinite(distance))
            for distance in distances
        ):
            raise ValueError("TSP-GLS behavior dataset contains invalid distances.")
        return tuple(distances)

    def _load_official_gls(self):
        # The deployed dataset lives in problems/tsp_gls/dataset.  Importing by
        # the real top-level name is required for Numba's on-disk cache.
        gls_path = self.dataset_path.parent.parent / "gls.py"
        if not gls_path.is_file():
            raise FileNotFoundError(
                f"Could not locate official TSP-GLS implementation: {gls_path}"
            )
        existing = sys.modules.get("gls")
        if existing is not None:
            existing_path = Path(getattr(existing, "__file__", "")).resolve()
            if existing_path != gls_path.resolve():
                raise RuntimeError(
                    f"A different top-level gls module is already loaded: {existing_path}"
                )
            return existing
        gls_dir = str(gls_path.parent)
        if gls_dir not in sys.path:
            sys.path.insert(0, gls_dir)
        return importlib.import_module("gls")

    @staticmethod
    def _make_fixed_states(gls, distance):
        work = np.asarray(distance, dtype=np.float32)
        tour = gls._init_nearest_neighbor(work, np.uint16(0))
        gls._local_search(work, tour, np.uint16(0), np.uint16(1000))
        tour = np.asarray(tour, dtype=np.int64)
        # _perturbation() intentionally examines only consecutive entries
        # 0..n-2; it never includes the closing edge tour[-1] -> tour[0].
        current = tour[:-1]
        following = tour[1:]
        edge_count = len(current)

        zero = np.zeros_like(work)

        patterned = np.zeros_like(work)
        patterned_values = (np.arange(edge_count) % 4 == 0).astype(np.float32)
        patterned_values += 2.0 * (np.arange(edge_count) % 11 == 0)
        patterned[current, following] = patterned_values
        patterned[following, current] = patterned_values

        length_conditioned = np.zeros_like(work)
        edge_lengths = work[current, following]
        order = np.argsort(edge_lengths, kind="mergesort")
        length_values = np.zeros(edge_count, dtype=np.float32)
        length_values[order[-max(1, edge_count // 3) :]] = 1.0
        length_values[order[-max(1, edge_count // 10) :]] = 3.0
        length_conditioned[current, following] = length_values
        length_conditioned[following, current] = length_values
        return tour, (zero, patterned, length_conditioned)

    @staticmethod
    def _compile_heuristic(code):
        source = _strip_code_fence(code)
        tree = ast.parse(source)
        namespace = {
            "np": np,
            "numpy": np,
            "math": math,
            "__builtins__": __builtins__,
        }
        exec(
            compile(tree, "<tsp_gls_candidate_heuristic>", "exec"),
            namespace,
            namespace,
        )
        for name in FUNCTION_NAMES:
            function = namespace.get(name)
            if callable(function):
                return function
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = namespace.get(node.name)
                if callable(function):
                    return function
        raise ValueError(
            f"No callable TSP-GLS heuristic found; expected one of {FUNCTION_NAMES}."
        )

    def _call_heuristic(self, function, distance, instance_index):
        # A shared deterministic stream is intentional: stochastic candidates
        # must see the same probes regardless of their source text.
        probe_seed = self.seed + 1009 * int(instance_index)
        with fixed_candidate_random_state("tsp_gls_shared_probe_stream", probe_seed):
            with _hard_timeout(self.heuristic_timeout):
                raw = function(distance.copy())
        raw = np.asarray(raw)
        expected_shape = (self.n_nodes, self.n_nodes)
        if raw.shape != expected_shape:
            raise ValueError(
                f"TSP-GLS guide shape {raw.shape} does not match {expected_shape}."
            )
        if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
            raise ValueError("TSP-GLS guide contains non-numeric, NaN, or infinite values.")
        # Match guided_local_search() exactly before deriving decisions.
        guide = np.asarray(raw, dtype=np.float32)
        if not np.all(np.isfinite(guide)):
            raise ValueError("TSP-GLS guide overflows when cast to evaluator float32.")
        return guide

    @staticmethod
    def _guide_features(guide):
        guide64 = np.asarray(guide, dtype=np.float64)
        mean_abs = float(np.mean(np.abs(guide64)))
        return np.asarray(
            [
                np.log1p(float(np.max(np.abs(guide64)))),
                float(np.mean(guide64 > 0.0)),
                float(np.mean(np.abs(guide64 - guide64.T))) / (mean_abs + 1e-12),
            ],
            dtype=np.float32,
        )

    def _state_features(self, guide, distance, tour, penalty):
        # Mirror gls._perturbation exactly: the closing edge is excluded;
        # max_util starts at zero; and ties/non-positive values do not replace
        # the initial edge index zero.
        current = tour[:-1]
        following = tour[1:]
        utility = guide[current, following] / (1.0 + penalty[current, following])
        utility64 = np.asarray(utility, dtype=np.float64)
        scale = float(np.max(np.abs(utility64)))
        normalized = utility64 / scale if scale > 0.0 else utility64.copy()
        ranks = _percentile_ranks(normalized)
        positions = np.linspace(
            0, len(normalized) - 1, self.probes_per_state, dtype=np.int64
        )
        maximum = float(np.max(normalized))
        if maximum > 0.0:
            selected_index = int(np.argmax(normalized))
            if len(normalized) >= 2:
                competitors = np.delete(normalized, selected_index)
                runner_up = max(0.0, float(np.max(competitors)))
            else:
                runner_up = 0.0
            margin = maximum - runner_up
        else:
            selected_index = 0
            # Distance to the evaluator's strict positive-utility threshold.
            margin = max(0.0, -maximum)
        edge_lengths = distance[current, following]
        return np.asarray(
            [
                *ranks[positions].tolist(),
                math.sin(2.0 * math.pi * selected_index / len(normalized)),
                math.cos(2.0 * math.pi * selected_index / len(normalized)),
                margin,
                _safe_correlation(normalized, edge_lengths),
            ],
            dtype=np.float32,
        )

    def _fallback_feature(self):
        return np.full(self.output_dim, -1.0, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        started = time.monotonic()
        try:
            function = self._compile_heuristic(text)
            instance_parts = []
            for instance_index, (distance, state_bank) in enumerate(
                zip(self._distances, self._state_banks)
            ):
                if self.encode_timeout > 0.0 and time.monotonic() - started > self.encode_timeout:
                    raise TSPGLSBehaviorTimeout("TSP-GLS behavior encoding timeout")
                guide = self._call_heuristic(function, distance, instance_index)
                tour, penalties = state_bank
                state_parts = [
                    self._state_features(guide, distance, tour, penalty)
                    for penalty in penalties
                ]
                instance_parts.append(
                    np.concatenate([*state_parts, self._guide_features(guide)])
                )
            feature = np.concatenate(instance_parts).astype(np.float32)
            if feature.shape != (self.output_dim,) or not np.all(np.isfinite(feature)):
                raise ValueError(
                    f"Invalid TSP-GLS behavior feature shape or values: {feature.shape}."
                )
            return feature
        except TSPGLSBehaviorTimeout as error:
            logging.info(
                "[BehaviorEmbedder] TSP-GLS timeout, using fallback feature: %s", error
            )
            return self._fallback_feature()
        except Exception as error:
            logging.warning(
                "[BehaviorEmbedder] TSP-GLS encode failed, using fallback feature: %s: %s",
                type(error).__name__,
                error,
            )
            return self._fallback_feature()

    def encode(self, texts: Union[str, List[str]], batch_size: int = None) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.output_dim), dtype=np.float32)
        return np.vstack([self.encode_single(text) for text in texts]).astype(np.float32)
