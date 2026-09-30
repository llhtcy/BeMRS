"""Deterministic OP-ACO policy trend features for surrogate modelling."""

from __future__ import annotations

import ast
import contextlib
import logging
import math
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

try:
    from .deterministic_behavior_rng import fixed_candidate_random_state
    from .behavior_instance_pooling import (
        aggregate_instance_features,
        instance_pooled_output_dim,
        resolve_instance_pooling,
    )
except ImportError:
    from bemrs.deterministic_behavior_rng import (
        fixed_candidate_random_state,
    )
    from bemrs.behavior_instance_pooling import (
        aggregate_instance_features,
        instance_pooled_output_dim,
        resolve_instance_pooling,
    )


TREND_FEATURE_NAMES = (
    "prize_rank_mean",
    "prize_rank_stage_slope",
    "edge_distance_rank_mean",
    "edge_distance_rank_stage_slope",
    "prize_detour_rank_mean",
    "prize_detour_rank_stage_slope",
    "remaining_slack_mean",
    "remaining_slack_stage_slope",
    "feasible_prior_mass_mean",
    "feasible_prior_mass_stage_slope",
    "entropy_mean",
    "entropy_stage_slope",
    "heuristic_asymmetry",
    "zero_support_ratio",
)
FUNCTION_NAMES = ("heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3")
REFERENCE_MODES = (
    "nearest",
    "prize_edge",
    "prize_detour",
    "low_detour_random",
)


class OPBehaviorTimeout(TimeoutError):
    pass


def _parse_ratios(value, default):
    if value is None:
        return tuple(float(item) for item in default)
    if isinstance(value, str):
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        values = [float(item) for item in value]
    if not values:
        raise ValueError("At least one OP route-stage ratio is required.")
    return tuple(values)


@contextlib.contextmanager
def _hard_timeout(seconds):
    """Bound candidate runtime while keeping the common path low-overhead.

    OP heuristics are often written with Python loops.  Applying ``sys.settrace``
    to every candidate line made those otherwise valid heuristics several times
    slower during feature extraction.  On Unix's main thread, an interval timer
    provides the same protection without per-line tracing.  The trace-based
    implementation remains as a portable fallback (and for worker threads).
    """
    seconds = float(seconds)
    if seconds <= 0.0:
        yield
        return

    deadline = time.monotonic() + seconds

    can_use_interval_timer = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and hasattr(signal, "ITIMER_REAL")
    )
    if can_use_interval_timer:
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        # Do not interfere with a timer owned by the surrounding application.
        if previous_timer[0] <= 0.0:
            previous_handler = signal.getsignal(signal.SIGALRM)

            def raise_timeout(signum, frame):
                raise OPBehaviorTimeout(
                    f"heuristic exceeded {seconds:g} seconds"
                )

            signal.signal(signal.SIGALRM, raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                yield
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)

            if time.monotonic() >= deadline:
                raise OPBehaviorTimeout(
                    f"heuristic exceeded {seconds:g} seconds"
                )
            return

    previous_trace = sys.gettrace()

    def trace_candidate(frame, event, arg):
        if (
            frame.f_code.co_filename == "<op_candidate_heuristic>"
            and time.monotonic() >= deadline
        ):
            raise OPBehaviorTimeout(f"heuristic exceeded {seconds:g} seconds")
        return trace_candidate

    sys.settrace(trace_candidate)
    try:
        yield
    finally:
        sys.settrace(previous_trace)

    if time.monotonic() >= deadline:
        raise OPBehaviorTimeout(f"heuristic exceeded {seconds:g} seconds")


def _strip_code_fence(code):
    code = str(code or "").strip()
    match = re.search(
        r"```(?:python)?\s*(.*?)```",
        code,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        code = match.group(1)
    return code.strip()


def _distribution(weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(len(weights), 1.0 / max(len(weights), 1), dtype=np.float64)
    return weights / total


def _percentile_ranks(values):
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


def _stage_slope(values, states):
    values = np.asarray(values, dtype=np.float64)
    stages = np.asarray([state["stage_ratio"] for state in states], dtype=np.float64)
    stage_values = []
    means = []
    for stage in np.unique(stages):
        selected = values[np.isclose(stages, stage)]
        if len(selected):
            stage_values.append(float(stage))
            means.append(float(np.mean(selected)))
    if len(means) < 2:
        return 0.0
    x = np.asarray(stage_values, dtype=np.float64)
    y = np.asarray(means, dtype=np.float64)
    x -= np.mean(x)
    denominator = float(x @ x)
    if denominator <= 0.0:
        return 0.0
    return float((x @ (y - np.mean(y))) / denominator)


def _get_max_len(n_nodes):
    for threshold, max_len in zip((50, 100, 200, 300), (3.0, 4.0, 5.0, 6.0)):
        if int(n_nodes) <= threshold:
            return float(max_len)
    return 7.0


class OPACOBehaviorEmbedder:
    """Map OP edge-heuristic code to 14 route-policy trends per instance."""

    def __init__(
        self,
        dataset_path: Union[str, os.PathLike],
        n_matrices: Optional[int] = None,
        n_nodes: Optional[int] = None,
        stage_ratios=None,
        probes_per_stage: Optional[int] = None,
        seed: Optional[int] = None,
        heuristic_timeout: Optional[float] = None,
        encode_timeout: Optional[float] = None,
        instance_pooling: Optional[str] = None,
    ):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f"OP behavior dataset does not exist: {self.dataset_path}"
            )

        self.n_matrices = int(
            n_matrices
            if n_matrices is not None
            else os.environ.get("BEMRS_OP_BEHAVIOR_MATRICES", 5)
        )
        self.n_nodes = int(
            n_nodes
            if n_nodes is not None
            else os.environ.get("BEMRS_OP_BEHAVIOR_NODES", 50)
        )
        self.stage_ratios = _parse_ratios(
            stage_ratios
            if stage_ratios is not None
            else os.environ.get("BEMRS_OP_STAGE_RATIOS"),
            (0.2, 0.5, 0.8),
        )
        self.probes_per_stage = int(
            probes_per_stage
            if probes_per_stage is not None
            else os.environ.get("BEMRS_OP_PROBES_PER_STAGE", 4)
        )
        self.seed = int(
            seed
            if seed is not None
            else os.environ.get(
                "BEMRS_OP_BEHAVIOR_SEED",
                os.environ.get("BEMRS_BEHAVIOR_SEED", 42),
            )
        )
        self.heuristic_timeout = float(
            heuristic_timeout
            if heuristic_timeout is not None
            else os.environ.get("BEMRS_OP_HEURISTIC_TIMEOUT", 0.5)
        )
        self.encode_timeout = float(
            encode_timeout
            if encode_timeout is not None
            else os.environ.get("BEMRS_OP_ENCODE_TIMEOUT", 5.0)
        )
        self.batch_size = int(os.environ.get("BEMRS_BEHAVIOR_BATCH_SIZE", 32))
        self.instance_pooling = resolve_instance_pooling(instance_pooling)
        self._validate_configuration()

        self._instances = self._load_instances()
        self._state_banks = [
            self._make_fixed_states(instance, self.seed + index * 1009)
            for index, instance in enumerate(self._instances)
        ]
        self.output_dim = instance_pooled_output_dim(
            self.n_matrices,
            len(TREND_FEATURE_NAMES),
            self.instance_pooling,
        )
        self.extractor_version = (
            "op_aco_policy_trend14_instance_mean_v2"
            if self.instance_pooling == "mean"
            else "op_aco_policy_trend14_v1"
        )
        self.extractor_description = (
            "Fourteen prize, edge-cost, detour-efficiency, remaining-budget, "
            "feasibility, entropy, asymmetry, and sparsity trends "
            + (
                "averaged coordinate-wise across fixed OP training instances."
                if self.instance_pooling == "mean"
                else "for each fixed OP training instance."
            )
        )
        logging.info(
            "[BehaviorEmbedder] Ready | task=op_aco extractor=%s dataset=%s "
            "matrices=%d nodes=%d states_per_matrix=%d pooling=%s output_dim=%d "
            "heuristic_timeout=%.3fs encode_timeout=%.3fs",
            self.extractor_version,
            self.dataset_path,
            self.n_matrices,
            self.n_nodes,
            len(self._state_banks[0]),
            self.instance_pooling,
            self.output_dim,
            self.heuristic_timeout,
            self.encode_timeout,
        )

    def _validate_configuration(self):
        if self.n_matrices <= 0:
            raise ValueError("OP behavior matrices must be positive.")
        if self.n_nodes <= 2:
            raise ValueError("OP behavior node count must be greater than two.")
        if self.probes_per_stage <= 0:
            raise ValueError("OP probes per stage must be positive.")
        if self.probes_per_stage > len(REFERENCE_MODES):
            raise ValueError(
                f"OP probes per stage cannot exceed {len(REFERENCE_MODES)} in v1."
            )
        if len(self.stage_ratios) < 2:
            raise ValueError("OP requires at least two route-stage ratios.")
        if any(not 0.0 < ratio < 1.0 for ratio in self.stage_ratios):
            raise ValueError("OP route-stage ratios must lie in (0, 1).")
        if any(
            earlier >= later
            for earlier, later in zip(self.stage_ratios, self.stage_ratios[1:])
        ):
            raise ValueError("OP route-stage ratios must be strictly increasing.")

    def _load_instances(self):
        with np.load(self.dataset_path, allow_pickle=False) as dataset:
            if "coordinates" not in dataset:
                raise ValueError("OP dataset must contain a 'coordinates' array.")
            coordinates = np.asarray(dataset["coordinates"], dtype=np.float64)

        if coordinates.ndim == 2:
            coordinates = coordinates[None, ...]
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError(
                "OP coordinates must have shape (instances, nodes, 2), got "
                f"{coordinates.shape}."
            )
        if self.n_matrices > len(coordinates):
            raise ValueError(
                f"Requested {self.n_matrices} OP instances from a dataset of "
                f"{len(coordinates)}."
            )
        if self.n_nodes > coordinates.shape[1]:
            raise ValueError(
                f"Requested {self.n_nodes} OP nodes from instances with "
                f"{coordinates.shape[1]} nodes."
            )

        instances = []
        for coordinate in coordinates[: self.n_matrices]:
            coordinate = coordinate[: self.n_nodes].copy()
            if not np.all(np.isfinite(coordinate)):
                raise ValueError("OP behavior dataset contains invalid coordinates.")
            delta = coordinate[:, None, :] - coordinate[None, :, :]
            distance = np.sqrt(np.sum(delta * delta, axis=-1))
            np.fill_diagonal(distance, 1e9)
            depot_distance = np.linalg.norm(coordinate - coordinate[0], axis=1)
            scale = float(np.max(depot_distance))
            if scale <= 0.0:
                raise ValueError("OP behavior instance has coincident nodes only.")
            prize = 1.0 + np.floor(99.0 * depot_distance / scale)
            prize /= float(np.max(prize))
            instances.append(
                {
                    "coordinate": coordinate,
                    "distance": distance,
                    "prize": prize,
                    "max_len": _get_max_len(self.n_nodes),
                }
            )
        return tuple(instances)

    @staticmethod
    def _feasible_candidates(instance, current, visited, travel_distance):
        distance = instance["distance"]
        available = np.flatnonzero(~np.asarray(visited, dtype=bool))
        available = available[available != 0]
        if not len(available):
            return available.astype(int)
        total = (
            float(travel_distance)
            + distance[int(current), available]
            + distance[available, 0]
        )
        return available[total <= float(instance["max_len"]) + 1e-12].astype(int)

    @staticmethod
    def _committed_ratio(instance, current, travel_distance):
        return_distance = (
            0.0 if int(current) == 0 else float(instance["distance"][int(current), 0])
        )
        return float(
            (float(travel_distance) + return_distance) / instance["max_len"]
        )

    def _reference_trace(self, instance, mode, seed):
        rng = np.random.default_rng(int(seed))
        n_nodes = len(instance["prize"])
        visited = np.zeros(n_nodes, dtype=bool)
        visited[0] = True
        current = 0
        travel_distance = 0.0
        trace = []

        for _ in range(n_nodes):
            feasible = self._feasible_candidates(
                instance, current, visited, travel_distance
            )
            if len(feasible) >= 2:
                trace.append(
                    {
                        "current": int(current),
                        "visited": visited.copy(),
                        "travel_distance": float(travel_distance),
                        "committed_ratio": self._committed_ratio(
                            instance, current, travel_distance
                        ),
                    }
                )
            if not len(feasible):
                break

            distance = instance["distance"]
            prize = instance["prize"]
            return_now = 0.0 if current == 0 else float(distance[current, 0])
            detour = (
                distance[current, feasible]
                + distance[feasible, 0]
                - return_now
            )
            if mode == "nearest":
                chosen = int(feasible[np.argmin(distance[current, feasible])])
            elif mode == "prize_edge":
                score = prize[feasible] / np.maximum(
                    distance[current, feasible], 1e-12
                )
                chosen = int(feasible[np.argmax(score)])
            elif mode == "prize_detour":
                score = prize[feasible] / np.maximum(detour, 1e-12)
                chosen = int(feasible[np.argmax(score)])
            elif mode == "low_detour_random":
                cutoff = float(np.quantile(detour, 0.60))
                pool = feasible[detour <= cutoff + 1e-12]
                chosen = int(rng.choice(pool))
            else:
                raise ValueError(f"Unknown OP reference mode: {mode}")

            travel_distance += float(distance[current, chosen])
            visited[chosen] = True
            current = chosen
        return trace

    def _make_fixed_states(self, instance, seed):
        states = []
        modes = REFERENCE_MODES[: self.probes_per_stage]
        traces = {
            mode: self._reference_trace(instance, mode, seed + index * 97)
            for index, mode in enumerate(modes)
        }
        for stage_ratio in self.stage_ratios:
            for mode in modes:
                trace = traces[mode]
                if not trace:
                    raise ValueError(
                        f"OP reference trace {mode!r} has no non-terminal states."
                    )
                chosen = min(
                    trace,
                    key=lambda row: (
                        abs(float(row["committed_ratio"]) - float(stage_ratio)),
                        float(row["committed_ratio"]) < float(stage_ratio),
                    ),
                )
                state = dict(chosen)
                state["stage_ratio"] = float(stage_ratio)
                state["reference_mode"] = mode
                states.append(state)
        return tuple(states)

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
        exec(compile(tree, "<op_candidate_heuristic>", "exec"), namespace, namespace)
        for name in FUNCTION_NAMES:
            function = namespace.get(name)
            if callable(function):
                return function
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function = namespace.get(node.name)
                if callable(function):
                    return function
        raise ValueError(
            f"No callable OP heuristic found; expected one of {FUNCTION_NAMES}."
        )

    def _call_heuristic(self, function, instance):
        with _hard_timeout(self.heuristic_timeout):
            raw = function(
                instance["prize"].copy(),
                instance["distance"].copy(),
                float(instance["max_len"]),
            )
        raw = np.asarray(raw, dtype=np.float64)
        expected_shape = (self.n_nodes, self.n_nodes)
        if raw.shape != expected_shape:
            raise ValueError(
                f"OP heuristic shape {raw.shape} does not match {expected_shape}."
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("OP heuristic contains NaN or infinity.")
        heuristic = raw + 1e-9
        heuristic[heuristic < 1e-9] = 1e-9
        return raw, heuristic

    def _state_core_features(self, heuristic, instance, state):
        current = int(state["current"])
        visited = np.asarray(state["visited"], dtype=bool)
        travel_distance = float(state["travel_distance"])
        available = np.flatnonzero(~visited)
        available = available[available != 0]
        feasible = self._feasible_candidates(
            instance, current, visited, travel_distance
        )
        if not len(feasible):
            raise ValueError("OP fixed state unexpectedly has no feasible actions.")

        available_probability = _distribution(heuristic[current, available])
        feasible_lookup = np.isin(available, feasible)
        feasible_prior_mass = float(np.sum(available_probability[feasible_lookup]))
        probabilities = _distribution(heuristic[current, feasible])

        distance = instance["distance"]
        prize = instance["prize"]
        return_now = 0.0 if current == 0 else float(distance[current, 0])
        detour = (
            distance[current, feasible]
            + distance[feasible, 0]
            - return_now
        )
        efficiency = prize[feasible] / np.maximum(detour, 1e-12)
        slack = np.clip(
            (
                float(instance["max_len"])
                - travel_distance
                - distance[current, feasible]
                - distance[feasible, 0]
            )
            / float(instance["max_len"]),
            0.0,
            1.0,
        )
        if len(probabilities) > 1:
            positive = probabilities[probabilities > 0.0]
            entropy = float(
                -np.sum(positive * np.log(positive)) / math.log(len(probabilities))
            )
        else:
            entropy = 0.0
        return np.asarray(
            [
                float(probabilities @ _percentile_ranks(prize[feasible])),
                float(
                    probabilities
                    @ _percentile_ranks(distance[current, feasible])
                ),
                float(probabilities @ _percentile_ranks(efficiency)),
                float(probabilities @ slack),
                feasible_prior_mass,
                entropy,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _matrix_features(raw_heuristic):
        raw = np.asarray(raw_heuristic, dtype=np.float64)
        n_nodes = len(raw)

        active_mask = np.zeros_like(raw, dtype=bool)
        active_mask[0, 1:] = True
        active_mask[1:, 1:] = True
        np.fill_diagonal(active_mask, False)
        zero_support_ratio = float(np.mean(raw[active_mask] <= 0.0))

        customer = np.where(raw[1:, 1:] > 0.0, raw[1:, 1:], 0.0).copy()
        np.fill_diagonal(customer, 0.0)
        row_sums = np.sum(customer, axis=1, keepdims=True)
        policy = np.divide(
            customer,
            row_sums,
            out=np.zeros_like(customer),
            where=row_sums > 0.0,
        )
        asymmetry = float(
            np.sum(np.abs(policy - policy.T)) / (2.0 * max(1, n_nodes - 1))
        )
        return asymmetry, zero_support_ratio

    @staticmethod
    def _trend_features(rows, states, raw_heuristic):
        asymmetry, zero_support_ratio = OPACOBehaviorEmbedder._matrix_features(
            raw_heuristic
        )
        values = []
        for column in range(6):
            values.extend(
                (
                    float(np.mean(rows[:, column])),
                    _stage_slope(rows[:, column], states),
                )
            )
        values.extend((asymmetry, zero_support_ratio))
        return np.asarray(values, dtype=np.float32)

    def _fallback_feature(self):
        return np.full(self.output_dim, -1.0, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        started = time.monotonic()
        try:
            with fixed_candidate_random_state(text, self.seed):
                function = self._compile_heuristic(text)
                parts = []
                for instance, states in zip(self._instances, self._state_banks):
                    if (
                        self.encode_timeout > 0.0
                        and time.monotonic() - started > self.encode_timeout
                    ):
                        raise OPBehaviorTimeout("OP behavior encoding timeout")
                    raw, heuristic = self._call_heuristic(function, instance)
                    rows = np.vstack(
                        [
                            self._state_core_features(heuristic, instance, state)
                            for state in states
                        ]
                    )
                    parts.append(self._trend_features(rows, states, raw))
            feature = aggregate_instance_features(parts, self.instance_pooling)
            if feature.shape != (self.output_dim,) or not np.all(np.isfinite(feature)):
                raise ValueError(
                    f"Invalid OP trend feature shape or values: {feature.shape}."
                )
            return feature
        except OPBehaviorTimeout as error:
            logging.info(
                "[BehaviorEmbedder] OP timeout, using fallback feature: %s", error
            )
            return self._fallback_feature()
        except Exception as error:
            logging.warning(
                "[BehaviorEmbedder] OP encode failed, using fallback feature: %s: %s",
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

    def finetune(self, *args, **kwargs):
        logging.info(
            "[BehaviorEmbedder] finetune skipped: OP trend features are deterministic."
        )


OPACOTrendEmbedder = OPACOBehaviorEmbedder
