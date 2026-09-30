"""Deterministic MKP-ACO policy trend features for surrogate modelling."""

import ast
import contextlib
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
    "density_rank_mean",
    "scarcity_density_rank_mean",
    "weight_rank_mean",
    "bottleneck_rank_mean",
    "feasible_prior_mass_mean",
    "entropy_mean",
    "prize_rank_stage_slope",
    "density_rank_stage_slope",
    "scarcity_density_rank_stage_slope",
    "weight_rank_stage_slope",
    "bottleneck_rank_stage_slope",
    "feasible_prior_mass_stage_slope",
    "entropy_stage_slope",
    "zero_support_ratio",
)
FUNCTION_NAMES = ("heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3")


class MKPBehaviorTimeout(TimeoutError):
    pass


def _parse_ratios(value, default):
    if value is None:
        return tuple(float(item) for item in default)
    if isinstance(value, str):
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        values = [float(item) for item in value]
    if not values:
        raise ValueError("At least one MKP stage ratio is required.")
    return tuple(values)


@contextlib.contextmanager
def _hard_timeout(seconds):
    """Interrupt Python loops in candidate code without interrupting NumPy internals."""
    seconds = float(seconds)
    if seconds <= 0.0:
        yield
        return

    deadline = time.monotonic() + seconds
    previous_trace = sys.gettrace()

    def trace_candidate(frame, event, arg):
        if (
            frame.f_code.co_filename == "<mkp_candidate_heuristic>"
            and time.monotonic() >= deadline
        ):
            raise MKPBehaviorTimeout(f"heuristic exceeded {seconds:g} seconds")
        return trace_candidate

    sys.settrace(trace_candidate)
    try:
        yield
    finally:
        sys.settrace(previous_trace)

    if time.monotonic() >= deadline:
        raise MKPBehaviorTimeout(f"heuristic exceeded {seconds:g} seconds")


def _strip_code_fence(code):
    code = str(code or "").strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", code, flags=re.IGNORECASE | re.DOTALL)
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


class MKPACOBehaviorEmbedder:
    """Map MKP heuristic code to 15 scale-invariant trends per instance."""

    def __init__(
        self,
        dataset_path: Union[str, os.PathLike],
        n_matrices: Optional[int] = None,
        n_items: Optional[int] = None,
        stage_ratios=None,
        probes_per_stage: Optional[int] = None,
        seed: Optional[int] = None,
        heuristic_timeout: Optional[float] = None,
        encode_timeout: Optional[float] = None,
        instance_pooling: Optional[str] = None,
    ):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"MKP behavior dataset does not exist: {self.dataset_path}")

        self.n_matrices = int(
            n_matrices
            if n_matrices is not None
            else os.environ.get("BEMRS_MKP_BEHAVIOR_MATRICES", 5)
        )
        self.n_items = int(
            n_items
            if n_items is not None
            else os.environ.get("BEMRS_MKP_BEHAVIOR_ITEMS", 50)
        )
        self.stage_ratios = _parse_ratios(
            stage_ratios
            if stage_ratios is not None
            else os.environ.get("BEMRS_MKP_STAGE_RATIOS"),
            (0.2, 0.5, 0.8),
        )
        self.probes_per_stage = int(
            probes_per_stage
            if probes_per_stage is not None
            else os.environ.get("BEMRS_MKP_PROBES_PER_STAGE", 4)
        )
        self.seed = int(
            seed
            if seed is not None
            else os.environ.get("BEMRS_MKP_BEHAVIOR_SEED", os.environ.get("BEMRS_BEHAVIOR_SEED", 42))
        )
        self.heuristic_timeout = float(
            heuristic_timeout
            if heuristic_timeout is not None
            else os.environ.get("BEMRS_MKP_HEURISTIC_TIMEOUT", 0.2)
        )
        self.encode_timeout = float(
            encode_timeout
            if encode_timeout is not None
            else os.environ.get("BEMRS_MKP_ENCODE_TIMEOUT", 2.0)
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
            "mkp_aco_policy_trend15_instance_mean_v3"
            if self.instance_pooling == "mean"
            else "mkp_aco_policy_trend15_v2"
        )
        self.extractor_description = (
            "Fifteen scale-invariant prize, density, capacity-feasibility, concentration, "
            "and sparsity trends "
            + (
                "averaged coordinate-wise across fixed MKP training instances."
                if self.instance_pooling == "mean"
                else "for each fixed MKP training instance."
            )
        )
        logging.info(
            "[BehaviorEmbedder] Ready | task=mkp_aco extractor=%s dataset=%s "
            "matrices=%d items=%d constraints=%d states_per_matrix=%d pooling=%s "
            "output_dim=%d "
            "heuristic_timeout=%.3fs encode_timeout=%.3fs",
            self.extractor_version,
            self.dataset_path,
            self.n_matrices,
            self.n_items,
            self._instances[0]["weight"].shape[1],
            len(self._state_banks[0]),
            self.instance_pooling,
            self.output_dim,
            self.heuristic_timeout,
            self.encode_timeout,
        )

    def _validate_configuration(self):
        if self.n_matrices <= 0:
            raise ValueError("MKP behavior matrices must be positive.")
        if self.n_items <= 1:
            raise ValueError("MKP behavior item count must be greater than one.")
        if self.probes_per_stage <= 0:
            raise ValueError("MKP probes per stage must be positive.")
        if any(not 0.0 <= ratio < 1.0 for ratio in self.stage_ratios):
            raise ValueError("MKP stage ratios must lie in [0, 1).")

    def _load_instances(self):
        with np.load(self.dataset_path, allow_pickle=False) as dataset:
            if "prizes" not in dataset or "weights" not in dataset:
                raise ValueError("MKP dataset must contain 'prizes' and 'weights'.")
            prizes = np.asarray(dataset["prizes"], dtype=np.float64)
            weights = np.asarray(dataset["weights"], dtype=np.float64)

        if prizes.ndim == 1:
            prizes = prizes[None, :]
        if weights.ndim == 2:
            weights = weights[None, :, :]
        if prizes.ndim != 2 or weights.ndim != 3:
            raise ValueError(
                "MKP dataset must have prizes (instances, items) and weights "
                f"(instances, items, constraints), got {prizes.shape} and {weights.shape}."
            )
        if weights.shape[:2] != prizes.shape:
            raise ValueError(
                f"MKP prize/weight shapes are inconsistent: {prizes.shape}, {weights.shape}."
            )
        if self.n_matrices > len(prizes):
            raise ValueError(
                f"Requested {self.n_matrices} MKP instances from a dataset of {len(prizes)}."
            )
        if self.n_items > prizes.shape[1]:
            raise ValueError(
                f"Requested {self.n_items} MKP items from instances with {prizes.shape[1]} items."
            )

        instances = []
        for prize, weight in zip(prizes[: self.n_matrices], weights[: self.n_matrices]):
            prize = prize[: self.n_items].copy()
            weight = weight[: self.n_items].copy()
            if (
                not np.all(np.isfinite(prize))
                or not np.all(np.isfinite(weight))
                or np.any(weight < 0.0)
            ):
                raise ValueError("MKP behavior dataset contains invalid values.")
            instances.append({"prize": prize, "weight": weight})
        return tuple(instances)

    @staticmethod
    def _reference_scores(instance, probe_index, rng):
        prize = instance["prize"]
        weight = instance["weight"]
        total_weight = np.sum(weight, axis=1)
        mode = int(probe_index) % 4
        if mode == 0:
            score = prize / np.maximum(total_weight, 1e-12)
        elif mode == 1:
            score = prize.copy()
        elif mode == 2:
            score = -total_weight
        else:
            score = rng.random(len(prize))
        score = np.asarray(score, dtype=np.float64)
        score += rng.uniform(-1e-12, 1e-12, size=len(score))
        return score

    def _make_fixed_states(self, instance, seed):
        rng = np.random.default_rng(int(seed))
        weight = instance["weight"]
        states = []
        for stage_ratio in self.stage_ratios:
            for probe_index in range(self.probes_per_stage):
                score = self._reference_scores(instance, probe_index, rng)
                order = np.argsort(-score, kind="mergesort")
                selected = np.zeros(len(weight), dtype=bool)
                used = np.zeros(weight.shape[1], dtype=np.float64)
                for item in order:
                    if float(np.mean(used)) >= float(stage_ratio):
                        break
                    candidate = used + weight[item]
                    if np.all(candidate <= 1.0 + 1e-12):
                        selected[item] = True
                        used = candidate
                states.append(
                    {
                        "selected": selected,
                        "remaining": np.maximum(1.0 - used, 0.0),
                        "stage_ratio": float(stage_ratio),
                    }
                )
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
        exec(compile(tree, "<mkp_candidate_heuristic>", "exec"), namespace, namespace)
        for name in FUNCTION_NAMES:
            function = namespace.get(name)
            if callable(function):
                return function
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function = namespace.get(node.name)
                if callable(function):
                    return function
        raise ValueError(f"No callable MKP heuristic found; expected one of {FUNCTION_NAMES}.")

    def _call_heuristic(self, function, instance):
        with _hard_timeout(self.heuristic_timeout):
            raw = function(instance["prize"].copy(), instance["weight"].copy())
        raw = np.asarray(raw, dtype=np.float64)
        expected_shape = (self.n_items,)
        if raw.shape != expected_shape:
            raise ValueError(f"MKP heuristic shape {raw.shape} does not match {expected_shape}.")
        if not np.all(np.isfinite(raw)):
            raise ValueError("MKP heuristic contains NaN or infinity.")

        # Match eval.py exactly while preserving raw zeros for the sparsity feature.
        heuristic = raw + 1e-9
        heuristic[heuristic < 1e-9] = 1e-9
        return raw, heuristic

    @staticmethod
    def _state_core_features(heuristic, instance, state):
        prize = instance["prize"]
        weight = instance["weight"]
        selected = state["selected"]
        remaining = state["remaining"]
        prior = _distribution(heuristic)

        available = ~selected
        feasible = available & np.all(weight <= remaining[None, :] + 1e-12, axis=1)
        feasible_prior_mass = float(np.sum(prior[feasible]))
        indices = np.flatnonzero(feasible)
        if not len(indices):
            return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, feasible_prior_mass, 0.0])

        probabilities = _distribution(heuristic[indices])
        candidate_prize = prize[indices]
        candidate_weight = weight[indices]
        total_weight = np.sum(candidate_weight, axis=1)
        density = candidate_prize / np.maximum(total_weight, 1e-12)
        relative_weight = candidate_weight / np.maximum(remaining[None, :], 1e-12)
        scarcity_density = candidate_prize / np.maximum(np.sum(relative_weight, axis=1), 1e-12)
        bottleneck = np.max(relative_weight, axis=1)

        expected_prize_rank = float(probabilities @ _percentile_ranks(candidate_prize))
        expected_density_rank = float(probabilities @ _percentile_ranks(density))
        expected_scarcity_rank = float(probabilities @ _percentile_ranks(scarcity_density))
        expected_weight_rank = float(probabilities @ _percentile_ranks(total_weight))
        expected_bottleneck_rank = float(probabilities @ _percentile_ranks(bottleneck))
        if len(probabilities) > 1:
            positive = probabilities[probabilities > 0.0]
            entropy = float(
                -np.sum(positive * np.log(positive)) / math.log(len(probabilities))
            )
        else:
            entropy = 0.0
        return np.asarray(
            [
                expected_prize_rank,
                expected_density_rank,
                expected_scarcity_rank,
                expected_weight_rank,
                expected_bottleneck_rank,
                feasible_prior_mass,
                entropy,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _trend_features(rows, states, raw_heuristic):
        return np.asarray(
            [
                np.mean(rows[:, 0]),
                np.mean(rows[:, 1]),
                np.mean(rows[:, 2]),
                np.mean(rows[:, 3]),
                np.mean(rows[:, 4]),
                np.mean(rows[:, 5]),
                np.mean(rows[:, 6]),
                _stage_slope(rows[:, 0], states),
                _stage_slope(rows[:, 1], states),
                _stage_slope(rows[:, 2], states),
                _stage_slope(rows[:, 3], states),
                _stage_slope(rows[:, 4], states),
                _stage_slope(rows[:, 5], states),
                _stage_slope(rows[:, 6], states),
                np.mean(raw_heuristic <= 0.0),
            ],
            dtype=np.float32,
        )

    def _fallback_feature(self):
        return np.full(self.output_dim, -1.0, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        started = time.monotonic()
        try:
            with fixed_candidate_random_state(text, self.seed):
                function = self._compile_heuristic(text)
                parts = []
                for instance, states in zip(self._instances, self._state_banks):
                    if self.encode_timeout > 0.0 and time.monotonic() - started > self.encode_timeout:
                        raise MKPBehaviorTimeout("MKP behavior encoding timeout")
                    raw, heuristic = self._call_heuristic(function, instance)
                    rows = np.vstack(
                        [self._state_core_features(heuristic, instance, state) for state in states]
                    )
                    parts.append(self._trend_features(rows, states, raw))
            feature = aggregate_instance_features(parts, self.instance_pooling)
            if feature.shape != (self.output_dim,) or not np.all(np.isfinite(feature)):
                raise ValueError(f"Invalid MKP trend feature shape or values: {feature.shape}.")
            return feature
        except MKPBehaviorTimeout as error:
            logging.info("[BehaviorEmbedder] MKP timeout, using fallback feature: %s", error)
            return self._fallback_feature()
        except Exception as error:
            logging.warning(
                "[BehaviorEmbedder] MKP encode failed, using fallback feature: %s: %s",
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
