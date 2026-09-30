"""Deterministic CVRP-ACO trend features for online surrogate modelling."""

import ast
import contextlib
import inspect
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
    "edge_rank_mean",
    "savings_rank_mean",
    "demand_rank_mean",
    "capacity_fill_mean",
    "depot_probability_mean",
    "infeasible_mass_mean",
    "entropy_mean",
    "edge_rank_stage_slope",
    "savings_rank_stage_slope",
    "demand_rank_load_slope",
    "capacity_fill_load_slope",
    "depot_probability_load_slope",
    "infeasible_mass_load_slope",
    "entropy_stage_slope",
    "heuristic_asymmetry",
)
FUNCTION_NAMES = ("heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3")


class CVRPBehaviorTimeout(TimeoutError):
    pass


def _parse_ratios(value, default):
    if value is None:
        return tuple(float(item) for item in default)
    if isinstance(value, str):
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        values = [float(item) for item in value]
    if not values:
        raise ValueError("At least one ratio is required.")
    return tuple(values)


@contextlib.contextmanager
def _hard_timeout(seconds):
    """Cooperatively time candidate Python code without interrupting C extensions.

    Raising from SIGALRM can interrupt SciPy while a C-backed object is only
    partially initialized, which can corrupt its cleanup state and segfault the
    whole search process.  Trace only frames compiled from candidate code, then
    check the elapsed time again after the call returns.  This still stops Python
    loops while allowing NumPy/SciPy calls to finish before timeout handling.
    """
    seconds = float(seconds)
    if seconds <= 0.0:
        yield
        return

    started = time.monotonic()
    deadline = started + seconds
    previous_trace = sys.gettrace()

    def trace_candidate(frame, event, arg):
        if (
            frame.f_code.co_filename == "<cvrp_candidate_heuristic>"
            and time.monotonic() >= deadline
        ):
            raise CVRPBehaviorTimeout(f"heuristic exceeded {seconds:g} seconds")
        return trace_candidate

    sys.settrace(trace_candidate)
    try:
        yield
    finally:
        sys.settrace(previous_trace)

    if time.monotonic() - started >= seconds:
        raise CVRPBehaviorTimeout(f"heuristic exceeded {seconds:g} seconds")


def _strip_code_fence(code):
    code = str(code or "").strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", code, flags=re.IGNORECASE | re.DOTALL)
    if match:
        code = match.group(1)
    return code.strip()


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


def _distribution(weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(len(weights), 1.0 / max(len(weights), 1), dtype=np.float64)
    return weights / total


def _group_slope(values, groups, n_groups):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=int)
    x_values = []
    y_values = []
    for group in range(int(n_groups)):
        selected = values[groups == group]
        if len(selected):
            x_values.append(float(group))
            y_values.append(float(np.mean(selected)))
    if len(y_values) < 2:
        return 0.0
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    x -= np.mean(x)
    denominator = float(x @ x)
    if denominator <= 0.0:
        return 0.0
    return float((x @ (y - np.mean(y))) / denominator)


class CVRPACOBehaviorEmbedder:
    """Map CVRP heuristic code to 15 paired trend/diagnostic features."""

    def __init__(
        self,
        dataset_path: Union[str, os.PathLike],
        n_matrices: Optional[int] = None,
        n_customers: Optional[int] = None,
        capacity: Optional[float] = None,
        stage_ratios=None,
        load_ratios=None,
        probes_per_cell: Optional[int] = None,
        seed: Optional[int] = None,
        heuristic_timeout: Optional[float] = None,
        encode_timeout: Optional[float] = None,
        instance_pooling: Optional[str] = None,
    ):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"CVRP behavior dataset does not exist: {self.dataset_path}")

        self.n_matrices = int(
            n_matrices
            if n_matrices is not None
            else os.environ.get("BEMRS_CVRP_BEHAVIOR_MATRICES", 8)
        )
        self.n_customers = int(
            n_customers
            if n_customers is not None
            else os.environ.get("BEMRS_CVRP_BEHAVIOR_NODES", 50)
        )
        self.capacity = float(
            capacity
            if capacity is not None
            else os.environ.get("BEMRS_CVRP_CAPACITY", 50.0)
        )
        self.stage_ratios = _parse_ratios(
            stage_ratios
            if stage_ratios is not None
            else os.environ.get("BEMRS_CVRP_STAGE_RATIOS"),
            (0.8, 0.5, 0.2),
        )
        self.load_ratios = _parse_ratios(
            load_ratios
            if load_ratios is not None
            else os.environ.get("BEMRS_CVRP_LOAD_RATIOS"),
            (0.1, 0.5, 0.85),
        )
        self.probes_per_cell = int(
            probes_per_cell
            if probes_per_cell is not None
            else os.environ.get("BEMRS_CVRP_PROBES_PER_CELL", 4)
        )
        self.seed = int(
            seed
            if seed is not None
            else os.environ.get(
                "BEMRS_CVRP_BEHAVIOR_SEED",
                os.environ.get("BEMRS_BEHAVIOR_SEED", 42),
            )
        )
        self.heuristic_timeout = float(
            heuristic_timeout
            if heuristic_timeout is not None
            else os.environ.get("BEMRS_CVRP_HEURISTIC_TIMEOUT", 0.5)
        )
        self.encode_timeout = float(
            encode_timeout
            if encode_timeout is not None
            else os.environ.get("BEMRS_CVRP_ENCODE_TIMEOUT", 5.0)
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
            "cvrp_aco_policy_trend15_instance_mean_v3"
            if self.instance_pooling == "mean"
            else "cvrp_aco_policy_trend15_v1"
        )
        self.extractor_description = (
            "Fifteen paired capacity-, depot-, demand-, and route-stage-conditioned policy "
            "trends "
            + (
                "averaged coordinate-wise across fixed CVRP training instances."
                if self.instance_pooling == "mean"
                else "for each fixed CVRP training instance."
            )
        )
        logging.info(
            "[BehaviorEmbedder] Ready | task=cvrp_aco extractor=%s dataset=%s "
            "matrices=%d customers=%d states_per_matrix=%d pooling=%s output_dim=%d "
            "heuristic_timeout=%.3fs encode_timeout=%.3fs",
            self.extractor_version,
            self.dataset_path,
            self.n_matrices,
            self.n_customers,
            len(self._state_banks[0]),
            self.instance_pooling,
            self.output_dim,
            self.heuristic_timeout,
            self.encode_timeout,
        )

    def _validate_configuration(self):
        if self.n_matrices < 1:
            raise ValueError("BEMRS_CVRP_BEHAVIOR_MATRICES must be positive.")
        if self.n_customers < 2:
            raise ValueError("BEMRS_CVRP_BEHAVIOR_NODES must be at least 2.")
        if self.capacity <= 0.0:
            raise ValueError("BEMRS_CVRP_CAPACITY must be positive.")
        if self.probes_per_cell < 1:
            raise ValueError("BEMRS_CVRP_PROBES_PER_CELL must be positive.")
        if len(self.stage_ratios) < 2 or any(not 0.0 < value <= 1.0 for value in self.stage_ratios):
            raise ValueError("CVRP stage ratios require at least two values in (0, 1].")
        if len(self.load_ratios) < 2 or any(not 0.0 <= value < 1.0 for value in self.load_ratios):
            raise ValueError("CVRP load ratios require at least two values in [0, 1).")

    def _load_instances(self):
        dataset = np.load(self.dataset_path, allow_pickle=False)
        if dataset.ndim == 2:
            dataset = dataset[None, ...]
        if dataset.ndim != 3 or dataset.shape[2] < 3:
            raise ValueError(
                "CVRP dataset must have shape (instances, depot+customers, >=3), "
                f"got {dataset.shape}."
            )
        if self.n_matrices > len(dataset):
            raise ValueError(
                f"Requested {self.n_matrices} CVRP instances from a dataset of {len(dataset)}."
            )

        instances = []
        for raw in dataset[: self.n_matrices]:
            node_count = min(raw.shape[0], self.n_customers + 1)
            demand = np.asarray(raw[:node_count, 0], dtype=np.float64).copy()
            coordinates = np.asarray(raw[:node_count, 1:3], dtype=np.float64).copy()
            delta = coordinates[:, None, :] - coordinates[None, :, :]
            distance = np.sqrt(np.sum(delta * delta, axis=2))
            np.fill_diagonal(distance, 1.0)
            demand[0] = 0.0
            if not np.all(np.isfinite(distance)) or not np.all(np.isfinite(demand)):
                raise ValueError("CVRP dataset contains non-finite values.")
            instances.append(
                {"distance": distance, "coordinates": coordinates, "demand": demand}
            )
        return instances

    def _make_fixed_states(self, instance, seed):
        customers = np.arange(1, len(instance["demand"]), dtype=int)
        rng = np.random.default_rng(int(seed))
        states = []
        for stage_index, stage_ratio in enumerate(self.stage_ratios):
            keep = int(np.clip(round(stage_ratio * len(customers)), 1, len(customers)))
            for load_index, load_ratio in enumerate(self.load_ratios):
                for probe_index in range(self.probes_per_cell):
                    unvisited = np.sort(rng.choice(customers, size=keep, replace=False))
                    visited = np.setdiff1d(customers, unvisited, assume_unique=True)
                    if len(visited):
                        current = int(visited[probe_index % len(visited)])
                    else:
                        current = int(rng.choice(customers))
                        unvisited = unvisited[unvisited != current]
                    states.append(
                        {
                            "current": current,
                            "unvisited": unvisited,
                            "used_capacity_ratio": float(load_ratio),
                            "stage_index": stage_index,
                            "load_index": load_index,
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
        exec(compile(tree, "<cvrp_candidate_heuristic>", "exec"), namespace, namespace)
        for name in FUNCTION_NAMES:
            function = namespace.get(name)
            if callable(function):
                return function
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function = namespace.get(node.name)
                if callable(function):
                    return function
        raise ValueError(f"No callable CVRP heuristic found; expected one of {FUNCTION_NAMES}.")

    def _call_heuristic(self, function, instance):
        try:
            signature = inspect.signature(function)
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(
                parameter.kind == inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
            arity = 4 if has_varargs else len(positional)
        except (TypeError, ValueError):
            arity = 1

        distance = instance["distance"]
        with _hard_timeout(self.heuristic_timeout):
            if arity >= 4:
                raw = function(
                    distance.copy(),
                    instance["coordinates"].copy(),
                    instance["demand"].copy(),
                    float(self.capacity),
                )
            elif arity == 2:
                raw = function(distance.copy(), instance["demand"].copy() / self.capacity)
            elif arity == 1:
                raw = function(distance.copy())
            else:
                raise ValueError(f"Unsupported CVRP heuristic arity: {arity}; expected 1, 2, or 4.")

        heuristic = np.asarray(raw, dtype=np.float64)
        if heuristic.shape != distance.shape:
            raise ValueError(
                f"CVRP heuristic shape {heuristic.shape} does not match {distance.shape}."
            )
        heuristic = heuristic + 1e-9
        heuristic[heuristic < 1e-9] = 1e-9
        if not np.all(np.isfinite(heuristic)):
            raise ValueError("CVRP heuristic contains NaN or infinity.")
        return heuristic

    def _state_core_features(self, heuristic, instance, state):
        distance = instance["distance"]
        demand = instance["demand"]
        current = int(state["current"])
        unvisited = np.asarray(state["unvisited"], dtype=int)
        remaining = max(0.0, self.capacity * (1.0 - state["used_capacity_ratio"]))

        visit_actions = np.concatenate((np.array([0], dtype=int), unvisited))
        prior = _distribution(heuristic[current, visit_actions])
        infeasible = (visit_actions != 0) & (demand[visit_actions] > remaining + 1e-12)
        infeasible_mass = float(np.sum(prior[infeasible]))

        feasible_actions = visit_actions[~infeasible]
        probabilities = _distribution(heuristic[current, feasible_actions])
        customer_mask = feasible_actions != 0
        customer_actions = feasible_actions[customer_mask]
        customer_probabilities = _distribution(probabilities[customer_mask])
        depot_probability = float(np.sum(probabilities[~customer_mask]))

        if len(customer_actions):
            edges = distance[current, customer_actions]
            savings = (
                distance[0, current]
                + distance[0, customer_actions]
                - distance[current, customer_actions]
            )
            demands = demand[customer_actions]
            edge_rank = float(customer_probabilities @ _percentile_ranks(edges))
            savings_rank = float(customer_probabilities @ _percentile_ranks(savings))
            demand_rank = float(customer_probabilities @ _percentile_ranks(demands))
            capacity_fill = float(
                customer_probabilities
                @ np.clip(demands / max(remaining, 1e-9), 0.0, 1.0)
            )
        else:
            edge_rank = savings_rank = demand_rank = capacity_fill = 0.0

        if len(probabilities) > 1:
            positive = probabilities[probabilities > 0.0]
            entropy = float(
                -np.sum(positive * np.log(positive)) / math.log(len(probabilities))
            )
        else:
            entropy = 0.0
        return np.asarray(
            [
                edge_rank,
                savings_rank,
                demand_rank,
                capacity_fill,
                depot_probability,
                infeasible_mass,
                entropy,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _heuristic_asymmetry(heuristic):
        policy = np.asarray(heuristic, dtype=np.float64).copy()
        np.fill_diagonal(policy, 0.0)
        row_sums = np.sum(policy, axis=1, keepdims=True)
        policy = np.divide(
            policy, row_sums, out=np.zeros_like(policy), where=row_sums > 0.0
        )
        return float(np.sum(np.abs(policy - policy.T)) / (2.0 * len(policy)))

    def _trend_features(self, rows, states, heuristic):
        stage = np.asarray([state["stage_index"] for state in states], dtype=int)
        load = np.asarray([state["load_index"] for state in states], dtype=int)
        return np.asarray(
            [
                np.mean(rows[:, 0]),
                np.mean(rows[:, 1]),
                np.mean(rows[:, 2]),
                np.mean(rows[:, 3]),
                np.mean(rows[:, 4]),
                np.mean(rows[:, 5]),
                np.mean(rows[:, 6]),
                _group_slope(rows[:, 0], stage, len(self.stage_ratios)),
                _group_slope(rows[:, 1], stage, len(self.stage_ratios)),
                _group_slope(rows[:, 2], load, len(self.load_ratios)),
                _group_slope(rows[:, 3], load, len(self.load_ratios)),
                _group_slope(rows[:, 4], load, len(self.load_ratios)),
                _group_slope(rows[:, 5], load, len(self.load_ratios)),
                _group_slope(rows[:, 6], stage, len(self.stage_ratios)),
                self._heuristic_asymmetry(heuristic),
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
                        raise CVRPBehaviorTimeout("CVRP behavior encoding timeout")
                    heuristic = self._call_heuristic(function, instance)
                    rows = np.vstack(
                        [
                            self._state_core_features(heuristic, instance, state)
                            for state in states
                        ]
                    )
                    parts.append(self._trend_features(rows, states, heuristic))
            feature = aggregate_instance_features(parts, self.instance_pooling)
            if feature.shape != (self.output_dim,) or not np.all(np.isfinite(feature)):
                raise ValueError(
                    f"Invalid CVRP trend feature shape or values: {feature.shape}."
                )
            return feature
        except CVRPBehaviorTimeout as error:
            logging.info("[BehaviorEmbedder] CVRP timeout, using fallback feature: %s", error)
            return self._fallback_feature()
        except Exception as error:
            logging.debug("[BehaviorEmbedder] CVRP encode failed: %s", error)
            return self._fallback_feature()

    def encode(self, texts: Union[str, List[str]], batch_size: int = None) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.output_dim), dtype=np.float32)
        return np.vstack([self.encode_single(text) for text in texts]).astype(np.float32)

    def finetune(self, *args, **kwargs):
        logging.info(
            "[BehaviorEmbedder] finetune skipped: CVRP trend features are deterministic."
        )


# Compatibility alias for the first CVRP trend implementation name.
CVRPACOTrendEmbedder = CVRPACOBehaviorEmbedder
