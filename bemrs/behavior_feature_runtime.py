"""Fixed runtime support for behavior feature extraction.

This file is infrastructure and is not an evolution target.  It owns timeout
handling, deterministic probe construction, fixed helper extractors, output
shape validation, and loading of the small evolvable feature program.
"""

import importlib
import os
import signal
from contextlib import contextmanager
from typing import Callable, Dict, List

import numpy as np


FEATURE_DIM_PER_PROBE = 10
DEFAULT_EXTRACTOR_NAME = "rollout3"

_ACTIVE_EXTRACTOR = None
_ACTIVE_DESCRIPTION = None
_ACTIVE_VERSION = "baseline_v0"

FIXED_EXTRACTOR_DESCRIPTIONS = {
    "generic": "Single-step generic decision ranks and normalized distances.",
    "future": "Single-step immediate-distance and destination-improvement behavior.",
    "dynamic_penalty": "Single-step dynamically weighted distance and connectivity behavior.",
}

def _rank01(values: np.ndarray, chosen_pos: int, ascending: bool = True) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return 0.0
    order = np.argsort(values, kind="mergesort")
    if not ascending:
        order = order[::-1]
    rank = int(np.where(order == int(chosen_pos))[0][0])
    return float(rank / (len(values) - 1))


def _z(value: float, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float((value - np.mean(values)) / (np.std(values) + 1e-12))


def _norm01(value: float, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float((value - np.min(values)) / (np.max(values) - np.min(values) + 1e-12))


class BehaviorProbeTimeout(Exception):
    pass


@contextmanager
def _behavior_timeout(seconds: float):
    seconds = float(seconds or 0.0)
    if seconds <= 0 or not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    old_handler = signal.getsignal(signal.SIGALRM)

    def _handler(signum, frame):
        raise BehaviorProbeTimeout("behavior probe timeout")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _call_selector_with_timeout(
    select_next_node_fn,
    current_node,
    destination_node,
    unvisited_nodes,
    matrix,
    timeout_s: float,
):
    with _behavior_timeout(timeout_s):
        return select_next_node_fn(
            current_node,
            destination_node,
            set(unvisited_nodes),
            np.asarray(matrix).copy(),
        )


def _make_probe_states(
    distance_matrix: np.ndarray,
    n_probes: int,
    seed: int,
    min_unvisited: int = 3,
):
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array.")

    n_nodes = int(distance_matrix.shape[0])
    if n_nodes < 4:
        raise ValueError("Need at least 4 nodes to build TSP behavior probes.")

    rng = np.random.default_rng(seed)
    min_unvisited = max(1, min(int(min_unvisited), n_nodes - 2))
    states = []
    for _ in range(int(n_probes)):
        current_node = int(rng.integers(0, n_nodes))
        destination_candidates = [node for node in range(n_nodes) if node != current_node]
        destination_node = int(rng.choice(destination_candidates))
        available = [node for node in range(n_nodes) if node not in {current_node, destination_node}]
        unvisited_count = int(rng.integers(min_unvisited, len(available) + 1))
        unvisited = set(int(x) for x in rng.choice(available, size=unvisited_count, replace=False))
        states.append((current_node, destination_node, unvisited, distance_matrix))
    return states


def _candidate_arrays(current_node, destination_node, unvisited_nodes, distance_matrix):
    candidates = np.array(sorted(int(x) for x in unvisited_nodes), dtype=int)
    d_cur = np.asarray(distance_matrix[current_node, candidates], dtype=np.float64)
    d_dest = np.asarray(distance_matrix[candidates, destination_node], dtype=np.float64)
    cur_to_dest = float(distance_matrix[current_node, destination_node])
    future_improvement = cur_to_dest - d_dest
    avg_to_unvisited = []
    for node in candidates:
        others = [int(x) for x in candidates if int(x) != int(node)]
        if others:
            avg_to_unvisited.append(float(np.mean(distance_matrix[int(node), others])))
        else:
            avg_to_unvisited.append(0.0)
    return candidates, d_cur, d_dest, future_improvement, np.asarray(avg_to_unvisited, dtype=np.float64)


def behavior_matrix_generic(
    select_next_node_fn,
    distance_matrix: np.ndarray,
    n_probes: int = 64,
    seed: int = 42,
    probe_timeout: float = None,
) -> np.ndarray:
    if probe_timeout is None:
        probe_timeout = float(os.environ.get("BEMRS_BEHAVIOR_PROBE_TIMEOUT", 0.05))
    rows = []
    penalty = np.full(FEATURE_DIM_PER_PROBE, -1.0, dtype=np.float32)
    for current_node, destination_node, unvisited_nodes, matrix in _make_probe_states(
        distance_matrix, n_probes, seed
    ):
        try:
            candidates, d_cur, d_dest, _, avg_unvisited = _candidate_arrays(
                current_node, destination_node, unvisited_nodes, matrix
            )
            chosen = int(
                _call_selector_with_timeout(
                    select_next_node_fn,
                    current_node,
                    destination_node,
                    unvisited_nodes,
                    matrix,
                    probe_timeout,
                )
            )
            if chosen not in unvisited_nodes:
                rows.append(penalty)
                continue
            chosen_pos = int(np.where(candidates == chosen)[0][0])
            rows.append(
                np.asarray(
                    [
                        _rank01(d_cur, chosen_pos, ascending=True),
                        _rank01(d_dest, chosen_pos, ascending=True),
                        _rank01(avg_unvisited, chosen_pos, ascending=True),
                        _norm01(float(d_cur[chosen_pos]), d_cur),
                        _norm01(float(d_dest[chosen_pos]), d_dest),
                        _norm01(float(avg_unvisited[chosen_pos]), avg_unvisited),
                        _z(float(d_cur[chosen_pos]), d_cur),
                        _z(float(d_dest[chosen_pos]), d_dest),
                        float(chosen_pos == int(np.argmin(d_cur))),
                        float(chosen_pos == int(np.argmin(d_dest))),
                    ],
                    dtype=np.float32,
                )
            )
        except Exception:
            rows.append(penalty)
    return np.nan_to_num(
        np.vstack(rows).astype(np.float32), nan=-1.0, posinf=1e6, neginf=-1e6
    )


def behavior_matrix_future_reduction(
    select_next_node_fn,
    distance_matrix: np.ndarray,
    n_probes: int = 64,
    seed: int = 42,
    probe_timeout: float = None,
) -> np.ndarray:
    if probe_timeout is None:
        probe_timeout = float(os.environ.get("BEMRS_BEHAVIOR_PROBE_TIMEOUT", 0.05))
    rows = []
    penalty = np.full(FEATURE_DIM_PER_PROBE, -1.0, dtype=np.float32)
    for current_node, destination_node, unvisited_nodes, matrix in _make_probe_states(
        distance_matrix, n_probes, seed
    ):
        try:
            candidates, d_cur, d_dest, future_improvement, _ = _candidate_arrays(
                current_node, destination_node, unvisited_nodes, matrix
            )
            chosen = int(
                _call_selector_with_timeout(
                    select_next_node_fn,
                    current_node,
                    destination_node,
                    unvisited_nodes,
                    matrix,
                    probe_timeout,
                )
            )
            if chosen not in unvisited_nodes:
                rows.append(penalty)
                continue
            chosen_pos = int(np.where(candidates == chosen)[0][0])
            score = d_cur + 0.5 * future_improvement
            chosen_score = float(score[chosen_pos])
            rows.append(
                np.asarray(
                    [
                        _rank01(d_cur, chosen_pos, ascending=True),
                        _rank01(d_dest, chosen_pos, ascending=True),
                        _rank01(future_improvement, chosen_pos, ascending=True),
                        _rank01(score, chosen_pos, ascending=True),
                        _norm01(float(d_cur[chosen_pos]), d_cur),
                        _norm01(float(d_dest[chosen_pos]), d_dest),
                        _z(float(future_improvement[chosen_pos]), future_improvement),
                        _z(chosen_score, score),
                        _norm01(chosen_score, score),
                        float(chosen_pos == int(np.argmin(score))),
                    ],
                    dtype=np.float32,
                )
            )
        except Exception:
            rows.append(penalty)
    return np.nan_to_num(
        np.vstack(rows).astype(np.float32), nan=-1.0, posinf=1e6, neginf=-1e6
    )


def behavior_matrix_dynamic_penalty(
    select_next_node_fn,
    distance_matrix: np.ndarray,
    n_probes: int = 64,
    seed: int = 42,
    probe_timeout: float = None,
) -> np.ndarray:
    if probe_timeout is None:
        probe_timeout = float(os.environ.get("BEMRS_BEHAVIOR_PROBE_TIMEOUT", 0.05))
    rows = []
    penalty = np.full(FEATURE_DIM_PER_PROBE, -1.0, dtype=np.float32)
    for current_node, destination_node, unvisited_nodes, matrix in _make_probe_states(
        distance_matrix, n_probes, seed
    ):
        try:
            candidates, d_cur, _, future_improvement, avg_unvisited = _candidate_arrays(
                current_node, destination_node, unvisited_nodes, matrix
            )
            chosen = int(
                _call_selector_with_timeout(
                    select_next_node_fn,
                    current_node,
                    destination_node,
                    unvisited_nodes,
                    matrix,
                    probe_timeout,
                )
            )
            if chosen not in unvisited_nodes:
                rows.append(penalty)
                continue
            chosen_pos = int(np.where(candidates == chosen)[0][0])
            num_unvisited = len(unvisited_nodes)
            weight_immediate = 0.4 + 0.1 * (num_unvisited / 10)
            weight_future = 0.3 - 0.05 * (num_unvisited / 10)
            weight_penalty = 0.3 + 0.05 * (num_unvisited / 10)
            score = (
                d_cur * weight_immediate
                + future_improvement * weight_future
                - avg_unvisited * weight_penalty
            )
            chosen_score = float(score[chosen_pos])
            rows.append(
                np.asarray(
                    [
                        _rank01(score, chosen_pos, ascending=True),
                        _rank01(d_cur, chosen_pos, ascending=True),
                        _rank01(future_improvement, chosen_pos, ascending=True),
                        _rank01(avg_unvisited, chosen_pos, ascending=False),
                        _norm01(chosen_score, score),
                        _z(chosen_score, score),
                        _norm01(float(avg_unvisited[chosen_pos]), avg_unvisited),
                        _z(float(avg_unvisited[chosen_pos]), avg_unvisited),
                        float(chosen_pos == int(np.argmin(score))),
                        float(num_unvisited) / max(1.0, float(matrix.shape[0] - 2)),
                    ],
                    dtype=np.float32,
                )
            )
        except Exception:
            rows.append(penalty)
    return np.nan_to_num(
        np.vstack(rows).astype(np.float32), nan=-1.0, posinf=1e6, neginf=-1e6
    )


def _load_baseline_extractor(reload_module: bool = False):
    module_name = (
        f"{__package__}.behavior_feature_extractor"
        if __package__
        else "behavior_feature_extractor"
    )
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    if reload_module:
        module = importlib.reload(module)
    extractor = getattr(module, "extract_behavior_features")
    description = str(getattr(module, "FEATURE_EXTRACTOR_DESCRIPTION", "")).strip()
    return extractor, description


def get_active_extractor():
    """Return the active function, description, and immutable version label."""
    if _ACTIVE_EXTRACTOR is None:
        extractor, description = _load_baseline_extractor()
        return extractor, description, "baseline_v0"
    return _ACTIVE_EXTRACTOR, str(_ACTIVE_DESCRIPTION or ""), str(_ACTIVE_VERSION)


def set_active_extractor(extractor, description: str, version: str):
    """Activate a validated extractor without modifying the baseline file."""
    if not callable(extractor):
        raise TypeError("Active behavior extractor must be callable.")
    global _ACTIVE_EXTRACTOR, _ACTIVE_DESCRIPTION, _ACTIVE_VERSION
    _ACTIVE_EXTRACTOR = extractor
    _ACTIVE_DESCRIPTION = str(description or "").strip()
    _ACTIVE_VERSION = str(version or "evolved")


def reset_to_baseline():
    """Restore the original extractor for the current process."""
    global _ACTIVE_EXTRACTOR, _ACTIVE_DESCRIPTION, _ACTIVE_VERSION
    _ACTIVE_EXTRACTOR = None
    _ACTIVE_DESCRIPTION = None
    _ACTIVE_VERSION = "baseline_v0"


def reload_baseline_extractor():
    """Reload and return the immutable baseline function and description."""
    return _load_baseline_extractor(reload_module=True)


def _extractor_registry() -> Dict[str, Callable]:
    evolvable_extractor, _, _ = get_active_extractor()
    return {
        "generic": behavior_matrix_generic,
        "future": behavior_matrix_future_reduction,
        "dynamic_penalty": behavior_matrix_dynamic_penalty,
        DEFAULT_EXTRACTOR_NAME: evolvable_extractor,
    }


def describe_extractors(extractor_names: List[str]) -> str:
    _, evolvable_description, _ = get_active_extractor()
    descriptions = dict(FIXED_EXTRACTOR_DESCRIPTIONS)
    descriptions[DEFAULT_EXTRACTOR_NAME] = evolvable_description
    return "\n\n".join(f"[{name}]\n{descriptions[name]}" for name in extractor_names)


def get_active_extractor_version() -> str:
    return get_active_extractor()[2]


def parse_extractor_names(text: str) -> List[str]:
    names = [part.strip() for part in str(text).replace(";", ",").split(",") if part.strip()]
    if not names:
        raise ValueError("No behavior extractors provided.")
    available = _extractor_registry()
    if "all" in names:
        names = list(available)
    bad = [name for name in names if name not in available]
    if bad:
        raise ValueError(
            f"Unknown behavior extractors {bad}. Available: {sorted(available)} or all."
        )
    return list(dict.fromkeys(names))


def _validate_feature_matrix(matrix: np.ndarray, n_probes: int, extractor_name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    expected_shape = (int(n_probes), FEATURE_DIM_PER_PROBE)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Extractor {extractor_name!r} returned shape {matrix.shape}; "
            f"expected {expected_shape}."
        )
    return np.nan_to_num(matrix, nan=-1.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def behavior_matrix_for_extractors(
    select_next_node_fn,
    distance_matrix: np.ndarray,
    extractor_names: List[str],
    n_probes: int = 64,
    seed: int = 42,
    probe_timeout: float = None,
) -> np.ndarray:
    registry = _extractor_registry()
    mats = []
    for name in extractor_names:
        matrix = registry[name](
            select_next_node_fn,
            distance_matrix,
            n_probes=n_probes,
            seed=seed,
            probe_timeout=probe_timeout,
        )
        mats.append(_validate_feature_matrix(matrix, n_probes, name))
    return np.concatenate(mats, axis=1).astype(np.float32)
