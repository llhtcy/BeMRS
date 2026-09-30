"""Diverse behavior feature components for constructive TSP heuristics.

The module never executes candidate code. A fixed runtime performs safe
rollouts and passes each valid selected node to ``decision_features``. Feature
selection can then use named components without allowing an LLM to rewrite
timeouts, probe generation, or rollout control flow.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


EPS = 1e-12
LIBRARY_VERSION = "1.1"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    scope: str
    category: str
    description: str


_SPECS = (
    # One-step decision descriptors. Every item depends on the selected node.
    FeatureSpec("edge_length_ratio", "decision", "local_cost", "Chosen edge length divided by the robust instance distance scale."),
    FeatureSpec("edge_length_rank", "decision", "local_order", "Percentile rank of the chosen edge among available candidates."),
    FeatureSpec("destination_progress_ratio", "decision", "goal_progress", "Signed reduction in destination distance caused by the choice."),
    FeatureSpec("destination_distance_rank", "decision", "goal_order", "Percentile rank of the chosen node by destination proximity."),
    FeatureSpec("destination_alignment_cosine", "decision", "geometry", "Cosine alignment of the selected edge with the direction to the destination."),
    FeatureSpec("connectivity_rank", "decision", "global_connectivity", "Rank of the selected node by mean distance to remaining candidates."),
    FeatureSpec("local_isolation_ratio", "decision", "local_density", "Selected node's nearest remaining-neighbor distance relative to candidate median isolation."),
    FeatureSpec("two_step_regret_ratio", "decision", "lookahead", "Excess selected edge plus best continuation cost over the best two-step candidate."),
    FeatureSpec("distance_profile_shift", "decision", "region_change", "Change between current-node and selected-node distance profiles over remaining candidates."),
    FeatureSpec("pareto_dominated_fraction", "decision", "multiobjective", "Fraction of candidates dominating the choice in cost, destination distance, and connectivity."),
    # Short-rollout descriptors. These summarize dynamics rather than repeating
    # the final value of an existing one-step measurement.
    FeatureSpec("trace_valid_fraction", "trace", "reliability", "Fraction of requested rollout steps that produced valid decisions."),
    FeatureSpec("trace_edge_ratio_mean", "trace", "cost_level", "Mean normalized selected-edge cost over the rollout."),
    FeatureSpec("trace_edge_ratio_cv", "trace", "cost_volatility", "Coefficient of variation of normalized selected-edge cost."),
    FeatureSpec("trace_edge_ratio_trend", "trace", "cost_dynamics", "Linear trend in selected-edge cost through the rollout."),
    FeatureSpec("trace_progress_total", "trace", "goal_progress", "Total signed destination progress accumulated by the rollout."),
    FeatureSpec("trace_backward_fraction", "trace", "goal_risk", "Fraction of valid steps that move farther from the destination."),
    FeatureSpec("trace_edge_rank_dispersion", "trace", "choice_consistency", "Variation in local edge-rank preference across steps."),
    FeatureSpec("trace_connectivity_trend", "trace", "connectivity_dynamics", "Linear trend in selected-node connectivity rank."),
    FeatureSpec("trace_regret_peak", "trace", "lookahead_risk", "Largest two-step regret incurred during the rollout."),
    FeatureSpec("trace_isolation_peak", "trace", "isolation_risk", "Largest local-isolation ratio encountered during the rollout."),
    FeatureSpec("trace_profile_shift_mean", "trace", "region_change", "Mean geometric region shift caused by selected nodes."),
    FeatureSpec("trace_dominated_fraction_mean", "trace", "multiobjective", "Mean fraction of candidates that Pareto-dominate each selected node."),
    # Fixed-state, stage-conditioned policy descriptors. These are invariant
    # to node labels and do not follow trajectories induced by the candidate.
    FeatureSpec("stage_edge_rank_mean", "stage_trend", "local_order", "Mean selected-edge percentile over fixed early, middle, and late states."),
    FeatureSpec("stage_edge_rank_slope", "stage_trend", "local_order_trend", "Change in selected-edge percentile from early to late fixed states."),
    FeatureSpec("stage_destination_rank_mean", "stage_trend", "goal_order", "Mean destination-proximity percentile over fixed states."),
    FeatureSpec("stage_destination_rank_slope", "stage_trend", "goal_order_trend", "Change in destination-proximity preference from early to late states."),
    FeatureSpec("stage_connectivity_rank_mean", "stage_trend", "global_connectivity", "Mean selected-node connectivity percentile over fixed states."),
    FeatureSpec("stage_connectivity_rank_slope", "stage_trend", "connectivity_trend", "Change in connectivity preference from early to late states."),
    FeatureSpec("stage_positive_progress_rate", "stage_trend", "goal_progress", "Fraction of fixed states where the choice moves closer to the destination."),
    FeatureSpec("stage_positive_progress_slope", "stage_trend", "goal_progress_trend", "Change in positive-progress rate from early to late states."),
    FeatureSpec("stage_two_step_regret_median", "stage_trend", "lookahead_level", "Median normalized two-step regret over fixed states."),
    FeatureSpec("stage_two_step_regret_q90", "stage_trend", "lookahead_tail", "Ninetieth percentile of normalized two-step regret over fixed states."),
    FeatureSpec("stage_pareto_efficient_rate", "stage_trend", "multiobjective", "Fraction of fixed-state choices not Pareto-dominated by another candidate."),
    FeatureSpec("stage_choice_rank_entropy", "stage_trend", "policy_dispersion", "Normalized entropy of selected-edge percentiles over fixed states."),
)

FEATURE_SPECS = {spec.name: spec for spec in _SPECS}
if len(FEATURE_SPECS) != len(_SPECS):
    raise RuntimeError("Constructive TSP feature names must be unique.")

DECISION_FEATURE_NAMES = tuple(spec.name for spec in _SPECS if spec.scope == "decision")
TRACE_FEATURE_NAMES = tuple(spec.name for spec in _SPECS if spec.scope == "trace")
STAGE_TREND_FEATURE_NAMES = tuple(
    spec.name for spec in _SPECS if spec.scope == "stage_trend"
)


def list_feature_specs(scope: Optional[str] = None):
    """Return JSON-serializable metadata for the selectable feature bank."""
    if scope not in {None, "decision", "trace", "stage_trend"}:
        raise ValueError(
            "scope must be None, 'decision', 'trace', or 'stage_trend'."
        )
    return [asdict(spec) for spec in _SPECS if scope is None or spec.scope == scope]


def feature_vector(
    feature_map: Mapping[str, float],
    selected_names: Sequence[str],
    missing_value: Optional[float] = None,
) -> np.ndarray:
    """Assemble a deterministic vector from a selected list of feature names."""
    unknown = [name for name in selected_names if name not in FEATURE_SPECS]
    if unknown:
        raise KeyError(f"Unknown constructive TSP features: {unknown}")
    values = []
    for name in selected_names:
        if name not in feature_map:
            if missing_value is None:
                raise KeyError(f"Feature {name!r} is unavailable in this context.")
            value = float(missing_value)
        else:
            value = float(feature_map[name])
        if not np.isfinite(value):
            value = float(0.0 if missing_value is None else missing_value)
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _square_matrix(distance_matrix) -> np.ndarray:
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distance_matrix must contain finite values.")
    return matrix


def _distance_scale(matrix: np.ndarray) -> float:
    positive = matrix[matrix > 0]
    return max(float(np.median(positive)) if positive.size else 1.0, EPS)


def _rank01(values, selected_index: int, ascending: bool = True) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return 0.0
    order = np.argsort(values, kind="mergesort")
    if not ascending:
        order = order[::-1]
    rank = int(np.where(order == int(selected_index))[0][0])
    return float(rank / (values.size - 1))


def _safe_correlation(left, right) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or np.std(left) < EPS or np.std(right) < EPS:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _linear_trend(values) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return 0.0
    x = np.linspace(0.0, 1.0, values.size)
    centered_x = x - np.mean(x)
    return float(
        np.sum(centered_x * (values - np.mean(values)))
        / (np.sum(centered_x**2) + EPS)
    )


def _candidate_connectivity(candidates: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = []
    for node in candidates:
        others = candidates[candidates != node]
        values.append(float(np.mean(matrix[int(node), others])) if others.size else 0.0)
    return np.asarray(values, dtype=np.float64)


def _candidate_isolation(
    candidates: np.ndarray, matrix: np.ndarray, destination_node: int
) -> np.ndarray:
    values = []
    for node in candidates:
        others = candidates[candidates != node]
        if others.size:
            values.append(float(np.min(matrix[int(node), others])))
        else:
            values.append(float(matrix[int(node), int(destination_node)]))
    return np.asarray(values, dtype=np.float64)


def decision_features(
    current_node: int,
    destination_node: int,
    unvisited_nodes: Iterable[int],
    distance_matrix,
    chosen_node: int,
) -> Dict[str, float]:
    """Compute all nonconstant one-step features for one selected TSP node."""
    matrix = _square_matrix(distance_matrix)
    current_node = int(current_node)
    destination_node = int(destination_node)
    candidates = np.asarray(sorted({int(node) for node in unvisited_nodes}), dtype=int)
    if candidates.size == 0:
        raise ValueError("unvisited_nodes must not be empty.")
    selected = np.flatnonzero(candidates == int(chosen_node))
    if selected.size != 1:
        raise ValueError("chosen_node must be in unvisited_nodes.")
    chosen_index = int(selected[0])
    chosen_node = int(chosen_node)
    scale = _distance_scale(matrix)

    edge_distances = matrix[current_node, candidates]
    destination_distances = matrix[candidates, destination_node]
    connectivity = _candidate_connectivity(candidates, matrix)
    isolation = _candidate_isolation(candidates, matrix, destination_node)

    chosen_edge = float(edge_distances[chosen_index])
    chosen_destination = float(destination_distances[chosen_index])
    current_destination = float(matrix[current_node, destination_node])
    median_isolation = max(float(np.median(isolation)), EPS)

    two_step_costs = []
    for node, edge_cost in zip(candidates, edge_distances):
        others = candidates[candidates != node]
        continuation = (
            float(np.min(matrix[int(node), others]))
            if others.size
            else float(matrix[int(node), destination_node])
        )
        two_step_costs.append(float(edge_cost) + continuation)
    two_step_costs = np.asarray(two_step_costs, dtype=np.float64)

    remaining = candidates[candidates != chosen_node]
    if remaining.size >= 2:
        current_profile = matrix[current_node, remaining]
        chosen_profile = matrix[chosen_node, remaining]
        profile_shift = 0.5 * (1.0 - _safe_correlation(current_profile, chosen_profile))
    else:
        profile_shift = 0.0

    chosen_objectives = np.asarray(
        [
            edge_distances[chosen_index],
            destination_distances[chosen_index],
            connectivity[chosen_index],
        ],
        dtype=np.float64,
    )
    objective_table = np.column_stack(
        [edge_distances, destination_distances, connectivity]
    )
    weakly_better = np.all(objective_table <= chosen_objectives + EPS, axis=1)
    strictly_better = np.any(objective_table < chosen_objectives - EPS, axis=1)
    dominated_fraction = float(np.mean(weakly_better & strictly_better))

    if chosen_edge > EPS and current_destination > EPS:
        alignment = (
            current_destination**2 + chosen_edge**2 - chosen_destination**2
        ) / (2.0 * current_destination * chosen_edge)
        alignment = float(np.clip(alignment, -1.0, 1.0))
    else:
        alignment = 0.0

    return {
        "edge_length_ratio": float(chosen_edge / scale),
        "edge_length_rank": _rank01(edge_distances, chosen_index, ascending=True),
        "destination_progress_ratio": float(
            (current_destination - chosen_destination) / scale
        ),
        "destination_distance_rank": _rank01(
            destination_distances, chosen_index, ascending=True
        ),
        "destination_alignment_cosine": alignment,
        "connectivity_rank": _rank01(connectivity, chosen_index, ascending=True),
        "local_isolation_ratio": float(isolation[chosen_index] / median_isolation),
        "two_step_regret_ratio": float(
            (two_step_costs[chosen_index] - np.min(two_step_costs)) / scale
        ),
        "distance_profile_shift": float(profile_shift),
        "pareto_dominated_fraction": dominated_fraction,
    }


def trace_features(
    decision_records: Sequence[Optional[Mapping[str, float]]],
) -> Dict[str, float]:
    """Summarize a short rollout using complementary level, risk, and trend features."""
    requested_steps = len(decision_records)
    if requested_steps == 0:
        raise ValueError("decision_records must include at least one requested step.")
    valid_rows = [row for row in decision_records if row is not None]
    if not valid_rows:
        return {
            "trace_valid_fraction": 0.0,
            **{name: -1.0 for name in TRACE_FEATURE_NAMES if name != "trace_valid_fraction"},
        }

    def column(name):
        values = np.asarray([float(row[name]) for row in valid_rows], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Trace field {name!r} contains non-finite values.")
        return values

    edge_ratio = column("edge_length_ratio")
    progress = column("destination_progress_ratio")
    edge_rank = column("edge_length_rank")
    connectivity = column("connectivity_rank")
    regret = column("two_step_regret_ratio")
    isolation = column("local_isolation_ratio")
    profile_shift = column("distance_profile_shift")
    dominated = column("pareto_dominated_fraction")

    return {
        "trace_valid_fraction": float(len(valid_rows) / requested_steps),
        "trace_edge_ratio_mean": float(np.mean(edge_ratio)),
        "trace_edge_ratio_cv": float(
            np.std(edge_ratio) / (abs(np.mean(edge_ratio)) + EPS)
        ),
        "trace_edge_ratio_trend": _linear_trend(edge_ratio),
        "trace_progress_total": float(np.sum(progress)),
        "trace_backward_fraction": float(np.mean(progress < 0.0)),
        "trace_edge_rank_dispersion": float(np.std(edge_rank)),
        "trace_connectivity_trend": _linear_trend(connectivity),
        "trace_regret_peak": float(np.max(regret)),
        "trace_isolation_peak": float(np.max(isolation)),
        "trace_profile_shift_mean": float(np.mean(profile_shift)),
        "trace_dominated_fraction_mean": float(np.mean(dominated)),
    }


def stage_trend_features(
    stage_records: Sequence[Sequence[Optional[Mapping[str, float]]]],
) -> Dict[str, float]:
    """Summarize candidate choices on aligned early, middle, and late states.

    Each inner sequence contains independent fixed states from one problem
    stage. Candidate choices must not be used to construct later states.
    """
    if len(stage_records) < 2:
        raise ValueError("stage_records must contain at least two stages.")

    valid_by_stage = [
        [row for row in rows if row is not None] for rows in stage_records
    ]
    all_rows = [row for rows in valid_by_stage for row in rows]
    if not all_rows:
        return {name: -1.0 for name in STAGE_TREND_FEATURE_NAMES}

    stage_positions = np.linspace(0.0, 1.0, len(valid_by_stage))

    def values(rows, name):
        result = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Stage field {name!r} contains non-finite values.")
        return result

    def overall(name):
        return values(all_rows, name)

    def stage_slope(name, transform=None):
        positions = []
        means = []
        for position, rows in zip(stage_positions, valid_by_stage):
            if not rows:
                continue
            column = values(rows, name)
            if transform is not None:
                column = transform(column)
            positions.append(float(position))
            means.append(float(np.mean(column)))
        if len(means) < 2:
            return 0.0
        x = np.asarray(positions, dtype=np.float64)
        y = np.asarray(means, dtype=np.float64)
        centered_x = x - np.mean(x)
        return float(
            np.sum(centered_x * (y - np.mean(y)))
            / (np.sum(centered_x**2) + EPS)
        )

    edge_rank = np.clip(overall("edge_length_rank"), 0.0, 1.0)
    destination_rank = np.clip(overall("destination_distance_rank"), 0.0, 1.0)
    connectivity_rank = np.clip(overall("connectivity_rank"), 0.0, 1.0)
    progress = overall("destination_progress_ratio")
    regret = np.maximum(overall("two_step_regret_ratio"), 0.0)
    dominated = np.clip(overall("pareto_dominated_fraction"), 0.0, 1.0)

    histogram, _ = np.histogram(edge_rank, bins=np.linspace(0.0, 1.0, 6))
    probabilities = histogram.astype(np.float64)
    probabilities /= max(float(np.sum(probabilities)), 1.0)
    nonzero = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(5.0))

    return {
        "stage_edge_rank_mean": float(np.mean(edge_rank)),
        "stage_edge_rank_slope": stage_slope("edge_length_rank"),
        "stage_destination_rank_mean": float(np.mean(destination_rank)),
        "stage_destination_rank_slope": stage_slope("destination_distance_rank"),
        "stage_connectivity_rank_mean": float(np.mean(connectivity_rank)),
        "stage_connectivity_rank_slope": stage_slope("connectivity_rank"),
        "stage_positive_progress_rate": float(np.mean(progress > 0.0)),
        "stage_positive_progress_slope": stage_slope(
            "destination_progress_ratio", lambda column: column > 0.0
        ),
        "stage_two_step_regret_median": float(np.median(regret)),
        "stage_two_step_regret_q90": float(np.quantile(regret, 0.9)),
        "stage_pareto_efficient_rate": float(np.mean(dominated <= EPS)),
        "stage_choice_rank_entropy": entropy,
    }
