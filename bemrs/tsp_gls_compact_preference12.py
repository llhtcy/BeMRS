"""12-dimensional compact preference descriptor for TSP-GLS.

The legacy full descriptor remains in :mod:`tsp_gls_behavior`, while the
offline 416D/52D compact analysis remains in ``scripts/``.  This module is a
self-contained online version of the compact probe calculations so a server
package does not need to import the offline scripts directory.

Each fixed probe first produces the same 26 preference features used by the
offline compact extractor.  Those rows are then mapped to twelve semantic
features and pooled by a coordinate-wise mean across the configured probes.
No full guided-local-search rollout is performed.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

try:
    from .tsp_gls_behavior import (
        GUIDE_FEATURE_NAMES,
        PENALTY_STATE_NAMES,
        TSPGLSBehaviorEmbedder,
        TSPGLSBehaviorTimeout,
        _percentile_ranks,
        _safe_correlation,
    )
except ImportError:
    from tsp_gls_behavior import (  # type: ignore
        GUIDE_FEATURE_NAMES,
        PENALTY_STATE_NAMES,
        TSPGLSBehaviorEmbedder,
        TSPGLSBehaviorTimeout,
        _percentile_ranks,
        _safe_correlation,
    )


STATE_PREFERENCE_NAMES = (
    "long_edge_preference",
    "local_long_edge_preference",
    "two_opt_preference",
    "topk_two_opt_quality",
    "selection_margin",
)
PENALTY_PREFERENCE_STATES = ("patterned", "length_conditioned")
STATE_PAIRS = tuple(combinations(PENALTY_STATE_NAMES, 2))


def compact_feature_names_per_instance():
    """Return the unchanged 26-feature compact probe layout."""
    names = []
    for state_name in PENALTY_STATE_NAMES:
        names.extend(
            f"{state_name}.{feature_name}"
            for feature_name in STATE_PREFERENCE_NAMES
        )
    names.extend(
        f"{state_name}.penalty_preference"
        for state_name in PENALTY_PREFERENCE_STATES
    )
    for left, right in STATE_PAIRS:
        names.append(f"{left}_vs_{right}.ranking_stability")
    for left, right in STATE_PAIRS:
        names.append(f"{left}_vs_{right}.topk_overlap")
    names.extend(f"guide.{name}" for name in GUIDE_FEATURE_NAMES)
    if len(names) != 26:
        raise AssertionError(f"Expected 26 feature names, got {len(names)}")
    return tuple(names)


def _topk_indices(values: np.ndarray, count: int) -> np.ndarray:
    """Return stable descending top-k indices."""
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    return order[-count:][::-1]


def _local_length_anomaly(distance: np.ndarray, tour: np.ndarray) -> np.ndarray:
    """Length divided by the mean length of four neighboring tour edges."""
    tour = np.asarray(tour, dtype=np.int64)
    n = len(tour)
    cycle_lengths = np.asarray(
        [distance[tour[i], tour[(i + 1) % n]] for i in range(n)],
        dtype=np.float64,
    )
    anomaly = np.empty(n - 1, dtype=np.float64)
    for index in range(n - 1):
        neighbors = cycle_lengths[
            [(index - 2) % n, (index - 1) % n, (index + 1) % n, (index + 2) % n]
        ]
        anomaly[index] = cycle_lengths[index] / (float(np.mean(neighbors)) + 1e-12)
    return anomaly


def _penalty_induced_two_opt_opportunity(
    distance: np.ndarray,
    tour: np.ndarray,
    penalty: np.ndarray,
) -> np.ndarray:
    """Best 2-opt gain after one GLS penalty increment on each scanned edge.

    The fixed probe tour is already 2-opt/relocate optimal under raw distance,
    so raw positive gains are normally zero.  The score instead asks the
    GLS-relevant question: if this edge were penalized next, how much 2-opt
    improvement would become available under ``distance + k * penalty``?
    """
    distance = np.asarray(distance, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)
    tour = np.asarray(tour, dtype=np.int64)
    n = len(tour)
    tour_cost = float(
        np.sum(distance[tour, np.roll(tour, -1)], dtype=np.float64)
    )
    penalty_weight = 0.1 * tour_cost / n
    guided = distance + penalty_weight * penalty
    opportunities = np.zeros(n - 1, dtype=np.float64)

    for edge_index in range(n - 1):
        a = int(tour[edge_index])
        b = int(tour[edge_index + 1])
        best_gain = 0.0
        for other_index in range(n):
            if other_index in {
                edge_index,
                (edge_index - 1) % n,
                (edge_index + 1) % n,
            }:
                continue
            c = int(tour[other_index])
            d = int(tour[(other_index + 1) % n])
            removed = guided[a, b] + penalty_weight + guided[c, d]
            added = guided[a, c] + guided[b, d]
            best_gain = max(best_gain, float(removed - added))
        opportunities[edge_index] = best_gain
    return opportunities


COMPACT12_FEATURE_NAMES = (
    "long_edge_preference_mean",
    "long_edge_preference_trend",
    "local_anomalous_long_preference",
    "penalty_preference",
    "two_opt_improvement_preference",
    "preferred_edge_two_opt_quality",
    "selection_margin",
    "utility_ranking_stability",
    "topk_overlap_stability",
    "heuristic_asymmetry",
    "guide_scale",
    "positive_support_ratio",
)


class TSPGLSCompactPreference12Embedder(TSPGLSBehaviorEmbedder):
    """Return a 12D mean-pooled compact TSP-GLS behavior descriptor.

    ``encode_both`` retains the intermediate twelve-feature row for each probe
    as a flattened diagnostic vector, followed by the final mean-pooled 12D
    vector.  ``encode_single`` and ``encode`` expose only the final vector for
    the online behavior filter.
    """

    output_dim = len(COMPACT12_FEATURE_NAMES)
    summary_dim = output_dim
    extractor_version = "tsp_gls_compact_preference12_mean_v1"

    def __init__(self, *args, topk_fraction: float = 0.2, **kwargs):
        self.topk_fraction = float(topk_fraction)
        if not 0.0 < self.topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0, 1].")
        super().__init__(*args, **kwargs)
        # The parent initializes the official probe bank and state matrices.
        # The online output is the mean-pooled semantic vector, not the legacy
        # 1008D parent layout.
        self.base_features_per_instance = 26
        self.features_per_instance = len(COMPACT12_FEATURE_NAMES)
        self.per_probe_dim = self.n_matrices * len(COMPACT12_FEATURE_NAMES)
        self.output_dim = len(COMPACT12_FEATURE_NAMES)
        self.summary_dim = self.output_dim
        self.extractor_version = "tsp_gls_compact_preference12_mean_v1"
        self.feature_names = COMPACT12_FEATURE_NAMES
        self.feature_names_per_probe = COMPACT12_FEATURE_NAMES
        self.base_feature_names_per_instance = compact_feature_names_per_instance()
        self.extractor_description = (
            "Deterministic TSP-GLS compact preference descriptor: twelve "
            "semantic probe features, mean-pooled across the configured "
            "fixed TSP50 probes."
        )
        self._probe_contexts = tuple(
            self._make_probe_context(distance, state_bank)
            for distance, state_bank in zip(self._distances, self._state_banks)
        )

    def _make_probe_context(self, distance, state_bank):
        tour, penalties = state_bank
        current = tour[:-1]
        following = tour[1:]
        edge_lengths = np.asarray(distance[current, following], dtype=np.float64)
        length_ranks = _percentile_ranks(edge_lengths)
        local_anomaly_ranks = _percentile_ranks(
            _local_length_anomaly(distance, tour)
        )
        two_opt_ranks = tuple(
            _percentile_ranks(
                _penalty_induced_two_opt_opportunity(distance, tour, penalty)
            )
            for penalty in penalties
        )
        topk_count = max(1, int(math.ceil(self.topk_fraction * len(current))))
        return {
            "tour": tour,
            "penalties": penalties,
            "length_ranks": length_ranks,
            "local_anomaly_ranks": local_anomaly_ranks,
            "two_opt_ranks": two_opt_ranks,
            "topk_count": topk_count,
        }

    @staticmethod
    def _margin(normalized: np.ndarray) -> float:
        maximum = float(np.max(normalized))
        if maximum <= 0.0:
            return max(0.0, -maximum)
        selected_index = int(np.argmax(normalized))
        if len(normalized) < 2:
            return maximum
        runner_up = max(0.0, float(np.max(np.delete(normalized, selected_index))))
        return maximum - runner_up

    def _instance_features(self, guide, distance, context):
        tour = context["tour"]
        current = tour[:-1]
        following = tour[1:]
        state_utility_ranks = []
        state_topk = []
        state_features = []
        penalty_preferences = []

        for state_index, (state_name, penalty) in enumerate(
            zip(PENALTY_STATE_NAMES, context["penalties"])
        ):
            utility = np.asarray(
                guide[current, following] / (1.0 + penalty[current, following]),
                dtype=np.float64,
            )
            scale = float(np.max(np.abs(utility)))
            normalized = utility / scale if scale > 0.0 else utility.copy()
            utility_ranks = _percentile_ranks(normalized)
            topk = _topk_indices(normalized, context["topk_count"])
            two_opt_ranks = context["two_opt_ranks"][state_index]

            state_utility_ranks.append(utility_ranks)
            state_topk.append(set(int(index) for index in topk))
            state_features.extend(
                [
                    _safe_correlation(utility_ranks, context["length_ranks"]),
                    _safe_correlation(
                        utility_ranks, context["local_anomaly_ranks"]
                    ),
                    _safe_correlation(utility_ranks, two_opt_ranks),
                    float(np.mean(two_opt_ranks[topk])),
                    self._margin(normalized),
                ]
            )

            if state_name in PENALTY_PREFERENCE_STATES:
                edge_penalty = np.asarray(
                    penalty[current, following], dtype=np.float64
                )
                penalized = edge_penalty > 0.0
                if np.any(penalized) and np.any(~penalized):
                    preference = float(
                        np.mean(utility_ranks[penalized])
                        - np.mean(utility_ranks[~penalized])
                    )
                else:
                    preference = 0.0
                penalty_preferences.append(preference)

        ranking_stability = [
            _safe_correlation(state_utility_ranks[left], state_utility_ranks[right])
            for left, right in combinations(range(len(PENALTY_STATE_NAMES)), 2)
        ]
        topk_overlap = []
        for left, right in combinations(range(len(PENALTY_STATE_NAMES)), 2):
            union = state_topk[left] | state_topk[right]
            overlap = len(state_topk[left] & state_topk[right]) / len(union)
            topk_overlap.append(float(overlap))

        feature = np.asarray(
            [
                *state_features,
                *penalty_preferences,
                *ranking_stability,
                *topk_overlap,
                *self._guide_features(guide).tolist(),
            ],
            dtype=np.float32,
        )
        if feature.shape != (26,):
            raise AssertionError(f"Expected 26 compact features, got {feature.shape}")
        return feature

    @staticmethod
    def _semantic_probe_features(matrix: np.ndarray) -> np.ndarray:
        """Convert ``(n_probes, 26)`` compact rows to ``(n_probes, 12)``."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != 26:
            raise ValueError(f"Expected compact probe matrix (n, 26), got {matrix.shape}")

        # Per-probe layout: three states x five values, two penalty preference
        # values, three ranking stabilities, three top-k overlaps, then guide
        # [scale, positive support, asymmetry].
        semantic = np.column_stack(
            [
                matrix[:, [0, 5, 10]].mean(axis=1),
                (matrix[:, 10] - matrix[:, 0]) / 2.0,
                matrix[:, [1, 6, 11]].mean(axis=1),
                matrix[:, [15, 16]].mean(axis=1),
                matrix[:, [2, 7, 12]].mean(axis=1),
                matrix[:, [3, 8, 13]].mean(axis=1),
                matrix[:, [4, 9, 14]].mean(axis=1),
                matrix[:, 17:20].mean(axis=1),
                matrix[:, 20:23].mean(axis=1),
                matrix[:, 25],
                matrix[:, 23],
                matrix[:, 24],
            ]
        )
        if semantic.shape[1] != len(COMPACT12_FEATURE_NAMES):
            raise AssertionError(f"Expected 12 semantic features, got {semantic.shape}")
        return semantic

    def _fallback_features(self):
        return (
            np.full(self.per_probe_dim, -1.0, dtype=np.float32),
            np.full(self.summary_dim, -1.0, dtype=np.float32),
        )

    def encode_both(self, text: str):
        """Return ``(per-probe12, mean-pooled12)`` or failure sentinels."""
        import time

        started = time.monotonic()
        try:
            function = self._compile_heuristic(text)
            rows = []
            for instance_index, (distance, context) in enumerate(
                zip(self._distances, self._probe_contexts)
            ):
                if (
                    self.encode_timeout > 0.0
                    and time.monotonic() - started > self.encode_timeout
                ):
                    raise TSPGLSBehaviorTimeout("TSP-GLS compact12 encoding timeout")
                guide = self._call_heuristic(function, distance, instance_index)
                rows.append(self._instance_features(guide, distance, context))
            matrix26 = np.vstack(rows).astype(np.float64)
            semantic = self._semantic_probe_features(matrix26)
            per_probe = semantic.reshape(-1).astype(np.float32)
            summary = np.mean(semantic, axis=0, dtype=np.float64).astype(np.float32)
            if (
                per_probe.shape != (self.per_probe_dim,)
                or summary.shape != (self.summary_dim,)
                or not np.all(np.isfinite(per_probe))
                or not np.all(np.isfinite(summary))
            ):
                raise ValueError("Invalid 12D compact TSP-GLS preference feature")
            return per_probe, summary
        except TSPGLSBehaviorTimeout:
            return self._fallback_features()
        except Exception:
            return self._fallback_features()

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode_both(text)[1]


__all__ = [
    "COMPACT12_FEATURE_NAMES",
    "TSPGLSCompactPreference12Embedder",
    "compact_feature_names_per_instance",
    "_penalty_induced_two_opt_opportunity",
]
