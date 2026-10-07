"""CVRP entropy/time plus existing distance, savings and demand preferences."""
import hashlib
import logging
import time

import numpy as np

from .cvrp_aco_behavior import CVRPBehaviorTimeout, _hard_timeout
from .deterministic_behavior_rng import fixed_candidate_random_state

NAMES = ('decision_entropy', 'feature_extraction_time_s', 'edge_rank_mean',
         'savings_rank_mean', 'demand_rank_mean')


class CVRPEntropyCostEmbedder:
    """Same timing definition as OP: core extraction, without bank setup/logs.

    Calculate only per-state core descriptors and average across probes and
    instances. Do not calculate slopes, matrix asymmetry, or perturbations.
    """
    def __init__(self, encoder, group='entropy_time_preferences5'):
        if group not in ('entropy_time2', 'entropy_time_preferences5'):
            raise ValueError(group)
        self.encoder = encoder
        self.output_dim = 2 if group == 'entropy_time2' else 5
        self.feature_names = NAMES[:self.output_dim]
        self.extractor_version = f'cvrp_{group}_{encoder.probe_mode}_v1'
        self.extractor_description = 'Entropy, extraction seconds, distance/savings/demand mean preferences'
        self.last_diagnostics = {}
        with fixed_candidate_random_state('cvrp_cost_preload', encoder.seed):
            pass
        logging.info('[CVRPEntropyCostFeatures] group=%s dimension=%s names=%s '
                     'probe_mode=%s time_unit=s timing_protocol=core_no_slopes',
                     group, self.output_dim, self.feature_names, encoder.probe_mode)

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def _fallback_feature(self):
        return np.full(self.output_dim, -1, dtype=np.float32)

    def encode_single(self, code):
        started = time.perf_counter()
        diagnostics = dict(dimension=self.output_dim, time_unit='s')
        try:
            with fixed_candidate_random_state(code, self.encoder.seed):
                with _hard_timeout(self.encoder.encode_timeout):
                    function = self.encoder._compile_heuristic(code)
                    parts = []
                    for instance, states in zip(self.encoder._instances, self.encoder._state_banks):
                        self._check_deadline(started)
                        heuristic = self.encoder._call_heuristic(function, instance)
                        self._check_deadline(started)
                        rows = np.vstack([self.encoder._state_core_features(heuristic, instance, state)
                                          for state in states])
                        parts.append(np.mean(rows[:, [6, 0, 1, 2]], axis=0))
                    means = np.mean(parts, axis=0)
            elapsed = time.perf_counter() - started
            self._check_deadline(started)
            complete = np.asarray([means[0], elapsed, *means[1:]], dtype=np.float32)
            if not np.isfinite(complete).all():
                raise ValueError('Nonfinite CVRP entropy/cost features')
            feature = complete[:self.output_dim].copy()
            diagnostics.update(status='ok', features=dict(zip(self.feature_names, map(float, feature))),
                               probes=sum(len(states) for states in self.encoder._state_banks))
        except Exception as error:
            feature = self._fallback_feature()
            diagnostics.update(status='failed', error=f'{type(error).__name__}: {error}',
                               feature_extraction_time_s=time.perf_counter()-started)
        self.last_diagnostics = diagnostics
        logging.info('[CVRPEntropyCostFeatures] code=%s metrics=%s',
                     hashlib.sha256(code.encode()).hexdigest()[:16], diagnostics)
        return feature

    def _check_deadline(self, started):
        if self.encoder.encode_timeout > 0 and time.perf_counter()-started > self.encoder.encode_timeout:
            raise CVRPBehaviorTimeout('CVRP entropy/cost encoding deadline exceeded')

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return (np.vstack([self.encode_single(text) for text in texts]) if texts
                else np.empty((0, self.output_dim), dtype=np.float32))

    def finetune(self, *args, **kwargs):
        pass
