"""Controlled OP 2D/5D features, sharing one extraction/timing protocol."""
import hashlib
import logging
import time

import numpy as np

from .deterministic_behavior_rng import fixed_candidate_random_state
from .op_aco_behavior import OPBehaviorTimeout, _hard_timeout

NAMES = ('decision_entropy', 'feature_extraction_time_ratio', 'prize_rank_mean',
         'distance_rank_mean', 'efficiency_rank_mean')


class OPEntropyCostEmbedder:
    """Same five-descriptor computation; only returned columns differ.

    Time excludes bank setup, lazy imports and logging. Both groups calculate
    existing core descriptors, without slopes, matrix diagnostics or noise.
    The time coordinate is elapsed seconds divided by the existing positive
    encoding deadline. Raw seconds remain available in diagnostics.
    """
    def __init__(self, encoder, group):
        if group not in ('entropy_time2', 'entropy_time_preferences5'):
            raise ValueError(group)
        self.encoder = encoder
        self.time_normalization_budget_s = float(encoder.encode_timeout)
        if not np.isfinite(self.time_normalization_budget_s) or self.time_normalization_budget_s <= 0:
            raise ValueError('Normalized OP extraction time requires a positive finite encode_timeout')
        self.output_dim = 2 if group == 'entropy_time2' else 5
        self.feature_names = NAMES[:self.output_dim]
        self.extractor_version = f'op_{group}_shared_protocol_time_ratio_v2'
        self.extractor_description = 'Entropy and normalized extraction time, optionally three mean decision preferences'
        self.last_diagnostics = {}
        with fixed_candidate_random_state('op_cost_preload', encoder.seed):
            pass
        logging.info('[EntropyCostFeatures] group=%s dimension=%s names=%s noise=False '
                     'timing_protocol=shared_core_no_slopes time_unit=s time_feature_unit=ratio '
                     'time_normalization=encode_timeout time_normalization_budget_s=%s extractor=%s',
                     group, self.output_dim, self.feature_names, self.time_normalization_budget_s,
                     self.extractor_version)

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def _fallback_feature(self):
        return np.full(self.output_dim, -1, dtype=np.float32)

    def _normalized_extraction_time(self, elapsed):
        return float(np.clip(elapsed / self.time_normalization_budget_s, 0.0, 1.0))

    def encode_single(self, code):
        started = time.perf_counter()
        diagnostics = dict(dimension=self.output_dim, time_unit='s', time_feature_unit='ratio',
                           time_normalization_budget_s=self.time_normalization_budget_s,
                           extractor_version=self.extractor_version)
        try:
            with fixed_candidate_random_state(code, self.encoder.seed):
                with _hard_timeout(self.encoder.encode_timeout):
                    fn = self.encoder._compile_heuristic(code)
                parts = []
                for instance, states in zip(self.encoder._instances, self.encoder._state_banks):
                    remaining = self.encoder.encode_timeout - (time.perf_counter()-started)
                    if self.encoder.encode_timeout > 0 and remaining <= 0:
                        raise OPBehaviorTimeout('Entropy/cost encoding deadline exceeded')
                    limit = self.encoder.heuristic_timeout
                    if self.encoder.encode_timeout > 0:
                        limit = min(limit, remaining) if limit > 0 else remaining
                    with _hard_timeout(limit):
                        raw = fn(instance['prize'].copy(),instance['distance'].copy(),float(instance['max_len']))
                    raw = np.asarray(raw, dtype=np.float64)
                    if raw.shape != (self.encoder.n_nodes,self.encoder.n_nodes) or not np.isfinite(raw).all():
                        raise ValueError('Invalid OP heuristic matrix')
                    heuristic = np.maximum(raw+1e-9,1e-9)
                    rows = np.vstack([self.encoder._state_core_features(heuristic,instance,state) for state in states])
                    # Existing entropy and reward/distance/efficiency definitions.
                    parts.append(np.mean(rows[:,[5,0,1,2]],axis=0))
                means = np.mean(parts,axis=0)
            elapsed = time.perf_counter()-started
            if self.encoder.encode_timeout > 0 and elapsed > self.encoder.encode_timeout:
                raise OPBehaviorTimeout('Entropy/cost encoding deadline exceeded')
            complete = np.asarray([means[0],self._normalized_extraction_time(elapsed),*means[1:]],dtype=np.float32)
            if not np.isfinite(complete).all():
                raise ValueError('Nonfinite OP entropy/cost features')
            feature = complete[:self.output_dim].copy()
            diagnostics.update(status='ok',feature_extraction_time_s=elapsed,
                               features=dict(zip(self.feature_names,map(float,feature))),
                               probes=sum(len(states) for states in self.encoder._state_banks))
        except Exception as error:
            feature = self._fallback_feature()
            diagnostics.update(status='failed',error=f'{type(error).__name__}: {error}',
                               feature_extraction_time_s=time.perf_counter()-started)
        self.last_diagnostics = diagnostics
        logging.info('[EntropyCostFeatures] code=%s metrics=%s',hashlib.sha256(code.encode()).hexdigest()[:16],diagnostics)
        return feature

    def encode(self, texts, **kwargs):
        if isinstance(texts,str):
            texts=[texts]
        return np.vstack([self.encode_single(text) for text in texts]) if texts else np.empty((0,self.output_dim),dtype=np.float32)

    def finetune(self,*args,**kwargs):
        pass
