"""OP decision entropy, extraction cost and joint-input rank stability.

The fixed bank is shared by candidates. Geometry is perturbed coherently,
not by independent noise on distance-matrix entries. No objective is evaluated.
"""
import hashlib
import logging
import time

import numpy as np

from .deterministic_behavior_rng import fixed_candidate_random_state
from .op_aco_behavior import (
    REFERENCE_MODES, OPBehaviorTimeout, _distribution, _hard_timeout,
    _percentile_ranks,
)

FEATURE_NAMES = ('decision_entropy', 'feature_extraction_time_s', 'input_stability')
PERTURBATION_FRACTION = 0.05


def perturb_state(instance, state, path, noise, fraction=PERTURBATION_FRACTION):
    """Perturb all continuous inputs; retain the discrete route prefix."""
    n = len(instance['prize'])
    scale = np.median(instance['distance'][~np.eye(n, dtype=bool)])
    coordinates = instance['coordinate'] + fraction * scale * noise['coordinates']
    coordinates[0] = instance['coordinate'][0]
    distance = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    np.fill_diagonal(distance, 1e9)
    changed = dict(instance, coordinate=coordinates, distance=distance,
                   prize=instance['prize'] * (1 + fraction * noise['prize']),
                   max_len=instance['max_len'] * (1 + fraction * noise['budget']))
    changed_state = dict(state, travel_distance=float(sum(
        distance[a, b] for a, b in zip(path, path[1:]))))
    return changed, changed_state


class OPResponseBehaviorEmbedder:
    """Three actual features, not a projection of the old preference vector.

Time is wall-clock seconds for compilation, paired calls and statistics,
excluding bank setup and logging. It is intentionally not deterministic.
"""
    output_dim = 3
    feature_names = FEATURE_NAMES
    extractor_version = 'op_response3_joint5pct_v1'
    extractor_description = 'Normalized entropy, extraction seconds, joint-input rank stability'

    def __init__(self, encoder):
        self.encoder = encoder
        # Load RNG infrastructure once; first-candidate timing must not include
        # the lazy Torch import used by the shared random-state sandbox.
        with fixed_candidate_random_state('op_response_preload', encoder.seed):
            pass
        self.last_diagnostics = {}
        self.bank = []
        for iid, (instance, states) in enumerate(zip(encoder._instances, encoder._state_banks)):
            traces = {mode: encoder._reference_trace(instance, mode, encoder.seed + iid*1009 + j*97)
                      for j, mode in enumerate(REFERENCE_MODES[:encoder.probes_per_stage])}
            cells = []
            for pid, state in enumerate(states):
                trace = traces[state['reference_mode']]
                index = next(i for i, row in enumerate(trace)
                             if row['current'] == state['current']
                             and np.array_equal(row['visited'], state['visited'])
                             and np.isclose(row['travel_distance'], state['travel_distance'], atol=1e-12, rtol=0))
                path = [row['current'] for row in trace[:index+1]]
                length = sum(instance['distance'][a,b] for a,b in zip(path,path[1:]))
                if not np.isclose(length, state['travel_distance'], atol=1e-10, rtol=0):
                    raise ValueError('OP probe prefix reconstruction failed')
                rng = np.random.default_rng(encoder.seed + iid*1009 + pid*97)
                noise = dict(prize=rng.uniform(-1,1,encoder.n_nodes),
                             coordinates=rng.uniform(-1,1,(encoder.n_nodes,2)),
                             budget=float(rng.uniform(-1,1)))
                changed, changed_state = perturb_state(instance, state, path, noise)
                original_actions = encoder._feasible_candidates(instance, state['current'], state['visited'], state['travel_distance'])
                changed_actions = encoder._feasible_candidates(changed, state['current'], changed_state['visited'], changed_state['travel_distance'])
                common = np.intersect1d(original_actions, changed_actions)
                cells.append((state, changed, original_actions, common))
            self.bank.append((instance, cells))
        logging.info('[ResponseFeatures] dimension=3 names=%s noise=5%% matrices=%s probes=%s time_unit=s',
                     FEATURE_NAMES, len(self.bank), sum(len(c) for _,c in self.bank))

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def _fallback_feature(self):
        return np.full(3, -1.0, dtype=np.float32)

    def encode_single(self, code):
        started = time.perf_counter()
        diagnostics = dict(valid_pairs=0, total_pairs=sum(len(c) for _,c in self.bank),
                           perturbation_fraction=PERTURBATION_FRACTION)
        try:
            with fixed_candidate_random_state('op_response_shared_rng', self.encoder.seed):
                with _hard_timeout(self.encoder.encode_timeout):
                    fn = self.encoder._compile_heuristic(code)
                entropies, stabilities, retained = [], [], []
                for iid, (instance, cells) in enumerate(self.bank):
                    def call(inputs):
                        remaining = self.encoder.encode_timeout - (time.perf_counter()-started)
                        if self.encoder.encode_timeout > 0 and remaining <= 0:
                            raise OPBehaviorTimeout('Response feature encoding deadline exceeded')
                        # Common random numbers for original and changed calls.
                        with fixed_candidate_random_state('op_response_shared_rng', self.encoder.seed+iid*1009):
                            limit = self.encoder.heuristic_timeout
                            if self.encoder.encode_timeout > 0:
                                limit = min(limit, remaining) if limit > 0 else remaining
                            with _hard_timeout(limit):
                                raw = fn(inputs['prize'].copy(), inputs['distance'].copy(), float(inputs['max_len']))
                            raw = np.asarray(raw,dtype=np.float64)
                            if raw.shape != (self.encoder.n_nodes,self.encoder.n_nodes) or not np.isfinite(raw).all():
                                raise ValueError('Invalid OP heuristic matrix')
                            return np.maximum(raw+1e-9,1e-9)
                    original = call(instance)
                    for state, changed, actions, common in cells:
                        current = state['current']
                        if len(actions) >= 2:
                            p = _distribution(original[current, actions])
                            positive = p[p > 0]
                            entropies.append(float(-np.sum(positive*np.log(positive))/np.log(len(actions))))
                        retained.append(len(common)/max(1,len(actions)))
                        if len(common) < 2:
                            continue
                        modified = call(changed)
                        difference = np.mean(np.abs(_percentile_ranks(original[current,common]) -
                                                    _percentile_ranks(modified[current,common])))
                        stabilities.append(float(1-difference))
                if not entropies or not stabilities:
                    raise ValueError('No valid entropy/stability probes')
                entropy, stability = float(np.mean(entropies)), float(np.mean(stabilities))
                diagnostics.update(valid_pairs=len(stabilities), retained_fraction=float(np.mean(retained)))
            elapsed = time.perf_counter()-started
            if self.encoder.encode_timeout > 0 and elapsed > self.encoder.encode_timeout:
                raise OPBehaviorTimeout('Response feature encoding deadline exceeded')
            feature = np.array([entropy, elapsed, stability], dtype=np.float32)
            if not np.isfinite(feature).all():
                raise ValueError('Nonfinite response features')
            diagnostics.update(status='ok', entropy=entropy, feature_extraction_time_s=elapsed, stability=stability)
        except Exception as error:
            feature = self._fallback_feature()
            diagnostics.update(status='failed', error=f'{type(error).__name__}: {error}',
                               feature_extraction_time_s=time.perf_counter()-started)
        self.last_diagnostics = diagnostics
        logging.info('[ResponseFeatures] code=%s metrics=%s', hashlib.sha256(code.encode()).hexdigest()[:16], diagnostics)
        return feature

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.vstack([self.encode_single(code) for code in texts]) if texts else np.empty((0,3),dtype=np.float32)

    def finetune(self, *args, **kwargs):
        pass
