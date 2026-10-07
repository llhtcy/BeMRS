"""Experiment-only uniform feasible rollouts and uniformly sampled states."""
import hashlib
import json
import logging
import time
import numpy as np


def make_random_states(self, instance, seed):
    rng = np.random.default_rng(int(seed))
    target = len(self.stage_ratios) * self.probes_per_stage
    n = len(instance['prize'])
    states, seen = [], set()
    # Generate reachable states, not arbitrary inconsistent visited/budget data.
    for attempt in range(max(100, target * 20)):
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current, travelled = 0, 0.0
        trace = []
        while True:
            feasible = self._feasible_candidates(instance, current, visited, travelled)
            if len(feasible) >= 2:
                trace.append(dict(current=current, visited=visited.copy(),
                                  travel_distance=travelled,
                                  committed_ratio=self._committed_ratio(instance,current,travelled)))
            if not len(feasible):
                break
            chosen = int(rng.choice(feasible))
            travelled += float(instance['distance'][current,chosen])
            visited[chosen] = True
            current = chosen
        if not trace:
            continue
        # Randomly sample a state from each random trajectory, without stage
        # targets or ranking by reward/distance/efficiency.
        state = trace[int(rng.integers(len(trace)))]
        key = (state['current'], state['visited'].tobytes(), state['travel_distance'])
        if key in seen:
            continue
        seen.add(key)
        state.update(stage_ratio=state['committed_ratio'],reference_mode='uniform_feasible_random')
        states.append(state)
        if len(states) == target:
            break
    if len(states) != target:
        raise ValueError(f'Only {len(states)}/{target} unique random reachable OP states available')
    serial = [dict(current=s['current'],visited=np.flatnonzero(s['visited']).tolist(),
                   travel_distance=s['travel_distance'],progress=s['committed_ratio']) for s in states]
    digest = hashlib.sha256(json.dumps(serial,sort_keys=True).encode()).hexdigest()
    logging.info('[RandomOPProbes] seed=%s states=%s unique=%s bank_hash=%s progress_range=%s selection=uniform_feasible_then_uniform_trace_state',
                 seed,len(states),len(seen),digest,
                 [min(s['committed_ratio'] for s in states),max(s['committed_ratio'] for s in states)])
    return tuple(states)


def make_stratified_states(self, instance, seed):
    started = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    count = self.probes_per_stage
    pools = [[], [], []]
    seen = set()
    n = len(instance['prize'])
    limit = max(100, 3 * count * 20)
    for attempt in range(limit):
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current, travelled = 0, 0.0
        while True:
            feasible = self._feasible_candidates(instance, current, visited, travelled)
            if len(feasible) >= 2:
                progress = self._committed_ratio(instance, current, travelled)
                key = (current, tuple(map(int, feasible)))
                if key not in seen:
                    seen.add(key)
                    layer = min(2, int(progress * 3))
                    pools[layer].append(dict(current=current, visited=visited.copy(),
                        travel_distance=travelled, committed_ratio=progress,
                        stage_ratio=progress, reference_mode='stratified_uniform_random',
                        probe_layer=layer, decision_key=key))
            if not len(feasible):
                break
            chosen = int(rng.choice(feasible))
            travelled += float(instance['distance'][current, chosen])
            visited[chosen] = True
            current = chosen
        # Seek enough different current nodes, not just enough near-duplicate
        # masks. If impossible within the cap, sample existing nodes again.
        if all(len(pool) >= count and len({s['current'] for s in pool}) >= count for pool in pools):
            break
    if any(len(pool) < count for pool in pools):
        raise ValueError(f'Random stratified probe shortage after {attempt+1} paths: {[len(p) for p in pools]}, need {count} per layer')
    selected = []
    used_nodes = set()
    for pool in pools:
        remaining = list(pool)
        layer_nodes = set()
        for _ in range(count):
            fresh = [s for s in remaining if s['current'] not in used_nodes]
            local_fresh = [s for s in remaining if s['current'] not in layer_nodes]
            options = fresh or local_fresh or remaining
            state = options[int(rng.integers(len(options)))]
            selected.append(state)
            used_nodes.add(state['current'])
            layer_nodes.add(state['current'])
            remaining = [s for s in remaining if s is not state]
    serial = [dict(current=s['current'],actions=list(s['decision_key'][1]),
                   travel_distance=s['travel_distance'],progress=s['committed_ratio'],layer=s['probe_layer']) for s in selected]
    digest = hashlib.sha256(json.dumps(serial,sort_keys=True).encode()).hexdigest()
    logging.info('[StratifiedRandomOPProbes] seed=%s states=%s layer_counts=%s pool_counts=%s paths=%s build_time_s=%.6f unique_decision_keys=%s unique_current_nodes=%s bank_hash=%s',
                 seed,len(selected),[count]*3,[len(p) for p in pools],attempt+1,time.perf_counter()-started,
                 len({s['decision_key'] for s in selected}),len(used_nodes),digest)
    return tuple(selected)


def install(stratified=False):
    from bemrs.op_aco_behavior import OPACOBehaviorEmbedder
    OPACOBehaviorEmbedder._make_fixed_states = make_stratified_states if stratified else make_random_states
