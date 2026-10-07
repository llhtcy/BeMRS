"""Reachable CVRP probes from uniformly random feasible route construction."""
import hashlib
import json
import logging
import time

import numpy as np


def make_stratified_states(encoder, instance, seed):
    """Sample equally from three served-customer progress intervals.

    Depot moves follow the real ACO mask: allowed away from the depot, but
    never depot-to-depot while customers remain. Load is accumulated along
    the sampled route and reset at depot returns. No candidate heuristic,
    distance ranking, or demand ranking guides these trajectories.
    """
    started = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    demand = instance['demand']
    n = len(demand) - 1
    capacity = encoder.capacity
    count = encoder.probes_per_stage
    if np.any(demand[1:] < 0) or np.any(demand[1:] > capacity):
        raise ValueError('Random CVRP probes require customer demands in [0, capacity]')
    pools, seen = [[], [], []], set()
    limit = max(100, 3 * count * 20)
    for attempt in range(limit):
        visited = np.zeros(n + 1, dtype=bool)
        visited[0] = True
        current, used = 0, 0.0
        path = [0]
        while not visited.all():
            unvisited = np.flatnonzero(~visited)
            feasible = unvisited[demand[unvisited] <= capacity - used + 1e-12]
            # At least two feasible customers: avoid forced decisions which
            # cannot distinguish the three percentile-rank preferences.
            if len(feasible) >= 2:
                actions = np.concatenate(([0], feasible)) if current else feasible
                key = (current, tuple(map(int, actions)))
                if key not in seen:
                    seen.add(key)
                    progress = float(np.count_nonzero(visited[1:])) / n
                    layer = min(2, int(progress * 3))
                    load_ratio = used / capacity
                    pools[layer].append(dict(
                        current=current, unvisited=unvisited.copy(),
                        used_capacity_ratio=load_ratio, progress=progress,
                        stage_index=layer, load_index=min(2, int(load_ratio * 3)),
                        probe_layer=layer, decision_key=key, path_prefix=tuple(path),
                        reference_mode='stratified_uniform_feasible_random',
                    ))
            # A depot return is a legal random choice even if customers fit;
            # a lack of fitting customers forces this move. Each subsequent
            # departure must visit a new customer, so trajectories terminate.
            actions = np.concatenate(([0], feasible)) if current else feasible
            if not len(actions):
                raise ValueError('No legal CVRP move from depot')
            chosen = int(rng.choice(actions))
            path.append(chosen)
            if chosen == 0:
                current, used = 0, 0.0
            else:
                visited[chosen] = True
                current, used = chosen, used + float(demand[chosen])
        if all(len(pool) >= count and len({s['current'] for s in pool}) >= count
               for pool in pools):
            break
    if any(len(pool) < count for pool in pools):
        raise ValueError(f'Random CVRP probe shortage after {attempt+1} paths: '
                         f'{[len(pool) for pool in pools]}, need {count} per layer')

    selected, used_nodes = [], set()
    for pool in pools:
        remaining, layer_nodes = list(pool), set()
        for _ in range(count):
            fresh = [s for s in remaining if s['current'] not in used_nodes]
            local_fresh = [s for s in remaining if s['current'] not in layer_nodes]
            options = fresh or local_fresh or remaining
            state = options[int(rng.integers(len(options)))]
            selected.append(state)
            used_nodes.add(state['current'])
            layer_nodes.add(state['current'])
            remaining = [s for s in remaining if s is not state]
    serial = [dict(current=s['current'], actions=list(s['decision_key'][1]),
                   used_capacity_ratio=s['used_capacity_ratio'], progress=s['progress'],
                   layer=s['probe_layer']) for s in selected]
    digest = hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()
    logging.info('[StratifiedRandomCVRPProbes] seed=%s states=%s layer_counts=%s '
                 'pool_counts=%s paths=%s build_time_s=%.6f unique_decision_keys=%s '
                 'unique_current_nodes=%s load_range=%s bank_hash=%s',
                 seed, len(selected), [count]*3, [len(p) for p in pools], attempt+1,
                 time.perf_counter()-started, len({s['decision_key'] for s in selected}),
                 len(used_nodes), [min(s['used_capacity_ratio'] for s in selected),
                                   max(s['used_capacity_ratio'] for s in selected)], digest)
    return tuple(selected)
