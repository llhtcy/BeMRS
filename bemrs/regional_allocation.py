"""BeMRS: annealed winner-take-most regional evaluation budgets."""

import math


def annealed_winner_take_most_quotas(
    ranked_region_ids, capacities, total_slots, used_real_evals,
    total_real_eval_budget, allocated_count, cursor=None,
):
    """Return quotas, diagnostics and next cursor without mutating history.

    The caller supplies real-objective rank and commits only regional slots
    to history. Cursor is a stable region ID, so inactive regions are skipped.
    """
    active = [int(r) for r in ranked_region_ids if capacities.get(r, 0) > 0]
    slots = max(0, int(total_slots))
    if total_real_eval_budget <= 0:
        raise ValueError("total_real_eval_budget must be positive")
    alpha = min(1.0, max(0.0, float(used_real_evals) / total_real_eval_budget))
    weights = {r: (1.0 - alpha) / len(active) for r in active}
    if active:
        weights[active[0]] += alpha
    stable = sorted(active)
    start = next((i for i, r in enumerate(stable) if cursor is None or r >= cursor), 0)
    rotated = stable[start:] + stable[:start]
    positions = {r: i for i, r in enumerate(rotated)}
    rank = {r: i + 1 for i, r in enumerate(active)}

    def fair_key(r):
        return (allocated_count.get(r, 0), positions[r], r)

    raw = {r: slots * weights[r] for r in active}
    initial = {r: math.floor(raw[r]) for r in active}
    integer = dict(initial)
    # Equal remainders can differ by floating-point roundoff (e.g. K=3,
    # B=4, alpha=.5). Weight wins before fairness for unequal weights.
    order = sorted(active, key=lambda r: (
        -round(raw[r] - initial[r], 12), -weights[r], *fair_key(r),
    ))
    last = None
    for r in order[:slots - sum(integer.values())]:
        integer[r] += 1
        last = r
    quotas = {r: min(integer[r], int(capacities[r])) for r in active}
    capped = dict(quotas)
    target = min(slots, sum(int(capacities[r]) for r in active))
    while sum(quotas.values()) < target:
        eligible = [r for r in active if quotas[r] < capacities[r]]
        r = min(eligible, key=lambda r: (-weights[r], rank[r], *fair_key(r)))
        quotas[r] += 1
        last = r
    if last is not None:
        cursor = stable[(stable.index(last) + 1) % len(stable)]
    counts = dict(allocated_count)
    for r in active:
        counts[r] = counts.get(r, 0) + quotas[r]
    return quotas, {
        "used_real_evals": used_real_evals,
        "total_real_eval_budget": total_real_eval_budget,
        "alpha": alpha,
        "active_region_ids": active,
        "K_active": len(active),
        "regional_rank": rank,
        "regional_weight": weights,
        "candidate_count_per_region": dict(capacities),
        "B_region": slots,
        "raw_quota": raw,
        "initial_integer_quota": initial,
        "largest_remainder_quota": integer,
        "redistributed_quota": {r: quotas[r] - capped[r] for r in active},
        "final_quota": dict(quotas),
        "regional_allocated_count": counts,
    }, cursor
