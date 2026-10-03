"""Deterministic ordered parent plans; no LLM calls or evaluation here."""
import math
import numpy as np


def evaluation_slots(candidate_count, ratio, remaining, available=None):
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('evaluation_ratio must be finite and in (0, 1]')
    available = candidate_count if available is None else available
    return min(max(0, int(remaining)), max(0, int(available)),
               math.ceil(max(0, candidate_count) * ratio))


def build_parent_plan(pools, canonical_code):
    """Freeze parent pools and give each region one BX and one BR slot per parent.

    The pool size, rather than a separate generation quota, controls the
    regional offspring count. BX visits every parent as the first parent and
    uses its farthest regional partner. BR keeps the best regional anchor and
    walks through the other parents; its singleton row is single-parent.
    """
    regions = {}
    seen_codes = set()
    for rid in sorted(pools):
        entries = sorted(pools[rid], key=lambda e: (
            float(e['objective']), canonical_code(e['parent'].get('code'))))
        unique = []
        for entry in entries:
            code = canonical_code(entry['parent'].get('code'))
            if not code or code in seen_codes or not np.isfinite(entry['objective']):
                continue
            seen_codes.add(code)
            unique.append(entry)
        if unique:
            regions[rid] = unique
    rows, signatures = [], set()

    def add(operator, kind, rid, entries):
        parents = [e['parent'] for e in entries]
        codes = tuple(canonical_code(p['code']) for p in parents)
        signature = (operator, codes)
        if len(set(codes)) != len(codes) or signature in signatures:
            return
        signatures.add(signature)
        rows.append(dict(operator=operator, kind=kind, region_id=rid, parents=parents))

    for rid, entries in regions.items():
        anchor = entries[0]
        if len(entries) == 1:
            add('bx', 'single_bx', rid, [anchor])
        add('br', 'single_br', rid, [anchor])
        for i, entry in enumerate(entries):
            others = [j for j in range(len(entries)) if j != i]
            if not others:
                continue
            distances = {j: float(np.linalg.norm(
                np.asarray(entry['point']) - np.asarray(entries[j]['point']))) for j in others}
            # Entries are quality/code ordered, so ties are deterministic.
            far = min(others, key=lambda j: (-distances[j], j))
            add('bx', 'intra_bx', rid, [entry, entries[far]])
        # Keep the BR anchor rule while covering exactly one slot per parent.
        for entry in entries[1:]:
            add('br', 'intra_br', rid, [anchor, entry])
    return rows
