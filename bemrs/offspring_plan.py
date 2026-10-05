"""Proportional evaluation budget for a combined candidate pool."""
import math


def evaluation_slots(candidate_count, ratio, remaining, available=None):
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('evaluation_ratio must be finite and in (0, 1]')
    available = candidate_count if available is None else available
    return min(max(0, int(remaining)), max(0, int(available)),
               math.ceil(max(0, candidate_count) * ratio))
