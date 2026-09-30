"""Order-independent random-state sandbox for candidate behavior probes."""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import random
import threading

import numpy as np


_RNG_LOCK = threading.RLock()


def behavior_seed(code: str, base_seed: int) -> int:
    """Return a stable per-code seed; unlike ``hash()``, it survives restarts."""
    digest = hashlib.blake2b(
        str(code or "").encode("utf-8", errors="replace"),
        digest_size=8,
        person=b"BeTR-probe",
    ).digest()
    code_seed = int.from_bytes(digest, byteorder="little", signed=False)
    return int((code_seed ^ int(base_seed)) & ((1 << 63) - 1))


@contextlib.contextmanager
def fixed_candidate_random_state(code: str, base_seed: int):
    """Make stochastic candidate code reproducible without leaking RNG state.

    The lock is necessary because NumPy's legacy RNG and Python's ``random``
    module are process-global. Calls to ``np.random.default_rng()`` without an
    explicit seed are also made deterministic, while explicitly seeded calls
    retain their requested seed. The original states and function are restored
    even when candidate execution fails or times out.
    """
    seed = behavior_seed(code, base_seed)
    numpy_seed = int(seed & 0xFFFFFFFF)

    with _RNG_LOCK:
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        original_default_rng = np.random.default_rng
        default_rng_counter = itertools.count()

        def deterministic_default_rng(requested_seed=None):
            if requested_seed is None:
                stream_index = next(default_rng_counter)
                requested_seed = (
                    seed + stream_index * 0x9E3779B97F4A7C15
                ) & ((1 << 64) - 1)
            return original_default_rng(requested_seed)

        torch_module = None
        torch_state = None
        try:
            import torch

            torch_module = torch
            torch_state = torch.random.get_rng_state()
        except Exception:
            torch_module = None
            torch_state = None

        np.random.seed(numpy_seed)
        np.random.default_rng = deterministic_default_rng
        random.seed(seed)
        if torch_module is not None:
            torch_module.manual_seed(seed)

        try:
            yield seed
        finally:
            np.random.default_rng = original_default_rng
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            if torch_module is not None and torch_state is not None:
                torch_module.random.set_rng_state(torch_state)
