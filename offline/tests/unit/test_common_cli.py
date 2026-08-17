from __future__ import annotations

import random

import numpy as np
from _common import resolve_seed, seed_everything


def test_config_seed_is_used_unless_cli_overrides_it() -> None:
    assert resolve_seed(None, {"seed": 123}) == 123
    assert resolve_seed(456, {"seed": 123}) == 456


def test_recorded_seed_is_applied_to_python_and_numpy() -> None:
    seed_everything(2026)
    first = (random.random(), np.random.random())
    seed_everything(2026)
    second = (random.random(), np.random.random())

    assert first == second
