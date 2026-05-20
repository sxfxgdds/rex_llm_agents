"""Random seeding utilities."""

from __future__ import annotations

import random


def set_global_seed(seed: int) -> random.Random:
    random.seed(seed)
    return random.Random(seed)
