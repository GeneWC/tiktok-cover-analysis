"""Canonical random seeds for training, evaluation, and experiments.

Every supported source of ML randomness in this repo should read from here
so a rerun with the same data and code produces the same numbers.

Environment (optional, documented for operators):
- PYTHONHASHSEED=0  — stabilize hash-randomized set/dict iteration in tooling
"""

from __future__ import annotations

import os
import random

import numpy as np

# Single project-wide seed. Creator splits, sklearn estimators, and experiment
# RNGs all use this value unless a test explicitly overrides it.
DEFAULT_SEED = 42
PYTHON_SEED = DEFAULT_SEED
NUMPY_SEED = DEFAULT_SEED
SKLEARN_RANDOM_STATE = DEFAULT_SEED


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and (when present) hash randomization."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def seed_record(seed: int = DEFAULT_SEED) -> dict[str, int | str]:
    """Metadata block written into training_metadata.json."""
    return {
        "python_seed": seed,
        "numpy_seed": seed,
        "sklearn_random_state": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
    }
