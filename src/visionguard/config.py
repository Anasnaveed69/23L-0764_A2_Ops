"""Configuration helpers: load params.yaml, apply CLI overrides, fix random seeds."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"


def load_params(overrides: list[str] | None = None, path: Path | None = None) -> dict:
    """Read params.yaml and apply `section.key=value` overrides (values parsed as YAML)."""
    with open(path or PARAMS_PATH, encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    params = copy.deepcopy(params)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        node: dict[str, Any] = params
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return params


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """{'train': {'lr': 1}} -> {'train.lr': 1}; used to log every param to MLflow."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def set_seeds(seed: int) -> None:
    """Fix every source of randomness we control (Python, NumPy, TensorFlow/Keras)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)  # seeds python, numpy and the TF backend together
    except ImportError:  # serving image without keras utils is fine
        pass


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
