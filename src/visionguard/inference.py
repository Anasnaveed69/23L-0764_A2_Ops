"""Inference module used by the Flask app (local mode) and by the cloud model API.

The model is loaded ONCE when the process starts. Loading a Keras model takes seconds and
hundreds of MB of RAM; doing it per request would make every request slow and could exhaust
memory under concurrent load.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .imaging import decode_and_resize

log = logging.getLogger("visionguard.inference")


@dataclass
class Prediction:
    label: str            # "AI-generated" | "Real"
    confidence: float     # probability of the predicted label, 0..1
    prob_ai: float        # raw sigmoid output
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    def __init__(self, model_path: str | os.PathLike, threshold: float | None = None):
        import keras  # imported lazily so the web container can start in remote mode without TF

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}. "
                                    "Run scripts/export_champion.py or check MODEL_PATH.")
        meta_path = model_path.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        t0 = time.perf_counter()
        self.model = keras.models.load_model(model_path, compile=False)
        self.image_size = int(self.meta.get("image_size", self.model.input_shape[1]))
        self.resize_mode = self.meta.get("resize_mode", "squash")
        # Env var wins over the threshold chosen at training time (lets ops adjust without rebuild)
        self.threshold = float(threshold if threshold is not None
                               else self.meta.get("threshold", 0.5))
        self.model_name = self.meta.get("model_name", model_path.stem)
        self.model_version = str(self.meta.get("model_version", "local"))
        # Warm-up: first call builds the TF graph; do it now instead of on the first user request.
        self.model.predict(np.zeros((1, self.image_size, self.image_size, 3), np.float32),
                           verbose=0)
        log.info("Loaded %s v%s from %s in %.1fs (size=%d, threshold=%.2f)", self.model_name,
                 self.model_version, model_path, time.perf_counter() - t0, self.image_size,
                 self.threshold)

    def predict_bytes(self, data: bytes) -> Prediction:
        t0 = time.perf_counter()
        x = decode_and_resize(data, self.image_size, self.resize_mode)[None, ...]
        p = float(self.model.predict(x, verbose=0)[0][0])
        is_ai = p >= self.threshold
        return Prediction(
            label="AI-generated" if is_ai else "Real",
            confidence=round(p if is_ai else 1 - p, 4),
            prob_ai=round(p, 4),
            threshold=self.threshold,
            model_name=self.model_name,
            model_version=self.model_version,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
