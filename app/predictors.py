"""Two interchangeable back-ends behind one interface: predict(bytes, filename) -> dict."""
from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger("visionguard.predictors")


class PredictorUnavailable(RuntimeError):
    pass


class LocalPredictor:
    """Loads the champion Keras model once, in-process (Task 4.1)."""

    mode = "local"

    def __init__(self, model_path: str, threshold: float | None):
        from visionguard.inference import Predictor  # TF imported only in local mode

        self._p = Predictor(model_path, threshold)
        self.info = {"model_name": self._p.model_name, "model_version": self._p.model_version,
                     "threshold": self._p.threshold, "image_size": self._p.image_size}

    def predict(self, data: bytes, filename: str) -> dict:
        return self._p.predict_bytes(data).to_dict()

    def healthy(self) -> bool:
        return True


class RemotePredictor:
    """Calls the cloud model API (Tasks 7/8). The API key travels in a header, never in a URL."""

    mode = "remote"

    def __init__(self, base_url: str, api_key: str, timeout: float = 30):
        if not base_url:
            raise PredictorUnavailable("INFERENCE_MODE=remote but MODEL_API_URL is empty")
        self.base_url, self.timeout = base_url, timeout
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key, "User-Agent": "VisionGuard-web/1.0"})
        self.info = {"model_name": "remote", "model_version": "?", "api": base_url}
        try:
            h = self.session.get(f"{base_url}/health", timeout=10).json()
            self.info.update({k: h.get(k) for k in ("model_name", "model_version", "threshold")})
        except Exception as exc:  # noqa: BLE001 — API may be cold; we retry per request
            log.warning("model API health check failed at start-up: %s", exc)

    def predict(self, data: bytes, filename: str) -> dict:
        t0 = time.perf_counter()
        try:
            r = self.session.post(f"{self.base_url}/v1/predict",
                                  files={"image": (filename, data)}, timeout=self.timeout)
        except requests.RequestException as exc:
            raise PredictorUnavailable(f"model API unreachable: {exc.__class__.__name__}") from exc
        if r.status_code != 200:
            try:
                detail = r.json().get("error", r.text[:200])
            except ValueError:
                detail = r.text[:200]
            raise PredictorUnavailable(f"model API returned {r.status_code}: {detail}")
        out = r.json()
        out["round_trip_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self.info.update({"model_name": out.get("model_name"),
                          "model_version": out.get("model_version")})
        return out

    def healthy(self) -> bool:
        try:
            return self.session.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False


def build_predictor(cfg) -> LocalPredictor | RemotePredictor:
    if cfg.INFERENCE_MODE == "remote":
        return RemotePredictor(cfg.MODEL_API_URL, cfg.MODEL_API_KEY, cfg.MODEL_API_TIMEOUT)
    return LocalPredictor(cfg.MODEL_PATH, cfg.THRESHOLD)
