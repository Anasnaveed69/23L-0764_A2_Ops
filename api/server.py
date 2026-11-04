"""VisionGuard model API — the part deployed to the cloud (Task 7).

  GET  /health          public, no key: liveness + which model is loaded
  POST /v1/predict      requires header X-API-Key; multipart field "image" -> JSON verdict

The champion model is baked into the image at build time and loaded once at start-up, so the API
never calls DagsHub while serving.
"""
from __future__ import annotations

import hmac
import logging
import os
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionguard.inference import Predictor  # noqa: E402
from visionguard.validation import UploadError, check_upload  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("visionguard.api")

MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
API_KEY = os.getenv("API_KEY", "")
STARTED = time.time()


def create_api(predictor: Predictor | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024 + 64 * 1024
    if not API_KEY:
        log.warning("API_KEY is empty: /v1/predict will reject every request")
    pred = predictor or Predictor(os.getenv("MODEL_PATH", "models/champion.keras"),
                                  float(os.environ["THRESHOLD"]) if os.getenv("THRESHOLD") else None)
    counters = {"requests": 0, "unauthorized": 0}

    @app.get("/")
    @app.get("/health")
    def health():
        return jsonify(status="ok", service="visionguard-model-api",
                       model_name=pred.model_name, model_version=pred.model_version,
                       threshold=pred.threshold, image_size=pred.image_size,
                       uptime_s=round(time.time() - STARTED), served=counters["requests"])

    @app.post("/v1/predict")
    def predict():
        supplied = request.headers.get("X-API-Key", "")
        # constant-time comparison: response time must not leak how much of the key matched
        if not API_KEY or not hmac.compare_digest(supplied.encode(), API_KEY.encode()):
            counters["unauthorized"] += 1
            return jsonify(error="unauthorized", detail="missing or invalid X-API-Key"), 401
        try:
            up = check_upload(request.files.get("image"), MAX_MB * 1024 * 1024)
        except UploadError as e:
            return jsonify(error=e.title, detail=e.message), e.status
        out = pred.predict_bytes(up.data).to_dict()
        out.update(filename=up.filename, sha256=up.sha256, width=up.width, height=up.height)
        counters["requests"] += 1
        log.info("predict %s -> %s (%.3f) in %.0f ms from %s", up.filename, out["label"],
                 out["confidence"], out["latency_ms"],
                 request.headers.get("X-Forwarded-For", request.remote_addr))
        return jsonify(out)

    @app.errorhandler(HTTPException)
    def http_err(e):
        return jsonify(error=e.name, detail=e.description), e.code

    return app


if __name__ == "__main__":
    create_api().run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
