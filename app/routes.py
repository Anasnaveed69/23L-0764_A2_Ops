"""Pages: upload (/), result (/result/<id>), history (/history), health check (/health)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from flask import (
    Blueprint, abort, current_app, jsonify, redirect, render_template, request, url_for,
)

from visionguard.validation import UploadError, check_upload

from .predictors import PredictorUnavailable

bp = Blueprint("main", __name__)
log = logging.getLogger("visionguard.web")


def _db():
    return current_app.extensions["db"]


def _predictor():
    return current_app.extensions["predictor"]


@bp.get("/")
def index():
    recent, _ = _db().page(1, 5)
    return render_template("index.html", recent=recent)


@bp.post("/predict")
def predict():
    cfg, p = current_app.config, _predictor()
    raw_name = (request.files.get("image").filename if request.files.get("image") else "") or ""
    try:
        up = check_upload(request.files.get("image"), cfg["MAX_UPLOAD_MB"] * 1024 * 1024)
    except UploadError as e:
        _db().add(status="rejected", filename=(raw_name or "-")[:120], inference_mode=p.mode,
                  error=f"{e.status} {e.title}")
        log.info("rejected upload %r: %s", raw_name, e.title)
        return render_template("error.html", status=e.status, title=e.title,
                               message=e.message), e.status
    try:
        out = p.predict(up.data, up.filename)
    except PredictorUnavailable as e:
        _db().add(status="error", filename=up.filename, image_sha256=up.sha256, width=up.width,
                  height=up.height, size_kb=up.size_kb, inference_mode=p.mode, error=str(e)[:255])
        log.error("prediction failed: %s", e)
        return render_template("error.html", status=503, title="Model service unavailable",
                               message="The model could not be reached. Please try again in a "
                                       "minute."), 503
    rec = _db().add(
        status="ok", filename=up.filename, image_sha256=up.sha256, width=up.width,
        height=up.height, size_kb=up.size_kb, label=out["label"], confidence=out["confidence"],
        prob_ai=out["prob_ai"], threshold=out["threshold"], model_name=out["model_name"],
        model_version=str(out["model_version"]), inference_mode=p.mode,
        latency_ms=out.get("round_trip_ms", out.get("latency_ms")))
    log.info("prediction id=%s file=%s label=%s conf=%.3f mode=%s", rec.id, up.filename,
             out["label"], out["confidence"], p.mode)
    # Post/Redirect/Get: refreshing the result page does not re-submit the upload.
    return redirect(url_for("main.result", rec_id=rec.id))


@bp.get("/result/<int:rec_id>")
def result(rec_id: int):
    rec = _db().get(rec_id)
    if rec is None or rec.status != "ok":
        abort(404)
    return render_template("result.html", r=rec)


@bp.get("/history")
def history():
    page = max(1, request.args.get("page", 1, type=int))
    label = request.args.get("label") or None
    if label not in (None, "AI-generated", "Real"):
        label = None
    include_rejected = request.args.get("all") == "1"
    size = current_app.config["HISTORY_PAGE_SIZE"]
    rows, total = _db().page(page, size, label, include_rejected)
    pages = max(1, -(-total // size))
    return render_template("history.html", rows=rows, total=total, page=page, pages=pages,
                           label=label, include_rejected=include_rejected, stats=_db().stats())


@bp.get("/api/history")
def history_json():
    rows, total = _db().page(1, min(request.args.get("limit", 50, type=int), 500),
                             include_rejected=True)
    cols = ["id", "created_at", "status", "filename", "label", "confidence", "prob_ai",
            "threshold", "model_name", "model_version", "inference_mode", "latency_ms", "error"]
    return jsonify(total=total, items=[{c: getattr(r, c) for c in cols} for r in rows])


@bp.get("/health")
def health():
    p = _predictor()
    db_ok, model_ok = _db().ping(), p.healthy()
    body = {
        "status": "ok" if db_ok and model_ok else "degraded",
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inference_mode": p.mode, "model": p.info, "database": "ok" if db_ok else "down",
        "model_backend": "ok" if model_ok else "down",
        "debug": current_app.debug,
    }
    return jsonify(body), 200 if db_ok and model_ok else 503
