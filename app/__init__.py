"""VisionGuard Flask application factory."""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from .config import Config  # noqa: E402
from .db import Database  # noqa: E402
from .predictors import build_predictor  # noqa: E402

STARTED = datetime.now(timezone.utc)


def create_app(config: type | None = None, predictor=None) -> Flask:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    app.config.from_object(config or Config)
    cfg = app.config

    # Heavy objects are created ONCE per process, at start-up, never per request.
    app.extensions["db"] = Database(cfg["DATABASE_URL"])
    app.extensions["predictor"] = predictor or build_predictor(_Cfg(cfg))
    app.logger.info("VisionGuard ready: inference=%s info=%s debug=%s",
                    app.extensions["predictor"].mode, app.extensions["predictor"].info,
                    cfg["DEBUG"])

    from .routes import bp

    app.register_blueprint(bp)

    @app.context_processor
    def _globals():
        p = app.extensions["predictor"]
        return {"model_info": p.info, "inference_mode": p.mode,
                "max_mb": cfg["MAX_UPLOAD_MB"], "year": datetime.now().year}

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_e):
        return _error(413, "File too large",
                      f"The limit is {cfg['MAX_UPLOAD_MB']} MB. Resize or compress the image.")

    @app.errorhandler(HTTPException)
    def http_error(e):
        return _error(e.code or 500, e.name, e.description)

    @app.errorhandler(Exception)
    def unhandled(e):
        app.logger.exception("unhandled error: %s", e)
        return _error(500, "Something went wrong",
                      "The server hit an unexpected error. The details were logged.")

    return app


def _error(status: int, title: str, message: str):
    if request.path.startswith("/api") or request.accept_mimetypes.best == "application/json":
        return jsonify(error=title, detail=message), status
    return render_template("error.html", status=status, title=title, message=message), status


class _Cfg:
    """Attribute access over Flask's config dict."""

    def __init__(self, d):
        self.__dict__.update(d)
