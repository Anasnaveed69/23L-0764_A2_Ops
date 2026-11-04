import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def image_bytes(fmt="JPEG", size=(64, 48), mode="RGB") -> bytes:
    arr = np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr).convert(mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class FakePredictor:
    """Stands in for the Keras model so web tests run in milliseconds without TensorFlow."""

    mode = "local"
    info = {"model_name": "fake-model", "model_version": "0", "threshold": 0.5}

    def predict(self, data, filename):
        return {"label": "AI-generated", "confidence": 0.91, "prob_ai": 0.91, "threshold": 0.5,
                "model_name": "fake-model", "model_version": "0", "latency_ms": 1.0}

    def healthy(self):
        return True


@pytest.fixture()
def client(tmp_path):
    from app import create_app
    from app.config import Config

    import os

    url = os.getenv("VG_TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'test.db'}"
    if not url.startswith("sqlite"):  # shared Postgres in CI: start every test from empty tables
        from sqlalchemy import create_engine

        from app.db import Base

        Base.metadata.drop_all(create_engine(url))

    class TestConfig(Config):
        TESTING = True
        DATABASE_URL = url
        MAX_UPLOAD_MB = 1
        MAX_CONTENT_LENGTH = 1 * 1024 * 1024 + 64 * 1024

    app = create_app(TestConfig, predictor=FakePredictor())
    return app.test_client()
