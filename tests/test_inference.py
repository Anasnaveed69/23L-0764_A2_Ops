"""Inference + model API tests with a tiny randomly initialised model (no download needed)."""
import io
import json
import os

import pytest

from conftest import image_bytes

keras = pytest.importorskip("keras")


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    os.environ["VG_BACKBONE_WEIGHTS"] = "none"
    from visionguard.models import build_baseline

    d = tmp_path_factory.mktemp("m")
    p = d / "champion.keras"
    build_baseline(32).save(p)
    p.with_suffix(".json").write_text(json.dumps(
        {"image_size": 32, "resize_mode": "squash", "threshold": 0.5, "model_name": "t",
         "model_version": "9"}))
    return p


def test_predictor_loads_once_and_predicts(model_path):
    from visionguard.inference import Predictor

    p = Predictor(model_path)
    out = p.predict_bytes(image_bytes())
    assert out.label in ("AI-generated", "Real")
    assert 0.5 <= out.confidence <= 1.0 and 0 <= out.prob_ai <= 1
    assert out.model_version == "9" and p.image_size == 32


def test_missing_model_file_is_clear(tmp_path):
    from visionguard.inference import Predictor

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        Predictor(tmp_path / "nope.keras")


def test_api_requires_key(model_path, monkeypatch):
    import api.server as srv
    from visionguard.inference import Predictor

    monkeypatch.setattr(srv, "API_KEY", "secret-key")
    c = srv.create_api(Predictor(model_path)).test_client()
    assert c.get("/health").status_code == 200
    files = {"image": (io.BytesIO(image_bytes()), "x.jpg")}
    assert c.post("/v1/predict", data=files, content_type="multipart/form-data").status_code == 401
    files = {"image": (io.BytesIO(image_bytes()), "x.jpg")}
    r = c.post("/v1/predict", data=files, content_type="multipart/form-data",
               headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200 and r.get_json()["label"] in ("AI-generated", "Real")
    files = {"image": (io.BytesIO(b"junk"), "x.txt")}
    r = c.post("/v1/predict", data=files, content_type="multipart/form-data",
               headers={"X-API-Key": "secret-key"})
    assert r.status_code == 415
