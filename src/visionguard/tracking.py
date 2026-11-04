"""MLflow helpers: connect to DagsHub (or local), reproducibility tags, manual logging callback."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import keras
import mlflow
import yaml

from .config import ROOT


def setup_mlflow(experiment: str) -> str:
    """Use MLFLOW_TRACKING_URI from the environment (.env) if present, else a local ./mlruns.

    Credentials are read by MLflow itself from MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD,
    so the DagsHub token never appears in code or in Git.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    uri = os.getenv("MLFLOW_TRACKING_URI") or (ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dvc_data_version(dvc_file: str = "data/raw.dvc") -> str:
    """The md5 that DVC recorded for the dataset directory = the data version."""
    p = ROOT / dvc_file
    if not p.exists():
        return "untracked"
    meta = yaml.safe_load(p.read_text())
    return str(meta["outs"][0].get("md5", "unknown"))


def repro_tags(seed: int, extra: dict | None = None) -> dict[str, str]:
    tags = {
        "git_commit": _git("rev-parse", "HEAD"),
        # tracked source files only; generated report assets / outputs do not count as "dirty"
        "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no", "--",
                                   "src", "scripts", "app", "api", "params.yaml"))),
        "dvc_data_version": dvc_data_version(),
        "seed": str(seed),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorflow": __import__("tensorflow").__version__,
        "keras": keras.__version__,
        "author": "Anas Naveed Butt (23L-0764)",
    }
    tags.update({k: str(v) for k, v in (extra or {}).items()})
    return tags


class MLflowEpochLogger(keras.callbacks.Callback):
    """Manual logging: every Keras metric at the end of every epoch, plus the learning rate.

    `offset` lets stage-2 (fine-tune) epochs continue numbering after stage-1 epochs, so the run
    shows one continuous curve.
    """

    def __init__(self, prefix: str = "", offset: int = 0):
        super().__init__()
        self.prefix = prefix
        self.offset = offset

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        try:
            logs["learning_rate"] = float(keras.ops.convert_to_numpy(
                self.model.optimizer.learning_rate))
        except Exception:  # noqa: BLE001
            pass
        mlflow.log_metrics({f"{self.prefix}{k}": float(v) for k, v in logs.items()},
                           step=self.offset + epoch + 1)


def log_keras_model(model: keras.Model, local_path: Path, registered_name: str | None = None):
    """Log the model twice: as the raw .keras file (simple to download for serving) and as an
    MLflow 'tensorflow' flavour model (needed for the Model Registry)."""
    import numpy as np
    from mlflow.models import infer_signature

    model.save(local_path)
    mlflow.log_artifact(str(local_path), artifact_path="keras_model")
    s = model.input_shape[1]
    # Signature documents the contract: float32 (N, s, s, 3) raw 0-255 pixels -> (N, 1) p(AI)
    sig = infer_signature(np.zeros((1, s, s, 3), np.float32), np.zeros((1, 1), np.float32))
    return mlflow.tensorflow.log_model(model, artifact_path="model", signature=sig,
                                       registered_model_name=registered_name)


def pip_freeze(path: Path) -> Path:
    out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    path.write_text(out)
    return path
