"""CI helper: write a tiny random-weight models/champion.keras (+ .json) so the Docker images can be
built and health-checked without the real (DVC/DagsHub-hosted) champion model."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VG_BACKBONE_WEIGHTS", "none")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visionguard.models import build_baseline  # noqa: E402

out = ROOT / "models" / "champion.keras"
out.parent.mkdir(exist_ok=True)
build_baseline(64).save(out)
out.with_suffix(".json").write_text(json.dumps({
    "model_name": "ci-dummy", "model_version": "0", "image_size": 64, "resize_mode": "squash",
    "threshold": 0.5}))
print(f"wrote {out}")
