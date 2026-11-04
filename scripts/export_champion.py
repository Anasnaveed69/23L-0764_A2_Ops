"""Copy the `champion` model out of the registry ONCE into models/champion.keras (+ .json).

The serving app reads this local file at start-up and never contacts DagsHub per request.
Run this whenever the champion alias moves, then rebuild/restart the container.

Usage: python scripts/export_champion.py [--alias champion] [--out models/champion.keras]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlflow.tracking import MlflowClient  # noqa: E402

from champion_utils import version_by_alias  # noqa: E402
from visionguard.config import ROOT, load_params  # noqa: E402
from visionguard.tracking import setup_mlflow  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--out", default="models/champion.keras")
    args = ap.parse_args()
    params = load_params()
    name = params["project"]["registered_model"]
    setup_mlflow(params["project"]["experiment"])
    client = MlflowClient()
    v = version_by_alias(client, name, args.alias)
    local = Path(client.download_artifacts(v.run_id, "keras_model"))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Re-save without optimizer state: serving never trains, so the file is smaller (~9 MB).
    import keras

    keras.models.load_model(local / "model.keras", compile=False).save(out)
    meta = json.loads((local / "model_meta.json").read_text())
    meta.update({"model_version": v.version, "alias": args.alias, "model_name": name})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"exported {name} v{v.version} ({meta['run_name']}) -> {out} "
          f"[{out.stat().st_size / 1e6:.1f} MB], threshold {meta['threshold']}")


if __name__ == "__main__":
    main()
