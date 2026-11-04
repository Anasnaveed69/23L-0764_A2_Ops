"""Task 3.6 — load the registered model BY ALIAS from the registry and classify one image.

Usage: python scripts/load_champion.py path/to/image.jpg [--alias champion]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from champion_utils import version_by_alias  # noqa: E402
from visionguard.config import load_params  # noqa: E402
from visionguard.imaging import decode_and_resize  # noqa: E402
from visionguard.inference import decision_confidence  # noqa: E402
from visionguard.tracking import setup_mlflow  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--alias", default="champion")
    args = ap.parse_args()
    params = load_params()
    name = params["project"]["registered_model"]
    print("tracking:", setup_mlflow(params["project"]["experiment"]))
    client = MlflowClient()
    v = version_by_alias(client, name, args.alias)
    uri = f"models:/{name}/{v.version}"
    print(f"alias '{args.alias}' -> {name} version {v.version} (run {v.run_id})")
    t0 = time.time()
    model = mlflow.tensorflow.load_model(uri)
    print(f"loaded {uri} in {time.time() - t0:.1f}s")
    meta = json.loads(Path(client.download_artifacts(v.run_id, "keras_model/model_meta.json"))
                      .read_text())
    x = decode_and_resize(Path(args.image).read_bytes(), meta["image_size"], meta["resize_mode"])
    p = float(model.predict(x[None, ...], verbose=0)[0][0])
    label = "AI-generated" if p >= meta["threshold"] else "Real"
    conf = decision_confidence(p, meta["threshold"])
    print(f"\nimage      : {args.image}\np(AI)      : {p:.4f}\nthreshold  : {meta['threshold']:.2f}"
          f"\nprediction : {label} (confidence {conf:.1%})")


if __name__ == "__main__":
    main()
