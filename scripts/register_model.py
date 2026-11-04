"""Task 3.6 — Model Registry: make sure >= 2 versions exist, put alias `champion` on the best
run's version and `challenger` on the runner-up.

Usage: python scripts/register_model.py               # best run from compare_runs.py
       python scripts/register_model.py --run-name mnv2_default
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from visionguard.config import load_params  # noqa: E402
from visionguard.reporting import GEN, write_table  # noqa: E402
from visionguard.tracking import setup_mlflow  # noqa: E402


def version_for_run(client, name, run_id):
    for v in client.search_model_versions(f"name='{name}'"):
        if v.run_id == run_id:
            return v
    print(f"registering run {run_id} as a new version of {name}")
    return mlflow.register_model(f"runs:/{run_id}/model", name)


def set_alias(client, name, alias, version):
    try:
        client.set_registered_model_alias(name, alias, version)
    except Exception as exc:  # noqa: BLE001  (some hosted servers lack the alias endpoint)
        print(f"alias API failed ({exc}); falling back to a version tag alias={alias}")
        client.set_model_version_tag(name, version, "alias", alias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name")
    args = ap.parse_args()
    params = load_params()
    name = params["project"]["registered_model"]
    setup_mlflow(params["project"]["experiment"])
    client = MlflowClient()

    exp = client.get_experiment_by_name(params["project"]["experiment"])
    runs = client.search_runs([exp.experiment_id], "attributes.status = 'FINISHED'",
                              order_by=["metrics.val_roc_auc DESC"])
    runs = [r for r in runs if "val_roc_auc" in r.data.metrics and r.data.tags.get("model_type")]
    if args.run_name:
        best = next(r for r in runs if r.data.tags.get("mlflow.runName") == args.run_name)
    elif (GEN / "best_run.json").exists():
        rid = json.loads((GEN / "best_run.json").read_text())["run_id"]
        best = client.get_run(rid)
    else:
        best = runs[0]
    runner_up = next(r for r in runs if r.info.run_id != best.info.run_id)

    champ = version_for_run(client, name, best.info.run_id)
    chall = version_for_run(client, name, runner_up.info.run_id)
    set_alias(client, name, "champion", champ.version)
    set_alias(client, name, "challenger", chall.version)
    client.update_registered_model(name, description=(
        "VisionGuard binary classifier: p(AI-generated) for an RGB image. "
        "Input: raw 0-255 pixels resized as in model_meta.json. Owner: 23L-0764."))
    client.set_model_version_tag(name, champ.version, "val_auc",
                                 f"{best.data.metrics['val_roc_auc']:.4f}")

    rows = []
    for v in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
        run = client.get_run(v.run_id)
        aliases = ", ".join(getattr(v, "aliases", []) or []) or v.tags.get("alias", "")
        rows.append([v.version, run.data.tags.get("mlflow.runName"), aliases,
                     f"{run.data.metrics.get('val_roc_auc', float('nan')):.3f}",
                     f"{run.data.metrics.get('test_accuracy', float('nan')):.3f}"])
        print(f"v{v.version:<3} {rows[-1][1]:<16} aliases=[{aliases}]")
    write_table("registry", ["Version", "Run", "Alias", "val AUC", "test acc"], rows, "rllrr")
    print(f"\nchampion -> {name} v{champ.version} ({best.data.tags.get('mlflow.runName')})")


if __name__ == "__main__":
    main()
