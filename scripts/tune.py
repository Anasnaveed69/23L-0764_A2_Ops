"""Task 2.7 (apply) — Keras Tuner random search for VisionGuard.

Searches: head learning rate, dropout, and how many backbone layers to unfreeze in stage 2.
Every trial is also logged to MLflow as a nested run under one parent "keras_tuner" run.

Usage: python scripts/tune.py [--set tuner.max_trials=10]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import keras  # noqa: E402
import keras_tuner as kt  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from visionguard.config import ROOT, load_params, set_seeds  # noqa: E402
from visionguard.data import class_weights, make_dataset, read_split  # noqa: E402
from visionguard.models import build_transfer, compile_model, unfreeze  # noqa: E402
from visionguard.reporting import FIG, write_macros, write_table  # noqa: E402
from visionguard.tracking import repro_tags, setup_mlflow  # noqa: E402


class VisionGuardHyperModel(kt.HyperModel):
    def __init__(self, params, class_weight):
        super().__init__()
        self.p = params
        self.cw = class_weight

    def build(self, hp):
        t = self.p["train"]
        hp.Float("lr_head", 1e-4, 3e-3, sampling="log")
        hp.Float("dropout", 0.1, 0.5, step=0.1)
        hp.Choice("n_unfrozen", [0, 11, 30, 60])  # 11 = MobileNetV2 last block
        model = build_transfer(self.p["image"]["size"], t["backbone"], hp.get("dropout"))
        compile_model(model, hp.get("lr_head"))
        return model

    def fit(self, hp, model, *args, **kwargs):
        """Two-stage training inside a single trial: head first, then optional fine-tuning."""
        epochs = kwargs.pop("epochs")
        stage1 = max(2, epochs // 2)
        h1 = model.fit(*args, epochs=stage1, class_weight=self.cw, **kwargs)
        n = hp.get("n_unfrozen")
        if n == 0:
            return h1
        unfreeze(model, self.p["train"]["backbone"], n)
        compile_model(model, hp.get("lr_head") / 100)
        return model.fit(*args, initial_epoch=stage1, epochs=epochs, class_weight=self.cw, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    params = load_params(args.set)
    seed = params["project"]["seed"]
    set_seeds(seed)
    tc = params["tuner"]
    tr_df, va_df = read_split(params, "train"), read_split(params, "val")
    train_ds = make_dataset(tr_df, params, training=True)
    val_ds = make_dataset(va_df, params, training=False)

    tuner = kt.RandomSearch(
        VisionGuardHyperModel(params, class_weights(tr_df)),
        objective=kt.Objective("val_loss", "min"),
        max_trials=tc["max_trials"], seed=seed, overwrite=True,
        directory=str(ROOT / tc["directory"]), project_name="visionguard",
    )
    tuner.search_space_summary()
    tuner.search(train_ds, validation_data=val_ds, epochs=tc["epochs"], verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=2)])
    tuner.results_summary(num_trials=5)

    rows = []
    for trial in tuner.oracle.trials.values():
        if trial.score is None:
            continue
        hp = trial.hyperparameters.values
        mt = trial.metrics
        auc = mt.get_best_value("val_auc") if mt.exists("val_auc") else np.nan
        acc = mt.get_best_value("val_accuracy") if mt.exists("val_accuracy") else np.nan
        rows.append({"trial": trial.trial_id, **hp, "val_loss": trial.score, "val_auc": auc,
                     "val_accuracy": acc})
    df = pd.DataFrame(rows).sort_values("val_loss").reset_index(drop=True)
    out = ROOT / "outputs" / "tuner"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tuner_trials.csv", index=False)
    print(df.to_string())

    # Which hyper-parameter mattered most? Spread of mean val_loss across its levels.
    df["lr_bin"] = pd.qcut(df.lr_head, q=min(3, df.lr_head.nunique()), duplicates="drop").astype(str)
    effects = {name: float(df.groupby(col).val_loss.mean().max() - df.groupby(col).val_loss.mean().min())
               for name, col in [("learning rate", "lr_bin"), ("dropout", "dropout"),
                                 ("unfrozen layers", "n_unfrozen")]}
    ranked = sorted(effects.items(), key=lambda kv: -kv[1])
    print("effect (range of mean val_loss across levels):", ranked)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, col, logx in zip(axes, ["lr_head", "dropout", "n_unfrozen"], [True, False, False]):
        ax.scatter(df[col], df.val_loss, s=40)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(col)
        ax.set_ylabel("best val_loss")
    fig.suptitle("Keras Tuner trials: validation loss vs each hyper-parameter")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "tuner_effects.png", dpi=130)
    plt.close(fig)

    write_table("tuner_trials", ["Rank", "lr\\_head", "Dropout", "Unfrozen", "val loss", "val AUC"],
                [[i + 1, f"{r.lr_head:.2e}", f"{r.dropout:.1f}", int(r.n_unfrozen),
                  f"{r.val_loss:.4f}", f"{r.val_auc:.3f}"] for i, r in df.iterrows()], "rrrrrr")
    write_table("tuner_effects", ["Hyper-parameter", "Range of mean val loss across levels"],
                [[k, f"{v:.4f}"] for k, v in ranked], "lr")
    best = df.iloc[0]
    write_macros("tuner_macros", {"vgTunerTrials": len(df), "vgTunerBestLr": f"{best.lr_head:.2e}",
                                  "vgTunerBestDrop": f"{best.dropout:.1f}",
                                  "vgTunerBestUnfrozen": int(best.n_unfrozen),
                                  "vgTunerBestLoss": f"{best.val_loss:.4f}",
                                  "vgTunerMostImportant": ranked[0][0]})

    setup_mlflow(params["project"]["experiment"])
    with mlflow.start_run(run_name="keras_tuner_search"):
        mlflow.set_tags(repro_tags(seed, {"stage": "tuning"}))
        mlflow.log_params({"max_trials": tc["max_trials"], "epochs_per_trial": tc["epochs"],
                           "algorithm": "RandomSearch"})
        for _, r in df.iterrows():
            with mlflow.start_run(run_name=f"trial_{r.trial}", nested=True):
                mlflow.log_params({"lr_head": r.lr_head, "dropout": r.dropout,
                                   "n_unfrozen": int(r.n_unfrozen)})
                mlflow.log_metrics({"val_loss": r.val_loss, "val_auc": r.val_auc})
        mlflow.log_artifact(str(out / "tuner_trials.csv"))
        mlflow.log_artifact(str(FIG / "tuner_effects.png"))
    print(f"\nBest: {best.to_dict()}")


if __name__ == "__main__":
    main()
