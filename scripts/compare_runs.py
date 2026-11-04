"""Task 3.5 — pull all training runs from MLflow, build the comparison table + chart, pick the best.

Selection rule: highest VALIDATION ROC-AUC (threshold-free), ties broken by lower validation
false-positive rate. Test metrics are reported but never used for choosing (that would turn the
test set into a second validation set).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from visionguard.config import ROOT, load_params  # noqa: E402
from visionguard.reporting import FIG, GEN, save_json, write_macros, write_table  # noqa: E402
from visionguard.tracking import setup_mlflow  # noqa: E402

CHANGE = {"mnv2_default": "reference (MobileNetV2, lr 1e-3, aug, 224, last block)",
          "baseline_cnn": "model: CNN from scratch", "mnv2_lr_1e-4": "head lr 1e-3 -> 1e-4",
          "mnv2_no_aug": "augmentation off", "mnv2_frozen": "no fine-tuning (all frozen)",
          "mnv2_img160": "image size 224 -> 160", "effb0_default": "backbone -> EfficientNetB0",
          "mnv2_autolog": "same as reference, autolog tracking",
          "mnv2_tuned": "tuner's best: lr 2.2e-3, dropout 0.1, 60 layers unfrozen"}


def main():
    params = load_params()
    setup_mlflow(params["project"]["experiment"])
    runs = mlflow.search_runs(experiment_names=[params["project"]["experiment"]],
                              filter_string="attributes.status = 'FINISHED'")
    runs = runs[runs["tags.model_type"].notna() & runs["metrics.val_roc_auc"].notna()]
    runs = runs.sort_values("start_time").drop_duplicates("tags.mlflow.runName", keep="last")
    cols = {"tags.mlflow.runName": "run", "metrics.val_roc_auc": "val_auc",
            "metrics.val_false_positive_rate": "val_fpr", "metrics.val_f1": "val_f1",
            "metrics.test_accuracy": "test_acc", "metrics.test_f1": "test_f1",
            "metrics.test_roc_auc": "test_auc", "metrics.test_false_positive_rate": "test_fpr",
            "metrics.best_epoch": "best_epoch", "metrics.train_seconds": "secs", "run_id": "run_id"}
    df = runs[list(cols)].rename(columns=cols)
    df = df.sort_values(["val_auc", "val_fpr"], ascending=[False, True]).reset_index(drop=True)
    print(df.drop(columns="run_id").round(3).to_string())
    best = df.iloc[0]
    print(f"\nBEST by validation AUC: {best.run} (run_id {best.run_id})")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "run_comparison.csv", index=False)
    write_table("runs", ["Run", "Change vs reference", "val AUC", "val FPR", "test acc", "test F1",
                         "test AUC", "best ep."],
                [[r.run, CHANGE.get(r.run, "-"), f"{r.val_auc:.3f}", f"{r.val_fpr:.3f}",
                  f"{r.test_acc:.3f}", f"{r.test_f1:.3f}", f"{r.test_auc:.3f}",
                  int(r.best_epoch)] for r in df.itertuples()],
                r"lp{0.3\linewidth}rrrrrr")
    write_macros("best_macros", {"vgBestRun": best.run, "vgBestRunId": best.run_id,
                                 "vgBestValAuc": float(best.val_auc),
                                 "vgBestTestAuc": float(best.test_auc),
                                 "vgBestTestAcc": float(best.test_acc),
                                 "vgBestTestFone": float(best.test_f1)})
    save_json("best_run", {"run": best.run, "run_id": best.run_id})

    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df.val_auc, width=0.4, label="validation ROC-AUC")
    ax.bar([i + 0.2 for i in x], df.test_f1, width=0.4, label="test F1")
    ax.set_xticks(list(x), df.run, rotation=25, ha="right")
    ax.set_ylim(max(0, min(df.val_auc.min(), df.test_f1.min()) - 0.1), 1.0)
    ax.legend()
    ax.set_title("Experiment comparison (sorted by validation AUC)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "runs_comparison.png", dpi=130)
    plt.close(fig)

    # Task 2.5 metric table for the two report models
    rows = []
    for key, name in [("baseline", "Baseline CNN"), ("transfer", "MobileNetV2 transfer")]:
        f = GEN / f"{key}_test_metrics.csv"
        if f.exists():
            m = pd.read_csv(f).iloc[0]
            rows.append([name, f"{m.threshold:.2f}", f"{m.accuracy:.3f}", f"{m.precision:.3f}",
                         f"{m.recall:.3f}", f"{m.f1:.3f}", f"{m.roc_auc:.3f}",
                         f"{m.false_positive_rate:.3f}"])
    if rows:
        write_table("model_metrics", ["Model", "Thr.", "Accuracy", "Precision", "Recall", "F1",
                                      "ROC-AUC", "FPR"], rows, "lrrrrrrr")


if __name__ == "__main__":
    main()
