"""Train + evaluate one VisionGuard model and log EVERYTHING to MLflow (DagsHub or local).

Examples
  python scripts/train.py --run-name baseline_cnn --set train.model=baseline --report-key baseline
  python scripts/train.py --run-name mnv2_default --report-key transfer
  python scripts/train.py --run-name mnv2_autolog --tracking autolog
  python scripts/train.py --run-name mnv2_lr3e-4 --set train.lr_head=0.0003

--tracking manual  : our own Keras callback logs per-epoch metrics (+ learning rate), and we log
                     params / tags / artifacts / model explicitly.
--tracking autolog : mlflow.tensorflow.autolog() captures fit() params, per-epoch metrics, the
                     model summary and early-stopping info automatically; we still add test
                     metrics, tags and evaluation artifacts that autolog cannot know about.
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import keras  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from visionguard import evaluation as ev  # noqa: E402
from visionguard.config import ROOT, flatten, load_params, resolve, set_seeds  # noqa: E402
from visionguard.data import class_weights, make_dataset, read_split  # noqa: E402
from visionguard.models import (  # noqa: E402
    build_baseline, build_transfer, compile_model, layer_table, unfreeze,
)
from visionguard.reporting import FIG, tex_escape, write_macros, write_table  # noqa: E402
from visionguard.tracking import (  # noqa: E402
    MLflowEpochLogger, log_keras_model, pip_freeze, repro_tags, setup_mlflow,
)


def callbacks(params, ckpt_path: Path, manual: bool, offset: int, ckpt=None):
    t = params["train"]
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=t["early_stop_patience"],
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                          patience=t["reduce_lr_patience"], min_lr=1e-7, verbose=1),
        ckpt or keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True,
                                                verbose=1),
    ]
    if manual:
        cbs.append(MLflowEpochLogger(offset=offset))
    return cbs


def predict(model, ds) -> np.ndarray:
    return model.predict(ds, verbose=0).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tracking", choices=["manual", "autolog"], default="manual")
    ap.add_argument("--set", nargs="*", default=[], help="param overrides, e.g. train.lr_head=1e-4")
    ap.add_argument("--report-key", default=None,
                    help="write report tables/figures under this key (baseline|transfer)")
    ap.add_argument("--register", action="store_true", help="also register in the Model Registry")
    ap.add_argument("--max-train", type=int, default=None, help="debug: subsample training set")
    args = ap.parse_args()

    params = load_params(args.set)
    seed = params["project"]["seed"]
    set_seeds(seed)
    t, size = params["train"], params["image"]["size"]
    uri = setup_mlflow(params["project"]["experiment"])
    print(f"MLflow tracking URI: {uri}")

    out = ROOT / "outputs" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "best.keras"

    tr_df, va_df, te_df = (read_split(params, s) for s in ("train", "val", "test"))
    if args.max_train:
        tr_df = tr_df.sample(min(args.max_train, len(tr_df)), random_state=seed)
    train_ds = make_dataset(tr_df, params, training=True)
    val_ds = make_dataset(va_df, params, training=False)
    test_ds = make_dataset(te_df, params, training=False)
    cw = class_weights(tr_df)

    manual = args.tracking == "manual"
    if not manual:
        # Autolog: per-epoch metrics, fit() arguments, optimizer config, model summary, the
        # EarlyStopping state. Models are logged by us below (same artifact layout for both
        # approaches -> the registry and serving code do not care which one produced the run).
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False, checkpoint=False)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(repro_tags(seed, {"tracking": args.tracking, "model_type": t["model"],
                                          "backbone": t["backbone"] if t["model"] == "transfer"
                                          else "none"}))
        mlflow.log_params({k: v for k, v in flatten(params).items()
                           if not k.startswith("data.class_map")})
        mlflow.log_params({"n_train": len(tr_df), "n_val": len(va_df), "n_test": len(te_df),
                           "class_weights": json.dumps({k: round(v, 3) for k, v in cw.items()})})
        for s in ("train", "val", "test"):
            mlflow.log_artifact(str(resolve(params["data"]["splits_dir"]) / f"{s}.csv"), "splits")
        mlflow.log_artifact(str(pip_freeze(out / "requirements-freeze.txt")), "environment")
        mlflow.log_artifact(str(ROOT / "params.yaml"), "environment")

        t0 = time.time()
        trainable_log = {}
        if t["model"] == "baseline":
            model = build_baseline(size, dropout=max(t["dropout"], 0.4))
            compile_model(model, t["lr_baseline"])
            model.summary(print_fn=lambda s, **_: print(s))
            h = model.fit(train_ds, validation_data=val_ds, epochs=t["epochs_baseline"],
                          class_weight=cw, callbacks=callbacks(params, ckpt_path, manual, 0),
                          verbose=2)
            history, boundary = h.history, None
        else:
            model = build_transfer(size, t["backbone"], t["dropout"])
            compile_model(model, t["lr_head"])
            trainable_log["stage1"] = [w.path for w in model.trainable_weights]
            model.summary(print_fn=lambda s, **_: print(s))
            ckpt = keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                                   save_best_only=True, verbose=1)
            print("\n=== Stage 1: backbone frozen, training the new head ===")
            h1 = model.fit(train_ds, validation_data=val_ds, epochs=t["epochs_head"],
                           class_weight=cw, callbacks=callbacks(params, ckpt_path, manual, 0, ckpt),
                           verbose=2)
            history, boundary = dict(h1.history), len(h1.history["loss"])
            if t["unfreeze"] not in ("none", 0) and t["epochs_finetune"] > 0:
                unfrozen = unfreeze(model, t["backbone"], t["unfreeze"])
                compile_model(model, t["lr_finetune"])  # recompile so trainable flags take effect
                trainable_log["stage2"] = [w.path for w in model.trainable_weights]
                trainable_log["stage2_backbone_layers"] = unfrozen
                print(f"\n=== Stage 2: fine-tuning {len(unfrozen)} backbone layers "
                      f"at lr={t['lr_finetune']} ===")
                # Autolog logs fit() arguments as params, and MLflow params are immutable, so a
                # second fit() in the same run collides ("epochs" changes). With autolog the
                # fine-tune stage therefore goes into a nested child run.
                stage2_ctx = (contextlib.nullcontext() if manual else
                              mlflow.start_run(run_name=f"{args.run_name}_stage2", nested=True))
                with stage2_ctx:
                    h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=boundary,
                                   epochs=boundary + t["epochs_finetune"], class_weight=cw,
                                   callbacks=callbacks(params, ckpt_path, manual, 0, ckpt),
                                   verbose=2)
                for k, v in h2.history.items():
                    history.setdefault(k, [np.nan] * boundary)
                    history[k] = list(history[k]) + list(v)
            else:
                boundary = None
        train_secs = time.time() - t0

        # Best weights across BOTH stages = the checkpoint with the lowest val_loss.
        model = keras.models.load_model(ckpt_path)
        title = f"{args.run_name}: loss / accuracy"
        best_epoch = ev.plot_history(history, out / "curves.png", title, boundary)
        mlflow.log_artifact(str(out / "curves.png"), "plots")
        (out / "trainable_layers.json").write_text(json.dumps(trainable_log, indent=1))
        mlflow.log_artifact(str(out / "trainable_layers.json"), "model_info")
        with open(out / "model_summary.txt", "w") as fh:
            model.summary(print_fn=lambda s, **_: fh.write(s + "\n"))
        mlflow.log_artifact(str(out / "model_summary.txt"), "model_info")

        # ----- threshold from VALIDATION, then the single test-set evaluation -----
        y_val = va_df.label.values
        p_val = predict(model, val_ds)
        thr, fb = ev.choose_threshold(y_val, p_val, params["eval"]["threshold_beta"])
        val_m = ev.compute_metrics(y_val, p_val, thr)
        y_te = te_df.label.values
        p_te = predict(model, test_ds)
        test_m = ev.compute_metrics(y_te, p_te, thr)
        test_m05 = ev.compute_metrics(y_te, p_te, 0.5)

        mlflow.log_metrics({"best_epoch": best_epoch, "train_seconds": train_secs,
                            "val_threshold": thr, f"val_f{params['eval']['threshold_beta']}": fb})
        mlflow.log_metrics({f"val_{k}": float(v) for k, v in val_m.items() if k != "threshold"})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_m.items() if k != "threshold"})
        mlflow.log_metrics({f"test_t0.5_{k}": float(v) for k, v in test_m05.items()
                            if k in ("accuracy", "f1", "precision", "recall")})

        ev.plot_confusion(y_te, p_te, thr, out / "confusion_matrix.png",
                          f"{args.run_name} — test (t={thr:.2f})")
        ev.plot_roc(y_te, p_te, out / "roc_curve.png", f"{args.run_name} — test ROC")
        for f in ("confusion_matrix.png", "roc_curve.png"):
            mlflow.log_artifact(str(out / f), "plots")
        preds = te_df[["path", "label"]].assign(p_ai=p_te, pred=(p_te >= thr).astype(int))
        preds.to_csv(out / "test_predictions.csv", index=False)
        mlflow.log_artifact(str(out / "test_predictions.csv"), "evaluation")

        meta = {"model_name": params["project"]["registered_model"], "run_id": run.info.run_id,
                "run_name": args.run_name, "image_size": size,
                "resize_mode": params["image"]["resize_mode"], "threshold": thr,
                "model_type": t["model"], "backbone": t["backbone"], "best_epoch": best_epoch,
                "test_metrics": {k: round(float(v), 4) for k, v in test_m.items()}}
        (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(out / "model_meta.json"), "keras_model")
        log_keras_model(model, out / "model.keras",
                        params["project"]["registered_model"] if args.register else None)

        print(f"\nbest epoch {best_epoch} | threshold {thr:.2f} (val F{params['eval']['threshold_beta']}"
              f"={fb:.3f})")
        print("TEST:", {k: round(float(v), 4) for k, v in test_m.items()})
        print(f"run id {run.info.run_id}")

        if args.report_key:
            write_report_assets(args.report_key, model, history, boundary, best_epoch, test_m,
                                val_m, test_m05, thr, trainable_log, out, title)


def write_report_assets(key, model, history, boundary, best_epoch, test_m, val_m, test_m05, thr,
                        trainable_log, out, title):
    import shutil

    FIG.mkdir(parents=True, exist_ok=True)
    for f in ("curves.png", "confusion_matrix.png", "roc_curve.png"):
        shutil.copy(out / f, FIG / f"{key}_{f}")
    if boundary:  # separate figures for the two transfer-learning stages (Task 2.3)
        h1 = {k: v[:boundary] for k, v in history.items()}
        h2 = {k: v[boundary:] for k, v in history.items()}
        ev.plot_history(h1, FIG / f"{key}_stage1_curves.png", "Stage 1 — frozen backbone")
        if h2.get("val_loss"):
            ev.plot_history(h2, FIG / f"{key}_stage2_curves.png", "Stage 2 — last block unfrozen")
    rows = [[n, typ, shp, f"{p:,}"] for n, typ, shp, p in layer_table(model)]
    write_table(f"{key}_layers", ["Layer", "Type", "Output shape", "Params"], rows, "lllr")
    m = {f"vg{key.capitalize()}{k.capitalize().replace('_', '')}": v for k, v in test_m.items()}
    m.update({f"vg{key.capitalize()}BestEpoch": best_epoch, f"vg{key.capitalize()}Thr": f"{thr:.2f}",
              f"vg{key.capitalize()}Params": f"{model.count_params():,}",
              f"vg{key.capitalize()}Trainable":
                  f"{sum(int(np.prod(w.shape)) for w in model.trainable_weights):,}",
              f"vg{key.capitalize()}ValFpr": val_m["false_positive_rate"],
              f"vg{key.capitalize()}AccHalf": test_m05["accuracy"]})
    write_macros(f"{key}_macros", m)
    if trainable_log:
        s2 = trainable_log.get("stage2_backbone_layers", [])
        write_table(f"{key}_trainable", ["Stage", "Trainable weight tensors", "Backbone layers unfrozen"],
                    [["1 (head only)", len(trainable_log.get("stage1", [])), "0"],
                     ["2 (fine-tune)", len(trainable_log.get("stage2", [])), len(s2)]], "lrr")
        (FIG.parent.parent / "generated" / f"{key}_unfrozen.tex").write_text(
            ", ".join(rf"\texttt{{{tex_escape(n)}}}" for n in s2) or "none")
    pd.DataFrame([test_m]).to_csv(FIG.parent.parent / "generated" / f"{key}_test_metrics.csv",
                                  index=False)


if __name__ == "__main__":
    main()
