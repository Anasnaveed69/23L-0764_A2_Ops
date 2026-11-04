"""Task 3.1 / 3.2 tool lab — tiny Keras model on Fashion-MNIST tracked with MLflow.

Local (3.1):   python labs/mlflow_toy.py --lr 0.001 --local
               python labs/mlflow_toy.py --lr 0.01  --local
               mlflow ui --backend-store-uri ./mlruns_toy      -> http://127.0.0.1:5000
DagsHub (3.2): python labs/mlflow_toy.py --lr 0.001             (reads MLFLOW_* from .env)
"""
import argparse
import os
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--units", type=int, default=64)
    ap.add_argument("--local", action="store_true", help="log to ./mlruns_toy instead of DagsHub")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if args.local or not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri((ROOT / "mlruns_toy").as_uri())
    mlflow.set_experiment("toy-fashion-mnist")
    keras.utils.set_random_seed(0)

    (x, y), (xt, yt) = keras.datasets.fashion_mnist.load_data()
    x, xt = x[:10000] / 255.0, xt[:2000] / 255.0
    y, yt = y[:10000], yt[:2000]
    model = keras.Sequential([
        keras.Input((28, 28)), keras.layers.Flatten(),
        keras.layers.Dense(args.units, activation="relu"), keras.layers.Dense(10, "softmax")])
    model.compile(keras.optimizers.Adam(args.lr), "sparse_categorical_crossentropy", ["accuracy"])

    with mlflow.start_run(run_name=f"toy_lr_{args.lr}") as run:
        mlflow.set_tags({"lab": "3.1", "author": "23L-0764", "dataset": "fashion_mnist[:10k]"})
        mlflow.log_params({"lr": args.lr, "epochs": args.epochs, "units": args.units,
                           "optimizer": "adam"})
        h = model.fit(x, y, validation_data=(xt, yt), epochs=args.epochs, batch_size=128,
                      callbacks=[EpochLogger()], verbose=2)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(np.arange(1, args.epochs + 1), h.history["loss"], label="train")
        ax.plot(np.arange(1, args.epochs + 1), h.history["val_loss"], label="val")
        ax.set_title(f"loss (lr={args.lr})")
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/loss_curve.png")
        mlflow.tensorflow.log_model(model, artifact_path="model")
        print(f"run {run.info.run_id} -> {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
