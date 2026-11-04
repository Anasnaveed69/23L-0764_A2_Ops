"""Task 2.7 toy part — Keras Tuner Hyperband on Fashion-MNIST: learning rate x hidden units."""
import sys
from pathlib import Path

import keras
import keras_tuner as kt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visionguard.reporting import write_table  # noqa: E402


def build(hp):
    model = keras.Sequential([
        keras.Input((28, 28)), keras.layers.Flatten(),
        keras.layers.Dense(hp.Int("units", 32, 512, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax")])
    model.compile(keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3, 1e-4])),
                  "sparse_categorical_crossentropy", ["accuracy"])
    return model


def main():
    keras.utils.set_random_seed(0)
    (x, y), (xt, yt) = keras.datasets.fashion_mnist.load_data()
    x, xt = x / 255.0, xt / 255.0
    tuner = kt.Hyperband(build, objective="val_accuracy", max_epochs=8, factor=3, seed=0,
                         directory=str(ROOT / "tuner_runs"), project_name="toy_fmnist",
                         overwrite=True)
    tuner.search(x, y, validation_split=0.2, verbose=1,
                 callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=2)])
    tuner.results_summary(num_trials=5)
    best = tuner.get_best_hyperparameters(1)[0]
    print(f"\nBEST: units={best.get('units')} lr={best.get('lr')}")
    rows = []
    for t in sorted(tuner.oracle.trials.values(), key=lambda t: -(t.score or 0))[:8]:
        rows.append([t.trial_id, t.hyperparameters.get("units"), t.hyperparameters.get("lr"),
                     t.hyperparameters.values.get("tuner/epochs", "-"), f"{t.score:.4f}"])
    write_table("tuner_toy", ["Trial", "Units", "LR", "Epochs", "val accuracy"], rows, "lrrrr")
    model = tuner.hypermodel.build(best)
    model.fit(x, y, epochs=5, validation_split=0.2, verbose=2)
    print("test accuracy with best settings:", round(model.evaluate(xt, yt, verbose=0)[1], 4))


if __name__ == "__main__":
    main()
