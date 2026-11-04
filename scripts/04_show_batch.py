"""Task 2.1 — render one augmented training batch (and the same images un-augmented)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from visionguard.config import load_params, set_seeds  # noqa: E402
from visionguard.data import CLASS_NAMES, build_augmenter, make_dataset, read_split  # noqa: E402
from visionguard.reporting import FIG  # noqa: E402


def grid(batch, labels, path, title, n=12):
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(np.clip(batch[i], 0, 255).astype("uint8"))
        ax.set_title(CLASS_NAMES[int(labels[i])], fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    params = load_params(sys.argv[1:])
    set_seeds(params["project"]["seed"])
    df = read_split(params, "train").sample(16, random_state=1)
    params["train"]["batch_size"] = 16
    FIG.mkdir(parents=True, exist_ok=True)
    # Same 16 files, no shuffle, so the two figures line up image-by-image.
    plain = make_dataset(df, params, training=False, augment=False)
    xb, yb = next(iter(plain))
    grid(xb.numpy(), yb.numpy(), FIG / "batch_plain.png", "Decoded + resized (no augmentation)")
    # Apply the exact augmenter the training pipeline uses to the same batch.
    xa = build_augmenter(params["project"]["seed"])(xb, training=True)
    grid(np.asarray(xa), yb.numpy(), FIG / "batch_augmented.png",
         "Same batch after random flip / rotation / zoom / translation / contrast")
    print(f"batch tensor shape {xb.shape}, dtype {xb.dtype}, value range "
          f"[{float(np.min(xb)):.0f}, {float(np.max(xb)):.0f}]")
    print(f"Saved {FIG/'batch_augmented.png'}")


if __name__ == "__main__":
    main()
