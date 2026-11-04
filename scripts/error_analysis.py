"""Task 2.6 — 10 misclassified test images with measured cues for a hypothesis each,
and the bonus Grad-CAM heat maps (4 correct + 4 wrong).

Usage: python scripts/error_analysis.py --run-dir outputs/mnv2_default
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from visionguard.config import ROOT  # noqa: E402
from visionguard.imaging import decode_and_resize, open_image  # noqa: E402
from visionguard.reporting import FIG, write_table  # noqa: E402

NAMES = {0: "Real", 1: "AI"}


def cues(path: Path) -> dict:
    """Cheap, measurable image properties that commonly explain errors."""
    img = open_image(path.read_bytes())
    gray = img.convert("L")
    small = gray.copy()
    small.thumbnail((512, 512))
    lap = np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    hsv = np.asarray(img.convert("HSV").resize((128, 128)), dtype=np.float32)
    with Image.open(path) as raw:
        fmt = raw.format
    rgb = np.asarray(img.resize((64, 64)), np.float32)
    channel_spread = float(np.abs(rgb - rgb.mean(axis=2, keepdims=True)).max())
    return {"w": img.width, "h": img.height, "sharpness": float(lap.var()),
            "saturation": float(hsv[..., 1].mean()), "format": fmt, "gray": channel_spread < 4}


def hypothesis(row, c) -> str:
    h = []
    if min(c["w"], c["h"]) < 300:
        h.append("low resolution: up-scaling to the input size hides texture cues")
    if c["sharpness"] < 60:
        h.append("soft / blurred image: generator-like smoothness")
    elif c["sharpness"] > 900:
        h.append("very sharp, heavy detail or noise looks 'camera-like'")
    if c["saturation"] > 140:
        h.append("highly saturated, painted / illustration style")
    if c["gray"]:
        h.append("grey-scale image: no colour statistics to rely on")
    if c["w"] / c["h"] > 1.6 or c["h"] / c["w"] > 1.6:
        h.append("extreme aspect ratio squashed by resizing")
    if c["format"] == "PNG" and row.label == 0:
        h.append("real photo saved as PNG (most PNGs in data are AI)")
    if not h:
        h.append("no obvious low-level cue: likely content/style ambiguity (check watermark/text)")
    return "; ".join(h[:2])


def gradcam(model: keras.Model, x: np.ndarray) -> np.ndarray:
    names = [lyr.name for lyr in model.layers]
    x = tf.convert_to_tensor(x[None, ...])
    if "backbone" in names:
        prep = next(lyr for lyr in model.layers if lyr.name.endswith("_preprocess"))
        base = model.get_layer("backbone")
        head = model.layers[names.index("backbone") + 1:]
        with tf.GradientTape() as tape:
            feat = base(prep(x), training=False)
            tape.watch(feat)
            y = feat
            for lyr in head:
                y = lyr(y, training=False)
    else:
        sub = keras.Model(model.inputs, [model.get_layer("relu4").output, model.output])
        with tf.GradientTape() as tape:
            feat, y = sub(x, training=False)
    grads = tape.gradient(y, feat)[0]
    w = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(feat[0] * w, axis=-1)).numpy()
    return cam / (cam.max() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    run_dir = ROOT / args.run_dir
    meta = json.loads((run_dir / "model_meta.json").read_text())
    df = pd.read_csv(run_dir / "test_predictions.csv")
    df["wrong_conf"] = np.where(df.label == 1, 1 - df.p_ai, df.p_ai)
    wrong = df[df.pred != df.label].sort_values("wrong_conf", ascending=False).head(args.n)
    print(f"{int((df.pred != df.label).sum())} misclassified of {len(df)} test images")

    FIG.mkdir(parents=True, exist_ok=True)
    cols = 5
    rows_n = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows_n, cols, figsize=(15, 3.6 * rows_n))
    table = []
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i >= len(wrong):
            continue
        r = wrong.iloc[i]
        p = ROOT / r.path
        c = cues(p)
        img = open_image(p.read_bytes())
        img.thumbnail((320, 320))
        ax.imshow(img)
        ax.set_title(f"#{i+1} true={NAMES[r.label]} pred={NAMES[r.pred]}\np(AI)={r.p_ai:.2f}  "
                     f"{c['w']}x{c['h']}", fontsize=9)
        table.append([i + 1, NAMES[r.label], f"{r.p_ai:.2f}", f"{c['w']}x{c['h']}",
                      f"{c['sharpness']:.0f}", hypothesis(r, c)])
    fig.suptitle(f"Most confident test errors — {meta['run_name']} (threshold {meta['threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "errors_grid.png", dpi=110)
    plt.close(fig)
    write_table("errors", ["\\#", "True", "p(AI)", "Size", "Sharpness", "Hypothesis (from measured cues)"],
                table, r"rlrrrp{0.5\linewidth}")
    for row in table:
        print(row)

    # ---- Bonus: Grad-CAM for 4 correct + 4 wrong ----
    model = keras.models.load_model(run_dir / "best.keras")
    right = df[df.pred == df.label].sample(4, random_state=0)
    picks = pd.concat([right, wrong.head(4)])
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, (_, r) in zip(axes.ravel(), picks.iterrows()):
        x = decode_and_resize((ROOT / r.path).read_bytes(), meta["image_size"], meta["resize_mode"])
        cam = gradcam(model, x)
        cam = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(x.shape[:2][::-1],
                                                                     Image.Resampling.BILINEAR))
        ax.imshow(x.astype("uint8"))
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ok = "correct" if r.pred == r.label else "WRONG"
        ax.set_title(f"{ok}: true={NAMES[r.label]} p(AI)={r.p_ai:.2f}", fontsize=10,
                     color="green" if ok == "correct" else "red")
        ax.axis("off")
    fig.suptitle("Grad-CAM: top row correct predictions, bottom row errors")
    fig.tight_layout()
    fig.savefig(FIG / "gradcam.png", dpi=110)
    plt.close(fig)
    print(f"Saved {FIG/'errors_grid.png'} and {FIG/'gradcam.png'}")


if __name__ == "__main__":
    main()
