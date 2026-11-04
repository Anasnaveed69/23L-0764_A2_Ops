"""Task 1.3 — visual EDA: sample grid per class + width/height/file-size/aspect histograms."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from visionguard.config import ROOT, load_params, resolve  # noqa: E402
from visionguard.imaging import open_image  # noqa: E402
from visionguard.reporting import FIG, write_macros  # noqa: E402

COLORS = {"AI-generated": "#d1495b", "Real": "#00798c"}


def main():
    params = load_params(sys.argv[1:])
    df = pd.read_csv(resolve(params["data"]["audit_csv"]))
    ok = df[df.readable & ~df.is_duplicate & df["class"].isin(COLORS)].copy()
    ok["aspect"] = ok.width / ok.height
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(params["project"]["seed"])

    # 1) sample grid: 2 rows x 6 columns per class
    n_cols, n_rows = 6, 2
    fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(13, 9.2))
    for c_i, cls in enumerate(["AI-generated", "Real"]):
        g = ok[ok["class"] == cls]
        pick = g.iloc[rng.choice(len(g), size=min(len(g), n_cols * n_rows), replace=False)]
        for k, (_, row) in enumerate(pick.iterrows()):
            ax = axes[c_i * n_rows + k // n_cols, k % n_cols]
            img = open_image((ROOT / row.path).read_bytes())
            img.thumbnail((300, 300))
            ax.imshow(img)
            ax.set_title(f"{cls}\n{row.width}x{row.height} {row.format}", fontsize=8,
                         color=COLORS[cls])
        for ax in axes[c_i * n_rows:(c_i + 1) * n_rows].ravel():
            ax.axis("off")
    fig.suptitle("Random samples per class (top: AI-generated, bottom: Real)")
    fig.tight_layout()
    fig.savefig(FIG / "eda_samples.png", dpi=110)
    plt.close(fig)

    # 2) histograms
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    specs = [("width", "Width (px)", True), ("height", "Height (px)", True),
             ("file_size_kb", "File size (KB)", True), ("aspect", "Aspect ratio (w/h)", False)]
    for ax, (col, title, logx) in zip(axes, specs):
        data_all = ok[col]
        bins = (np.logspace(np.log10(max(data_all.min(), 1)), np.log10(data_all.max()), 35)
                if logx else 35)
        for cls, color in COLORS.items():
            vals = ok.loc[ok["class"] == cls, col]
            ax.hist(vals, bins=bins, alpha=0.55, color=color, label=f"{cls} (med {vals.median():.0f})"
                    if col != "aspect" else f"{cls} (med {vals.median():.2f})")
        if logx:
            ax.set_xscale("log")
        ax.set_title(title)
        ax.set_ylabel("images")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "eda_histograms.png", dpi=130)
    plt.close(fig)

    med = ok.groupby("class")[["width", "height", "file_size_kb", "aspect"]].median()
    print(med.round(2))
    write_macros("eda_macros", {
        "vgMedWAI": int(med.loc["AI-generated", "width"]),
        "vgMedHAI": int(med.loc["AI-generated", "height"]),
        "vgMedWReal": int(med.loc["Real", "width"]),
        "vgMedHReal": int(med.loc["Real", "height"]),
        "vgMedKBAI": int(med.loc["AI-generated", "file_size_kb"]),
        "vgMedKBReal": int(med.loc["Real", "file_size_kb"]),
        "vgSquarePctAI": f"{100 * (ok[ok['class'] == 'AI-generated'].aspect.between(0.98, 1.02)).mean():.0f}",
        "vgSquarePctReal": f"{100 * (ok[ok['class'] == 'Real'].aspect.between(0.98, 1.02)).mean():.0f}",
    })
    print(f"Saved {FIG/'eda_samples.png'} and {FIG/'eda_histograms.png'}")


if __name__ == "__main__":
    main()
