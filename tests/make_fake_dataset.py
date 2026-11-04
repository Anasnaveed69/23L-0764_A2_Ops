"""Generate a tiny synthetic dataset with the same folder layout as the Kaggle one.

Used by CI and for smoke-testing the pipeline without the real data. It deliberately contains
the awkward cases the audit must catch: a corrupt file, an exact duplicate, a WEBP, an RGBA PNG,
a grey-scale image and a non-image file.

Usage: python tests/make_fake_dataset.py data/raw 60
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def smooth(rng, w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    a = rng.uniform(0.005, 0.03, 3)
    img = np.stack([127 + 120 * np.sin(xx * a[i] + yy * a[(i + 1) % 3]) for i in range(3)], -1)
    return Image.fromarray(img.astype("uint8"))


def noisy(rng, w, h):
    base = rng.integers(40, 200, 3)
    img = base + rng.normal(0, 45, (h, w, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"))


def main(root="data/raw", n=60):
    root, n = Path(root), int(n)
    rng = np.random.default_rng(0)
    ai, real = root / "AiArtData" / "AiArtData", root / "RealArt" / "RealArt"
    ai.mkdir(parents=True, exist_ok=True)
    real.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        w, h = (512, 512) if i % 3 else (int(rng.integers(300, 900)), int(rng.integers(300, 700)))
        smooth(rng, w, h).save(ai / f"ai_{i:03d}.{'png' if i % 2 else 'jpg'}")
        w, h = int(rng.integers(250, 1000)), int(rng.integers(250, 800))
        noisy(rng, w, h).save(real / f"real_{i:03d}.jpg", quality=85)
    smooth(rng, 400, 400).save(ai / "ai_webp.webp")
    smooth(rng, 300, 300).convert("RGBA").save(ai / "ai_rgba.png")
    noisy(rng, 400, 300).convert("L").save(real / "real_gray.jpg")
    (real / "real_corrupt.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"garbage" * 50)
    (ai / "notes.txt").write_text("not an image")
    (real / "real_dup.jpg").write_bytes((real / "real_000.jpg").read_bytes())
    print(f"fake dataset written to {root} ({2 * n + 6} files)")


if __name__ == "__main__":
    main(*sys.argv[1:])
