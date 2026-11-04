"""Image decoding shared by training AND serving.

Both the tf.data training pipeline and the Flask/API inference path call `decode_and_resize`,
so the pixels the model sees in production are produced by exactly the same code that produced
the pixels it was trained on (no training/serving skew).

Pillow is used instead of tf.io.decode_image because the dataset contains formats TensorFlow
cannot decode (e.g. WEBP, CMYK JPEGs, palette PNGs with transparency).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 50_000_000  # refuse decompression bombs (~7000x7000)

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "MPO"}


class ImageDecodeError(ValueError):
    """Raised when bytes look like an image but cannot be decoded."""


def open_image(data: bytes) -> Image.Image:
    """Fully decode bytes into an RGB PIL image (raises ImageDecodeError)."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # cheap structural check (truncated / corrupt headers)
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()  # forces the full pixel decode, catches truncated pixel data
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError) as exc:
        raise ImageDecodeError(f"cannot decode image: {exc}") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ImageDecodeError(f"unsupported image format {fmt}")
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)  # animated GIF/WEBP: use the first frame
    img = ImageOps.exif_transpose(img)  # respect camera orientation
    if img.mode in ("RGBA", "LA", "P"):
        # Composite transparency onto white so transparent pixels do not become black.
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def resize(img: Image.Image, size: int, mode: str = "squash") -> Image.Image:
    if mode == "pad":
        # Keep aspect ratio, letterbox with black borders to a square.
        return ImageOps.pad(img, (size, size), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
    # Plain resize: aspect ratio is NOT preserved; wide images are squeezed horizontally.
    return img.resize((size, size), Image.Resampling.BILINEAR)


def decode_and_resize(data: bytes, size: int, mode: str = "squash") -> np.ndarray:
    """bytes -> float32 array (size, size, 3) with raw 0..255 pixel values.

    Scaling to the range a backbone expects happens INSIDE the Keras model (Rescaling layer),
    so callers never have to remember backbone-specific preprocessing.
    """
    img = resize(open_image(data), size, mode)
    return np.asarray(img, dtype=np.float32)
