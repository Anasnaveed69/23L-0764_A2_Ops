"""tf.data input pipeline built from the split lists produced by scripts/03_make_splits.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from .config import resolve
from .imaging import decode_and_resize

AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = {0: "Real", 1: "AI-generated"}


def read_split(params: dict, name: str) -> pd.DataFrame:
    df = pd.read_csv(resolve(params["data"]["splits_dir"]) / f"{name}.csv")
    return df


def _loader(size: int, mode: str):
    def _load(path: bytes) -> np.ndarray:
        data = Path(resolve(path.decode())).read_bytes()
        # uint8 keeps the in-memory cache 4x smaller than float32
        return decode_and_resize(data, size, mode).astype(np.uint8)

    def _tf_load(path, label):
        img = tf.numpy_function(_load, [path], tf.uint8, stateful=False)
        img.set_shape((size, size, 3))
        return img, tf.cast(label, tf.float32)

    return _tf_load


def build_augmenter(seed: int) -> keras.Sequential:
    """Mild, label-preserving augmentation. Deliberately NO blur / JPEG re-compression / noise:
    those would destroy the high-frequency generator artefacts the classifier relies on."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed),
            keras.layers.RandomRotation(0.05, fill_mode="reflect", seed=seed),
            keras.layers.RandomZoom((-0.1, 0.1), fill_mode="reflect", seed=seed),
            keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed),
            keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="augmentation",
    )


def make_dataset(df: pd.DataFrame, params: dict, training: bool,
                 image_size: int | None = None, augment: bool | None = None) -> tf.data.Dataset:
    """Stages: list -> decode+resize (parallel) -> cache -> shuffle -> augment -> batch -> prefetch."""
    size = image_size or params["image"]["size"]
    mode = params["image"]["resize_mode"]
    seed = params["project"]["seed"]
    bs = params["train"]["batch_size"]
    do_aug = params["train"]["augment"] if augment is None else augment

    ds = tf.data.Dataset.from_tensor_slices((df["path"].astype(str).values, df["label"].values))
    ds = ds.map(_loader(size, mode), num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()  # decode each file once; later epochs read from RAM
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(bs)
    if training and do_aug:
        aug = build_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(tf.cast(x, tf.float32), training=True), y),
                    num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def class_weights(df: pd.DataFrame) -> dict[int, float]:
    """Inverse-frequency weights so the minority class is not ignored."""
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    return {int(k): total / (2.0 * v) for k, v in counts.items()}
