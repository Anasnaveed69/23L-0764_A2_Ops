"""Model builders: a from-scratch baseline CNN and a transfer-learning model."""
from __future__ import annotations

import os

import keras
from keras import layers

# Offline tests set VG_BACKBONE_WEIGHTS=none so no ImageNet download is attempted.
BACKBONE_WEIGHTS = None if os.getenv("VG_BACKBONE_WEIGHTS", "imagenet") == "none" else "imagenet"

# Name of the first layer of the LAST block of each backbone (used for "unfreeze last block").
LAST_BLOCK_START = {
    "mobilenetv2": "block_16_expand",
    "efficientnetb0": "block7a_expand_conv",
}


def build_baseline(size: int, dropout: float = 0.4) -> keras.Model:
    """Small CNN from scratch: 4 conv blocks (conv-BN-ReLU-pool), GAP head, sigmoid output."""
    inputs = keras.Input((size, size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="scale_0_1")(inputs)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout / 2, name="dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def _backbone(name: str, size: int) -> tuple[keras.Model, keras.layers.Layer]:
    shape = (size, size, 3)
    if name == "mobilenetv2":
        base = keras.applications.MobileNetV2(input_shape=shape, include_top=False,
                                              weights=BACKBONE_WEIGHTS, name="backbone")
        # mobilenet_v2.preprocess_input maps [0,255] -> [-1,1]; same thing as a saveable layer.
        prep = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")
    elif name == "efficientnetb0":
        # NOT name="backbone" here: EfficientNet derives the ImageNet weights file name from the
        # model name ("<name>_notop.h5"), so a custom name gives a 403 on download. Load it under
        # its own name, then wrap it in a sub-model called "backbone" (same layers, same weights).
        eff = keras.applications.EfficientNetB0(input_shape=shape, include_top=False,
                                                weights=BACKBONE_WEIGHTS)
        base = keras.Model(eff.input, eff.output, name="backbone")
        # EfficientNet has its own Rescaling+Normalization inside and expects raw [0,255].
        prep = layers.Identity(name="efficientnet_preprocess")
    else:
        raise ValueError(f"unknown backbone {name}")
    return base, prep


def build_transfer(size: int, backbone: str = "mobilenetv2", dropout: float = 0.3) -> keras.Model:
    base, prep = _backbone(backbone, size)
    base.trainable = False
    inputs = keras.Input((size, size, 3), name="image")
    x = prep(inputs)
    # training=False keeps BatchNorm statistics frozen even after we unfreeze layers, which is
    # essential on a small dataset (otherwise batch stats from 32 images overwrite ImageNet ones).
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name=f"transfer_{backbone}")


def unfreeze(model: keras.Model, backbone: str, how: str | int) -> list[str]:
    """Unfreeze the last block (or last N layers) of the backbone. BatchNorm layers stay frozen.
    Returns names of backbone layers that are now trainable."""
    base = model.get_layer("backbone")
    base.trainable = True
    names = [lyr.name for lyr in base.layers]
    if how == "none" or how == 0:
        start = len(names)
    elif how == "last_block":
        start = names.index(LAST_BLOCK_START[backbone])
    else:
        start = max(0, len(names) - int(how))
    for i, lyr in enumerate(base.layers):
        lyr.trainable = i >= start and not isinstance(lyr, layers.BatchNormalization)
    return [lyr.name for lyr in base.layers if lyr.trainable]


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def trainable_summary(model: keras.Model) -> list[tuple[str, str, int]]:
    """(layer, trainable?, #params) for top-level layers plus trainable backbone layers."""
    rows = []
    for lyr in model.layers:
        rows.append((lyr.name, "yes" if lyr.trainable and lyr.trainable_weights else "no",
                     int(sum(int(w.numpy().size) for w in lyr.trainable_weights))))
    return rows


def layer_table(model: keras.Model) -> list[tuple[str, str, str, int]]:
    """(name, type, output shape, params) for the report's per-layer table."""
    rows = []
    for lyr in model.layers:
        try:
            shape = str(tuple(lyr.output.shape))
        except Exception:  # noqa: BLE001
            shape = "-"
        rows.append((lyr.name, lyr.__class__.__name__, shape, int(lyr.count_params())))
    return rows
