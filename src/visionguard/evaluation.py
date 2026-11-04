"""Metrics, threshold selection and report figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

LABELS = ["Real", "AI-generated"]


def choose_threshold(y_true: np.ndarray, p: np.ndarray, beta: float = 0.5) -> tuple[float, float]:
    """Pick the threshold on VALIDATION data that maximises F-beta (beta<1 favours precision of
    the AI verdict, i.e. fewer real photos wrongly called fake). Returns (threshold, score)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [fbeta_score(y_true, (p >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def compute_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan"),
        "false_positive_rate": fp / max(fp + tn, 1),  # real photos called AI
        "false_negative_rate": fn / max(fn + tp, 1),  # AI images called real
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": threshold,
    }


def plot_confusion(y_true, p, threshold, path: Path, title: str = "") -> Path:
    cm = confusion_matrix(y_true, (p >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or f"Confusion matrix (t={threshold:.2f})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc(y_true, p, path: Path, title: str = "ROC curve") -> Path:
    fpr, tpr, _ = roc_curve(y_true, p)
    auc = roc_auc_score(y_true, p)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate (real called AI)")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_history(history: dict, path: Path, title: str, stage_boundary: int | None = None) -> int:
    """Loss + accuracy curves with the best epoch (min val_loss) marked. Returns best epoch (1-based)."""
    val_loss = history["val_loss"]
    best = int(np.argmin(val_loss)) + 1
    epochs = np.arange(1, len(val_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[key], label=f"train {key}")
        ax.plot(epochs, history[f"val_{key}"], label=f"val {key}")
        ax.axvline(best, color="red", ls="--", lw=1, label=f"best epoch {best}")
        if stage_boundary:
            ax.axvline(stage_boundary + 0.5, color="grey", ls=":", lw=1, label="fine-tune starts")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return best
