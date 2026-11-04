"""Task 1.1 — print exact library versions and whether TensorFlow can see a GPU."""
import getpass
import importlib
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKGS = ["tensorflow", "keras", "numpy", "pandas", "PIL", "sklearn", "matplotlib", "mlflow",
        "keras_tuner", "dvc", "flask", "sqlalchemy"]

lines = [
    f"VisionGuard environment check — {datetime.now():%Y-%m-%d %H:%M:%S}",
    f"user@host      : {getpass.getuser()}@{socket.gethostname()}",
    f"python         : {sys.version.split()[0]} ({sys.executable})",
    f"platform       : {platform.platform()}",
    f"in virtualenv  : {sys.prefix != sys.base_prefix}",
]
for name in PKGS:
    try:
        mod = importlib.import_module(name)
        lines.append(f"{name:<15}: {getattr(mod, '__version__', '?')}")
    except ImportError:
        lines.append(f"{name:<15}: NOT INSTALLED")

import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
lines.append(f"TF built with CUDA: {tf.test.is_built_with_cuda()}")
lines.append(f"GPUs visible   : {len(gpus)} {[g.name for g in gpus]}")
lines.append(f"CPUs visible   : {[d.name for d in tf.config.list_physical_devices('CPU')]}")

text = "\n".join(lines)
print(text)
out = ROOT / "reports" / "env.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text + "\n")
