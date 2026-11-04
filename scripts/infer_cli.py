"""Task 4.1 evidence — load the inference module ONCE, then classify several images.

Usage: python scripts/infer_cli.py path/to/ai.png path/to/real.jpg
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionguard.inference import Predictor  # noqa: E402

t0 = time.perf_counter()
pred = Predictor("models/champion.keras")
print(f"model loaded once in {time.perf_counter() - t0:.2f}s "
      f"({pred.model_name} v{pred.model_version}, size {pred.image_size}, threshold {pred.threshold})")
for path in sys.argv[1:]:
    out = pred.predict_bytes(Path(path).read_bytes())
    print(f"{Path(path).name:<30} -> {out.label:<13} confidence {out.confidence:6.1%}  "
          f"p(AI)={out.prob_ai:.3f}  {out.latency_ms:.0f} ms")
