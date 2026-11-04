"""Task 4.1 / 7 / 8 evidence helper — send several test images and print a results table.

  # through the Flask UI (local or remote inference, whatever the web app is configured for):
  python scripts/send_samples.py --target flask --url http://localhost:5000 --n 6
  # straight to the cloud model API:
  python scripts/send_samples.py --target api --url https://<ip>.sslip.io --n 6

Images are taken from the TEST split (never seen in training), half AI and half real.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["flask", "api"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--files", nargs="*", help="explicit image paths instead of test-split samples")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    if args.files:
        items = [(Path(f), "?") for f in args.files]
    else:
        test = pd.read_csv(ROOT / "data/splits/test.csv")
        k = args.n // 2
        pick = pd.concat([test[test.label == 1].sample(k, random_state=7),
                          test[test.label == 0].sample(args.n - k, random_state=7)])
        items = [(ROOT / p, "AI-generated" if lbl == 1 else "Real")
                 for p, lbl in zip(pick.path, pick.label)]

    s = requests.Session()
    print(f"{'file':<34} {'truth':<13} {'verdict':<13} {'conf':>6} {'ms':>7}")
    for path, truth in items:
        t0 = time.perf_counter()
        with open(path, "rb") as fh:
            files = {"image": (path.name, fh)}
            if args.target == "api":
                r = s.post(f"{args.url.rstrip('/')}/v1/predict", files=files, timeout=60,
                           headers={"X-API-Key": os.environ["MODEL_API_KEY"]})
                j = r.json()
                verdict, conf = j.get("label", r.status_code), j.get("confidence", 0)
            else:
                r = s.post(f"{args.url.rstrip('/')}/predict", files=files, timeout=60)
                html = r.text
                verdict = ("AI-generated" if "Likely AI-generated" in html else
                           "Real" if "Likely a real photo" in html else f"HTTP {r.status_code}")
                m = re.search(r"<strong>([\d.]+)%</strong> confidence", html)
                conf = float(m.group(1)) / 100 if m else 0
        ms = (time.perf_counter() - t0) * 1000
        print(f"{path.name[:33]:<34} {truth:<13} {verdict:<13} {conf:>6.1%} {ms:>7.0f}")
    if args.target == "flask":
        print(f"\nOpen {args.url.rstrip('/')}/history to see these rows in the database trail.")


if __name__ == "__main__":
    sys.exit(main())
