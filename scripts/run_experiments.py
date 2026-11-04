"""Task 3.5 — the experiment grid. Each run differs from the reference run (mnv2_default) in
exactly ONE meaningful setting, so any metric difference can be attributed to that change.

Usage: python scripts/run_experiments.py            # all runs
       python scripts/run_experiments.py --only mnv2_no_aug mnv2_img160
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name: (overrides, extra flags)
RUNS = {
    # reference run (also written into the report as "transfer")
    "mnv2_default":   ([], ["--report-key", "transfer", "--register"]),
    # from-scratch baseline (report key "baseline")
    "baseline_cnn":   (["train.model=baseline"], ["--report-key", "baseline", "--register"]),
    # one change each vs mnv2_default:
    "mnv2_lr_1e-4":   (["train.lr_head=0.0001"], []),                    # learning rate
    "mnv2_no_aug":    (["train.augment=false"], []),                     # augmentation
    "mnv2_frozen":    (["train.unfreeze=none"], []),                     # frozen layers
    "mnv2_img160":    (["image.size=160"], []),                          # image size
    "effb0_default":  (["train.backbone=efficientnetb0"], []),           # backbone
    # same config as mnv2_default but tracked by MLflow autologging (Task 3.3)
    "mnv2_autolog":   ([], ["--tracking", "autolog"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--extra", nargs="*", default=[], help="overrides applied to every run")
    args = ap.parse_args()
    failed = []
    for name, (overrides, flags) in RUNS.items():
        if args.only and name not in args.only:
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--run-name", name, *flags,
               "--set", *overrides, *args.extra]
        print("\n" + "=" * 80 + f"\n>>> {' '.join(cmd)}\n" + "=" * 80, flush=True)
        # keep going if one run fails, so a single problem does not cancel the rest of the grid
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            failed.append(name)
    if failed:
        print(f"\nFAILED runs: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
