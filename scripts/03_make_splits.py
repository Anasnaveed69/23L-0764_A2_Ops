"""Create stratified train/val/test split lists from the audit table.

Cleaning rules (decided from the audit):
  * drop unreadable files and images smaller than data.min_side
  * drop files whose folder did not map to a class
  * drop ALL copies of an image that appears under both classes (label noise)
  * keep only the first copy of exact duplicates, so the same image can never be in train AND
    test (that would leak and inflate test accuracy)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from visionguard.config import load_params, resolve  # noqa: E402
from visionguard.reporting import write_macros, write_table  # noqa: E402


def main():
    params = load_params(sys.argv[1:])
    d = params["data"]
    seed = params["project"]["seed"]
    df = pd.read_csv(resolve(d["audit_csv"]))
    before = len(df)
    df = df[df.readable & ~df.too_small & df.label.notna() & ~df.label_conflict & ~df.is_duplicate]
    df = df.assign(label=df.label.astype(int))[["path", "label", "class", "sha256", "width",
                                                 "height"]]
    print(f"kept {len(df)}/{before} files after cleaning")

    train, temp = train_test_split(df, test_size=d["val_frac"] + d["test_frac"],
                                   stratify=df.label, random_state=seed)
    rel_test = d["test_frac"] / (d["val_frac"] + d["test_frac"])
    val, test = train_test_split(temp, test_size=rel_test, stratify=temp.label, random_state=seed)

    out = resolve(d["splits_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sort_values("path").to_csv(out / f"{name}.csv", index=False)
        n_ai, n_real = int((part.label == 1).sum()), int((part.label == 0).sum())
        rows.append([name, len(part), n_ai, n_real, f"{100 * n_ai / len(part):.1f}"])
        print(f"{name:<5} {len(part):5d}  AI={n_ai}  Real={n_real}")
    assert not set(train.sha256) & set(test.sha256), "leak: same image in train and test"
    write_table("splits", ["Split", "Images", "AI-generated", "Real", "% AI"], rows)
    write_macros("split_macros", {"vgNTrain": len(train), "vgNVal": len(val), "vgNTest": len(test),
                                  "vgNClean": len(df)})


if __name__ == "__main__":
    main()
