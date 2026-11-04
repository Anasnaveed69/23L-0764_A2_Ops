"""Task 1.2 — dataset audit.

Walks data/raw, and for EVERY file records: class, file size, width, height, colour mode, format,
SHA-256 content hash, and whether it can be fully decoded. Then reports per-class counts,
min/median/max of width/height/size, corrupt files and exact duplicates (same SHA-256).

Usage: python scripts/01_audit.py
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from visionguard.config import ROOT, load_params, resolve  # noqa: E402
from visionguard.imaging import ImageDecodeError, open_image  # noqa: E402
from visionguard.reporting import save_json, write_macros, write_table  # noqa: E402

CLASS_NAME = {0: "Real", 1: "AI-generated"}


def label_for(rel: Path, class_map: dict) -> int | None:
    keys = sorted(class_map, key=len, reverse=True)  # "aiart" before "ai"
    for part in rel.parts[:-1]:
        low = part.lower()
        for k in keys:
            if k in low:
                return int(class_map[k])
    return None


def audit_file(path: Path, rel: Path, label) -> dict:
    data = path.read_bytes()
    row = {
        "path": str(rel.as_posix()),
        "label": label,
        "class": CLASS_NAME.get(label, "unknown"),
        "ext": path.suffix.lower(),
        "file_size_kb": round(len(data) / 1024, 2),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": None, "height": None, "mode": None, "format": None,
        "readable": False, "error": "",
    }
    try:
        with Image.open(path) as im:  # header info (original mode, before RGB conversion)
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
        open_image(data)  # full decode with the SAME function used in training/serving
        row["readable"] = True
    except (ImageDecodeError, OSError, ValueError, SyntaxError) as exc:
        row["error"] = str(exc)[:120]
    return row


def main():
    params = load_params(sys.argv[1:])
    raw = resolve(params["data"]["raw_dir"])
    if not raw.exists():
        sys.exit(f"{raw} not found. Download the Kaggle dataset into data/raw first (see RUNBOOK).")
    files = sorted(p for p in raw.rglob("*") if p.is_file() and not p.name.startswith("."))
    rows = []
    for i, p in enumerate(files, 1):
        rel = p.relative_to(ROOT)
        rows.append(audit_file(p, rel, label_for(p.relative_to(raw), params["data"]["class_map"])))
        if i % 200 == 0:
            print(f"  audited {i}/{len(files)}")
    df = pd.DataFrame(rows)

    # Duplicates: every file after the first with the same hash is a duplicate.
    df["dup_rank"] = df.groupby("sha256").cumcount()
    df["is_duplicate"] = df["dup_rank"] > 0
    grp = df.groupby("sha256")["label"].nunique()
    conflict_hashes = set(grp[grp > 1].index)  # same bytes, different labels = label noise
    df["label_conflict"] = df["sha256"].isin(conflict_hashes)
    min_side = params["data"]["min_side"]
    df["too_small"] = df["readable"] & ((df["width"].fillna(0) < min_side) |
                                        (df["height"].fillna(0) < min_side))

    out = resolve(params["data"]["audit_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # ---------- summary tables ----------
    print("\n=== Files per class ===")
    rows_tex = []
    for cls, g in df.groupby("class"):
        r = [cls, len(g), int(g.readable.sum()), int((~g.readable).sum()),
             int(g.is_duplicate.sum()), int(g.too_small.sum())]
        rows_tex.append(r)
        print(f"{cls:<13} total={r[1]:5d} readable={r[2]:5d} corrupt={r[3]:3d} dup={r[4]:3d}")
    rows_tex.append(["Total", len(df), int(df.readable.sum()), int((~df.readable).sum()),
                     int(df.is_duplicate.sum()), int(df.too_small.sum())])
    write_table("audit_counts", ["Class", "Files", "Readable", "Corrupt", "Duplicates", "Too small"],
                rows_tex)

    ok = df[df.readable]
    stat_rows = []
    for cls, g in ok.groupby("class"):
        for col, name in [("width", "Width (px)"), ("height", "Height (px)"),
                          ("file_size_kb", "File size (KB)")]:
            stat_rows.append([cls, name, f"{g[col].min():.0f}", f"{g[col].median():.0f}",
                              f"{g[col].max():.0f}"])
    write_table("audit_stats", ["Class", "Quantity", "Min", "Median", "Max"], stat_rows, "llrrr")
    print("\n=== min / median / max (readable files) ===")
    for r in stat_rows:
        print("  ", " | ".join(map(str, r)))

    fmt_rows = []
    for (cls, f, m), n in ok.groupby(["class", "format", "mode"]).size().items():
        fmt_rows.append([cls, f, m, n])
    write_table("audit_formats", ["Class", "Format", "Colour mode", "Files"], fmt_rows, "lllr")

    corrupt = df[~df.readable][["path", "error"]].head(15).values.tolist()
    write_table("audit_corrupt", ["File", "Error"], corrupt or [["none", "-"]],
                r"p{0.42\linewidth}p{0.42\linewidth}")
    dup_groups = df[df.sha256.duplicated(keep=False)].groupby("sha256")["path"].apply(list)
    print(f"\nexact-duplicate groups: {len(dup_groups)}, cross-class conflicts: "
          f"{len(conflict_hashes)}, unknown-label files: {int((df.label.isna()).sum())}")
    summary = {
        "n_files": len(df), "n_readable": int(df.readable.sum()),
        "n_corrupt": int((~df.readable).sum()), "n_duplicates": int(df.is_duplicate.sum()),
        "n_dup_groups": int(len(dup_groups)), "n_label_conflicts": len(conflict_hashes),
        "n_unknown_label": int(df.label.isna().sum()),
        "per_class": df.groupby("class").size().to_dict(),
        "example_duplicate_groups": [v for v in dup_groups.head(5).tolist()],
    }
    save_json("audit_summary", summary)
    write_macros("audit_macros", {
        "vgNFiles": summary["n_files"], "vgNReadable": summary["n_readable"],
        "vgNCorrupt": summary["n_corrupt"], "vgNDup": summary["n_duplicates"],
        "vgNDupGroups": summary["n_dup_groups"], "vgNConflicts": summary["n_label_conflicts"],
        "vgNAI": summary["per_class"].get("AI-generated", 0),
        "vgNReal": summary["per_class"].get("Real", 0),
    })
    print(f"\nWrote {out} and report/generated/audit_*.tex")


if __name__ == "__main__":
    main()
