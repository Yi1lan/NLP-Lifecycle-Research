#!/usr/bin/env python3
"""Create final Stage 5 submission files in required locations."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize final dev.txt/test.txt files.")
    parser.add_argument(
        "--ensemble-dir",
        type=str,
        default="outputs/stage4/final_ensemble",
        help="Directory containing ensemble `dev.txt`, `test.txt`, and `ensemble_summary.json`.",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/stage5/submission")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--no-root-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensemble_dir = Path(args.ensemble_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dev = ensemble_dir / "dev.txt"
    src_test = ensemble_dir / "test.txt"
    summary_path = ensemble_dir / "ensemble_summary.json"
    if not src_dev.exists() or not src_test.exists():
        raise FileNotFoundError(
            f"Could not find ensemble predictions in {ensemble_dir}. "
            "Run `scripts/stage4_run_matrix.py` first."
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing ensemble summary: {summary_path}")

    dst_dev = out_dir / "dev.txt"
    dst_test = out_dir / "test.txt"
    shutil.copyfile(src_dev, dst_dev)
    shutil.copyfile(src_test, dst_test)

    root_dev = PROJECT_ROOT / "dev.txt"
    root_test = PROJECT_ROOT / "test.txt"
    if not args.no_root_copy:
        shutil.copyfile(src_dev, root_dev)
        shutil.copyfile(src_test, root_test)

    validate_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "stage5_validate_submission.py"),
        "--dev",
        str(dst_dev),
        "--test",
        str(dst_test),
        "--data-dir",
        str(Path(args.data_dir).resolve()),
        "--ensemble-summary",
        str(summary_path),
        "--json-out",
        str(out_dir / "validation_report.json"),
    ]
    subprocess.run(validate_cmd, check=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dev_metrics = summary.get("dev_metrics", {})
    report_payload = {
        "source_ensemble_dir": str(ensemble_dir),
        "dev_txt": str(dst_dev),
        "test_txt": str(dst_test),
        "root_dev_txt": str(root_dev) if not args.no_root_copy else None,
        "root_test_txt": str(root_test) if not args.no_root_copy else None,
        "dev_metrics": dev_metrics,
        "baseline_dev_f1": 0.48,
        "beats_baseline_dev": float(dev_metrics.get("f1", 0.0)) > 0.48,
    }
    (out_dir / "submission_report.json").write_text(
        json.dumps(report_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(report_payload, indent=2))


if __name__ == "__main__":
    main()

