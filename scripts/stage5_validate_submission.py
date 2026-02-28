#!/usr/bin/env python3
"""Validate `dev.txt`/`test.txt` formatting and ordering consistency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage3.data import dev_order_hash, load_stage3_dataset, test_order_hash  # pylint: disable=wrong-import-position
from src.stage3.submission import validate_prediction_file  # pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 5 submission files.")
    parser.add_argument("--dev", type=str, default="dev.txt")
    parser.add_argument("--test", type=str, default="test.txt")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument(
        "--ensemble-summary",
        type=str,
        default="outputs/stage4/final_ensemble/ensemble_summary.json",
    )
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_bundle = load_stage3_dataset(args.data_dir)
    expected_dev = len(data_bundle.dev)
    expected_test = len(data_bundle.test)
    expected_dev_hash = dev_order_hash(data_bundle)
    expected_test_hash = test_order_hash(data_bundle)

    errors = []
    errors.extend(validate_prediction_file(args.dev, expected_dev, "dev"))
    errors.extend(validate_prediction_file(args.test, expected_test, "test"))

    summary_path = Path(args.ensemble_summary).resolve()
    summary_payload = None
    if summary_path.exists():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        reported_dev_hash = summary_payload.get("dev_order_hash")
        reported_test_hash = summary_payload.get("test_order_hash")
        if reported_dev_hash and reported_dev_hash != expected_dev_hash:
            errors.append(
                "dev order hash mismatch between ensemble metadata and official dev order."
            )
        if reported_test_hash and reported_test_hash != expected_test_hash:
            errors.append(
                "test order hash mismatch between ensemble metadata and official test order."
            )
    else:
        errors.append(f"ensemble summary not found at: {summary_path}")

    report = {
        "dev_file": str(Path(args.dev).resolve()),
        "test_file": str(Path(args.test).resolve()),
        "expected_dev_lines": expected_dev,
        "expected_test_lines": expected_test,
        "expected_dev_order_hash": expected_dev_hash,
        "expected_test_order_hash": expected_test_hash,
        "ensemble_summary": str(summary_path),
        "errors": errors,
        "ok": len(errors) == 0,
    }
    if args.json_out:
        output = Path(args.json_out).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

