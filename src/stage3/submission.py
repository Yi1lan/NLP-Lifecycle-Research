"""Submission I/O helpers for `dev.txt` and `test.txt`."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_binary_labels(path: str | Path) -> np.ndarray:
    """Load binary labels from `.npy`, `.txt`, or `.csv` files."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(file_path), dtype=np.int64)
    if suffix == ".txt":
        labels: list[int] = []
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                text = line.strip()
                if text == "":
                    continue
                labels.append(int(text))
        return np.asarray(labels, dtype=np.int64)
    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            return np.asarray([], dtype=np.int64)
        for key in ["label", "label_binary", "target", "y"]:
            if key in rows[0]:
                return np.asarray([int(float(row[key])) for row in rows], dtype=np.int64)
        # Fallback to first column when no standard label key exists.
        first_key = list(rows[0].keys())[0]
        return np.asarray([int(float(row[first_key])) for row in rows], dtype=np.int64)
    raise ValueError(f"Unsupported label file format: {file_path}")


def write_prediction_file(preds: np.ndarray, output_path: str | Path) -> Path:
    """Write one binary prediction per line."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in np.asarray(preds, dtype=np.int64).tolist():
            handle.write(f"{int(value)}\n")
    return path


def validate_prediction_file(
    prediction_path: str | Path,
    expected_count: int,
    label_name: str,
) -> list[str]:
    """Validate line count and binary format for a prediction file."""

    path = Path(prediction_path)
    errors: list[str] = []
    if not path.exists():
        return [f"{label_name}: file does not exist at {path}"]

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) != expected_count:
        errors.append(
            f"{label_name}: expected {expected_count} lines but found {len(lines)} in {path}"
        )
    for line_no, value in enumerate(lines, start=1):
        if value not in {"0", "1"}:
            errors.append(
                f"{label_name}: invalid label '{value}' at line {line_no} (must be 0 or 1)"
            )
            break
    return errors

