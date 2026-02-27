"""Probability ensembling and threshold selection for final submission files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from .config import THRESHOLD_END, THRESHOLD_START, THRESHOLD_STEP
from .losses import binary_metrics_from_probs, search_best_threshold
from .submission import load_binary_labels, write_prediction_file


def _expand_paths(spec: str) -> list[Path]:
    paths: list[Path] = []
    for token in spec.split(","):
        candidate = token.strip()
        if not candidate:
            continue
        if any(char in candidate for char in {"*", "?", "["}):
            for path in sorted(glob.glob(candidate)):
                paths.append(Path(path).resolve())
        else:
            paths.append(Path(candidate).resolve())
    unique = []
    seen = set()
    for path in paths:
        if str(path) in seen:
            continue
        seen.add(str(path))
        unique.append(path)
    return unique


def _load_probs(paths: list[Path], label: str) -> np.ndarray:
    if not paths:
        raise ValueError(f"No {label} probability files were provided.")
    arrays = [np.asarray(np.load(path), dtype=np.float64) for path in paths]
    expected = arrays[0].shape[0]
    for path, array in zip(paths, arrays):
        if array.ndim != 1:
            raise ValueError(f"{label} probs must be 1-D: {path}")
        if array.shape[0] != expected:
            raise ValueError(
                f"{label} length mismatch: expected {expected}, got {array.shape[0]} at {path}"
            )
    return np.mean(np.vstack(arrays), axis=0)


def _load_order_hashes(paths: list[Path], split_name: str) -> str | None:
    hashes = set()
    for path in paths:
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            continue
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("split") != split_name:
            continue
        order_hash = payload.get("order_hash")
        if order_hash:
            hashes.add(str(order_hash))
    if not hashes:
        return None
    if len(hashes) > 1:
        raise ValueError(f"{split_name} order hash mismatch across probability files: {hashes}")
    return list(hashes)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average probabilities and emit dev.txt/test.txt.")
    parser.add_argument("--dev-probs", required=True, type=str)
    parser.add_argument("--test-probs", required=True, type=str)
    parser.add_argument("--dev-labels", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--threshold-start", type=float, default=THRESHOLD_START)
    parser.add_argument("--threshold-end", type=float, default=THRESHOLD_END)
    parser.add_argument("--threshold-step", type=float, default=THRESHOLD_STEP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_prob_files = _expand_paths(args.dev_probs)
    test_prob_files = _expand_paths(args.test_probs)
    dev_probs = _load_probs(dev_prob_files, "dev")
    test_probs = _load_probs(test_prob_files, "test")

    dev_labels = load_binary_labels(args.dev_labels).astype(np.int64)
    if dev_labels.shape[0] != dev_probs.shape[0]:
        raise ValueError(
            f"Dev label length mismatch: labels={dev_labels.shape[0]}, probs={dev_probs.shape[0]}"
        )

    best = search_best_threshold(
        probs=dev_probs,
        labels=dev_labels,
        start=args.threshold_start,
        end=args.threshold_end,
        step=args.threshold_step,
    )
    dev_metrics = binary_metrics_from_probs(dev_probs, dev_labels, threshold=best.threshold)

    dev_preds = (dev_probs >= best.threshold).astype(np.int64)
    test_preds = (test_probs >= best.threshold).astype(np.int64)

    ensemble_dev_probs_path = out_dir / "ensemble_dev_probs.npy"
    ensemble_test_probs_path = out_dir / "ensemble_test_probs.npy"
    np.save(ensemble_dev_probs_path, dev_probs)
    np.save(ensemble_test_probs_path, test_probs)

    dev_txt_path = write_prediction_file(dev_preds, out_dir / "dev.txt")
    test_txt_path = write_prediction_file(test_preds, out_dir / "test.txt")

    dev_order_hash = _load_order_hashes(dev_prob_files, split_name="dev")
    test_order_hash = _load_order_hashes(test_prob_files, split_name="test")

    summary = {
        "n_models": len(dev_prob_files),
        "dev_prob_files": [str(path) for path in dev_prob_files],
        "test_prob_files": [str(path) for path in test_prob_files],
        "dev_labels_path": str(Path(args.dev_labels).resolve()),
        "selected_threshold": best.threshold,
        "dev_metrics": dev_metrics.to_dict(),
        "dev_positive_predictions": int(dev_preds.sum()),
        "test_positive_predictions": int(test_preds.sum()),
        "dev_order_hash": dev_order_hash,
        "test_order_hash": test_order_hash,
        "ensemble_dev_probs_path": str(ensemble_dev_probs_path),
        "ensemble_test_probs_path": str(ensemble_test_probs_path),
        "dev_txt_path": str(dev_txt_path),
        "test_txt_path": str(test_txt_path),
    }
    summary_path = out_dir / "ensemble_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ensemble_summary": str(summary_path),
                "dev_f1": round(dev_metrics.f1, 6),
                "threshold": round(best.threshold, 6),
                "dev_count": int(dev_preds.shape[0]),
                "test_count": int(test_preds.shape[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

