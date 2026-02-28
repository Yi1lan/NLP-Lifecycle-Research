"""Probability ensembling and threshold selection for final submission files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_MAX_POSITIVE_RATE,
    DEFAULT_MIN_DEV_F1,
    DEFAULT_MIN_DEV_PROB_STD,
    DEFAULT_MIN_POSITIVE_RATE,
    THRESHOLD_END,
    THRESHOLD_START,
    THRESHOLD_STEP,
)
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
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _infer_run_id(path: Path, split_name: str) -> str:
    suffix = f"_{split_name}"
    stem = path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _load_prob_map(paths: list[Path], split_name: str) -> dict[str, dict]:
    if not paths:
        raise ValueError(f"No {split_name} probability files were provided.")
    mapping: dict[str, dict] = {}
    for path in paths:
        array = np.asarray(np.load(path), dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"{split_name} probs must be 1-D: {path}")
        run_id = _infer_run_id(path, split_name)
        if run_id in mapping:
            raise ValueError(f"Duplicate {split_name} run_id '{run_id}' from path: {path}")
        mapping[run_id] = {"path": path, "probs": array}
    return mapping


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


def _resolve_run_summary_path(prob_path: Path, run_id: str) -> Path | None:
    # Typical layout: outputs/stage4/probs/{run_id}_dev.npy -> outputs/stage4/runs/{run_id}/run_summary.json
    if prob_path.parent.name == "probs":
        candidate = prob_path.parent.parent / "runs" / run_id / "run_summary.json"
        if candidate.exists():
            return candidate
    # Fallback local neighbor.
    neighbor = prob_path.parent / run_id / "run_summary.json"
    if neighbor.exists():
        return neighbor
    return None


def _run_health(
    dev_probs: np.ndarray,
    dev_labels: np.ndarray,
    *,
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
) -> dict:
    best = search_best_threshold(
        probs=dev_probs,
        labels=dev_labels,
        start=threshold_start,
        end=threshold_end,
        step=threshold_step,
    )
    best_metrics = binary_metrics_from_probs(dev_probs, dev_labels, threshold=best.threshold)
    positive_rate = float(np.mean((dev_probs >= best.threshold).astype(np.float64)))
    prob_std = float(np.std(dev_probs))
    return {
        "best_threshold": float(best.threshold),
        "dev_f1": float(best_metrics.f1),
        "dev_precision": float(best_metrics.precision),
        "dev_recall": float(best_metrics.recall),
        "dev_prob_std": prob_std,
        "dev_positive_rate_at_best_threshold": positive_rate,
    }


def _is_run_eligible(
    run_health: dict,
    *,
    min_dev_f1: float,
    min_dev_prob_std: float,
    min_positive_rate: float,
    max_positive_rate: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if run_health["dev_f1"] < min_dev_f1:
        reasons.append(f"dev_f1<{min_dev_f1:.2f}")
    if run_health["dev_prob_std"] < min_dev_prob_std:
        reasons.append(f"dev_prob_std<{min_dev_prob_std:.2f}")
    positive_rate = run_health["dev_positive_rate_at_best_threshold"]
    if positive_rate < min_positive_rate:
        reasons.append(f"positive_rate<{min_positive_rate:.2f}")
    if positive_rate > max_positive_rate:
        reasons.append(f"positive_rate>{max_positive_rate:.2f}")
    return len(reasons) == 0, reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average probabilities and emit dev.txt/test.txt.")
    parser.add_argument("--dev-probs", required=True, type=str)
    parser.add_argument("--test-probs", required=True, type=str)
    parser.add_argument("--dev-labels", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--threshold-start", type=float, default=THRESHOLD_START)
    parser.add_argument("--threshold-end", type=float, default=THRESHOLD_END)
    parser.add_argument("--threshold-step", type=float, default=THRESHOLD_STEP)
    parser.add_argument("--min-dev-f1", type=float, default=DEFAULT_MIN_DEV_F1)
    parser.add_argument("--min-dev-prob-std", type=float, default=DEFAULT_MIN_DEV_PROB_STD)
    parser.add_argument("--min-positive-rate", type=float, default=DEFAULT_MIN_POSITIVE_RATE)
    parser.add_argument("--max-positive-rate", type=float, default=DEFAULT_MAX_POSITIVE_RATE)
    parser.add_argument("--disable-health-filtering", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_prob_files = _expand_paths(args.dev_probs)
    test_prob_files = _expand_paths(args.test_probs)
    dev_map = _load_prob_map(dev_prob_files, split_name="dev")
    test_map = _load_prob_map(test_prob_files, split_name="test")

    run_ids = sorted(set(dev_map) & set(test_map))
    if not run_ids:
        raise ValueError("No overlapping run IDs between dev and test probability files.")
    missing_test = sorted(set(dev_map) - set(test_map))
    if missing_test:
        raise ValueError(f"Missing test probabilities for run IDs: {missing_test}")
    missing_dev = sorted(set(test_map) - set(dev_map))
    if missing_dev:
        raise ValueError(f"Missing dev probabilities for run IDs: {missing_dev}")

    # Validate run lengths match.
    expected_dev_len = None
    expected_test_len = None
    for run_id in run_ids:
        dev_arr = dev_map[run_id]["probs"]
        test_arr = test_map[run_id]["probs"]
        if expected_dev_len is None:
            expected_dev_len = dev_arr.shape[0]
        if expected_test_len is None:
            expected_test_len = test_arr.shape[0]
        if dev_arr.shape[0] != expected_dev_len:
            raise ValueError(f"Dev length mismatch at run '{run_id}'.")
        if test_arr.shape[0] != expected_test_len:
            raise ValueError(f"Test length mismatch at run '{run_id}'.")

    dev_labels = load_binary_labels(args.dev_labels).astype(np.int64)
    if dev_labels.shape[0] != expected_dev_len:
        raise ValueError(
            f"Dev label length mismatch: labels={dev_labels.shape[0]}, probs={expected_dev_len}"
        )

    selected_run_ids: list[str] = []
    excluded_run_ids: list[str] = []
    manifest_rows: list[dict] = []
    for run_id in run_ids:
        run_health = _run_health(
            dev_probs=dev_map[run_id]["probs"],
            dev_labels=dev_labels,
            threshold_start=args.threshold_start,
            threshold_end=args.threshold_end,
            threshold_step=args.threshold_step,
        )
        include = True
        reasons: list[str] = []
        if not args.disable_health_filtering:
            include, reasons = _is_run_eligible(
                run_health,
                min_dev_f1=args.min_dev_f1,
                min_dev_prob_std=args.min_dev_prob_std,
                min_positive_rate=args.min_positive_rate,
                max_positive_rate=args.max_positive_rate,
            )
        summary_path = _resolve_run_summary_path(dev_map[run_id]["path"], run_id)
        summary_payload = None
        if summary_path is not None:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "run_id": run_id,
            "dev_prob_file": str(dev_map[run_id]["path"]),
            "test_prob_file": str(test_map[run_id]["path"]),
            "include": include,
            "exclusion_reasons": reasons,
            "run_health": run_health,
            "run_summary_path": str(summary_path) if summary_path is not None else None,
            "reported_degenerate_run": (
                None if summary_payload is None else summary_payload.get("degenerate_run")
            ),
            "reported_degenerate_reason": (
                None if summary_payload is None else summary_payload.get("degenerate_reason")
            ),
        }
        manifest_rows.append(row)
        if include:
            selected_run_ids.append(run_id)
        else:
            excluded_run_ids.append(run_id)

    if not selected_run_ids:
        raise ValueError("No runs passed health filtering. Relax thresholds or disable filtering.")

    selected_dev_probs = np.vstack([dev_map[run_id]["probs"] for run_id in selected_run_ids])
    selected_test_probs = np.vstack([test_map[run_id]["probs"] for run_id in selected_run_ids])
    dev_probs = np.mean(selected_dev_probs, axis=0)
    test_probs = np.mean(selected_test_probs, axis=0)

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

    selected_dev_paths = [dev_map[run_id]["path"] for run_id in selected_run_ids]
    selected_test_paths = [test_map[run_id]["path"] for run_id in selected_run_ids]
    dev_order_hash = _load_order_hashes(selected_dev_paths, split_name="dev")
    test_order_hash = _load_order_hashes(selected_test_paths, split_name="test")

    criteria = {
        "min_dev_f1": args.min_dev_f1,
        "min_dev_prob_std": args.min_dev_prob_std,
        "min_positive_rate": args.min_positive_rate,
        "max_positive_rate": args.max_positive_rate,
        "disable_health_filtering": args.disable_health_filtering,
    }
    selected_manifest_path = out_dir / "selected_runs.json"
    selected_manifest = {
        "criteria": criteria,
        "n_runs_total": len(run_ids),
        "n_runs_selected": len(selected_run_ids),
        "selected_run_ids": selected_run_ids,
        "excluded_run_ids": excluded_run_ids,
        "runs": manifest_rows,
    }
    selected_manifest_path.write_text(json.dumps(selected_manifest, indent=2), encoding="utf-8")

    summary = {
        "n_models_total": len(run_ids),
        "n_models_selected": len(selected_run_ids),
        "all_run_ids": run_ids,
        "selected_run_ids": selected_run_ids,
        "excluded_run_ids": excluded_run_ids,
        "dev_prob_files_input": [str(path) for path in dev_prob_files],
        "test_prob_files_input": [str(path) for path in test_prob_files],
        "dev_prob_files": [str(path) for path in selected_dev_paths],
        "test_prob_files": [str(path) for path in selected_test_paths],
        "dev_labels_path": str(Path(args.dev_labels).resolve()),
        "health_filtering_criteria": criteria,
        "selected_runs_manifest": str(selected_manifest_path),
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
                "selected_runs_manifest": str(selected_manifest_path),
                "n_models_total": len(run_ids),
                "n_models_selected": len(selected_run_ids),
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

