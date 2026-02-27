#!/usr/bin/env python3
"""Orchestrate the balanced Stage 4 run matrix and final ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage3.data import dev_labels, load_stage3_dataset  # pylint: disable=wrong-import-position


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print("$", " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _read_run_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run summary not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _train_and_predict(
    *,
    run_id: str,
    model: str,
    seed: int,
    loss: str,
    lex_drop: bool,
    max_len: int,
    data_dir: Path,
    out_root: Path,
    python_exe: str,
    epochs: int,
    skip_existing: bool,
    dry_run: bool,
    lr: float | None = None,
    class_weight_scale: float = 1.0,
    force_fast_tokenizer: bool = False,
    force_slow_tokenizer: bool = False,
) -> dict:
    run_dir = out_root / "runs" / run_id
    probs_dir = out_root / "probs"
    dev_probs = probs_dir / f"{run_id}_dev.npy"
    test_probs = probs_dir / f"{run_id}_test.npy"
    if (
        skip_existing
        and run_dir.joinpath("run_summary.json").exists()
        and dev_probs.exists()
        and test_probs.exists()
    ):
        print(f"[skip] {run_id} already has run summary + dev/test probs.")
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "dev_probs": str(dev_probs),
            "test_probs": str(test_probs),
            "summary": _read_run_summary(run_dir),
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    probs_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        python_exe,
        "-m",
        "src.stage3.train",
        "--model",
        model,
        "--seed",
        str(seed),
        "--max-len",
        str(max_len),
        "--loss",
        loss,
        "--lex-drop",
        str(lex_drop).lower(),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(run_dir),
        "--epochs",
        str(epochs),
    ]
    if lr is not None:
        train_cmd.extend(["--lr", str(lr)])
    if class_weight_scale != 1.0:
        train_cmd.extend(["--class-weight-scale", str(class_weight_scale)])
    if force_fast_tokenizer:
        train_cmd.append("--force-fast-tokenizer")
    if force_slow_tokenizer:
        train_cmd.append("--force-slow-tokenizer")
    _run(train_cmd, dry_run=dry_run)

    for split, output_path in [("dev", dev_probs), ("test", test_probs)]:
        predict_cmd = [
            python_exe,
            "-m",
            "src.stage3.predict",
            "--checkpoint",
            str(run_dir),
            "--split",
            split,
            "--data-dir",
            str(data_dir),
            "--out-probs",
            str(output_path),
            "--max-len",
            str(max_len),
        ]
        if split == "dev":
            predict_cmd.extend(["--out-labels", str(out_root / "dev_labels.npy")])
        _run(predict_cmd, dry_run=dry_run)

    summary = {}
    if not dry_run:
        summary = _read_run_summary(run_dir)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "dev_probs": str(dev_probs),
        "test_probs": str(test_probs),
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 matrix and build final ensemble.")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out-root", type=str, default="outputs/stage4")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--probe-epochs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-max-len-probe", action="store_true")
    parser.add_argument("--force-max-len", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "runs").mkdir(parents=True, exist_ok=True)
    (out_root / "probs").mkdir(parents=True, exist_ok=True)

    bundle = load_stage3_dataset(data_dir)
    np.save(out_root / "dev_labels.npy", np.asarray(dev_labels(bundle), dtype=np.int64))

    chosen_max_len = args.force_max_len
    max_len_probe_path = out_root / "max_len_selection.json"
    probe_epochs = args.probe_epochs if args.probe_epochs is not None else args.epochs

    if chosen_max_len is None and max_len_probe_path.exists() and args.skip_existing:
        payload = json.loads(max_len_probe_path.read_text(encoding="utf-8"))
        chosen_max_len = int(payload["selected_max_len"])

    if chosen_max_len is None and not args.skip_max_len_probe:
        probe_192 = _train_and_predict(
            run_id="probe_roberta_len192",
            model="roberta",
            seed=42,
            loss="focal",
            lex_drop=True,
            max_len=192,
            data_dir=data_dir,
            out_root=out_root,
            python_exe=args.python_exe,
            epochs=probe_epochs,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        probe_256 = _train_and_predict(
            run_id="probe_roberta_len256",
            model="roberta",
            seed=42,
            loss="focal",
            lex_drop=True,
            max_len=256,
            data_dir=data_dir,
            out_root=out_root,
            python_exe=args.python_exe,
            epochs=probe_epochs,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            chosen_max_len = 256
        else:
            f1_192 = float(probe_192["summary"]["dev_metrics"]["f1"])
            f1_256 = float(probe_256["summary"]["dev_metrics"]["f1"])
            drop = f1_256 - f1_192
            chosen_max_len = 192 if drop <= 0.005 else 256
            max_len_probe_path.write_text(
                json.dumps(
                    {
                        "probe_epochs": probe_epochs,
                        "dev_f1_len192": f1_192,
                        "dev_f1_len256": f1_256,
                        "drop_len256_minus_len192": drop,
                        "rule": "choose_192_if_drop<=0.005_else_256",
                        "selected_max_len": chosen_max_len,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    elif chosen_max_len is None:
        chosen_max_len = 256

    run_matrix = [
        {"run_id": "b0_roberta_seed42", "model": "roberta", "seed": 42, "loss": "ce", "lex_drop": False},
        {"run_id": "b1_roberta_seed42", "model": "roberta", "seed": 42, "loss": "focal", "lex_drop": False},
        {"run_id": "roberta_seed42", "model": "roberta", "seed": 42, "loss": "focal", "lex_drop": True},
        {"run_id": "roberta_seed2024", "model": "roberta", "seed": 2024, "loss": "focal", "lex_drop": True},
        {"run_id": "deberta_seed42", "model": "deberta", "seed": 42, "loss": "focal", "lex_drop": True},
        {"run_id": "deberta_seed2024", "model": "deberta", "seed": 2024, "loss": "focal", "lex_drop": True},
    ]

    results = []
    for config in run_matrix:
        result = _train_and_predict(
            run_id=config["run_id"],
            model=config["model"],
            seed=config["seed"],
            loss=config["loss"],
            lex_drop=config["lex_drop"],
            max_len=chosen_max_len,
            data_dir=data_dir,
            out_root=out_root,
            python_exe=args.python_exe,
            epochs=args.epochs,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        results.append(result)

    final_run_ids = [
        "roberta_seed42",
        "roberta_seed2024",
        "deberta_seed42",
        "deberta_seed2024",
    ]
    final_dev_probs = [str(out_root / "probs" / f"{run_id}_dev.npy") for run_id in final_run_ids]
    final_test_probs = [str(out_root / "probs" / f"{run_id}_test.npy") for run_id in final_run_ids]

    final_ensemble_dir = out_root / "final_ensemble"
    ensemble_cmd = [
        args.python_exe,
        "-m",
        "src.stage3.ensemble",
        "--dev-probs",
        ",".join(final_dev_probs),
        "--test-probs",
        ",".join(final_test_probs),
        "--dev-labels",
        str(out_root / "dev_labels.npy"),
        "--out-dir",
        str(final_ensemble_dir),
    ]
    _run(ensemble_cmd, dry_run=args.dry_run)

    if args.dry_run:
        return

    ablation_csv_path = out_root / "ablation_summary.csv"
    with ablation_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "model",
                "seed",
                "loss",
                "lex_drop",
                "dev_f1",
                "dev_precision",
                "dev_recall",
                "threshold",
                "dev_prob_std",
                "dev_positive_rate_at_best_threshold",
                "degenerate_run",
                "degenerate_reason",
                "is_final_submission_candidate",
            ],
        )
        writer.writeheader()
        for item in run_matrix:
            summary = _read_run_summary(out_root / "runs" / item["run_id"])
            metrics = summary["dev_metrics"]
            writer.writerow(
                {
                    "run_id": item["run_id"],
                    "model": item["model"],
                    "seed": item["seed"],
                    "loss": item["loss"],
                    "lex_drop": item["lex_drop"],
                    "dev_f1": metrics["f1"],
                    "dev_precision": metrics["precision"],
                    "dev_recall": metrics["recall"],
                    "threshold": metrics["threshold"],
                    "dev_prob_std": summary.get("dev_prob_std"),
                    "dev_positive_rate_at_best_threshold": summary.get(
                        "dev_positive_rate_at_best_threshold"
                    ),
                    "degenerate_run": summary.get("degenerate_run"),
                    "degenerate_reason": summary.get("degenerate_reason"),
                    "is_final_submission_candidate": item["run_id"] in final_run_ids,
                }
            )
        ensemble_summary = json.loads(
            (final_ensemble_dir / "ensemble_summary.json").read_text(encoding="utf-8")
        )
        metrics = ensemble_summary["dev_metrics"]
        writer.writerow(
            {
                "run_id": "final_ensemble",
                "model": "roberta+deberta",
                "seed": "42,2024",
                "loss": "focal",
                "lex_drop": True,
                "dev_f1": metrics["f1"],
                "dev_precision": metrics["precision"],
                "dev_recall": metrics["recall"],
                "threshold": metrics["threshold"],
                "dev_prob_std": None,
                "dev_positive_rate_at_best_threshold": None,
                "degenerate_run": False,
                "degenerate_reason": None,
                "is_final_submission_candidate": True,
            }
        )

    matrix_summary = {
        "data_dir": str(data_dir),
        "out_root": str(out_root),
        "selected_max_len": chosen_max_len,
        "run_ids": [item["run_id"] for item in run_matrix],
        "final_run_ids": final_run_ids,
        "ablation_summary_csv": str(ablation_csv_path),
        "final_ensemble_summary_json": str(final_ensemble_dir / "ensemble_summary.json"),
        "selected_runs_manifest_json": str(final_ensemble_dir / "selected_runs.json"),
    }
    (out_root / "run_matrix_summary.json").write_text(
        json.dumps(matrix_summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(matrix_summary, indent=2))


if __name__ == "__main__":
    main()
