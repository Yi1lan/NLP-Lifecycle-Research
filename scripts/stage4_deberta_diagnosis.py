#!/usr/bin/env python3
"""Focused DeBERTa diagnosis matrix for failure analysis."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print("$", " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _train_and_predict(
    *,
    run_id: str,
    data_dir: Path,
    out_root: Path,
    python_exe: str,
    seed: int,
    max_len: int,
    epochs: int,
    loss: str,
    lex_drop: bool,
    lr: float | None,
    class_weight_scale: float,
    force_fast_tokenizer: bool,
    skip_existing: bool,
    dry_run: bool,
) -> dict:
    run_dir = out_root / "runs" / run_id
    probs_dir = out_root / "probs"
    dev_probs = probs_dir / f"{run_id}_dev.npy"
    test_probs = probs_dir / f"{run_id}_test.npy"
    run_summary = run_dir / "run_summary.json"

    if skip_existing and run_summary.exists() and dev_probs.exists() and test_probs.exists():
        print(f"[skip] {run_id} already exists")
        return _read_json(run_summary)

    run_dir.mkdir(parents=True, exist_ok=True)
    probs_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        python_exe,
        "-m",
        "src.stage3.train",
        "--model",
        "deberta",
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
        "--class-weight-scale",
        str(class_weight_scale),
    ]
    if lr is not None:
        train_cmd.extend(["--lr", str(lr)])
    if force_fast_tokenizer:
        train_cmd.append("--force-fast-tokenizer")
    _run(train_cmd, dry_run=dry_run)

    for split, target in [("dev", dev_probs), ("test", test_probs)]:
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
            str(target),
            "--max-len",
            str(max_len),
        ]
        _run(predict_cmd, dry_run=dry_run)

    if dry_run:
        return {
            "run_id": run_id,
            "seed": seed,
            "loss": loss,
            "lex_drop": lex_drop,
            "lr": lr,
            "class_weight_scale": class_weight_scale,
            "force_fast_tokenizer": force_fast_tokenizer,
            "dev_metrics": {"f1": None},
        }
    return _read_json(run_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused DeBERTa diagnosis matrix.")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out-root", type=str, default="outputs/stage4/deberta_diagnosis")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--promote-seed", type=int, default=2024)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-weight-half", action="store_true")
    parser.add_argument("--promote-best-two", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    phase1 = [
        {
            "name": "deberta_ce_no_lexdrop",
            "loss": "ce",
            "lex_drop": False,
            "lr": None,
            "class_weight_scale": 1.0,
            "force_fast_tokenizer": False,
        },
        {
            "name": "deberta_focal_no_lexdrop",
            "loss": "focal",
            "lex_drop": False,
            "lr": None,
            "class_weight_scale": 1.0,
            "force_fast_tokenizer": False,
        },
        {
            "name": "deberta_focal_lexdrop",
            "loss": "focal",
            "lex_drop": True,
            "lr": None,
            "class_weight_scale": 1.0,
            "force_fast_tokenizer": False,
        },
        {
            "name": "deberta_focal_lexdrop_fast_tokenizer",
            "loss": "focal",
            "lex_drop": True,
            "lr": None,
            "class_weight_scale": 1.0,
            "force_fast_tokenizer": True,
        },
        {
            "name": "deberta_focal_lexdrop_lr8e6",
            "loss": "focal",
            "lex_drop": True,
            "lr": 8e-6,
            "class_weight_scale": 1.0,
            "force_fast_tokenizer": False,
        },
    ]
    if args.include_weight_half:
        phase1.append(
            {
                "name": "deberta_focal_lexdrop_weight_half",
                "loss": "focal",
                "lex_drop": True,
                "lr": None,
                "class_weight_scale": 0.5,
                "force_fast_tokenizer": False,
            }
        )

    phase1_results: list[dict] = []
    for config in phase1:
        run_id = f"{config['name']}_seed{args.seed}"
        summary = _train_and_predict(
            run_id=run_id,
            data_dir=data_dir,
            out_root=out_root,
            python_exe=args.python_exe,
            seed=args.seed,
            max_len=args.max_len,
            epochs=args.epochs,
            loss=config["loss"],
            lex_drop=config["lex_drop"],
            lr=config["lr"],
            class_weight_scale=config["class_weight_scale"],
            force_fast_tokenizer=config["force_fast_tokenizer"],
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        phase1_results.append({"config": config, "run_id": run_id, "summary": summary})

    promoted_results: list[dict] = []
    if args.promote_best_two:
        if args.dry_run:
            promoted_base = phase1[:2]
        else:
            ranked = sorted(
                phase1_results,
                key=lambda item: float(item["summary"]["dev_metrics"]["f1"]),
                reverse=True,
            )
            promoted_base = [item["config"] for item in ranked[:2]]
        for config in promoted_base:
            run_id = f"{config['name']}_seed{args.promote_seed}"
            summary = _train_and_predict(
                run_id=run_id,
                data_dir=data_dir,
                out_root=out_root,
                python_exe=args.python_exe,
                seed=args.promote_seed,
                max_len=args.max_len,
                epochs=args.epochs,
                loss=config["loss"],
                lex_drop=config["lex_drop"],
                lr=config["lr"],
                class_weight_scale=config["class_weight_scale"],
                force_fast_tokenizer=config["force_fast_tokenizer"],
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )
            promoted_results.append({"config": config, "run_id": run_id, "summary": summary})

    rows = []
    for collection in [phase1_results, promoted_results]:
        for item in collection:
            summary = item["summary"]
            metrics = summary.get("dev_metrics", {})
            rows.append(
                {
                    "run_id": item["run_id"],
                    "seed": summary.get("seed"),
                    "loss": summary.get("loss"),
                    "lex_drop": summary.get("lex_drop"),
                    "learning_rate": summary.get("learning_rate"),
                    "class_weight_scale": summary.get("class_weight_scale"),
                    "tokenizer_mode": summary.get("tokenizer_mode"),
                    "dev_f1": metrics.get("f1"),
                    "dev_precision": metrics.get("precision"),
                    "dev_recall": metrics.get("recall"),
                    "threshold": metrics.get("threshold"),
                    "dev_prob_std": summary.get("dev_prob_std"),
                    "dev_positive_rate_at_best_threshold": summary.get(
                        "dev_positive_rate_at_best_threshold"
                    ),
                    "degenerate_run": summary.get("degenerate_run"),
                    "degenerate_reason": summary.get("degenerate_reason"),
                }
            )

    csv_path = out_root / "diagnosis_summary.csv"
    if rows and not args.dry_run:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "data_dir": str(data_dir),
        "out_root": str(out_root),
        "phase1_seed": args.seed,
        "promote_seed": args.promote_seed if args.promote_best_two else None,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "phase1_configs": phase1,
        "phase1_run_ids": [item["run_id"] for item in phase1_results],
        "promoted_run_ids": [item["run_id"] for item in promoted_results],
        "summary_csv": str(csv_path),
    }
    manifest_path = out_root / "diagnosis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "summary_csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()

