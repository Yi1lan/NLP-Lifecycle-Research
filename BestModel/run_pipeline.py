#!/usr/bin/env python3
"""Run the full proposal pipeline (Stage 4 matrix + final ensemble)."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BestModel.data import dev_labels, load_stage3_dataset


RUN_VARIANTS = [
    {
        "variant_id": "b0_roberta",
        "model_family": "b0_roberta",
        "model": "roberta",
        "loss": "ce",
        "lex_drop": False,
        "include_in_ensemble": False,
    },
    {
        "variant_id": "b1_roberta",
        "model_family": "b1_roberta",
        "model": "roberta",
        "loss": "focal",
        "lex_drop": False,
        "include_in_ensemble": False,
    },
    {
        "variant_id": "roberta",
        "model_family": "roberta_base",
        "model": "roberta",
        "loss": "focal",
        "lex_drop": True,
        "include_in_ensemble": True,
    },
    {
        "variant_id": "roberta_large",
        "model_family": "roberta_large",
        "model": "roberta_large",
        "loss": "ce",
        "lex_drop": False,
        "include_in_ensemble": True,
    },
    {
        "variant_id": "deberta",
        "model_family": "deberta_base",
        "model": "deberta",
        "loss": "focal",
        "lex_drop": True,
        "include_in_ensemble": True,
    },
]


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


def _parse_seeds(seed_spec: str) -> list[int]:
    seeds = []
    for token in seed_spec.split(","):
        value = token.strip()
        if not value:
            continue
        seeds.append(int(value))
    if not seeds:
        raise ValueError("No seeds were provided.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return seeds


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
        "BestModel.train",
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
    _run(train_cmd, dry_run=dry_run)

    for split, output_path in [("dev", dev_probs), ("test", test_probs)]:
        predict_cmd = [
            python_exe,
            "-m",
            "BestModel.predict",
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


def _extract_row(variant: dict, seed: int, run_id: str, summary: dict) -> dict:
    metrics = summary.get("dev_metrics", {})
    return {
        "run_id": run_id,
        "variant_id": variant["variant_id"],
        "model_family": variant["model_family"],
        "model": variant["model"],
        "seed": seed,
        "loss": variant["loss"],
        "lex_drop": variant["lex_drop"],
        "dev_f1": metrics.get("f1"),
        "dev_precision": metrics.get("precision"),
        "dev_recall": metrics.get("recall"),
        "threshold": metrics.get("threshold"),
        "dev_prob_std": summary.get("dev_prob_std"),
        "dev_positive_rate_at_best_threshold": summary.get("dev_positive_rate_at_best_threshold"),
        "degenerate_run": summary.get("degenerate_run"),
        "degenerate_reason": summary.get("degenerate_reason"),
        "include_in_ensemble": variant["include_in_ensemble"],
        "is_final_submission_candidate": variant["include_in_ensemble"],
    }


def _compute_model_seed_statistics(rows: list[dict]) -> list[dict]:
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        family = str(row["model_family"])
        by_family.setdefault(family, []).append(row)

    stats_rows = []
    for family, family_rows in sorted(by_family.items()):
        f1s = np.asarray([float(item["dev_f1"]) for item in family_rows], dtype=np.float64)
        precisions = np.asarray([float(item["dev_precision"]) for item in family_rows], dtype=np.float64)
        recalls = np.asarray([float(item["dev_recall"]) for item in family_rows], dtype=np.float64)
        prob_stds = np.asarray(
            [float(item["dev_prob_std"]) for item in family_rows], dtype=np.float64
        )
        pos_rates = np.asarray(
            [float(item["dev_positive_rate_at_best_threshold"]) for item in family_rows],
            dtype=np.float64,
        )
        best_row = max(
            family_rows,
            key=lambda item: (float(item["dev_f1"]), float(item["dev_precision"])),
        )
        std_ddof = 1 if len(f1s) > 1 else 0
        stats_rows.append(
            {
                "model_family": family,
                "n_seeds": int(len(family_rows)),
                "seeds": ",".join(str(item["seed"]) for item in sorted(family_rows, key=lambda x: int(x["seed"]))),
                "mean_dev_f1": float(np.mean(f1s)),
                "std_dev_f1": float(np.std(f1s, ddof=std_ddof)),
                "max_dev_f1": float(np.max(f1s)),
                "best_seed": int(best_row["seed"]),
                "best_run_id": str(best_row["run_id"]),
                "mean_dev_precision": float(np.mean(precisions)),
                "std_dev_precision": float(np.std(precisions, ddof=std_ddof)),
                "mean_dev_recall": float(np.mean(recalls)),
                "std_dev_recall": float(np.std(recalls, ddof=std_ddof)),
                "mean_dev_prob_std": float(np.mean(prob_stds)),
                "mean_dev_positive_rate": float(np.mean(pos_rates)),
                "num_degenerate_runs": int(
                    sum(bool(item.get("degenerate_run")) for item in family_rows)
                ),
            }
        )
    return stats_rows


def _write_best_model_artifacts(out_root: Path, best_row: dict) -> dict:
    run_id = str(best_row["run_id"])
    run_dir = out_root / "runs" / run_id
    run_summary = _read_run_summary(run_dir)
    threshold = float(run_summary["dev_metrics"]["threshold"])

    best_dir = out_root / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    src_dev_txt = run_dir / "dev_preds.txt"
    dst_dev_txt = best_dir / "dev.txt"
    if not src_dev_txt.exists():
        raise FileNotFoundError(f"Missing dev predictions for best run: {src_dev_txt}")
    shutil.copyfile(src_dev_txt, dst_dev_txt)

    test_probs_path = out_root / "probs" / f"{run_id}_test.npy"
    if not test_probs_path.exists():
        raise FileNotFoundError(f"Missing test probabilities for best run: {test_probs_path}")
    test_probs = np.asarray(np.load(test_probs_path), dtype=np.float64)
    test_preds = (test_probs >= threshold).astype(np.int64)
    dst_test_txt = best_dir / "test.txt"
    with dst_test_txt.open("w", encoding="utf-8") as handle:
        for value in test_preds:
            handle.write(f"{int(value)}\n")

    payload = {
        "best_run_id": run_id,
        "model_family": best_row["model_family"],
        "model": best_row["model"],
        "seed": int(best_row["seed"]),
        "loss": best_row["loss"],
        "lex_drop": bool(best_row["lex_drop"]),
        "dev_f1": float(best_row["dev_f1"]),
        "dev_precision": float(best_row["dev_precision"]),
        "dev_recall": float(best_row["dev_recall"]),
        "threshold": threshold,
        "degenerate_run": bool(best_row.get("degenerate_run")),
        "degenerate_reason": best_row.get("degenerate_reason"),
        "run_summary_path": str(run_dir / "run_summary.json"),
        "dev_txt_path": str(dst_dev_txt),
        "test_txt_path": str(dst_test_txt),
        "test_positive_predictions": int(test_preds.sum()),
        "test_count": int(test_preds.shape[0]),
    }
    (best_dir / "best_model_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 matrix and build final ensemble.")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out-root", type=str, default="outputs/stage4")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--probe-epochs", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="42,123,2024,3407,777")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-max-len-probe", action="store_true")
    parser.add_argument("--force-max-len", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
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

    rows: list[dict] = []
    all_run_ids: list[str] = []
    ensemble_run_ids: list[str] = []

    for variant in RUN_VARIANTS:
        for seed in seeds:
            run_id = f"{variant['variant_id']}_seed{seed}"
            result = _train_and_predict(
                run_id=run_id,
                model=variant["model"],
                seed=seed,
                loss=variant["loss"],
                lex_drop=variant["lex_drop"],
                max_len=chosen_max_len,
                data_dir=data_dir,
                out_root=out_root,
                python_exe=args.python_exe,
                epochs=args.epochs,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )
            all_run_ids.append(run_id)
            if variant["include_in_ensemble"]:
                ensemble_run_ids.append(run_id)
            if not args.dry_run:
                rows.append(_extract_row(variant, seed, run_id, result["summary"]))

    final_ensemble_dir = out_root / "final_ensemble"
    final_dev_probs = [str(out_root / "probs" / f"{run_id}_dev.npy") for run_id in ensemble_run_ids]
    final_test_probs = [str(out_root / "probs" / f"{run_id}_test.npy") for run_id in ensemble_run_ids]
    ensemble_cmd = [
        args.python_exe,
        "-m",
        "BestModel.ensemble",
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
        dry_payload = {
            "data_dir": str(data_dir),
            "out_root": str(out_root),
            "selected_max_len": chosen_max_len,
            "seeds": seeds,
            "all_run_ids": all_run_ids,
            "ensemble_run_ids": ensemble_run_ids,
        }
        print(json.dumps(dry_payload, indent=2))
        return

    ablation_csv_path = out_root / "ablation_summary.csv"
    with ablation_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats_rows = _compute_model_seed_statistics(rows)
    stats_csv_path = out_root / "model_seed_statistics.csv"
    with stats_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    best_row = max(rows, key=lambda item: (float(item["dev_f1"]), float(item["dev_precision"])))
    best_model_payload = _write_best_model_artifacts(out_root, best_row)
    best_model_summary_path = out_root / "best_model" / "best_model_summary.json"

    matrix_summary = {
        "data_dir": str(data_dir),
        "out_root": str(out_root),
        "selected_max_len": chosen_max_len,
        "seeds": seeds,
        "run_variants": RUN_VARIANTS,
        "all_run_ids": all_run_ids,
        "ensemble_run_ids": ensemble_run_ids,
        "ablation_summary_csv": str(ablation_csv_path),
        "model_seed_statistics_csv": str(stats_csv_path),
        "best_model_summary_json": str(best_model_summary_path),
        "best_model": best_model_payload,
        "final_ensemble_summary_json": str(final_ensemble_dir / "ensemble_summary.json"),
        "selected_runs_manifest_json": str(final_ensemble_dir / "selected_runs.json"),
    }
    (out_root / "run_matrix_summary.json").write_text(
        json.dumps(matrix_summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(matrix_summary, indent=2))


if __name__ == "__main__":
    main()
