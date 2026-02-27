#!/usr/bin/env python3
"""Generate Stage 5.2 local evaluation artifacts and markdown report."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage3.augment import contains_trigger_token, load_trigger_tokens  # pylint: disable=wrong-import-position
from src.stage3.data import load_stage3_dataset  # pylint: disable=wrong-import-position
from src.stage3.losses import binary_metrics_from_probs  # pylint: disable=wrong-import-position


def _metrics_for_subset(df: pd.DataFrame, threshold: float) -> dict:
    labels = df["label"].to_numpy(dtype=np.int64)
    probs = df["prob"].to_numpy(dtype=np.float64)
    if labels.size == 0:
        return {
            "count": 0,
            "positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fpr": 0.0,
        }
    metrics = binary_metrics_from_probs(probs, labels, threshold=threshold)
    fpr_denominator = metrics.fp + metrics.tn
    fpr = (metrics.fp / fpr_denominator) if fpr_denominator > 0 else 0.0
    return {
        "count": int(labels.size),
        "positives": int(labels.sum()),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "fpr": float(fpr),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local evaluation package for Stage 5.2.")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument(
        "--ensemble-summary",
        type=str,
        default="outputs/stage4/final_ensemble/ensemble_summary.json",
    )
    parser.add_argument(
        "--ablation-summary",
        type=str,
        default="outputs/stage4/ablation_summary.csv",
    )
    parser.add_argument(
        "--baseline-dev-probs",
        type=str,
        default="outputs/stage4/probs/b0_roberta_seed42_dev.npy",
    )
    parser.add_argument(
        "--baseline-run-summary",
        type=str,
        default="outputs/stage4/runs/b0_roberta_seed42/run_summary.json",
    )
    parser.add_argument(
        "--lexical-csv",
        type=str,
        default="outputs/stage2/tables/lexical_analysis.csv",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/stage5/local_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    ensemble_summary_path = Path(args.ensemble_summary).resolve()
    if not ensemble_summary_path.exists():
        raise FileNotFoundError(
            f"Missing ensemble summary: {ensemble_summary_path}. Run Stage 4 first."
        )
    ensemble_summary = json.loads(ensemble_summary_path.read_text(encoding="utf-8"))
    threshold = float(ensemble_summary["selected_threshold"])
    dev_probs = np.asarray(np.load(ensemble_summary["ensemble_dev_probs_path"]), dtype=np.float64)

    bundle = load_stage3_dataset(args.data_dir)
    dev_rows = bundle.dev
    dev_labels = np.asarray([row.label for row in dev_rows], dtype=np.int64)
    if dev_probs.shape[0] != dev_labels.shape[0]:
        raise ValueError(
            f"Dev probability length mismatch: probs={dev_probs.shape[0]}, labels={dev_labels.shape[0]}"
        )

    df = pd.DataFrame(
        {
            "par_id": [row.par_id for row in dev_rows],
            "keyword": [row.keyword for row in dev_rows],
            "country": [row.country for row in dev_rows],
            "orig_label": [row.orig_label for row in dev_rows],
            "text": [row.text for row in dev_rows],
            "label": dev_labels,
            "prob": dev_probs,
        }
    )
    df["pred"] = (df["prob"] >= threshold).astype(int)
    df["token_count"] = df["text"].str.split().map(len)
    df["error_type"] = np.where(
        (df["label"] == 1) & (df["pred"] == 0),
        "FN",
        np.where((df["label"] == 0) & (df["pred"] == 1), "FP", "OK"),
    )

    final_metrics = binary_metrics_from_probs(dev_probs, dev_labels, threshold=threshold)
    confusion = confusion_matrix(df["label"], df["pred"], labels=[0, 1])
    confusion_df = pd.DataFrame(
        confusion,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    confusion_path = tables_dir / "confusion_matrix.csv"
    confusion_df.to_csv(confusion_path, index=True)

    plt.figure(figsize=(5, 4))
    plt.imshow(confusion, cmap="Blues")
    plt.title("Confusion Matrix (Dev)")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(confusion[i, j]), ha="center", va="center")
    plt.tight_layout()
    confusion_fig = figures_dir / "confusion_matrix.png"
    plt.savefig(confusion_fig, dpi=200)
    plt.close()

    report_dict = classification_report(
        df["label"],
        df["pred"],
        output_dict=True,
        target_names=["No PCL", "PCL"],
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = tables_dir / "classification_report.csv"
    report_df.to_csv(report_path, index=True)

    precision_curve, recall_curve, _ = precision_recall_curve(df["label"], df["prob"])
    pr_fig = figures_dir / "precision_recall_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(recall_curve, precision_curve)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Dev)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(pr_fig, dpi=200)
    plt.close()

    thresholds = np.arange(0.05, 0.95 + 1e-9, 0.005)
    f1_values = []
    for candidate in thresholds:
        metrics = binary_metrics_from_probs(df["prob"], df["label"], threshold=float(candidate))
        f1_values.append(metrics.f1)
    threshold_fig = figures_dir / "threshold_vs_f1.png"
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, f1_values)
    plt.axvline(threshold, color="red", linestyle="--", label=f"Selected: {threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Positive-class F1")
    plt.title("Threshold vs F1 (Dev)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(threshold_fig, dpi=200)
    plt.close()

    error_df = df[df["error_type"].isin(["FP", "FN"])].copy()
    error_df["confidence"] = np.where(error_df["pred"] == 1, error_df["prob"], 1 - error_df["prob"])
    manual_review = error_df.sort_values("confidence", ascending=False).head(12).copy()
    manual_review["manual_note"] = ""
    manual_review["review_status"] = "pending"
    manual_error_path = tables_dir / "manual_error_review_12.csv"
    manual_review.to_csv(manual_error_path, index=False)

    keyword_rows = []
    for keyword, subset in df.groupby("keyword"):
        metrics = _metrics_for_subset(subset, threshold=threshold)
        keyword_rows.append({"keyword": keyword, **metrics})
    keyword_table = pd.DataFrame(keyword_rows).sort_values("f1", ascending=False)
    keyword_table_path = tables_dir / "slice_by_keyword.csv"
    keyword_table.to_csv(keyword_table_path, index=False)

    df["length_bin"] = pd.cut(
        df["token_count"],
        bins=[-1, 64, 128, 10_000],
        labels=["<=64", "65-128", ">128"],
    )
    length_rows = []
    for length_bin, subset in df.groupby("length_bin", dropna=False):
        metrics = _metrics_for_subset(subset, threshold=threshold)
        length_rows.append({"length_bin": str(length_bin), **metrics})
    length_table = pd.DataFrame(length_rows)
    length_table_path = tables_dir / "slice_by_length_bin.csv"
    length_table.to_csv(length_table_path, index=False)

    trigger_tokens = load_trigger_tokens(args.lexical_csv, top_k=20)
    df["has_trigger"] = df["text"].apply(lambda text: contains_trigger_token(text, trigger_tokens))
    trigger_rows = []
    for has_trigger, subset in df.groupby("has_trigger"):
        metrics = _metrics_for_subset(subset, threshold=threshold)
        trigger_rows.append({"has_trigger": bool(has_trigger), **metrics})
    trigger_table = pd.DataFrame(trigger_rows)
    trigger_table_path = tables_dir / "slice_by_lexical_trigger.csv"
    trigger_table.to_csv(trigger_table_path, index=False)

    severity_rows = []
    for severity in [2, 3, 4]:
        subset = df[df["orig_label"] == severity]
        total = int(len(subset))
        recall = float((subset["pred"] == 1).mean()) if total else 0.0
        severity_rows.append({"orig_label": severity, "count": total, "recall": recall})
    severity_table = pd.DataFrame(severity_rows)
    severity_table_path = tables_dir / "recall_by_original_severity.csv"
    severity_table.to_csv(severity_table_path, index=False)

    baseline_comparison_path = tables_dir / "baseline_vs_final.csv"
    baseline_note = None
    baseline_probs_path = Path(args.baseline_dev_probs).resolve()
    baseline_summary_path = Path(args.baseline_run_summary).resolve()
    if baseline_probs_path.exists():
        baseline_probs = np.asarray(np.load(baseline_probs_path), dtype=np.float64)
        if baseline_probs.shape[0] != df.shape[0]:
            baseline_note = (
                f"Baseline probability length mismatch: baseline={baseline_probs.shape[0]}, "
                f"final={df.shape[0]}"
            )
            pd.DataFrame([{"note": baseline_note}]).to_csv(baseline_comparison_path, index=False)
        else:
            if baseline_summary_path.exists():
                baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
                baseline_threshold = float(baseline_summary["dev_metrics"]["threshold"])
            else:
                baseline_threshold = 0.5
            baseline_pred = (baseline_probs >= baseline_threshold).astype(int)
            final_pred = df["pred"].to_numpy(dtype=np.int64)
            truth = df["label"].to_numpy(dtype=np.int64)
            baseline_correct = baseline_pred == truth
            final_correct = final_pred == truth
            comparison = pd.DataFrame(
                [
                    {
                        "both_correct": int(np.sum(baseline_correct & final_correct)),
                        "both_wrong": int(np.sum((~baseline_correct) & (~final_correct))),
                        "baseline_only_correct": int(np.sum(baseline_correct & (~final_correct))),
                        "final_only_correct": int(np.sum((~baseline_correct) & final_correct)),
                        "baseline_threshold": baseline_threshold,
                        "final_threshold": threshold,
                    }
                ]
            )
            comparison.to_csv(baseline_comparison_path, index=False)
    else:
        baseline_note = f"Baseline dev probs not found: {baseline_probs_path}"
        pd.DataFrame([{"note": baseline_note}]).to_csv(baseline_comparison_path, index=False)

    ablation_source = Path(args.ablation_summary).resolve()
    copied_ablation_path = tables_dir / "ablation_summary.csv"
    method_validity_md = ""
    failure_analysis_md = ""
    if ablation_source.exists():
        shutil.copyfile(ablation_source, copied_ablation_path)
        ablation_df = pd.read_csv(copied_ablation_path)
        if "dev_f1" in ablation_df.columns:
            table_cols = [
                col
                for col in [
                    "run_id",
                    "model",
                    "loss",
                    "lex_drop",
                    "dev_f1",
                    "dev_precision",
                    "dev_recall",
                    "dev_prob_std",
                    "degenerate_run",
                    "degenerate_reason",
                ]
                if col in ablation_df.columns
            ]
            if table_cols:
                method_validity_path = tables_dir / "method_validity_table.csv"
                ablation_df[table_cols].sort_values("dev_f1", ascending=False).to_csv(
                    method_validity_path, index=False
                )
                method_validity_md = (
                    "\n## Method Validity\n\n"
                    "- Summary table: `tables/method_validity_table.csv`\n"
                )
                deg_mask = pd.Series(False, index=ablation_df.index)
                if "degenerate_run" in ablation_df.columns:
                    deg_mask = deg_mask | ablation_df["degenerate_run"].fillna(False).astype(bool)
                if "dev_prob_std" in ablation_df.columns:
                    deg_mask = deg_mask | (ablation_df["dev_prob_std"].fillna(0.0) < 0.01)
                if "dev_f1" in ablation_df.columns:
                    deg_mask = deg_mask | (ablation_df["dev_f1"].fillna(0.0) < 0.4)
                deg_df = ablation_df[deg_mask].copy()
                if not deg_df.empty:
                    collapse_path = tables_dir / "deberta_collapse_candidates.csv"
                    selected_cols = [col for col in table_cols if col in deg_df.columns]
                    deg_df[selected_cols].to_csv(collapse_path, index=False)
                    failure_analysis_md = (
                        "\n## Failure Analysis\n\n"
                        "- Potentially degenerate runs: `tables/deberta_collapse_candidates.csv`\n"
                        "- Typical indicators: low `dev_f1`, near-constant probabilities (`dev_prob_std`),"
                        " and extreme positive prediction rate.\n"
                    )

    markdown_path = out_dir / "stage5_local_eval.md"
    markdown = f"""# Stage 5.2 Local Evaluation

## Core Dev Metrics

- Threshold: **{threshold:.3f}**
- Precision (PCL): **{final_metrics.precision:.4f}**
- Recall (PCL): **{final_metrics.recall:.4f}**
- F1 (PCL): **{final_metrics.f1:.4f}**
- Confusion counts: TP={final_metrics.tp}, FP={final_metrics.fp}, FN={final_metrics.fn}, TN={final_metrics.tn}

## Artifacts

- Confusion matrix table: `tables/confusion_matrix.csv`
- Confusion matrix figure: `figures/confusion_matrix.png`
- Classification report: `tables/classification_report.csv`
- PR curve: `figures/precision_recall_curve.png`
- Threshold-vs-F1: `figures/threshold_vs_f1.png`
- Manual error review (12 cases): `tables/manual_error_review_12.csv`
- Slice by keyword: `tables/slice_by_keyword.csv`
- Slice by length bin: `tables/slice_by_length_bin.csv`
- Slice by lexical trigger: `tables/slice_by_lexical_trigger.csv`
- Recall by original severity: `tables/recall_by_original_severity.csv`
- Baseline vs final comparison: `tables/baseline_vs_final.csv`

## Notes

- The manual-review file contains the highest-confidence FP/FN cases and blank `manual_note` cells for qualitative annotation.
- Trigger tokens for lexical-slice analysis were loaded from: `{args.lexical_csv}`.
"""
    if ablation_source.exists():
        markdown += "- Ablation summary copy: `tables/ablation_summary.csv`\n"
    if method_validity_md:
        markdown += method_validity_md
    if failure_analysis_md:
        markdown += failure_analysis_md
    if baseline_note:
        markdown += f"- Baseline comparison note: {baseline_note}\n"

    markdown_path.write_text(markdown, encoding="utf-8")

    summary_payload = {
        "out_dir": str(out_dir),
        "markdown_report": str(markdown_path),
        "final_metrics": final_metrics.to_dict(),
        "threshold": threshold,
    }
    (out_dir / "local_eval_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
