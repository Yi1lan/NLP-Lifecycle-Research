"""Stage 2 report artifact generation."""

from __future__ import annotations

from pathlib import Path

from .eda import EDAOutputs
from .preprocessing import PreprocessOutputs


def _pct(value: float) -> str:
    return f"{value:.2f}%"


def _num(value: float) -> str:
    return f"{value:.2f}"


def write_stage2_summary(
    preprocess_outputs: PreprocessOutputs,
    eda_outputs: EDAOutputs,
    summary_path: Path,
) -> None:
    """Write a report-ready markdown summary for Stage 2 Exercise 2."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    train_pos_pct = float(eda_outputs.key_metrics["train_positive_pct"])
    dev_pos_pct = float(eda_outputs.key_metrics["dev_positive_pct"])
    train_p95 = float(eda_outputs.key_metrics["train_p95_token_count"])
    train_pos_mean = float(eda_outputs.key_metrics["train_pos_mean_tokens"])
    train_neg_mean = float(eda_outputs.key_metrics["train_neg_mean_tokens"])
    lexical_vocab_size = int(eda_outputs.key_metrics["lexical_vocab_size"])
    top_pcl_token = str(eda_outputs.key_metrics["top_pcl_token"])
    top_no_pcl_token = str(eda_outputs.key_metrics["top_no_pcl_token"])
    majority_class = "PCL" if train_pos_pct >= 50.0 else "No PCL"

    dropped_rows = int(preprocess_outputs.summary_stats["dropped_rows_total"])
    duplicate_rows = int(preprocess_outputs.summary_stats["duplicate_par_id_rows"])

    content = f"""# Stage 2 Summary: Data Acquisition, Exploration, and Preprocessing

## Data Acquisition and Preprocessing Overview

- Raw dataset downloaded to: `data/raw/dontpatronizeme_pcl.tsv`
- Practice splits downloaded to: `data/raw/practice_splits/`
- Official test set downloaded to: `data/raw/task4_test.tsv`
- Preprocessing summary JSON: `data/processed/preprocessing_summary.json`
- Rows removed during cleanup: **{dropped_rows}**
- Duplicate paragraph IDs detected: **{duplicate_rows}**

---

## EDA Technique 1: Class Distribution

### Visual/Tabular Evidence

- Figure: `outputs/stage2/figures/class_distribution.png`
- Table: `outputs/stage2/tables/class_distribution.csv`

### Analysis

- Train positive (PCL) rate: **{_pct(train_pos_pct)}**
- Dev positive (PCL) rate: **{_pct(dev_pos_pct)}**
- The class ratio is imbalanced toward **{majority_class}**, so accuracy alone would be misleading.

### Impact Statement

- Use class-sensitive evaluation (positive-class F1 as the primary metric).
- Prefer class-weighted loss and/or sampling strategies in Stage 3/4 to improve recall on the PCL class.

---

## EDA Technique 2: Token-Length Profiling

### Visual/Tabular Evidence

- Figure: `outputs/stage2/figures/length_distribution.png`
- Table: `outputs/stage2/tables/length_summary.csv`

### Analysis

- Train p95 token length: **{_num(train_p95)}**
- Mean tokens (Train, PCL): **{_num(train_pos_mean)}**
- Mean tokens (Train, No PCL): **{_num(train_neg_mean)}**
- Length spread indicates that truncation settings will materially affect model behavior.

### Impact Statement

- Set model `max_length` using observed percentile statistics instead of arbitrary defaults.
- Evaluate truncation trade-offs during Stage 4 tuning to avoid losing critical context in long examples.

---

## EDA Technique 3: Lexical Signal Analysis

### Visual/Tabular Evidence

- Figure: `outputs/stage2/figures/lexical_analysis.png`
- Table: `outputs/stage2/tables/lexical_analysis.csv`

### Analysis

- Train lexical vocabulary size after filtering: **{lexical_vocab_size}**
- Top PCL-indicative token: **{top_pcl_token}**
- Top No-PCL-indicative token: **{top_no_pcl_token}**
- Distinct token usage between labels indicates that lexical cues can support early feature engineering and targeted error analysis.

### Impact Statement

- Use high-signal lexical terms to guide baseline feature checks and qualitative inspection.
- Review false positives/negatives against these lexical signals to confirm whether the model is learning meaningful patterns or shortcut cues.
"""

    summary_path.write_text(content, encoding="utf-8")
