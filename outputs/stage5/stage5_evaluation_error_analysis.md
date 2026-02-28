# Stage 5 Evaluation and Error Analysis

## 1) Model Summary

### Final Ensemble Result (official dev-set local mirror)

| Item | Value |
|---|---:|
| Decision threshold | 0.470 |
| Precision (PCL=1) | 0.6368 |
| Recall (PCL=1) | 0.6432 |
| F1 (PCL=1) | **0.6400** |
| TP / FP / FN / TN | 128 / 73 / 71 / 1821 |
| Accuracy | 0.9312 |

Source: [local_eval_summary.json](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/local_eval_summary.json), [classification_report.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/classification_report.csv), [confusion_matrix.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/confusion_matrix.csv)

### Submission Checks (Stage 5.1 format and ordering)

- `dev.txt` lines: **2093** (expected 2093)
- `test.txt` lines: **3832** (expected 3832)
- Values are binary (`0`/`1`)
- Order hashes match expected official order
- Validation status: **ok = true**, no errors

Source: [validation_report.json](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/submission/validation_report.json), [dev.txt](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/submission/dev.txt), [test.txt](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/submission/test.txt)

### Baseline Comparison

- Baseline dev F1 in coursework brief: **0.48**
- Final ensemble dev F1: **0.64**
- Absolute gain: **+0.16 F1**
- Relative gain over baseline: **+33.3%**
- `beats_baseline_dev`: **true**

Source: [submission_report.json](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/submission/submission_report.json)

### Model Performance Table (run-family summary from ablation/method-validity outputs)

The table below summarizes 25 matrix runs grouped by run family (`*_seed*` prefixes):

| Run family | #Runs | Mean F1 | Best F1 | Mean Precision | Mean Recall | Degenerate runs |
|---|---:|---:|---:|---:|---:|---:|
| `roberta_large` | 5 | 0.5992 | **0.6289** | 0.6241 | 0.5819 | 0 |
| `b1_roberta` | 5 | 0.6156 | 0.6276 | 0.5832 | 0.6543 | 0 |
| `b0_roberta` | 5 | 0.6124 | 0.6273 | 0.5977 | 0.6352 | 0 |
| `roberta` (lex-drop focal) | 5 | 0.6151 | 0.6219 | 0.5686 | 0.6724 | 0 |
| `deberta` | 5 | 0.1741 | 0.1759 | 0.0953 | 1.0000 | 5 |

Notes:
- Best single run F1 = **0.6289** (`roberta_large_seed777`).
- Final ensemble F1 = **0.6400**, which is +0.0111 above the best single run.
- All DeBERTa diagnosis runs were flagged as degenerate in this experiment.

Source: [method_validity_table.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/method_validity_table.csv), [deberta_collapse_candidates.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/deberta_collapse_candidates.csv)

---

## 2) Error Analysis (Stage 5.2, required)

### Global Error Profile

- Total dev errors: **144** / 2093
- False Positives (FP): **73**
- False Negatives (FN): **71**
- False Positive Rate (class 0): **0.0385**
- False Negative Rate (class 1): **0.3568**

Interpretation:
- Precision and recall are reasonably balanced at the selected threshold.
- Remaining weakness is mainly minority-class recall: ~35.7% of positives are still missed.

### Slice-Based Failure Patterns

#### By keyword

Lower-recall keywords:

| Keyword | Count | Positives | Recall | F1 |
|---|---:|---:|---:|---:|
| `immigrant` | 218 | 7 | 0.2857 | 0.4444 |
| `women` | 233 | 14 | 0.3571 | 0.4545 |
| `migrant` | 206 | 5 | 0.4000 | 0.4444 |

Higher-recall keywords:

| Keyword | Count | Positives | Recall | F1 |
|---|---:|---:|---:|---:|
| `in-need` | 226 | 33 | 0.9394 | 0.7949 |
| `refugee` | 188 | 13 | 0.7692 | 0.7143 |

Interpretation:
- The model is strongest on explicit aid-language patterns (especially `in-need`).
- It struggles more on categories where PCL may be subtler or less lexically explicit (`immigrant`, `women`, `migrant`).

Source: [slice_by_keyword.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/slice_by_keyword.csv)

#### By text length

| Length bin | Count | Recall | F1 |
|---|---:|---:|---:|
| `<=64` | 1673 | 0.6571 | 0.6502 |
| `65-128` | 386 | 0.5849 | 0.6019 |
| `>128` | 34 | 0.8333 | 0.7143 |

Interpretation:
- Mid-length paragraphs (65-128) are the weakest range.
- Very long texts look good, but sample size is very small (`n=34`), so confidence is limited.

Source: [slice_by_length_bin.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/slice_by_length_bin.csv)

#### By lexical-trigger presence

| Has trigger token | Count | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| `False` | 2047 | 0.6067 | 0.6136 | 0.6102 | 0.0374 |
| `True` | 46 | 0.8696 | 0.8696 | 0.8696 | 0.1304 |

Interpretation:
- Trigger-rich examples are easier for positive detection, but they also increase false-positive tendency (higher FPR).
- Most data has no trigger (`n=2047`), where F1 drops to 0.6102.

Source: [slice_by_lexical_trigger.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/slice_by_lexical_trigger.csv)

#### By original severity label

| Original severity | Count | Recall |
|---|---:|---:|
| 2 | 18 | 0.2222 |
| 3 | 89 | 0.5506 |
| 4 | 92 | 0.8152 |

Interpretation:
- The model detects strong/severe PCL (label 4) well.
- It under-detects weaker/marginal PCL (label 2), consistent with subtle-cue misses.

Source: [recall_by_original_severity.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/recall_by_original_severity.csv)

### Manual Case Review (high-confidence errors)

From [manual_error_review_12.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/manual_error_review_12.csv):

- High-confidence FPs are mostly humanitarian/helping narratives containing words like "in need", but without clearly patronising framing.
- High-confidence FNs include subtle institutional/policy wording about vulnerable groups where condescension is implied rather than explicit.

Representative pattern-level diagnosis:

- FP mechanism: lexical over-triggering on compassion/help vocabulary.
- FN mechanism: misses context-dependent tone and implicit power dynamics.

### Baseline-vs-final disagreement analysis (spec-suggested)

| Category | Count | Share |
|---|---:|---:|
| Both correct | 1917 | 91.59% |
| Both wrong | 107 | 5.11% |
| Baseline only correct | 37 | 1.77% |
| Final only correct | 32 | 1.53% |

Interpretation:
- Most examples are stable across both models.
- Small disagreement region indicates improvements came from rebalancing precision/recall on positives rather than broad sample flips.

Source: [baseline_vs_final.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/baseline_vs_final.csv)

---

## 3) Local Evaluation (Stage 5.2, additional analyses)

The local evaluation package already includes the key components expected in the coursework brief:

- Error analysis with concrete examples and slice tables.
- Confusion matrix and classification report.
- Precision-Recall curve.
- Threshold-vs-F1 analysis supporting the selected threshold (0.470).
- Ablation/method-validity table plus explicit failure analysis for degenerate runs.

Main local-evaluation conclusions:

1. The ensemble clearly surpasses the baseline on dev F1 (0.64 vs 0.48).
2. Performance is strongest on explicit lexical cues, but weaker on subtle PCL and specific topical slices (`immigrant`, `women`, `migrant`).
3. DeBERTa configuration in this run collapsed (near-constant probabilities), and the diagnostics correctly surfaced this failure mode.
4. Remaining headroom is primarily recall on subtle positives (especially original severity 2/3), not broad formatting or ordering issues.

Key artifacts for report insertion:

- [stage5_local_eval.md](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/stage5_local_eval.md)
- [confusion_matrix.png](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/figures/confusion_matrix.png)
- [precision_recall_curve.png](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/figures/precision_recall_curve.png)
- [threshold_vs_f1.png](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/figures/threshold_vs_f1.png)
- [method_validity_table.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/method_validity_table.csv)
- [manual_error_review_12.csv](/Users/yi1lan/Desktop/NLP-Lifecycle-Research/outputs/stage5/local_eval/tables/manual_error_review_12.csv)
