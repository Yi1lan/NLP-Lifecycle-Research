# Stage 5.2 Local Evaluation

## Core Dev Metrics

- Threshold: **0.680**
- Precision (PCL): **0.5776**
- Recall (PCL): **0.6734**
- F1 (PCL): **0.6218**
- Confusion counts: TP=134, FP=98, FN=65, TN=1796

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
- Trigger tokens for lexical-slice analysis were loaded from: `outputs/stage2/tables/lexical_analysis.csv`.
- Ablation summary copy: `tables/ablation_summary.csv`
