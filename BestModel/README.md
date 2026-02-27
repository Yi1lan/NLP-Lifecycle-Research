# BestModel (Coursework Submission Package)

This folder contains the submission-facing entrypoints for the final Stage 3-5 pipeline.

## Files

- `train.py`: wraps `src.stage3.train`
- `predict.py`: wraps `src.stage3.predict`
- `ensemble.py`: wraps `src.stage3.ensemble`
- `config.json`: pinned method choices and health-filter criteria

## Recommended Reproduction Flow

Run from repo root.

```bash
python3 scripts/stage4_run_matrix.py \
  --data-dir data/raw \
  --out-root outputs/stage4 \
  --skip-existing
```

```bash
python3 scripts/stage5_make_submission.py \
  --ensemble-dir outputs/stage4/final_ensemble \
  --out-dir outputs/stage5/submission \
  --data-dir data/raw
```

```bash
python3 scripts/stage5_local_eval.py \
  --data-dir data/raw \
  --ensemble-summary outputs/stage4/final_ensemble/ensemble_summary.json \
  --out-dir outputs/stage5/local_eval
```

## Focused DeBERTa Diagnosis (for report failure analysis)

```bash
python3 scripts/stage4_deberta_diagnosis.py \
  --data-dir data/raw \
  --out-root outputs/stage4/deberta_diagnosis \
  --promote-best-two \
  --include-weight-half
```

Outputs:

- `outputs/stage4/deberta_diagnosis/diagnosis_summary.csv`
- `outputs/stage4/deberta_diagnosis/diagnosis_manifest.json`

