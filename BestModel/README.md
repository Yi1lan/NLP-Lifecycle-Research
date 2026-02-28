# BestModel

This folder contains the **full proposal pipeline** used for the final result, not just a single best checkpoint.

Final dev result from this pipeline:

- Positive-class F1: `0.64`
- Precision: `0.6368`
- Recall: `0.6432`
- Ensemble threshold: `0.47`

## What is included

- Full training/inference/ensemble modules:
  - `train.py` (single run training)
  - `predict.py` (single run probability inference)
  - `ensemble.py` (health-filtered multi-run ensemble)
- Full pipeline runner:
  - `run_pipeline.py` (Stage 4 matrix + final ensemble generation)
- Submission utilities:
  - `make_submission.py`
  - `validate_submission.py`
- Shared helpers:
  - `augment.py`, `config.py`, `data.py`, `losses.py`, `submission.py`
- Reference outputs from the final ensemble:
  - `ensemble_summary.json`
  - `selected_runs.json`
  - `run_matrix_summary.json`
  - `dev.txt`, `test.txt`

## Storage note

Large model checkpoints are kept in `outputs/stage4/runs/` and are **not copied** into `BestModel`.

## Run the full proposal pipeline

Run from repository root:

```bash
python3 -m BestModel.run_pipeline \
  --data-dir data/raw \
  --out-root outputs/stage4 \
  --seeds 42,123,2024,3407,777 \
  --skip-existing
```

## Materialize and validate submission files

```bash
python3 -m BestModel.make_submission \
  --ensemble-dir outputs/stage4/final_ensemble \
  --out-dir outputs/stage5/submission \
  --data-dir data/raw
```

```bash
python3 -m BestModel.validate_submission \
  --dev outputs/stage5/submission/dev.txt \
  --test outputs/stage5/submission/test.txt \
  --data-dir data/raw \
  --ensemble-summary outputs/stage4/final_ensemble/ensemble_summary.json
```

## Component-level usage

Single run training:

```bash
python3 -m BestModel.train \
  --model roberta_large \
  --seed 777 \
  --max-len 192 \
  --loss ce \
  --lex-drop false \
  --data-dir data/raw \
  --out-dir outputs/stage4/runs/roberta_large_seed777
```

Single run inference:

```bash
python3 -m BestModel.predict \
  --checkpoint outputs/stage4/runs/roberta_large_seed777 \
  --split dev \
  --data-dir data/raw \
  --out-probs outputs/stage4/probs/roberta_large_seed777_dev.npy \
  --out-labels outputs/stage4/dev_labels.npy \
  --max-len 192
```

Ensemble from multiple runs:

```bash
python3 -m BestModel.ensemble \
  --dev-probs "outputs/stage4/probs/*_dev.npy" \
  --test-probs "outputs/stage4/probs/*_test.npy" \
  --dev-labels outputs/stage4/dev_labels.npy \
  --out-dir outputs/stage4/final_ensemble
```
