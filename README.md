# NLP-Lifecycle-Research

Stage-by-stage binary classification of Patronising and Condescending Language (PCL).

## Stage 2 (Data Acquisition, Exploration, and Preprocessing)

Stage 2 includes:

- Data acquisition from the official sources
- Preprocessing and binary-label normalization
- EDA Technique 1: Class distribution
- EDA Technique 2: Token-length profiling
- EDA Technique 3: Lexical signal analysis (class-discriminative unigrams)

## Repository Layout

```
NLP-Research-Lifecycle/
├── environment.yml
├── scripts/
│   └── stage2_pipeline.py
├── src/
│   └── stage2/
│       ├── __init__.py
│       ├── acquisition.py
│       ├── eda.py
│       ├── preprocessing.py
│       └── reporting.py
└── data/
    ├── raw/          # downloaded files
    └── processed/    # cleaned/split files used for later stages
```

## Stage 2 Run Instructions (Conda)

1. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate nlp-research-lifecycle
```

2. Run Stage 2 pipeline:

```bash
python scripts/stage2_pipeline.py
```

3. Inspect outputs:

- Processed data: `data/processed/`
- EDA tables: `outputs/stage2/tables/`
- EDA figures: `outputs/stage2/figures/`
- Report-ready summary: `outputs/stage2/stage2_summary.md`

## Notes

- This stage only implements data and EDA workflows.
- No training is performed in Stage 2.

## Stage 3-5 Pipeline (IARF)

Stage 3-5 implementation is in `src/stage3/` and the orchestration scripts:

- `scripts/stage4_run_matrix.py`
- `scripts/stage4_deberta_diagnosis.py`
- `scripts/stage5_make_submission.py`
- `scripts/stage5_validate_submission.py`
- `scripts/stage5_local_eval.py`
- Submission package: `BestModel/`

### Run Stage 4 Matrix (training + probs + ensemble)

```bash
python3 scripts/stage4_run_matrix.py \
  --data-dir data/raw \
  --out-root outputs/stage4 \
  --skip-existing
```

If running DeBERTa-v3 and tokenizer loading reports missing backend deps, install:

```bash
python -m pip install -U protobuf
```

Some `transformers` builds may also require:

```bash
python -m pip install -U tiktoken
```

Key outputs:

- Per-run checkpoints/metadata: `outputs/stage4/runs/`
- Per-run probabilities: `outputs/stage4/probs/`
- Final ensemble summary + `dev.txt`/`test.txt`: `outputs/stage4/final_ensemble/`
- Run-selection manifest (health-filter based): `outputs/stage4/final_ensemble/selected_runs.json`
- Ablation table: `outputs/stage4/ablation_summary.csv`

### Focused DeBERTa Diagnosis Matrix (for failure analysis)

```bash
python3 scripts/stage4_deberta_diagnosis.py \
  --data-dir data/raw \
  --out-root outputs/stage4/deberta_diagnosis \
  --promote-best-two \
  --include-weight-half
```

This runs a compact diagnosis matrix across CE/focal, lexical-drop toggles, tokenizer mode,
learning-rate adjustment, and class-weight scaling.

### Create Stage 5 Submission Files

```bash
python3 scripts/stage5_make_submission.py \
  --ensemble-dir outputs/stage4/final_ensemble \
  --out-dir outputs/stage5/submission \
  --data-dir data/raw
```

This writes:

- `outputs/stage5/submission/dev.txt`
- `outputs/stage5/submission/test.txt`
- Root-level `dev.txt` and `test.txt` for GTA visibility

### Validate Submission Format

```bash
python3 scripts/stage5_validate_submission.py \
  --dev outputs/stage5/submission/dev.txt \
  --test outputs/stage5/submission/test.txt \
  --data-dir data/raw \
  --ensemble-summary outputs/stage4/final_ensemble/ensemble_summary.json
```

### Generate Stage 5.2 Local Evaluation Package

```bash
python3 scripts/stage5_local_eval.py \
  --data-dir data/raw \
  --ensemble-summary outputs/stage4/final_ensemble/ensemble_summary.json \
  --out-dir outputs/stage5/local_eval
```

Report-ready markdown is generated at:

- `outputs/stage5/local_eval/stage5_local_eval.md`

## Spec-facing Checklist

- `BestModel/` present with train/predict/ensemble entrypoints
- Root `dev.txt` and `test.txt` present (0/1 per line)
- Submission copies under `outputs/stage5/submission/`
