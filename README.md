# NLP-Lifecycle-Research

Stage-by-stage binary classification for Patronising and Condescending Language (PCL), covering data preparation, model training, ensembling, submission generation, and local evaluation.

## 1) Key Reproducibility Artifacts

This section provides direct access to the primary artifacts used for academic review and reproducibility.

### Final submission files

- Root `dev.txt`: [`dev.txt`](./dev.txt)
- Root `test.txt`: [`test.txt`](./test.txt)

### BestModel package (full proposal pipeline)

- BestModel folder: [`BestModel/`](./BestModel)
- Stage 4 pipeline runner (matrix + final ensemble): [`BestModel/run_pipeline.py`](./BestModel/run_pipeline.py)
- Core model entrypoints (`train` / `predict` / `ensemble`):
  [`BestModel/train.py`](./BestModel/train.py),
  [`BestModel/predict.py`](./BestModel/predict.py),
  [`BestModel/ensemble.py`](./BestModel/ensemble.py)
- Submission materialization + validation:
  [`BestModel/make_submission.py`](./BestModel/make_submission.py),
  [`BestModel/validate_submission.py`](./BestModel/validate_submission.py)
- BestModel setting file: [`BestModel/config.json`](./BestModel/config.json)
- BestModel package note: [`BestModel/README.md`](./BestModel/README.md)
- In-package ensemble artifacts:
  [`BestModel/ensemble_summary.json`](./BestModel/ensemble_summary.json),
  [`BestModel/selected_runs.json`](./BestModel/selected_runs.json),
  [`BestModel/run_matrix_summary.json`](./BestModel/run_matrix_summary.json),
  [`BestModel/dev.txt`](./BestModel/dev.txt),
  [`BestModel/test.txt`](./BestModel/test.txt)
- Checkpoint storage note: large model checkpoints are stored under `outputs/stage4/runs/` and are not duplicated inside `BestModel`.

### Best model (single model without ensemble) and ensemble JSON artifacts

- Best model JSON summary: [`outputs/stage4/best_model/best_model_summary.json`](./outputs/stage4/best_model/best_model_summary.json)
- Final ensemble JSON summary: [`outputs/stage4/final_ensemble/ensemble_summary.json`](./outputs/stage4/final_ensemble/ensemble_summary.json)
- Final ensemble selected-runs JSON: [`outputs/stage4/final_ensemble/selected_runs.json`](./outputs/stage4/final_ensemble/selected_runs.json)
- Stage 4 ablation summary: [`outputs/stage4/ablation_summary.csv`](./outputs/stage4/ablation_summary.csv)

## 2) Summary Results of the Proposed Approach

The following summary is taken from the final generated artifacts:
[`outputs/stage4/final_ensemble/ensemble_summary.json`](./outputs/stage4/final_ensemble/ensemble_summary.json),
[`outputs/stage4/best_model/best_model_summary.json`](./outputs/stage4/best_model/best_model_summary.json),
[`outputs/stage4/run_matrix_summary.json`](./outputs/stage4/run_matrix_summary.json).

### Stage 3 Novelty to BestModel Code Mapping

The Stage 3 proposed method in the report (IARF) is implemented in `BestModel` as follows:

- Imbalance-aware objective (class-weighted focal loss, `gamma=2.0`): [`BestModel/train.py`](./BestModel/train.py) + [`BestModel/losses.py`](./BestModel/losses.py)
  - `compute_balanced_class_weights(...)` + `WeightedFocalLoss(...)` are used when `--loss focal`.
- Lexical-robust augmentation (trigger-token dropout on positive samples): [`BestModel/train.py`](./BestModel/train.py) + [`BestModel/augment.py`](./BestModel/augment.py)
  - Trigger tokens come from Stage 2 lexical analysis; default dropout probability is `0.2`.
- Dev-threshold optimization for positive-class F1: [`BestModel/train.py`](./BestModel/train.py), [`BestModel/ensemble.py`](./BestModel/ensemble.py), [`BestModel/losses.py`](./BestModel/losses.py)
  - `search_best_threshold(...)` sweeps `0.05 -> 0.95` (step `0.005`) instead of fixed threshold `0.5`.
- Probability-level ensemble with health filtering + auditability: [`BestModel/ensemble.py`](./BestModel/ensemble.py)
  - Run health checks (`dev_f1`, `dev_prob_std`, positive rate bounds), probability averaging, and manifest export in `selected_runs.json`.

| Item | Value |
| --- | --- |
| Final ensemble (dev) F1 | 0.6400 |
| Final ensemble (dev) precision | 0.6368 |
| Final ensemble (dev) recall | 0.6432 |
| Final ensemble threshold | 0.4700 |
| Selected ensemble runs | 10 / 15 |
| Best single model ID | `roberta_large_seed777` |
| Best single model (dev) F1 | 0.6289 |
| Best single model threshold | 0.2050 |
| Selected sequence length | 192 |
| Ensemble predicted positives on test | 319 |

Note: official test labels are not available in this repository, so official test F1 is not reported locally.

## 3) How To Run The Whole Project

Run from repository root.

### 3.1 Environment setup

```bash
conda env create -f environment.yml
conda activate nlp-research-lifecycle
```

If DeBERTa tokenizer dependencies are missing in your environment:

```bash
python -m pip install -U protobuf
python -m pip install -U tiktoken
```

### 3.2 End-to-end pipeline

```bash
# Stage 2: data acquisition + preprocessing + EDA
python scripts/stage2_pipeline.py

# Stage 4: run training matrix and build ensemble outputs
python scripts/stage4_run_matrix.py \
  --data-dir data/raw \
  --out-root outputs/stage4 \
  --seeds 42,123,2024,3407,777 \
  --skip-existing

# Stage 5: create final submission files (also copies to root dev.txt/test.txt)
python scripts/stage5_make_submission.py \
  --ensemble-dir outputs/stage4/final_ensemble \
  --out-dir outputs/stage5/submission \
  --data-dir data/raw

# Validate submission format/order
python scripts/stage5_validate_submission.py \
  --dev outputs/stage5/submission/dev.txt \
  --test outputs/stage5/submission/test.txt \
  --data-dir data/raw \
  --ensemble-summary outputs/stage4/final_ensemble/ensemble_summary.json

# Stage 5.2: local evaluation package
python scripts/stage5_local_eval.py \
  --data-dir data/raw \
  --ensemble-summary outputs/stage4/final_ensemble/ensemble_summary.json \
  --out-dir outputs/stage5/local_eval
```

Optional diagnostic experiment (failure analysis):

```bash
python scripts/stage4_deberta_diagnosis.py \
  --data-dir data/raw \
  --out-root outputs/stage4/deberta_diagnosis \
  --promote-best-two \
  --include-weight-half
```

## 4) Stage Descriptions

### Stage 2: Data Acquisition, Preprocessing, and EDA

- Script: [`scripts/stage2_pipeline.py`](./scripts/stage2_pipeline.py)
- Modules: [`src/stage2/`](./src/stage2)
- What it does:
  - Downloads/loads official Task 4 data
  - Converts ordinal labels to binary labels (PCL vs non-PCL)
  - Produces EDA tables/figures and stage summary
- Main outputs:
  - `outputs/stage2/tables/`
  - `outputs/stage2/figures/`
  - [`outputs/stage2/stage2_summary.md`](./outputs/stage2/stage2_summary.md)

### Stage 3: Core Modeling Components

- Code location: [`src/stage3/`](./src/stage3)
- Key modules:
  - Training: [`src/stage3/train.py`](./src/stage3/train.py)
  - Inference: [`src/stage3/predict.py`](./src/stage3/predict.py)
  - Ensembling: [`src/stage3/ensemble.py`](./src/stage3/ensemble.py)
  - Data/order handling: [`src/stage3/data.py`](./src/stage3/data.py)
- Notes:
  - Stage 3 provides reusable model components used by Stage 4 and Stage 5 scripts.

### Stage 4: Training Matrix, Run Selection, and Best Model

- Main script: [`scripts/stage4_run_matrix.py`](./scripts/stage4_run_matrix.py)
- What it does:
  - Runs multi-model, multi-seed training/inference matrix
  - Applies health filtering for ensemble run selection
  - Builds final ensemble predictions and metrics
  - Exports single best-model artifact set
- Main outputs:
  - `outputs/stage4/runs/` (per-run models and run summaries)
  - `outputs/stage4/probs/` (per-run dev/test probabilities)
  - [`outputs/stage4/final_ensemble/ensemble_summary.json`](./outputs/stage4/final_ensemble/ensemble_summary.json)
  - [`outputs/stage4/final_ensemble/selected_runs.json`](./outputs/stage4/final_ensemble/selected_runs.json)
  - [`outputs/stage4/best_model/best_model_summary.json`](./outputs/stage4/best_model/best_model_summary.json)

### Stage 5: Submission Materialization, Validation, and Local Evaluation

- Submission builder: [`scripts/stage5_make_submission.py`](./scripts/stage5_make_submission.py)
- Format/order validator: [`scripts/stage5_validate_submission.py`](./scripts/stage5_validate_submission.py)
- Local evaluation package: [`scripts/stage5_local_eval.py`](./scripts/stage5_local_eval.py)
- What it does:
  - Writes final `dev.txt`/`test.txt` to `outputs/stage5/submission/` and root
  - Validates line count, binary format, and ordering consistency
  - Produces figures/tables/report for local error analysis
- Main outputs:
  - [`outputs/stage5/submission/submission_report.json`](./outputs/stage5/submission/submission_report.json)
  - [`outputs/stage5/submission/validation_report.json`](./outputs/stage5/submission/validation_report.json)
  - [`outputs/stage5/local_eval/stage5_local_eval.md`](./outputs/stage5/local_eval/stage5_local_eval.md)
