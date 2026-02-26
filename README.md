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
