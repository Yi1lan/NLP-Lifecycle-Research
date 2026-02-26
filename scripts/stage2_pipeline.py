#!/usr/bin/env python3
"""Stage 2 pipeline: acquisition, preprocessing, and EDA generation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2.acquisition import download_stage2_data
from src.stage2.preprocessing import run_preprocessing
from src.stage2.eda import run_eda
from src.stage2.reporting import write_stage2_summary


def main() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    outputs_dir = PROJECT_ROOT / "outputs" / "stage2"

    raw_paths = download_stage2_data(raw_dir)
    preprocess_outputs = run_preprocessing(raw_paths, processed_dir)
    eda_outputs = run_eda(preprocess_outputs, outputs_dir)
    write_stage2_summary(preprocess_outputs, eda_outputs, outputs_dir / "stage2_summary.md")


if __name__ == "__main__":
    main()
