"""Preprocessing utilities for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd
from pandas.errors import ParserError

from .acquisition import RawDataPaths


ID_CANDIDATES = ["par_id", "paragraph_id", "id", "sample_id"]
TEXT_CANDIDATES = ["txt", "text", "sentence", "content"]
LABEL_CANDIDATES = ["label", "labels", "class", "binary_label", "target"]


@dataclass(frozen=True)
class PreprocessOutputs:
    full_dataset_path: Path
    train_path: Path
    dev_path: Path
    test_path: Path
    summary_json_path: Path
    summary_stats: dict[str, int | float]


def _read_table(path: Path) -> pd.DataFrame:
    def _safe_read(
        file_path: Path,
        *,
        sep: str,
        skiprows: int = 0,
    ) -> pd.DataFrame:
        return pd.read_csv(
            file_path,
            sep=sep,
            engine="python",
            skiprows=skiprows,
        )

    def _has_html_signature(lines: list[str]) -> bool:
        head = "\n".join(lines[:10]).lower()
        return "<html" in head or "<!doctype html" in head

    def _find_first_data_line(lines: list[str], sep: str) -> int:
        for idx, line in enumerate(lines):
            if line.count(sep) > 0:
                return idx
        return 0

    default_sep = "\t" if path.suffix.lower() == ".tsv" else ","
    alternate_sep = "," if default_sep == "\t" else "\t"

    try:
        df = _safe_read(path, sep=default_sep)
        if df.shape[1] > 1:
            return df
    except ParserError:
        pass

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if _has_html_signature(lines):
        raise RuntimeError(
            f"{path.name} appears to be an HTML document instead of a data file. "
            "Delete it and rerun the pipeline to re-download."
        )

    first_data_line = _find_first_data_line(lines, default_sep)
    first_data_line_alt = _find_first_data_line(lines, alternate_sep)
    candidate_attempts = [
        (default_sep, first_data_line),
        (default_sep, 0),
        (alternate_sep, first_data_line_alt),
        (alternate_sep, 0),
    ]

    tried: list[str] = []
    for sep, skiprows in candidate_attempts:
        key = f"sep={repr(sep)}, skiprows={skiprows}"
        if key in tried:
            continue
        tried.append(key)
        try:
            df = _safe_read(path, sep=sep, skiprows=skiprows)
            if df.shape[1] > 1:
                return df
        except ParserError:
            continue

    raise RuntimeError(
        f"Could not parse {path.name} as structured data. "
        f"Tried {', '.join(tried)}. "
        "Please check that the downloaded file is complete and in CSV/TSV format."
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _coerce_binary_label(raw_label: object) -> int | None:
    if pd.isna(raw_label):
        return None

    value_str = str(raw_label).strip().lower()
    if value_str in {"", "nan", "none"}:
        return None
    if value_str.startswith("[") and value_str.endswith("]"):
        payload = value_str[1:-1].strip()
        if payload == "":
            return 0
        try:
            values = [int(float(part.strip())) for part in payload.split(",")]
        except ValueError:
            return None
        return 1 if sum(values) > 0 else 0
    if value_str in {"pcl", "positive", "pos", "yes", "true"}:
        return 1
    if value_str in {"no_pcl", "non-pcl", "negative", "neg", "no", "false"}:
        return 0

    try:
        numeric = float(value_str)
    except ValueError:
        return None

    if numeric in {0.0, 1.0}:
        return int(numeric)
    return 1 if numeric > 0.0 else 0


def _coerce_main_dataset_label(raw_label: object) -> int | None:
    """Map DPM ordinal labels to binary: 0/1 -> No PCL, 2/3/4 -> PCL."""
    if pd.isna(raw_label):
        return None

    value_str = str(raw_label).strip().lower()
    if value_str in {"", "nan", "none"}:
        return None
    if value_str in {"pcl", "positive", "pos", "yes", "true"}:
        return 1
    if value_str in {"no_pcl", "non-pcl", "negative", "neg", "no", "false"}:
        return 0

    try:
        numeric = float(value_str)
    except ValueError:
        return None

    return 1 if numeric >= 2.0 else 0


def _clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("\u200b", "", regex=False)
        .str.strip()
    )


def _read_main_dataset_headerless(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    first_data_line = None
    for idx, line in enumerate(lines):
        # Official DPM file rows have 6 tab-separated fields.
        if line.count("\t") >= 5:
            first_data_line = idx
            break

    if first_data_line is None:
        raise RuntimeError(
            f"Could not locate tabular rows in {path.name} for headerless parsing."
        )

    raw = pd.read_csv(
        path,
        sep="\t",
        header=None,
        skiprows=first_data_line,
        engine="python",
        on_bad_lines="skip",
    )
    if raw.shape[1] < 6:
        raise RuntimeError(
            f"Unexpected column count in {path.name}: expected at least 6, got {raw.shape[1]}."
        )

    output = raw.iloc[:, [0, 4, 5]].copy()
    output.columns = ["par_id", "text", "label_raw"]
    return output


def _load_split_dataframe(path: Path) -> pd.DataFrame:
    df = _normalize_columns(_read_table(path))
    if df.empty:
        return df

    id_col = _find_column(df, ID_CANDIDATES)
    label_col = _find_column(df, LABEL_CANDIDATES)

    if id_col is None:
        id_col = str(df.columns[0])
    if label_col is None and len(df.columns) > 1:
        label_col = str(df.columns[1])

    columns_to_keep = [id_col]
    if label_col is not None:
        columns_to_keep.append(label_col)

    split_df = df[columns_to_keep].copy()
    split_df = split_df.rename(columns={id_col: "par_id"})
    if label_col is not None:
        split_df = split_df.rename(columns={label_col: "split_label_raw"})
    else:
        split_df["split_label_raw"] = None

    split_df["par_id"] = pd.to_numeric(split_df["par_id"], errors="coerce")
    split_df = split_df.dropna(subset=["par_id"])
    split_df["par_id"] = split_df["par_id"].astype("int64")
    split_df["split_label_binary"] = split_df["split_label_raw"].apply(_coerce_binary_label)
    split_df = split_df.drop_duplicates(subset=["par_id"], keep="first")
    return split_df


def _collect_split_files(
    split_files: list[Path],
) -> tuple[list[Path], list[Path]]:
    train_files: list[Path] = []
    dev_files: list[Path] = []
    for path in split_files:
        name = path.name.lower()
        if "train" in name:
            train_files.append(path)
        elif "dev" in name or "valid" in name:
            dev_files.append(path)
    return train_files, dev_files


def _merge_split_files(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["par_id", "split_label_raw", "split_label_binary"])
    frames = [_load_split_dataframe(path) for path in paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["par_id"], keep="first")
    return merged


def _build_main_binary_dataset(raw_paths: RawDataPaths) -> tuple[pd.DataFrame, dict[str, int | float]]:
    try:
        raw_df = _normalize_columns(_read_table(raw_paths.dataset_tsv))
    except RuntimeError:
        raw_df = pd.DataFrame()

    id_col = _find_column(raw_df, ID_CANDIDATES)
    text_col = _find_column(raw_df, TEXT_CANDIDATES)
    label_col = _find_column(raw_df, LABEL_CANDIDATES)

    missing = [
        name
        for name, col in {"id": id_col, "text": text_col, "label": label_col}.items()
        if col is None
    ]
    if missing:
        df = _read_main_dataset_headerless(raw_paths.dataset_tsv)
    else:
        df = raw_df[[id_col, text_col, label_col]].copy()
        df.columns = ["par_id", "text", "label_raw"]

    df["par_id"] = pd.to_numeric(df["par_id"], errors="coerce")
    df["text"] = _clean_text_series(df["text"])
    df["label_binary"] = df["label_raw"].apply(_coerce_main_dataset_label)

    rows_before = len(df)
    df = df.dropna(subset=["par_id", "text", "label_binary"])
    df = df[df["text"].ne("")]
    df["par_id"] = df["par_id"].astype("int64")
    df["label_binary"] = df["label_binary"].astype("int64")
    rows_after_dropna = len(df)

    duplicate_par_ids = int(df["par_id"].duplicated().sum())
    duplicate_texts = int(df["text"].duplicated().sum())
    df = df.drop_duplicates(subset=["par_id"], keep="first")
    rows_after_dedup = len(df)

    df["token_count"] = df["text"].str.split().map(len).astype("int64")
    df = df.sort_values("par_id").reset_index(drop=True)

    summary = {
        "raw_rows": int(rows_before),
        "rows_after_dropna": int(rows_after_dropna),
        "rows_after_dedup": int(rows_after_dedup),
        "dropped_rows_total": int(rows_before - rows_after_dedup),
        "duplicate_par_id_rows": int(duplicate_par_ids),
        "duplicate_text_rows": int(duplicate_texts),
    }
    return df, summary


def _build_test_dataset(raw_paths: RawDataPaths) -> pd.DataFrame:
    try:
        test_df = _normalize_columns(_read_table(raw_paths.test_tsv))
    except RuntimeError:
        test_df = pd.DataFrame()
    text_col = _find_column(test_df, TEXT_CANDIDATES)
    id_col = _find_column(test_df, ID_CANDIDATES)

    if text_col is None:
        raw = pd.read_csv(
            raw_paths.test_tsv,
            sep="\t",
            header=None,
            engine="python",
            on_bad_lines="skip",
        )
        if raw.shape[1] < 2:
            raise RuntimeError(
                f"Missing text column in {raw_paths.test_tsv.name}. "
                f"Tried named columns: {', '.join(TEXT_CANDIDATES)} and headerless fallback."
            )
        output = pd.DataFrame({"sample_id": raw.iloc[:, 0], "text": raw.iloc[:, -1]})
    else:
        if id_col is None:
            output = pd.DataFrame({"sample_id": range(len(test_df)), "text": test_df[text_col]})
        else:
            output = pd.DataFrame({"sample_id": test_df[id_col], "text": test_df[text_col]})

    output["sample_id"] = output["sample_id"].astype(str).str.strip()
    output["text"] = _clean_text_series(output["text"])
    output = output[output["text"].ne("")].copy()
    output["token_count"] = output["text"].str.split().map(len).astype("int64")
    output = output.reset_index(drop=True)
    return output


def run_preprocessing(raw_paths: RawDataPaths, processed_root: Path) -> PreprocessOutputs:
    """Create processed train/dev/test files and preprocessing metadata."""
    processed_root.mkdir(parents=True, exist_ok=True)

    full_df, preprocess_summary = _build_main_binary_dataset(raw_paths)
    train_files, dev_files = _collect_split_files(raw_paths.split_files)
    if not train_files or not dev_files:
        raise RuntimeError(
            "Could not identify both train and dev split files in practice_splits. "
            "Expected filenames containing 'train' and 'dev'."
        )

    train_split_df = _merge_split_files(train_files)
    dev_split_df = _merge_split_files(dev_files)

    train_df = full_df.merge(train_split_df[["par_id"]], on="par_id", how="inner")
    dev_df = full_df.merge(dev_split_df[["par_id"]], on="par_id", how="inner")

    train_label_mismatch = 0
    dev_label_mismatch = 0
    if "split_label_binary" in train_split_df.columns:
        compare = train_df.merge(
            train_split_df[["par_id", "split_label_binary"]],
            on="par_id",
            how="left",
        )
        compare = compare.dropna(subset=["split_label_binary"])
        train_label_mismatch = int(
            (compare["label_binary"] != compare["split_label_binary"].astype("int64")).sum()
        )
    if "split_label_binary" in dev_split_df.columns:
        compare = dev_df.merge(
            dev_split_df[["par_id", "split_label_binary"]],
            on="par_id",
            how="left",
        )
        compare = compare.dropna(subset=["split_label_binary"])
        dev_label_mismatch = int(
            (compare["label_binary"] != compare["split_label_binary"].astype("int64")).sum()
        )

    test_df = _build_test_dataset(raw_paths)

    full_dataset_path = processed_root / "full_dataset_binary.csv"
    train_path = processed_root / "train_binary.csv"
    dev_path = processed_root / "dev_binary.csv"
    test_path = processed_root / "test_unlabeled.csv"
    summary_json_path = processed_root / "preprocessing_summary.json"

    full_df.to_csv(full_dataset_path, index=False)
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary_stats: dict[str, int | float] = {
        **preprocess_summary,
        "full_dataset_size": int(len(full_df)),
        "train_size": int(len(train_df)),
        "dev_size": int(len(dev_df)),
        "test_size": int(len(test_df)),
        "train_positive_rate": float(train_df["label_binary"].mean()) if len(train_df) else 0.0,
        "dev_positive_rate": float(dev_df["label_binary"].mean()) if len(dev_df) else 0.0,
        "train_label_mismatch_vs_split_file": int(train_label_mismatch),
        "dev_label_mismatch_vs_split_file": int(dev_label_mismatch),
    }

    with summary_json_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary_stats, summary_file, indent=2, sort_keys=True)

    return PreprocessOutputs(
        full_dataset_path=full_dataset_path,
        train_path=train_path,
        dev_path=dev_path,
        test_path=test_path,
        summary_json_path=summary_json_path,
        summary_stats=summary_stats,
    )
