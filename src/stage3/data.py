"""Data loading and ordering utilities for the Stage 3-5 pipeline."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DataPaths:
    """Resolved paths for official Task 4 files."""

    pcl: Path
    train_split: Path
    dev_split: Path
    test: Path

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key, value in payload.items():
            payload[key] = str(value)
        return payload


@dataclass(frozen=True)
class LabeledExample:
    """One train/dev sample with all fields required for slice analysis."""

    par_id: str
    text: str
    label: int
    keyword: str
    country: str
    orig_label: int


@dataclass(frozen=True)
class TestExample:
    """One unlabeled official test sample."""

    sample_id: str
    text: str


@dataclass(frozen=True)
class Stage3Dataset:
    """Container for train/dev/test data in official ordering."""

    paths: DataPaths
    train: list[LabeledExample]
    dev: list[LabeledExample]
    test: list[TestExample]
    train_missing_ids: list[str]
    dev_missing_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "paths": self.paths.to_dict(),
            "train_size": len(self.train),
            "dev_size": len(self.dev),
            "test_size": len(self.test),
            "train_missing_ids": self.train_missing_ids,
            "dev_missing_ids": self.dev_missing_ids,
            "dev_order_hash": dev_order_hash(self),
            "test_order_hash": test_order_hash(self),
        }


def _normalize_id(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    try:
        parsed = float(text)
        if parsed.is_integer():
            return str(int(parsed))
    except ValueError:
        pass
    return text


def _resolve_first_existing(candidates: Iterable[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {label}. Checked:\n{joined}")


def resolve_data_paths(data_dir: str | Path) -> DataPaths:
    """Resolve official file paths from a root directory with flexible layouts."""

    root = Path(data_dir).expanduser().resolve()
    pcl = _resolve_first_existing(
        [
            root / "dontpatronizeme_pcl.tsv",
            root / "raw" / "dontpatronizeme_pcl.tsv",
            root
            / "NLPLabs-2024"
            / "Dont_Patronize_Me_Trainingset"
            / "dontpatronizeme_pcl.tsv",
        ],
        "main dataset TSV",
    )
    train_split = _resolve_first_existing(
        [
            root / "train_semeval_parids-labels.csv",
            root / "practice_splits" / "train_semeval_parids-labels.csv",
            root / "raw" / "practice_splits" / "train_semeval_parids-labels.csv",
            root
            / "dontpatronizeme"
            / "semeval-2022"
            / "practice splits"
            / "train_semeval_parids-labels.csv",
        ],
        "train split CSV",
    )
    dev_split = _resolve_first_existing(
        [
            root / "dev_semeval_parids-labels.csv",
            root / "practice_splits" / "dev_semeval_parids-labels.csv",
            root / "raw" / "practice_splits" / "dev_semeval_parids-labels.csv",
            root
            / "dontpatronizeme"
            / "semeval-2022"
            / "practice splits"
            / "dev_semeval_parids-labels.csv",
        ],
        "dev split CSV",
    )
    test = _resolve_first_existing(
        [
            root / "task4_test.tsv",
            root / "raw" / "task4_test.tsv",
            root / "TEST" / "task4_test.tsv",
            root / "semeval-2022" / "TEST" / "task4_test.tsv",
            root / "dontpatronizeme" / "semeval-2022" / "TEST" / "task4_test.tsv",
        ],
        "official test TSV",
    )
    return DataPaths(pcl=pcl, train_split=train_split, dev_split=dev_split, test=test)


def _binary_label_from_ordinal(raw_value: str) -> int | None:
    value = str(raw_value).strip()
    if value == "":
        return None
    try:
        ordinal = int(float(value))
    except ValueError:
        return None
    return 1 if ordinal >= 2 else 0


def load_main_dataset(pcl_path: str | Path) -> dict[str, LabeledExample]:
    """Load official training TSV into a map keyed by normalized paragraph ID."""

    path = Path(pcl_path)
    examples: dict[str, LabeledExample] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if len(parts) < 6:
                continue
            par_id = _normalize_id(parts[0])
            if not par_id:
                continue
            label = _binary_label_from_ordinal(parts[5])
            if label is None:
                continue
            text = str(parts[4]).strip()
            if not text:
                continue
            try:
                orig_label = int(float(parts[5]))
            except ValueError:
                continue
            if par_id in examples:
                continue
            examples[par_id] = LabeledExample(
                par_id=par_id,
                text=text,
                label=label,
                keyword=str(parts[2]).strip(),
                country=str(parts[3]).strip(),
                orig_label=orig_label,
            )
    return examples


def load_split_ids(split_path: str | Path) -> list[str]:
    """Load split IDs from CSV while preserving on-disk row order."""

    path = Path(split_path)
    ids: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader):
            if not row:
                continue
            maybe_id = _normalize_id(row[0])
            if not maybe_id:
                continue
            if row_index == 0 and maybe_id.lower() in {"par_id", "paragraph_id", "id"}:
                continue
            ids.append(maybe_id)
    return ids


def load_test_rows(test_path: str | Path) -> list[TestExample]:
    """Load official test rows in exact file order."""

    path = Path(test_path)
    rows: list[TestExample] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_index, parts in enumerate(reader):
            if not parts:
                continue
            if row_index == 0 and str(parts[0]).strip().lower() in {
                "id",
                "par_id",
                "sample_id",
            }:
                continue
            if len(parts) >= 5:
                sample_id = _normalize_id(parts[0])
                text = str(parts[4]).strip()
            elif len(parts) >= 2:
                sample_id = _normalize_id(parts[0])
                text = str(parts[-1]).strip()
            else:
                continue
            if not text:
                continue
            if not sample_id:
                sample_id = str(len(rows))
            rows.append(TestExample(sample_id=sample_id, text=text))
    return rows


def load_stage3_dataset(data_dir: str | Path) -> Stage3Dataset:
    """Load full train/dev/test data bundle with official ordering guarantees."""

    paths = resolve_data_paths(data_dir)
    main_map = load_main_dataset(paths.pcl)
    train_ids = load_split_ids(paths.train_split)
    dev_ids = load_split_ids(paths.dev_split)

    train = [main_map[row_id] for row_id in train_ids if row_id in main_map]
    dev = [main_map[row_id] for row_id in dev_ids if row_id in main_map]
    train_missing = [row_id for row_id in train_ids if row_id not in main_map]
    dev_missing = [row_id for row_id in dev_ids if row_id not in main_map]
    test = load_test_rows(paths.test)

    return Stage3Dataset(
        paths=paths,
        train=train,
        dev=dev,
        test=test,
        train_missing_ids=train_missing,
        dev_missing_ids=dev_missing,
    )


def id_order_hash(ids: Sequence[str]) -> str:
    """Create a deterministic hash for a sequence of IDs."""

    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dev_order_hash(data: Stage3Dataset) -> str:
    return id_order_hash([example.par_id for example in data.dev])


def test_order_hash(data: Stage3Dataset) -> str:
    return id_order_hash([example.sample_id for example in data.test])


def dev_labels(data: Stage3Dataset) -> list[int]:
    return [example.label for example in data.dev]


def train_labels(data: Stage3Dataset) -> list[int]:
    return [example.label for example in data.train]

