"""Data acquisition utilities for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests


DATASET_URL = (
    "https://raw.githubusercontent.com/CRLala/NLPLabs-2024/main/"
    "Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/"
    "semeval-2022/TEST/task4_test.tsv"
)
SPLITS_API_URL = (
    "https://api.github.com/repos/Perez-AlmendrosC/dontpatronizeme/contents/"
    "semeval-2022/practice%20splits"
)
SPLITS_RAW_FALLBACK_URLS = [
    "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/"
    "semeval-2022/practice%20splits/train_semeval_parids-labels.csv",
    "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/"
    "semeval-2022/practice%20splits/dev_semeval_parids-labels.csv",
]

REQUEST_TIMEOUT_SECONDS = 60
USER_AGENT = "nlp-research-lifecycle-stage2/1.0"


@dataclass(frozen=True)
class RawDataPaths:
    dataset_tsv: Path
    test_tsv: Path
    splits_dir: Path
    split_files: list[Path]


def _download_file(url: str, destination: Path, overwrite: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    with requests.get(
        url,
        stream=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    output_file.write(chunk)
    return destination


def _looks_like_html(path: Path) -> bool:
    if not path.exists():
        return False
    head = path.read_text(encoding="utf-8", errors="replace")[:300].lower()
    return "<html" in head or "<!doctype html" in head


def _looks_like_delimited_data(path: Path) -> bool:
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:30]
    if not lines:
        return False
    return any(("\t" in line or "," in line) for line in lines)


def _ensure_valid_file(url: str, destination: Path) -> Path:
    path = _download_file(url, destination)
    if _looks_like_html(path) or not _looks_like_delimited_data(path):
        path = _download_file(url, destination, overwrite=True)
    return path


def _download_split_files_from_api(splits_dir: Path) -> list[Path]:
    response = requests.get(
        SPLITS_API_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    entries = response.json()
    if not isinstance(entries, list):
        raise RuntimeError("Unexpected GitHub API response for practice splits")

    paths: list[Path] = []
    for entry in entries:
        if entry.get("type") != "file":
            continue
        download_url = entry.get("download_url")
        name = entry.get("name")
        if not download_url or not name:
            continue
        destination = splits_dir / str(name)
        path = _ensure_valid_file(download_url, destination)
        if _looks_like_delimited_data(path):
            paths.append(path)

    if not paths:
        raise RuntimeError("No split files were available from the GitHub API listing")
    return sorted(paths)


def _download_split_files_fallback(splits_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for url in SPLITS_RAW_FALLBACK_URLS:
        name = url.rsplit("/", maxsplit=1)[-1]
        destination = splits_dir / name
        try:
            path = _ensure_valid_file(url, destination)
            if _looks_like_delimited_data(path):
                paths.append(path)
        except requests.RequestException:
            continue
    if not paths:
        raise RuntimeError(
            "Unable to download practice split files from API or fallback URLs"
        )
    return sorted(paths)


def _download_split_files(splits_dir: Path) -> list[Path]:
    try:
        return _download_split_files_from_api(splits_dir)
    except (requests.RequestException, RuntimeError):
        return _download_split_files_fallback(splits_dir)


def download_stage2_data(raw_root: Path) -> RawDataPaths:
    """Download all Stage 2 data assets into data/raw."""
    raw_root.mkdir(parents=True, exist_ok=True)
    splits_dir = raw_root / "practice_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    dataset_tsv = _ensure_valid_file(DATASET_URL, raw_root / "dontpatronizeme_pcl.tsv")
    test_tsv = _ensure_valid_file(TEST_URL, raw_root / "task4_test.tsv")
    split_files = _download_split_files(splits_dir)

    return RawDataPaths(
        dataset_tsv=dataset_tsv,
        test_tsv=test_tsv,
        splits_dir=splits_dir,
        split_files=split_files,
    )
