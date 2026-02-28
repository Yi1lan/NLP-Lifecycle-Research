"""Lexical trigger utilities and augmentation for shortcut-robust training."""

from __future__ import annotations

import csv
import random
import string
from pathlib import Path
from typing import Iterable


DEFAULT_TRIGGER_TOKENS = [
    "darkness",
    "compassion",
    "hungry",
    "christmas",
    "donate",
    "destitute",
    "blessings",
    "christ",
    "hearts",
    "fortunate",
    "dole",
    "teresa",
]


def _normalize_token(token: str) -> str:
    return token.strip().strip(string.punctuation).lower()


def load_trigger_tokens(
    lexical_csv_path: str | Path | None,
    top_k: int = 20,
) -> list[str]:
    """Load top PCL-indicative tokens from Stage 2 lexical analysis output."""

    if lexical_csv_path is None:
        return DEFAULT_TRIGGER_TOKENS[:]
    path = Path(lexical_csv_path)
    if not path.exists():
        return DEFAULT_TRIGGER_TOKENS[:]

    loaded: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("label_name", "")).strip() != "PCL":
                continue
            token = _normalize_token(str(row.get("token", "")))
            if not token:
                continue
            loaded.append(token)
            if len(loaded) >= top_k:
                break
    if not loaded:
        return DEFAULT_TRIGGER_TOKENS[:]
    return loaded


def lexical_trigger_dropout(
    text: str,
    trigger_tokens: Iterable[str],
    drop_prob: float,
    rng: random.Random,
) -> str:
    """Drop trigger tokens from a single text with probability `drop_prob`."""

    trigger_set = {token.lower() for token in trigger_tokens}
    words = text.split()
    kept: list[str] = []
    dropped = 0

    for word in words:
        normalized = _normalize_token(word)
        if normalized in trigger_set and rng.random() < drop_prob:
            dropped += 1
            continue
        kept.append(word)

    if dropped == 0 or not kept:
        return text
    return " ".join(kept)


def apply_lexical_dropout(
    texts: list[str],
    labels: list[int],
    trigger_tokens: Iterable[str],
    drop_prob: float = 0.2,
    positive_only: bool = True,
    seed: int = 42,
) -> list[str]:
    """Apply lexical dropout over training texts for robustness."""

    rng = random.Random(seed)
    output: list[str] = []
    for text, label in zip(texts, labels):
        if positive_only and int(label) != 1:
            output.append(text)
            continue
        augmented = lexical_trigger_dropout(
            text=text,
            trigger_tokens=trigger_tokens,
            drop_prob=drop_prob,
            rng=rng,
        )
        output.append(augmented)
    return output


def contains_trigger_token(text: str, trigger_tokens: Iterable[str]) -> bool:
    """Whether a text contains at least one trigger token."""

    trigger_set = {token.lower() for token in trigger_tokens}
    for token in text.split():
        if _normalize_token(token) in trigger_set:
            return True
    return False

