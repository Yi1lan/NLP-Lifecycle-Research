"""Configuration objects and constants for the Stage 3-5 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """Hyperparameters for one backbone model."""

    key: str
    name: str
    lr: float
    batch_size: int
    max_len: int
    use_slow_tokenizer: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainSpec:
    """Shared default training settings."""

    epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    focal_gamma: float = 2.0
    seeds: Tuple[int, ...] = (42, 2024)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        return payload


MODELS: Dict[str, ModelSpec] = {
    "roberta": ModelSpec(
        key="roberta",
        name="roberta-base",
        lr=2e-5,
        batch_size=32,
        max_len=256,
        use_slow_tokenizer=False,
    ),
    "deberta": ModelSpec(
        key="deberta",
        name="microsoft/deberta-v3-base",
        lr=1.8e-5,
        batch_size=24,
        max_len=256,
        # DeBERTa-v3 tokenization is typically safer with the slow tokenizer.
        use_slow_tokenizer=True,
    ),
}


TRAIN_SPEC = TrainSpec()
BASELINE_DEV_F1 = 0.48
BASELINE_TEST_F1 = 0.49
THRESHOLD_START = 0.05
THRESHOLD_END = 0.95
THRESHOLD_STEP = 0.005
DEFAULT_MIN_DEV_F1 = 0.40
DEFAULT_MIN_DEV_PROB_STD = 0.01
DEFAULT_MIN_POSITIVE_RATE = 0.02
DEFAULT_MAX_POSITIVE_RATE = 0.60


def parse_bool(value: str | bool) -> bool:
    """Parse common string booleans for CLI arguments."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def get_model_spec(model_key: str) -> ModelSpec:
    """Get one model spec by key and raise a clear error if unknown."""

    if model_key not in MODELS:
        supported = ", ".join(sorted(MODELS))
        raise KeyError(f"Unsupported model '{model_key}'. Supported: {supported}")
    return MODELS[model_key]
