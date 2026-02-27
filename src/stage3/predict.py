"""Inference entrypoint for Stage 4/5 probabilities."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import dev_order_hash, load_stage3_dataset, test_order_hash


class InferenceDataset(Dataset):
    """Simple tensor dataset for batched inference."""

    def __init__(self, encodings: dict[str, list[list[int]]]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][idx], dtype=torch.long
            ),
        }


def _softmax_positive(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    return probs[:, 1]


def _load_transformers() -> tuple:
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    return AutoModelForSequenceClassification, AutoTokenizer


def _resolve_checkpoint_path(checkpoint_arg: str | Path) -> Path:
    path = Path(checkpoint_arg).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if path.is_file():
        return path.parent

    # Run directory with run_summary.json.
    summary_path = path / "run_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        model_dir = payload.get("paths", {}).get("model_dir")
        if model_dir and Path(model_dir).exists():
            return Path(model_dir)

    # Direct model directory.
    if (path / "config.json").exists() and (
        (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    ):
        return path

    # Checkpoint subdirectory fallback.
    candidates = []
    candidates.extend(path.glob("checkpoint-*"))
    candidates.extend((path / "checkpoints").glob("checkpoint-*"))
    candidates = [candidate for candidate in candidates if candidate.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No usable checkpoint found under: {path}")
    latest = sorted(candidates, key=lambda item: int(item.name.split("-")[-1]))[-1]
    state_file = latest / "trainer_state.json"
    if state_file.exists():
        payload = json.loads(state_file.read_text(encoding="utf-8"))
        best_path = payload.get("best_model_checkpoint")
        if best_path and Path(best_path).exists():
            return Path(best_path)
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and save probabilities.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--out-probs", required=True, type=str)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-labels", type=str, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_probs = Path(args.out_probs).resolve()
    out_probs.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_stage3_dataset(args.data_dir)
    if args.split == "dev":
        rows = bundle.dev
        ids = [row.par_id for row in rows]
        labels = np.asarray([row.label for row in rows], dtype=np.int64)
        order_hash = dev_order_hash(bundle)
    else:
        rows = bundle.test
        ids = [row.sample_id for row in rows]
        labels = None
        order_hash = test_order_hash(bundle)
    texts = [row.text for row in rows]

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    summary_path = checkpoint_path.parent / "run_summary.json"
    max_len = args.max_len
    if max_len is None and summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        maybe_len = payload.get("max_len")
        if isinstance(maybe_len, int):
            max_len = maybe_len
    if max_len is None:
        max_len = 256

    AutoModelForSequenceClassification, AutoTokenizer = _load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=2)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and (not args.no_cuda)
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model.to(device)
    model.eval()

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors=None,
    )
    dataset = InferenceDataset(encodings)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            use_amp = device.type == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(**batch)
            probs = _softmax_positive(outputs.logits.detach().cpu().numpy())
            all_probs.append(probs)
    merged = np.concatenate(all_probs) if all_probs else np.asarray([], dtype=np.float64)
    np.save(out_probs, merged)

    if args.out_labels and labels is not None:
        out_labels = Path(args.out_labels).resolve()
        out_labels.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_labels, labels)

    metadata = {
        "split": args.split,
        "count": len(merged),
        "checkpoint_arg": str(args.checkpoint),
        "resolved_checkpoint": str(checkpoint_path),
        "max_len": max_len,
        "order_hash": order_hash,
        "first_ids": ids[:5],
        "last_ids": ids[-5:] if len(ids) >= 5 else ids,
    }
    sidecar_path = out_probs.with_suffix(".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "split": args.split,
                "out_probs": str(out_probs),
                "count": int(len(merged)),
                "order_hash": order_hash,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

