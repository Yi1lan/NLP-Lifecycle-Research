"""Single-run training entrypoint for Stage 4 (Exercise 4)."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .augment import apply_lexical_dropout, load_trigger_tokens
from .config import (
    THRESHOLD_END,
    THRESHOLD_START,
    THRESHOLD_STEP,
    TRAIN_SPEC,
    get_model_spec,
    parse_bool,
)
from .data import dev_order_hash, load_stage3_dataset, test_order_hash
from .losses import (
    WeightedFocalLoss,
    binary_metrics_from_probs,
    compute_balanced_class_weights,
    search_best_threshold,
)


class EncodedTextDataset(Dataset):
    """Pre-tokenized dataset for Hugging Face Trainer."""

    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int] | None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        key = "input_ids"
        return len(self.encodings[key])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        output = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][idx], dtype=torch.long
            ),
        }
        if self.labels is not None:
            output["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return output


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _softmax_positive(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    return probs[:, 1]


def _load_transformers() -> tuple:
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    return AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def _build_compute_metrics(
    start: float,
    end: float,
    step: float,
) -> Callable:
    def _compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        probs = _softmax_positive(np.asarray(logits, dtype=np.float64))
        result = search_best_threshold(
            probs=probs,
            labels=np.asarray(labels, dtype=np.int64),
            start=start,
            end=end,
            step=step,
        )
        return {
            "f1": float(result.f1),
            "precision": float(result.precision),
            "recall": float(result.recall),
            "threshold": float(result.threshold),
        }

    return _compute_metrics


def _make_focal_trainer_class(base_trainer_cls):
    class FocalTrainer(base_trainer_cls):
        def __init__(self, focal_loss: WeightedFocalLoss, **kwargs):
            super().__init__(**kwargs)
            self.focal_loss = focal_loss

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.float()
            loss = self.focal_loss(logits, labels)
            return (loss, outputs) if return_outputs else loss

    return FocalTrainer


def smoke_train_step(seed: int = 123) -> float:
    """Tiny deterministic smoke step used by tests."""

    _seed_everything(seed)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(8, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0], dtype=torch.long)
    class_weights = compute_balanced_class_weights(labels.tolist())
    loss_fn = WeightedFocalLoss(class_weights=class_weights, gamma=2.0)

    logits = model(x)
    loss = loss_fn(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Stage 3 run.")
    parser.add_argument("--model", required=True, choices=["roberta", "deberta"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-len", required=True, type=int)
    parser.add_argument("--loss", required=True, choices=["ce", "focal"])
    parser.add_argument("--lex-drop", required=True, type=str)
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)

    parser.add_argument("--epochs", type=int, default=TRAIN_SPEC.epochs)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=TRAIN_SPEC.warmup_ratio)
    parser.add_argument("--weight-decay", type=float, default=TRAIN_SPEC.weight_decay)
    parser.add_argument("--focal-gamma", type=float, default=TRAIN_SPEC.focal_gamma)
    parser.add_argument("--lex-drop-prob", type=float, default=0.2)
    parser.add_argument("--lexical-csv", type=str, default="outputs/stage2/tables/lexical_analysis.csv")
    parser.add_argument("--lex-drop-positive-only", type=str, default="true")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lex_drop_enabled = parse_bool(args.lex_drop)
    lex_drop_positive_only = parse_bool(args.lex_drop_positive_only)

    model_spec = get_model_spec(args.model)
    learning_rate = args.lr if args.lr is not None else model_spec.lr
    batch_size = args.batch_size if args.batch_size is not None else model_spec.batch_size
    run_dir = Path(args.out_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    model_dir = run_dir / "model"

    _seed_everything(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    data_bundle = load_stage3_dataset(args.data_dir)
    train_examples = data_bundle.train
    dev_examples = data_bundle.dev
    if args.max_train_samples is not None:
        train_examples = train_examples[: args.max_train_samples]
    if args.max_dev_samples is not None:
        dev_examples = dev_examples[: args.max_dev_samples]

    train_texts = [row.text for row in train_examples]
    train_labels = [row.label for row in train_examples]
    dev_texts = [row.text for row in dev_examples]
    dev_labels = [row.label for row in dev_examples]

    trigger_tokens: list[str] = []
    if lex_drop_enabled:
        trigger_tokens = load_trigger_tokens(args.lexical_csv, top_k=20)
        train_texts = apply_lexical_dropout(
            texts=train_texts,
            labels=train_labels,
            trigger_tokens=trigger_tokens,
            drop_prob=args.lex_drop_prob,
            positive_only=lex_drop_positive_only,
            seed=args.seed,
        )

    AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments = (
        _load_transformers()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.name,
        use_fast=not model_spec.use_slow_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_spec.name, num_labels=2)

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        max_length=args.max_len,
        padding="max_length",
        return_tensors=None,
    )
    dev_encodings = tokenizer(
        dev_texts,
        truncation=True,
        max_length=args.max_len,
        padding="max_length",
        return_tensors=None,
    )
    train_dataset = EncodedTextDataset(train_encodings, train_labels)
    dev_dataset = EncodedTextDataset(dev_encodings, dev_labels)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        bf16=torch.cuda.is_available() and (not args.no_cuda),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        max_grad_norm=1.0,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    compute_metrics = _build_compute_metrics(
        start=THRESHOLD_START,
        end=THRESHOLD_END,
        step=THRESHOLD_STEP,
    )
    class_weights = compute_balanced_class_weights(train_labels)
    if args.loss == "focal":
        focal_trainer_cls = _make_focal_trainer_class(Trainer)
        trainer = focal_trainer_cls(
            focal_loss=WeightedFocalLoss(class_weights=class_weights, gamma=args.focal_gamma),
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
        )

    trainer.train()
    dev_predictions = trainer.predict(dev_dataset)
    dev_probs = _softmax_positive(np.asarray(dev_predictions.predictions, dtype=np.float64))
    best_metrics = search_best_threshold(
        probs=dev_probs,
        labels=np.asarray(dev_labels, dtype=np.int64),
        start=THRESHOLD_START,
        end=THRESHOLD_END,
        step=THRESHOLD_STEP,
    )
    dev_metric_details = binary_metrics_from_probs(
        probs=dev_probs,
        labels=np.asarray(dev_labels, dtype=np.int64),
        threshold=best_metrics.threshold,
    )

    # Save a stable, single model directory for the prediction script.
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    np.save(run_dir / "dev_probs.npy", dev_probs)
    np.save(run_dir / "dev_labels.npy", np.asarray(dev_labels, dtype=np.int64))
    np.save(run_dir / "dev_preds.npy", (dev_probs >= best_metrics.threshold).astype(np.int64))
    with (run_dir / "dev_preds.txt").open("w", encoding="utf-8") as handle:
        for value in (dev_probs >= best_metrics.threshold).astype(np.int64):
            handle.write(f"{int(value)}\n")

    summary = {
        "model_key": args.model,
        "model_name": model_spec.name,
        "seed": args.seed,
        "max_len": args.max_len,
        "loss": args.loss,
        "lex_drop": lex_drop_enabled,
        "lex_drop_positive_only": lex_drop_positive_only,
        "lex_drop_prob": args.lex_drop_prob,
        "trigger_tokens": trigger_tokens,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "focal_gamma": args.focal_gamma,
        "class_weights": class_weights.tolist(),
        "train_size": len(train_examples),
        "dev_size": len(dev_examples),
        "test_size": len(data_bundle.test),
        "train_missing_ids": data_bundle.train_missing_ids,
        "dev_missing_ids": data_bundle.dev_missing_ids,
        "dev_order_hash": dev_order_hash(data_bundle),
        "test_order_hash": test_order_hash(data_bundle),
        "dev_metrics": dev_metric_details.to_dict(),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "paths": {
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "model_dir": str(model_dir),
            "dev_probs_npy": str(run_dir / "dev_probs.npy"),
            "dev_labels_npy": str(run_dir / "dev_labels.npy"),
            "dev_preds_npy": str(run_dir / "dev_preds.npy"),
            "dev_preds_txt": str(run_dir / "dev_preds.txt"),
        },
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "model": args.model,
                "seed": args.seed,
                "dev_f1": round(dev_metric_details.f1, 6),
                "dev_precision": round(dev_metric_details.precision, 6),
                "dev_recall": round(dev_metric_details.recall, 6),
                "threshold": round(dev_metric_details.threshold, 6),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

