"""Unit tests for Stage 3-5 utility logic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.stage3.augment import apply_lexical_dropout
from src.stage3.data import load_stage3_dataset
from src.stage3.losses import search_best_threshold
from src.stage3.submission import validate_prediction_file, write_prediction_file
from src.stage3.train import smoke_train_step


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class Stage3PipelineTests(unittest.TestCase):
    def test_data_loader_preserves_official_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(
                root / "dontpatronizeme_pcl.tsv",
                (
                    "meta1\nmeta2\nmeta3\nmeta4\n"
                    "1\ta\twomen\tuk\tfirst text\t0\n"
                    "2\ta\thomeless\tuk\tsecond text\t3\n"
                    "3\ta\tmigrant\tus\tthird text\t1\n"
                    "4\ta\tpoor-families\tus\tfourth text\t4\n"
                ),
            )
            _write(
                root / "train_semeval_parids-labels.csv",
                "par_id,label\n2,1\n1,0\n",
            )
            _write(
                root / "dev_semeval_parids-labels.csv",
                "par_id,label\n3,0\n4,1\n",
            )
            _write(
                root / "task4_test.tsv",
                "100\tx\tx\tx\ttest one\n101\tx\tx\tx\ttest two\n",
            )

            data = load_stage3_dataset(root)
            self.assertEqual([row.par_id for row in data.train], ["2", "1"])
            self.assertEqual([row.par_id for row in data.dev], ["3", "4"])
            self.assertEqual([row.sample_id for row in data.test], ["100", "101"])

    def test_submission_validator_catches_count_and_label_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "preds.txt"
            file_path.write_text("0\n2\n", encoding="utf-8")
            errors = validate_prediction_file(file_path, expected_count=3, label_name="dev")
            self.assertTrue(any("expected 3 lines" in msg for msg in errors))
            self.assertTrue(any("invalid label" in msg for msg in errors))

    def test_threshold_search_is_deterministic(self) -> None:
        probs = np.asarray([0.10, 0.40, 0.60, 0.90], dtype=np.float64)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        result = search_best_threshold(probs, labels)
        self.assertAlmostEqual(result.f1, 1.0, places=6)
        self.assertGreaterEqual(result.threshold, 0.4)
        self.assertLessEqual(result.threshold, 0.405)

    def test_lexical_dropout_only_removes_trigger_tokens(self) -> None:
        texts = ["compassion anti compassion", "compassion untouched negative"]
        labels = [1, 0]
        outputs = apply_lexical_dropout(
            texts=texts,
            labels=labels,
            trigger_tokens=["compassion"],
            drop_prob=1.0,
            positive_only=True,
            seed=7,
        )
        self.assertEqual(outputs[0], "anti")
        self.assertEqual(outputs[1], texts[1])

    def test_smoke_train_step_runs(self) -> None:
        loss = smoke_train_step(seed=13)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_augmentation_reproducibility_with_seed(self) -> None:
        texts = ["darkness and compassion", "just text", "hungry hearts"]
        labels = [1, 1, 1]
        first = apply_lexical_dropout(
            texts,
            labels,
            trigger_tokens=["darkness", "compassion", "hungry"],
            drop_prob=0.5,
            seed=42,
        )
        second = apply_lexical_dropout(
            texts,
            labels,
            trigger_tokens=["darkness", "compassion", "hungry"],
            drop_prob=0.5,
            seed=42,
        )
        self.assertEqual(first, second)

    def test_prediction_write_uses_one_label_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dev.txt"
            write_prediction_file(np.asarray([1, 0, 1], dtype=np.int64), output)
            content = output.read_text(encoding="utf-8")
            self.assertEqual(content, "1\n0\n1\n")


if __name__ == "__main__":
    unittest.main()
