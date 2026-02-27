"""Unit tests for Stage 3-5 utility logic."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.stage3.augment import apply_lexical_dropout
from src.stage3.data import load_stage3_dataset
from src.stage3.ensemble import _is_run_eligible, _run_health
from src.stage3.losses import search_best_threshold
from src.stage3.submission import validate_prediction_file, write_prediction_file
from src.stage3.train import smoke_train_step


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class Stage3PipelineTests(unittest.TestCase):
    def test_degeneracy_detection_flags_constant_probs(self) -> None:
        dev_labels = np.asarray([0, 1, 0, 1, 0, 0, 1, 0], dtype=np.int64)
        constant_probs = np.asarray([0.69] * len(dev_labels), dtype=np.float64)
        health = _run_health(
            constant_probs,
            dev_labels,
            threshold_start=0.05,
            threshold_end=0.95,
            threshold_step=0.005,
        )
        eligible, reasons = _is_run_eligible(
            health,
            min_dev_f1=0.4,
            min_dev_prob_std=0.01,
            min_positive_rate=0.02,
            max_positive_rate=0.6,
        )
        self.assertFalse(eligible)
        self.assertTrue(any("dev_prob_std" in reason for reason in reasons))

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

    def test_ensemble_filters_degenerate_and_fallbacks_to_good_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dev_labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 0], dtype=np.int64)

            good_dev = np.asarray([0.05, 0.12, 0.81, 0.75, 0.11, 0.69, 0.08, 0.22])
            bad_dev = np.asarray([0.69] * len(dev_labels))
            good_test = np.asarray([0.1, 0.8, 0.2, 0.9], dtype=np.float64)
            bad_test = np.asarray([0.69] * len(good_test), dtype=np.float64)

            good_dev_path = root / "roberta_seed42_dev.npy"
            bad_dev_path = root / "deberta_seed42_dev.npy"
            good_test_path = root / "roberta_seed42_test.npy"
            bad_test_path = root / "deberta_seed42_test.npy"
            labels_path = root / "dev_labels.npy"
            out_dir = root / "ensemble"

            np.save(good_dev_path, good_dev)
            np.save(bad_dev_path, bad_dev)
            np.save(good_test_path, good_test)
            np.save(bad_test_path, bad_test)
            np.save(labels_path, dev_labels)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.stage3.ensemble",
                    "--dev-probs",
                    f"{good_dev_path},{bad_dev_path}",
                    "--test-probs",
                    f"{good_test_path},{bad_test_path}",
                    "--dev-labels",
                    str(labels_path),
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
            )

            manifest = json.loads((out_dir / "selected_runs.json").read_text(encoding="utf-8"))
            self.assertIn("roberta_seed42", manifest["selected_run_ids"])
            self.assertIn("deberta_seed42", manifest["excluded_run_ids"])

            dev_file = out_dir / "dev.txt"
            test_file = out_dir / "test.txt"
            self.assertTrue(dev_file.exists())
            self.assertTrue(test_file.exists())
            self.assertTrue(all(line in {"0", "1"} for line in dev_file.read_text().splitlines()))
            self.assertTrue(all(line in {"0", "1"} for line in test_file.read_text().splitlines()))

    def test_spec_artifacts_present(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        required = [
            repo_root / "BestModel" / "README.md",
            repo_root / "BestModel" / "train.py",
            repo_root / "BestModel" / "predict.py",
            repo_root / "BestModel" / "ensemble.py",
            repo_root / "dev.txt",
            repo_root / "test.txt",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing required artifact: {path}")


if __name__ == "__main__":
    unittest.main()
