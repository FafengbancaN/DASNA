"""CPU integration checks for augmentation; run in the augmentation Conda environment.

Run ``python -m unittest discover -s tests -p test_augmentation_pipeline.py``.
All fixture data, model checkpoints and generated graphs live in temporary
directories. No project dataset or cached augmentation is read or overwritten.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.io
import scipy.sparse as sp
import torch

from dasna.augmentation.common import (
    load_mat_with_masks, make_loader, training_metadata, training_signature,
)
from dasna.augmentation.generate import build_parser, export_standalone
from dasna.augmentation.models import VGAE


REPO_ROOT = Path(__file__).resolve().parents[1]


class AugmentationPipelineTests(unittest.TestCase):
    def run_cli(self, module, arguments, cwd):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, "-B", "-m", module, *arguments], cwd=cwd,
            env=environment, text=True, encoding="utf-8", errors="replace",
            capture_output=True, timeout=120,
        )

    def assert_cli_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cross_process_checkpoints_and_anomaly_only_generation(self):
        with tempfile.TemporaryDirectory(prefix="dasna_augmentation_test_") as directory:
            temporary = Path(directory)
            raw = temporary / "raw"
            raw.mkdir()
            n = 64
            rng = np.random.default_rng(5)
            features = rng.normal(size=(n, 4)).astype(np.float32)
            adjacency = np.zeros((n, n), dtype=np.float32)
            for index in range(n - 1):
                adjacency[index, index + 1] = adjacency[index + 1, index] = 1
            labels = np.zeros(n, dtype=np.int64)
            labels[:20] = labels[48:] = 1  # 20 training anomalies, 36 total.
            indexes = np.arange(n)
            fixture = raw / "Amazon.mat"
            scipy.io.savemat(fixture, {
                "homo": adjacency, "features": features, "label": labels,
                "train_mask": indexes < 48,
                "val_mask": (indexes >= 48) & (indexes < 56),
                "test_mask": indexes >= 56,
            })
            before = hashlib.sha256(fixture.read_bytes()).hexdigest()
            checkpoints = temporary / "checkpoints"
            generated = temporary / "generated"
            common = ["--dataset", "amazon", "--data-dir", str(raw),
                      "--checkpoint-dir", str(checkpoints), "--output-dir", str(generated),
                      "--device", "-1", "--sample-steps", "2"]
            trained = self.run_cli("dasna.augmentation.train", common + [
                "--vae-epochs", "1", "--diff-epochs", "1", "--batch-size", "128", "--train-only",
            ], temporary)
            self.assert_cli_ok(trained)
            ae = torch.load(checkpoints / "amazon_ae.pt", map_location="cpu", weights_only=True)
            dm = torch.load(checkpoints / "amazon_dm.pt", map_location="cpu", weights_only=True)
            self.assertIsInstance(ae, dict)
            self.assertEqual(ae["kind"], "ae")
            self.assertIn("state_dict", ae)
            self.assertEqual(ae["metadata"]["training_id"], dm["metadata"]["training_id"])
            self.assertEqual(ae["metadata"]["n_train_anomalies"], 20)
            np.testing.assert_allclose(ae["metadata"]["mean"].numpy(), features[:48].mean(0), atol=1e-6)
            np.testing.assert_allclose(ae["metadata"]["std"].numpy(), features[:48].std(0), atol=1e-6)
            # A tiny threshold makes every within-batch pair an edge, so offset
            # assertions below cannot pass merely because the output is edgeless.
            generation_arguments = common + ["--batch-size", "2", "--fixed-threshold", "--threshold", "0.000001"]
            result = self.run_cli("dasna.augmentation.generate", generation_arguments, temporary)
            self.assert_cli_ok(result)
            for ratio, expected_nodes in ((5, 1), (10, 2), (25, 5)):
                output = scipy.io.loadmat(generated / f"amazon_{ratio}pct_standalone.mat")
                self.assertEqual(output["features"].shape, (expected_nodes, 4))
                self.assertTrue(np.isfinite(output["features"]).all())
                self.assertTrue((output["label"] == 1).all())
                self.assertTrue(output["train_mask"].all())
                self.assertFalse(output["val_mask"].any())
                self.assertFalse(output["test_mask"].any())
                self.assertEqual(output["source_training_anomalies"].item(), 20)
            largest = generated / "amazon_25pct_standalone.mat"
            output = scipy.io.loadmat(largest)
            source, target = output["homo"].nonzero()
            self.assertEqual(set(zip(source.tolist(), target.tolist())), {(0, 1), (1, 0), (2, 3), (3, 2)})
            before_output = hashlib.sha256(largest.read_bytes()).hexdigest()
            refused = self.run_cli("dasna.augmentation.generate", generation_arguments, temporary)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("Output exists", refused.stderr)
            self.assertEqual(hashlib.sha256(largest.read_bytes()).hexdigest(), before_output)
            self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), before)

    def test_sparse_loader_missing_masks_and_edgeless_partitions(self):
        with tempfile.TemporaryDirectory(prefix="dasna_loader_test_") as directory:
            temporary = Path(directory)
            n = 40
            path = temporary / "input.mat"
            scipy.io.savemat(path, {
                "Network": sp.diags([np.ones(n - 1)], [-1], shape=(n, n), format="csr"),
                "Attributes": np.ones((n, 4), dtype=np.float32),
                "Label": np.tile([0, 1], n // 2),
            })
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            data, again = load_mat_with_masks(path), load_mat_with_masks(path)
            self.assertTrue(torch.equal(data.train_mask, again.train_mask))
            self.assertEqual(int(data.train_mask.sum()), 24)
            self.assertEqual(tuple(data.edge_index.shape), (2, 78))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            sub, metadata = training_metadata(data, "amazon", SimpleNamespace(seed=42, threshold=0.75))
            self.assertTrue(torch.isfinite(sub.x).all())
            self.assertIn("training_id", metadata)
            held = data.clone()
            held.x[~held.train_mask] = 99
            held.y[~held.train_mask] = 1 - held.y[~held.train_mask]
            self.assertEqual(training_signature(data), training_signature(held))
            changed = data.clone()
            changed.x[changed.train_mask] += 1
            self.assertNotEqual(training_signature(data), training_signature(changed))
            # METIS makes some edgeless partitions for this sparse training graph.
            batches = list(make_loader(sub, 4))
            self.assertEqual(sum(batch.num_nodes for batch in batches), 24)
            self.assertTrue(any(batch.edge_index.numel() == 0 for batch in batches))
            model = VGAE(4, 2, temporal=False)
            features, edges, _, _ = model.sample(torch.zeros((1, 2)), torch.ones((1, 1)))
            export_standalone(temporary / "single.mat", features.detach().numpy(), edges.numpy(), 5, 1, 0.75, 42)
            single = scipy.io.loadmat(temporary / "single.mat")
            self.assertEqual(single["homo"].shape, (1, 1))
            self.assertEqual(single["homo"].nnz, 0)
            self.assertEqual(build_parser().parse_args(["--dataset", "musae_facebook"]).dataset, "facebook")


if __name__ == "__main__":
    unittest.main()
