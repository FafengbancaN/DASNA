"""Tiny, temporary regression fixtures; this does not validate paper results.

Run from the repository root: python -m unittest discover -s tests -p test_detection_pipeline.py -v
"""

from contextlib import redirect_stdout
import hashlib
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
from scipy.io import savemat
import torch
import yaml

from dasna.preprocessing import preprocess_dataset
from dasna.detection.model import DASNALayer
from dasna.detection.model_handler import ModelHandler
from dasna.detection.train import get_config, grid, REPOSITORY_ROOT, set_random_seed
from dasna.detection.utils import load_data, merge_standalone_train


class DetectionPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix='dasna_detection_test_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw, self.processed, self.generated = [self.root / name for name in ('raw', 'processed', 'generated')]
        for directory in (self.raw, self.processed, self.generated):
            directory.mkdir()
        n = 12
        self.features = np.arange(n * 4, dtype=np.float32).reshape(n, 4) / 100 + 0.1
        self.labels = np.array([0, 1] * 6, dtype=np.int64)
        self.base = dict(
            features=self.features, label=self.labels,
            homo=sp.csr_matrix(np.ones((n, n), dtype=np.float32)),
            net_upu=sp.eye(n, format='csr'),
            net_usu=sp.csr_matrix(np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)),
            net_uvu=sp.csr_matrix(np.eye(n) + np.eye(n, k=2) + np.eye(n, k=-2)),
            train_mask=np.isin(np.arange(n), range(6)),
            val_mask=np.isin(np.arange(n), range(6, 9)),
            test_mask=np.isin(np.arange(n), range(9, 12)),
        )
        self.source = self.raw / 'Amazon.mat'
        savemat(self.source, self.base)
        with redirect_stdout(io.StringIO()):
            preprocess_dataset('amazon', self.raw, self.processed)
        self.extra_features = np.arange(16, dtype=np.float32).reshape(4, 4) / 50
        savemat(self.generated / 'amazon_10pct_standalone.mat', dict(
            features=self.extra_features,
            label=np.array([0, 1, 1, 0]),
            homo=sp.csr_matrix(([1, 1], ([1, 3], [3, 1])), shape=(4, 4)),
            train_mask=np.array([0, 1, 0, 1]),
        ))

    def configuration(self, **overrides):
        values = dict(
            data_name='amazon', raw_dir=str(self.raw), processed_dir=str(self.processed),
            generated_dir=str(self.generated), save_dir=str(self.root / 'checkpoints'),
            multi_relation='GNN', choose_type='residual', no_cuda=True,
            use_dm=True, data_suffix='_10pct_dm', batch_size=3,
            emb_size=4, num_epochs=1, valid_epochs=1,
        )
        values.update(overrides)
        path = self.root / 'fixture.yaml'
        path.write_text(yaml.safe_dump(values), encoding='utf-8')
        return get_config(str(path))

    def test_preprocessing_preserves_distinct_relations_and_source(self):
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        with redirect_stdout(io.StringIO()):
            outputs = preprocess_dataset('amazon', self.raw, self.processed)
        self.assertEqual(before, hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(len(outputs), 4)
        graphs, features, labels = load_data('amazon', self.raw, processed_dir=self.processed)
        self.assertEqual(graphs[1][0], {0})
        self.assertEqual(graphs[2][0], {0, 1})
        self.assertEqual(graphs[3][0], {0, 2})
        np.testing.assert_array_equal(features, self.features)
        np.testing.assert_array_equal(labels, self.labels)

    def test_generated_mask_subsets_features_labels_and_both_graph_axes(self):
        features, labels, graph = merge_standalone_train(self.generated, 'amazon', '_10pct_dm')
        np.testing.assert_array_equal(features, self.extra_features[[1, 3]])
        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(graph, {0: {0, 1}, 1: {0, 1}})
        with self.assertRaises(FileNotFoundError):
            merge_standalone_train(self.generated, 'amazon', '_25pct_dm')

    def test_augmented_training_exact_batch_boundary_and_checkpoint_reload(self):
        set_random_seed(72)
        with redirect_stdout(io.StringIO()):
            handler = ModelHandler(self.configuration())
            self.assertEqual(len(handler.dataset['idx_train']), 8)
            self.assertEqual(handler.dataset['idx_valid'], [6, 7, 8])
            self.assertEqual(handler.dataset['idx_test'], [9, 10, 11])
            self.assertEqual(handler.dataset['y_train'][-2:].tolist(), [1, 0])
            metrics = handler.train()
        self.assertEqual(len(metrics), 6)
        self.assertTrue(np.isfinite(metrics).all())
        self.assertTrue((self.root / 'checkpoints' / 'fixture.ckpt').is_file())
        # Validation/test size 3 divides batch_size 3 exactly; no empty batch is emitted.

    def test_base_training_singletons_eval_mode_and_no_grad(self):
        # No standalone MAT is available at this path: use_dm=False must avoid loading it.
        absent_generated = self.root / 'absent_generated'
        configuration = self.configuration(use_dm=False, generated_dir=str(absent_generated), batch_size=1)
        observations = []
        original = DASNALayer.to_prob

        def checked(model, nodes, labels, train_flag=True):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(train_flag)
            self.assertEqual(len(nodes), 1)
            observations.append(model)
            return original(model, nodes, labels, train_flag)

        set_random_seed(72)
        with patch.object(DASNALayer, 'to_prob', checked), redirect_stdout(io.StringIO()):
            handler = ModelHandler(configuration)
            self.assertEqual(len(handler.dataset['idx_train']), 6)
            metrics = handler.train()
        self.assertEqual(len(observations), 6)
        self.assertTrue(observations[-1].training)  # Evaluation restores the previous mode.
        self.assertTrue(np.isfinite(metrics).all())
        self.assertFalse(absent_generated.exists())

    def test_cpu_mamba_request_is_explicitly_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            ModelHandler(self.configuration(multi_relation='Mamba', no_cuda=True))

    def test_configs_resolve_paths_from_repository_root(self):
        original_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            for name in ('amazon', 'yelpchi', 'weibo', 'questions', 'elliptic', 'facebook'):
                configuration = get_config(f'configs/{name}.yaml')
                self.assertEqual(configuration['data_name'], name)
                self.assertEqual(configuration['model'], 'DASNA')
                self.assertEqual(configuration['multi_relation'], 'Mamba')
                self.assertEqual(Path(configuration['raw_dir']), REPOSITORY_ROOT / 'data' / 'raw')
        finally:
            os.chdir(original_cwd)
        self.assertEqual(len(list(grid({'a': 1}))), 1)
        self.assertEqual(len(list(grid({'a': [1, 2], 'b': [3, 4]}))), 4)


if __name__ == '__main__':
    unittest.main()
