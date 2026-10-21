"""Generate standalone synthetic anomaly nodes from training-only checkpoints.

Run with ``python -m dasna.augmentation.generate --dataset amazon --ratio 10``.
Every output node has label 1 and belongs to the training split. Ratios use only
the number of anomalies in the original TRAINING mask. Original nodes and edges
are not merged into the output; downstream alignment consumes this separate graph.

Modified for DASNA: anomaly-only export, shared loaders, portable state checkpoints
and cumulative batch edge offsets. The old merged-graph export is retired.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import torch

from dasna.augmentation.common import (
    DATASET_FILES, DEFAULT_DATASETS, add_common_arguments, extract_train_subgraph,
    load_mat_with_masks, load_models, resolve_device, seed_everything,
)
from dasna.augmentation.diffusion import sample_dm


def select_adaptive_threshold(ae, train_sub, device, n_sample=512,
                              thresholds=None, min_density=1e-4, max_density=0.95):
    """Select a threshold by balanced edge classification on training nodes.

    This retains the inherited thresholded ROC AUC criterion (equivalent to
    balanced accuracy for binary predictions), using the same label-conditioned
    latent representation as the decoder. No held-out labels or features enter.
    """
    thresholds = thresholds or [round(t, 2) for t in np.arange(0.05, 0.95, 0.05)]
    ae.eval()
    n = min(n_sample, train_sub.num_nodes)
    if n < 2:
        return ae.threshold, {}
    idx = torch.randperm(train_sub.num_nodes)[:n]
    sub = train_sub.subgraph(idx)
    edges = sub.edge_index.to(device)
    labels = sub.y.float().unsqueeze(1).to(device)
    with torch.no_grad():
        mean, _ = ae.encode(sub.x.to(device), edges, labels)
        latent = mean + ae.map_label_d(labels)
        pairs = torch.cat((latent[:, None, :].expand(-1, n, -1),
                           latent[None, :, :].expand(n, -1, -1)), dim=2)
        scores = torch.sigmoid(ae.dec_stru(pairs)).squeeze(-1).cpu().numpy()
    truth = np.zeros((n, n), dtype=bool)
    if edges.numel():
        source, target = edges.cpu().numpy()
        truth[source, target] = True
        truth[target, source] = True
    upper = np.triu_indices(n, k=1)
    target = truth[upper]
    if np.unique(target).size < 2:
        return ae.threshold, {}
    best_threshold, best_score = ae.threshold, -1.0
    for threshold in thresholds:
        predictions = scores[upper] > threshold
        density = float(predictions.mean())
        if min_density <= density <= max_density:
            score = float(roc_auc_score(target, predictions))
            if score > best_score:
                best_threshold, best_score = threshold, score
    return best_threshold, {"balanced_accuracy": best_score,
                            "score_auc": float(roc_auc_score(target, scores[upper]))}


def check_output_paths(name, args):
    paths = [args.output_dir / f"{name}_{ratio}pct_standalone.mat"
             for ratio in ((args.ratio,) if args.ratio else (5, 10, 25))]
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists: {existing[0]}. Use another --output-dir or explicit --overwrite.")
    return paths


def export_standalone(path, features, edge_index, ratio, n_training_anomalies, threshold, seed):
    """Write the sparse MAT schema expected by downstream alignment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(features)
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(2, -1)
    keep = edge_index[0] != edge_index[1]
    rows, cols = edge_index[:, keep]
    adjacency = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    if adjacency.nnz:
        adjacency.data[:] = 1
    scipy.io.savemat(path, {
        "homo": adjacency,
        "features": np.asarray(features, dtype=np.float32),
        "label": np.ones((1, n), dtype=np.int64),
        "train_mask": np.ones((1, n), dtype=bool),
        "val_mask": np.zeros((1, n), dtype=bool),
        "test_mask": np.zeros((1, n), dtype=bool),
        "gen_nodes": np.array([[n]], dtype=np.int64),
        "gen_ratio_percent": np.array([[ratio]], dtype=np.int64),
        "source_training_anomalies": np.array([[n_training_anomalies]], dtype=np.int64),
        "generation_threshold": np.array([[threshold]], dtype=np.float32),
        "generation_seed": np.array([[seed]], dtype=np.int64),
        "generation_mode": "standalone_anomalies",
    }, do_compression=True)
    return path


def generate_dataset(data, name, args, device):
    paths = check_output_paths(name, args)
    ae, dm, metadata = load_models(args.checkpoint_dir, name, data, device)
    seed_everything(args.seed)
    n_training_anomalies = int((data.y[data.train_mask] == 1).sum())
    if n_training_anomalies < 1:
        raise ValueError("Training mask contains no anomalies to use as a generation reference.")
    sub = extract_train_subgraph(data)
    sub.x = (sub.x - metadata["mean"]) / metadata["std"].clamp_min(1e-8)
    ae.threshold = args.threshold
    if not args.fixed_threshold:
        ae.threshold, diagnostics = select_adaptive_threshold(ae, sub, device)
        print(f"[{name}] adaptive threshold={ae.threshold:.2f}; diagnostics={diagnostics}")
    ratios = (args.ratio,) if args.ratio else (5, 10, 25)
    for ratio, path in zip(ratios, paths):
        seed_everything(args.seed)
        n_nodes = max(1, int(round(n_training_anomalies * ratio / 100.0)))
        features, batches = [], []
        offset = 0
        with torch.no_grad():
            while offset < n_nodes:
                size = min(args.batch_size, n_nodes - offset)
                noise = torch.randn(size, metadata["hid_dim"], device=device)
                labels = torch.ones((size, 1), device=device)
                latent = sample_dm(dm.denoise_fn_D, noise, labels, args.sample_steps)
                x, edges, _, _ = ae.sample(latent, labels)
                features.append(x.cpu())
                batches.append(edges.cpu() + offset)
                offset += size
        x = torch.cat(features).numpy() * metadata["std"].numpy() + metadata["mean"].numpy()
        lower = (metadata["mean"] - 3 * metadata["std"]).numpy()
        upper = (metadata["mean"] + 3 * metadata["std"]).numpy()
        x = np.clip(x, lower, upper)
        if not np.isfinite(x).all():
            raise RuntimeError("Generated features are not finite; check the trained checkpoints.")
        edges = torch.cat(batches, dim=1).numpy()
        export_standalone(path, x, edges, ratio, n_training_anomalies, ae.threshold, args.seed)
        print(f"[{name}] {ratio}% of {n_training_anomalies} training anomalies -> {n_nodes} synthetic anomalies: {path}")
    return paths


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sample_steps < 2:
        parser.error("--sample-steps must be at least 2")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must lie strictly between 0 and 1")
    seed_everything(args.seed)
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    print(f"Device: {device}")
    completed = 0
    for name in (args.dataset,) if args.dataset else DEFAULT_DATASETS:
        path = args.data_dir / DATASET_FILES[name]
        if not path.is_file():
            if args.dataset:
                parser.error(f"Data file not found: {path}")
            print(f"[{name}] SKIP: data file not found: {path}")
            continue
        try:
            data = load_mat_with_masks(path)
            generate_dataset(data, name, args, device)
        except (ValueError, RuntimeError, FileNotFoundError, OSError) as error:
            parser.error(f"{name}: {error}")
        completed += 1
    if not completed:
        parser.error(f"No dataset files found in {args.data_dir}. See data/README.md for input preparation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
