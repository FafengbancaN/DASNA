"""Shared data, training and checkpoint helpers for graph anomaly synthesis.

Modified for DASNA: portable paths, validated MAT input, state checkpoints and
train-only normalization. The VAE and diffusion objectives follow the inherited
implementation; provenance and license information is in THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import random
import time
import uuid

import numpy as np
import scipy.io
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, DataLoader
from torch_geometric.utils import negative_sampling, to_undirected

from dasna.augmentation.diffusion import MLPDiffusion, Model
from dasna.augmentation.models import VGAE


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILES = {
    "amazon": "Amazon.mat", "yelpchi": "YelpChi.mat", "weibo": "weibo.mat",
    "questions": "questions.mat", "pubmed": "pubmed.mat", "reddit": "Reddit.mat",
    "facebook": "facebook.mat", "elliptic": "elliptic.mat", "acm": "ACM.mat",
    "blogcatalog": "BlogCatalog.mat",
}
DEFAULT_DATASETS = ("amazon", "yelpchi", "weibo", "questions", "facebook", "elliptic")
CHECKPOINT_VERSION = 1


def canonical_dataset(value):
    value = value.lower()
    return "facebook" if value == "musae_facebook" else value


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def add_common_arguments(parser):
    parser.add_argument("--dataset", type=canonical_dataset,
                        choices=tuple(DATASET_FILES), default=None,
                        help="One dataset; musae_facebook is an alias for facebook.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=REPO_ROOT / "artifacts" / "checkpoints" / "augmentation")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "generated")
    parser.add_argument("--batch-size", "--batch_size", type=positive_int, default=512,
                        help="Training partition or generation batch size.")
    parser.add_argument("--sample-steps", "--sample_steps", type=positive_int, default=50)
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Fixed edge probability threshold when adaptive selection is disabled.")
    parser.add_argument("--fixed-threshold", action="store_true",
                        help="Use --threshold instead of selecting it on the training subgraph.")
    parser.add_argument("--ratio", type=int, choices=(5, 10, 25), default=None,
                        help="Percent of TRAINING anomalies to synthesize; default: 5, 10, 25.")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, or a CUDA device index.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing generated MAT files; otherwise refuse to overwrite.")


def resolve_device(index):
    if index == -1:
        return torch.device("cpu")
    if index < -1 or not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} is unavailable; use --device -1 for CPU.")
    capability = torch.cuda.get_device_capability(index)
    arch = f"sm_{capability[0]}{capability[1]}"
    if arch not in torch.cuda.get_arch_list():
        raise ValueError(f"This PyTorch build does not list {arch}; use --device -1 or a compatible build.")
    return torch.device(f"cuda:{index}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mat_with_masks(filepath):
    """Read common MAT schemas; missing masks get a deterministic 60/20/20 split.

    Input files are never rewritten. Unlabelled nodes (label -1) are excluded
    from automatically created masks. Only the training mask enters synthesis.
    """
    filepath = Path(filepath)
    mat = scipy.io.loadmat(filepath)
    def field(*keys):
        for key in keys:
            if key in mat:
                return mat[key]
        raise ValueError(f"{filepath.name}: expected one of {keys}")
    raw_features = field("Attributes", "features")
    features = raw_features.toarray() if sp.issparse(raw_features) else np.asarray(raw_features)
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(field("Label", "label")).reshape(-1)
    if features.ndim != 2 or len(labels) != len(features):
        raise ValueError(f"{filepath.name}: feature rows and label length must agree")
    if not np.isfinite(features).all() or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError(f"{filepath.name}: features must be finite; labels must be -1, 0, or 1")
    n = len(labels)
    adjacency = sp.coo_matrix(field("homo", "Network"))
    if adjacency.shape != (n, n):
        raise ValueError(f"{filepath.name}: adjacency must have shape ({n}, {n})")
    nonzero = np.asarray(adjacency.data != 0)
    rows, cols = adjacency.row[nonzero], adjacency.col[nonzero]
    keep = rows != cols
    edge_index = torch.tensor(np.stack((rows[keep], cols[keep])), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=n)
    names = ("train_mask", "val_mask", "test_mask")
    present = [name in mat for name in names]
    if any(present) and not all(present):
        raise ValueError(f"{filepath.name}: supply all three masks or none")
    if all(present):
        masks = []
        for name in names:
            values = np.asarray(mat[name]).reshape(-1)
            if len(values) != n or not np.isin(values, [0, 1]).all():
                raise ValueError(f"{filepath.name}: {name} must be a binary vector of length {n}")
            masks.append(torch.as_tensor(values.astype(bool)))
    else:
        labeled = np.flatnonzero(labels >= 0)
        try:
            train_idx, held_idx = train_test_split(
                labeled, test_size=0.4, random_state=42, stratify=labels[labeled])
            val_idx, test_idx = train_test_split(
                held_idx, test_size=0.5, random_state=42, stratify=labels[held_idx])
        except ValueError as error:
            raise ValueError(f"{filepath.name}: cannot stratify these labels; supply explicit masks") from error
        masks = []
        for indexes in (train_idx, val_idx, test_idx):
            mask = torch.zeros(n, dtype=torch.bool)
            mask[indexes] = True
            masks.append(mask)
    if torch.stack(masks).sum(0).max().item() > 1:
        raise ValueError(f"{filepath.name}: train, validation and test masks overlap")
    if not masks[0].any() or np.any(labels[masks[0].numpy()] < 0):
        raise ValueError(f"{filepath.name}: training mask must contain only labelled nodes")
    return Data(x=torch.from_numpy(features), y=torch.as_tensor(labels, dtype=torch.long),
                edge_index=edge_index, train_mask=masks[0], val_mask=masks[1], test_mask=masks[2])


def extract_train_subgraph(data):
    sub = data.subgraph(data.train_mask)
    sub.train_mask = torch.ones(sub.num_nodes, dtype=torch.bool)
    sub.val_mask = torch.zeros(sub.num_nodes, dtype=torch.bool)
    sub.test_mask = torch.zeros(sub.num_nodes, dtype=torch.bool)
    return sub


def training_signature(data):
    digest = hashlib.sha256()
    digest.update(data.train_mask.numpy().tobytes())
    digest.update(data.y[data.train_mask].numpy().tobytes())
    digest.update(data.x[data.train_mask].numpy().tobytes())
    src, dst = data.edge_index
    edges = data.edge_index[:, data.train_mask[src] & data.train_mask[dst]]
    digest.update(edges.numpy().tobytes())
    return digest.hexdigest()


def training_metadata(data, name, args):
    sub = extract_train_subgraph(data)
    if sub.num_node_features < 2:
        raise ValueError("At least two input features are required for latent diffusion.")
    hid_dim = min(2 ** max(0, int(math.log2(sub.num_node_features)) - 1), 512)
    mean, std = sub.x.mean(0), sub.x.std(0, unbiased=False)
    metadata = {
        "version": CHECKPOINT_VERSION, "dataset": name, "training_id": uuid.uuid4().hex,
        "signature": training_signature(data), "seed": args.seed,
        "in_dim": sub.num_node_features, "hid_dim": hid_dim, "diff_dim": 2 * hid_dim,
        "mean": mean, "std": std, "threshold": args.threshold,
        "n_train_anomalies": int((sub.y == 1).sum()),
    }
    sub.x = (sub.x - mean) / std.clamp_min(1e-8)
    return sub, metadata


def make_loader(sub, batch_size):
    if sub.num_nodes <= batch_size:
        return DataLoader([sub], batch_size=1, shuffle=False)
    try:
        parts = max(1, math.ceil(sub.num_nodes / batch_size))
        clusters = ClusterData(sub, num_parts=parts, log=False)
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("Graph partitioning needs a matching pyg-lib or torch-sparse build with METIS. "
                           "Install it in the augmentation environment, or use a batch size that fits the full graph.") from error
    # The inherited pipeline uses one partition per batch. Reading each
    # ClusterData item directly preserves that induced subgraph and also handles
    # edgeless partitions, for which PyG 2.7 ClusterLoader._collate calls max()
    # on an empty edge-index tensor. Empty METIS partitions are omitted.
    partptr = clusters.partition.partptr
    nonempty = (partptr[1:] > partptr[:-1]).nonzero(as_tuple=True)[0].tolist()
    return DataLoader(torch.utils.data.Subset(clusters, nonempty), batch_size=1,
                      shuffle=True, num_workers=0)


def save_checkpoint(path, kind, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"kind": kind, "metadata": metadata, "state_dict": state}, path)


def read_checkpoint(path, kind):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != kind or checkpoint.get("metadata", {}).get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported {kind} checkpoint format: {path}")
    return checkpoint


def load_models(directory, name, data, device):
    directory = Path(directory)
    ae_checkpoint = read_checkpoint(directory / f"{name}_ae.pt", "ae")
    dm_checkpoint = read_checkpoint(directory / f"{name}_dm.pt", "dm")
    metadata = ae_checkpoint["metadata"]
    if metadata["dataset"] != name or metadata["signature"] != training_signature(data):
        raise ValueError("Input training graph differs from the checkpoint; train again with this data/mask.")
    if dm_checkpoint["metadata"] != {key: value for key, value in metadata.items() if key not in ("mean", "std")}:
        raise ValueError("VAE and diffusion checkpoints do not belong to the same training graph/configuration.")
    ae = VGAE(metadata["in_dim"], metadata["hid_dim"], temporal=False,
              threshold=metadata["threshold"]).to(device)
    dm = Model(MLPDiffusion(metadata["hid_dim"], metadata["diff_dim"]), metadata["hid_dim"]).to(device)
    ae.load_state_dict(ae_checkpoint["state_dict"])
    dm.load_state_dict(dm_checkpoint["state_dict"])
    return ae.eval(), dm.eval(), metadata


def train_ae(ae, loader, args, device, path, metadata):
    optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr, weight_decay=args.wd)
    for module in ae.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    best_loss, wait = float("inf"), 0
    for epoch in range(args.vae_epochs):
        ae.train()
        total_loss, n_nodes = 0.0, 0
        start = time.monotonic()
        for batch in loader:
            x, edges = batch.x.to(device), batch.edge_index.to(device)
            negatives = negative_sampling(edges, num_nodes=len(x))
            labels = torch.cat((torch.ones(edges.size(1), device=device), torch.zeros(negatives.size(1), device=device)))
            y = batch.y.float().unsqueeze(1).to(device)
            reconstructed, edge_logits, _, _ = ae(x, edges, negatives, y)
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, labels) if labels.numel() else x.new_zeros(())
            loss = args.wx * F.mse_loss(reconstructed, x) + args.we * edge_loss
            kl = (0.5 / len(x) * (1 + 2 * ae.log_std - ae.mean ** 2 - torch.exp(ae.log_std) ** 2).sum(1).mean())
            loss = loss - args.beta * kl
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite VAE loss; lower --lr and check the training features.")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
            n_nodes += len(x)
        epoch_loss = total_loss / n_nodes
        if epoch_loss < best_loss:
            best_loss, wait = epoch_loss, 0
            save_checkpoint(path, "ae", ae, metadata)
        else:
            wait += 1
        print(f"  VAE {epoch + 1:03d}: loss={epoch_loss:.6f}, seconds={time.monotonic() - start:.2f}")
        if wait >= args.patience:
            break
    ae.load_state_dict(read_checkpoint(path, "ae")["state_dict"])
    return ae.eval()


def train_dm(dm, ae, loader, args, device, path, metadata):
    optimizer = torch.optim.Adam(dm.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=20)
    best_loss, wait = float("inf"), 0
    ae.eval()
    dm_metadata = {key: value for key, value in metadata.items() if key not in ("mean", "std")}
    for epoch in range(args.diff_epochs):
        dm.train()
        total_loss, n_nodes = 0.0, 0
        for batch in loader:
            x, edges = batch.x.to(device), batch.edge_index.to(device)
            y = batch.y.float().unsqueeze(1).to(device)
            with torch.no_grad():
                latent, _ = ae.encode(x, edges, y)
            loss = dm(latent, y).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite diffusion loss; lower --lr and check the VAE checkpoint.")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dm.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
            n_nodes += len(x)
        epoch_loss = total_loss / n_nodes
        scheduler.step(epoch_loss)
        if epoch_loss < best_loss:
            best_loss, wait = epoch_loss, 0
            save_checkpoint(path, "dm", dm, dm_metadata)
        else:
            wait += 1
        print(f"  Diffusion {epoch + 1:03d}: loss={epoch_loss:.6f}")
        if wait >= args.patience:
            break
    dm.load_state_dict(read_checkpoint(path, "dm")["state_dict"])
    return dm.eval()
