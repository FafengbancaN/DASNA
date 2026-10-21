"""Dataset loading, sampling, and binary detection metrics.

Modified for DASNA: data paths, masks, synthetic-node remapping and evaluation.
Source and component licenses: THIRD_PARTY_NOTICES.md.
"""

from pathlib import Path
import pickle
import random
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
import torch

DATASETS = {
    'amazon': ('Amazon.mat', ('upu', 'usu', 'uvu')),
    'yelpchi': ('YelpChi.mat', ('rur', 'rtr', 'rsr')),
    'weibo': ('weibo.mat', ('rel1', 'rel2', 'rel3')),
    'questions': ('questions.mat', ('rel1', 'rel2', 'rel3')),
    'elliptic': ('elliptic.mat', ('rel1', 'rel2', 'rel3')),
    'facebook': ('facebook.mat', ('rel1', 'rel2', 'rel3')),
}


def _safe_feat(raw):
    return np.asarray(raw.toarray() if sp.issparse(raw) else raw, dtype=np.float32)


def _labels(mat):
    labels = np.asarray(mat['label'] if 'label' in mat else mat['Label']).reshape(-1)
    # Retain the supplied experiment convention: class 1 is positive.
    if labels.max() >= 2 and labels.min() == 0:
        labels = labels == 1
    labels = labels.astype(np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Detection requires binary labels (0 and 1).')
    return labels


def _mask_to_indices(mask):
    return np.flatnonzero(np.asarray(mask).reshape(-1) == 1).tolist()


def dataset_path(data, raw_dir):
    if data not in DATASETS:
        raise ValueError(f'Unknown dataset {data!r}; choose from {sorted(DATASETS)}.')
    return Path(raw_dir) / DATASETS[data][0]


def load_mask(data, prefix='data/raw', suffix=''):
    """Return fixed train/validation/test splits, or None when masks are absent."""
    if suffix:
        raise ValueError('Load base MAT masks without a suffix; use generated_dir for augmentation.')
    mat = loadmat(dataset_path(data, prefix), variable_names=[
        'train_mask', 'val_mask', 'test_mask', 'label', 'Label',
    ])
    keys = ('train_mask', 'val_mask', 'test_mask')
    if not all(key in mat for key in keys):
        return None
    labels = _labels(mat)
    if any(np.asarray(mat[key]).size != labels.size for key in keys):
        raise ValueError('Split mask lengths must match the number of labels.')
    splits = [_mask_to_indices(mat[key]) for key in keys]
    if any(not split for split in splits):
        raise ValueError('Base train/validation/test masks must all contain nodes.')
    if any(set(splits[i]) & set(splits[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Base train/validation/test masks overlap.')
    return (*splits, *(labels[split] for split in splits))


def load_data(data, prefix='data/raw', suffix='', processed_dir='data/processed'):
    """Load base MAT features and relation adjacency lists prepared by preprocessing."""
    if suffix:
        raise ValueError('Augmented graphs use base data plus generated_dir; set use_dm: true.')
    mat = loadmat(dataset_path(data, prefix))
    labels = _labels(mat)
    features = _safe_feat(mat['features'] if 'features' in mat else mat['Attributes'])
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError('Feature rows and labels must have the same node count.')
    graphs = []
    for relation in ('homo', *DATASETS[data][1]):
        path = Path(processed_dir) / f'{data}_{relation}_adjlists.pickle'
        if not path.is_file():
            raise FileNotFoundError(f'{path} is missing. Run python -m dasna.preprocessing --datasets {data}.')
        with path.open('rb') as handle:
            graph = pickle.load(handle)
        graphs.append({int(k): set(map(int, neighbors)) for k, neighbors in graph.items()})
    return graphs, features, labels


def merge_standalone_train(generated_dir, dm_name, suffix):
    """Read generated training nodes and remap their induced graph locally."""
    suffix = suffix[:-3] if suffix.endswith('_dm') else suffix
    if not suffix or '/' in suffix or '\\' in suffix:
        raise ValueError('data_suffix must identify an augmentation, for example _10pct_dm.')
    path = Path(generated_dir) / f'{dm_name}{suffix}_standalone.mat'
    if not path.is_file():
        raise FileNotFoundError(f'Generated training data is missing: {path}')
    mat = loadmat(path)
    if 'train_mask' not in mat:
        raise ValueError(f'{path} has no train_mask.')
    labels = _labels(mat)
    if np.asarray(mat['train_mask']).size != labels.size:
        raise ValueError('Generated train_mask length does not match labels.')
    indices = _mask_to_indices(mat['train_mask'])
    if not indices:
        raise ValueError(f'{path} contains no training nodes.')
    features = _safe_feat(mat['features'] if 'features' in mat else mat['Attributes'])
    if features.shape[0] != labels.size:
        raise ValueError('Generated feature rows and labels must match.')
    adjacency = mat.get('homo', mat.get('Network'))
    if adjacency is None:
        raise ValueError(f'{path} has no homo or Network adjacency matrix.')
    adjacency = sp.csr_matrix(adjacency)
    if adjacency.shape != (labels.size, labels.size):
        raise ValueError('Generated adjacency shape does not match labels.')
    # Subset both axes before renumbering; noncontiguous masks are supported.
    adjacency = adjacency[indices][:, indices]
    graph = {i: {i} for i in range(len(indices))}
    rows, columns = adjacency.nonzero()
    for i, j in zip(rows, columns):
        graph[int(i)].add(int(j))
        graph[int(j)].add(int(i))
    return features[indices], labels[indices], graph


def normalize(matrix):
    """Retain the supplied row normalization, including its 0.01 offset."""
    rowsum = np.asarray(matrix.sum(1)).reshape(-1) + 0.01
    inverse = np.zeros_like(rowsum, dtype=np.float32)
    np.divide(1.0, rowsum, out=inverse, where=rowsum != 0)
    return inverse[:, None] * matrix


def sparse_to_adjlist(matrix, filename):
    """Write an undirected adjacency list with self loops."""
    matrix = sp.csr_matrix(matrix)
    graph = {i: {i} for i in range(matrix.shape[0])}
    rows, columns = matrix.nonzero()
    for i, j in zip(rows, columns):
        graph[int(i)].add(int(j))
        graph[int(j)].add(int(i))
    with Path(filename).open('wb') as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pos_neg_split(nodes, labels):
    return ([node for node, label in zip(nodes, labels) if label == 1],
            [node for node, label in zip(nodes, labels) if label == 0])


def pick_step(idx_train, y_train, adj_list, size):
    degree_train = np.array([len(adj_list[node]) for node in idx_train])
    lf_train = (y_train.sum() - len(y_train)) * y_train + len(y_train)
    return random.choices(idx_train, weights=degree_train / lf_train, k=size)


def prob2pred(y_prob, thres=0.5):
    return (np.asarray(y_prob) >= thres).astype(np.int32)


def conf_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    denominator = (tp + fn) * (tn + fp)
    return float(np.sqrt(float(tp) * float(tn) / denominator)) if denominator else 0.0


def _evaluate(test_cases, labels, model, batch_size, thres, relation_model):
    if not test_cases:
        raise ValueError('Cannot evaluate an empty split.')
    probabilities = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(test_cases), batch_size):
                nodes = test_cases[start:start + batch_size]
                if relation_model:
                    scores, _ = model.to_prob(nodes, labels[start:start + batch_size], train_flag=False)
                else:
                    scores = model.to_prob(nodes)
                probabilities.extend(scores.cpu().numpy()[:, 1].tolist())
    finally:
        model.train(was_training)
    predictions = prob2pred(probabilities, thres)
    auc = roc_auc_score(labels, probabilities)
    ap = average_precision_score(labels, probabilities)
    macro = f1_score(labels, predictions, average='macro', zero_division=0)
    positive = f1_score(labels, predictions, pos_label=1, average='binary', zero_division=0)
    negative = f1_score(labels, predictions, pos_label=0, average='binary', zero_division=0)
    confusion = confusion_matrix(labels, predictions, labels=[0, 1])
    gmean = conf_gmean(confusion)
    print(f'  F1-macro={macro:.4f} AUC={auc:.4f} AUC-PR={ap:.4f} G-Mean={gmean:.4f}')
    return macro, positive, negative, auc, gmean, ap


def test_sage(test_cases, labels, model, batch_size, thres=0.5):
    return _evaluate(test_cases, labels, model, batch_size, thres, False)


def test_dasna(test_cases, labels, model, batch_size, thres=0.5):
    return _evaluate(test_cases, labels, model, batch_size, thres, True)
