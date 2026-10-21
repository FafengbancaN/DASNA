# Modified for DASNA: preserve original relations and separate local cache paths.
# Source and component licenses: THIRD_PARTY_NOTICES.md.
"""Prepare base graph adjacency lists without changing source MAT files.

Generated standalone MAT files are read directly by the detector; they do not
need adjacency pickle preprocessing.
"""

import argparse
from pathlib import Path
import pickle
import scipy.sparse as sp
from scipy.io import loadmat
from dasna.detection.utils import DATASETS, dataset_path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def sparse_csr_to_adjlist(matrix):
    """Use CSR row slices; preserve supplied relations and add self loops."""
    matrix = sp.csr_matrix(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Adjacency must be square.')
    matrix = matrix + sp.eye(matrix.shape[0], format='csr')
    matrix.eliminate_zeros()
    return {i: set(matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]].tolist())
            for i in range(matrix.shape[0])}


def preprocess_dataset(name, raw_dir, processed_dir):
    source = dataset_path(name, raw_dir)
    if not source.is_file():
        raise FileNotFoundError(f'Dataset is missing: {source}')
    relations = DATASETS[name][1]
    relation_keys = [f'net_{relation}' for relation in relations]
    mat = loadmat(source, variable_names=['homo', 'Network', *relation_keys])
    homo = mat.get('homo', mat.get('Network'))
    if homo is None:
        raise ValueError(f'{source} has no homo or Network matrix.')
    supplied_relations = [key in mat for key in relation_keys]
    if any(supplied_relations) and not all(supplied_relations):
        raise ValueError(f'{source} provides only some relation matrices: {relation_keys}')
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    homo_graph = sparse_csr_to_adjlist(homo)
    outputs = []
    for relation, key in [('homo', 'homo'), *zip(relations, relation_keys)]:
        graph = homo_graph if relation == 'homo' or not all(supplied_relations) else sparse_csr_to_adjlist(mat[key])
        output = output_dir / f'{name}_{relation}_adjlists.pickle'
        with output.open('wb') as handle:
            pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        outputs.append(output)
        print(f'{name}: {output.name} ({len(graph)} nodes)')
    if not any(supplied_relations):
        print(f'{name}: no distinct relation matrices supplied; the three relation slots reuse the base graph.')
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument('--raw-dir', default='data/raw', help='MAT directory, relative to repository root or absolute')
    parser.add_argument('--processed-dir', default='data/processed', help='Adjacency output directory')
    args = parser.parse_args(argv)
    for name in args.datasets:
        preprocess_dataset(name, resolve_path(args.raw_dir), resolve_path(args.processed_dir))


if __name__ == '__main__':
    main()
