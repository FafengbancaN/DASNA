# Modified for DASNA: package CLI, configuration templates and path overrides.
# Source and component licenses: THIRD_PARTY_NOTICES.md.
"""Train DASNA from a YAML experiment template.

Example: python -m dasna.detection.train --config configs/amazon.yaml
Templates are runnable starting points, not validated paper reproduction settings.
"""

import argparse
from itertools import product
import os
from pathlib import Path
import random
import numpy as np
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PATH_DEFAULTS = {
    'raw_dir': 'data/raw', 'processed_dir': 'data/processed',
    'generated_dir': 'data/generated', 'save_dir': 'artifacts/checkpoints/detection',
}
DEFAULTS = dict(
    model='DASNA', multi_relation='Mamba', train_ratio=0.4, test_ratio=0.67,
    emb_size=64, thres=0.5, rho=0.5, seed=72, lr=0.005,
    weight_decay=0.0005, batch_size=256, num_epochs=51, valid_epochs=5,
    alpha=2.0, no_cuda=False, cuda_id='0', use_dm=True,
    data_suffix='_10pct_dm', use_pick=True, use_neighbor_filter=True,
    choose_type='residual', feat_residual_alpha=0.5, feat_compress_dim=0,
    mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
)


def resolve_path(path):
    path = Path(path).expanduser()
    return str(path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve())


def get_config(config_path):
    path = Path(resolve_path(config_path))
    with path.open(encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict) or 'data_name' not in loaded:
        raise ValueError('Configuration must be a mapping with data_name.')
    config = {**DEFAULTS, **PATH_DEFAULTS, **loaded}
    config['_config_name'] = path.stem
    for key in PATH_DEFAULTS:
        if not isinstance(config[key], str):
            raise ValueError(f'{key} must be a single path string.')
        config[key] = resolve_path(config[key])
    return config


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grid(config):
    keys = [key for key, value in config.items() if isinstance(value, list)]
    values = [config[key] for key in keys]
    for combination in product(*values):
        yield {**config, **dict(zip(keys, combination))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', '-config', required=True, help='YAML path, relative to repository root or absolute')
    parser.add_argument('--multi-run', '--multi_run', action='store_true', help='Run the Cartesian product of YAML list values')
    parser.add_argument('--aggregation', choices=['GNN', 'Mamba'], help='Explicit aggregation override; GNN supports CPU smoke tests')
    parser.add_argument('--cpu', action='store_true', help='Use CPU; combine with --aggregation GNN for DASNA')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    for key in PATH_DEFAULTS:
        parser.add_argument('--' + key.replace('_', '-'), help=f'Override {key}; repository-relative or absolute')
    args = parser.parse_args(argv)
    config = get_config(args.config)
    if args.aggregation:
        config['multi_relation'] = args.aggregation
    if args.cpu:
        config['no_cuda'] = True
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    for key in PATH_DEFAULTS:
        if getattr(args, key) is not None:
            config[key] = resolve_path(getattr(args, key))
    if not args.multi_run and any(isinstance(value, list) for value in config.values()):
        parser.error('YAML list values require --multi-run.')
    from dasna.detection.model_handler import ModelHandler
    results = []
    for index, current in enumerate(grid(config)):
        if any(current[key] < 1 for key in ('num_epochs', 'valid_epochs', 'batch_size')):
            raise ValueError('num_epochs, valid_epochs, and batch_size must be positive.')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(current['cuda_id'])
        set_random_seed(current['seed'])
        if args.multi_run:
            current['_config_name'] += f'_run{index + 1}_seed{current["seed"]}'
        print('Configuration:')
        for key, value in sorted(current.items()):
            print(f'  {key}: {value}')
        result = ModelHandler(current).train()
        results.append(result)
        print(f'Test F1-macro={result[0]:.4f} AUC={result[3]:.4f} AUC-PR={result[5]:.4f} G-Mean={result[4]:.4f}')
    if len(results) > 1:
        values = np.asarray(results)
        print(f'Multi-run mean={values.mean(axis=0).tolist()} std={values.std(axis=0, ddof=1).tolist()}')
    return results


if __name__ == '__main__':
    main()
