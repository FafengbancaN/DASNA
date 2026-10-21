"""Training orchestration for DASNA and the retained SAGE/GCN baselines.

Modified for DASNA: portable paths, augmentation alignment and training plumbing.
Source and component licenses: THIRD_PARTY_NOTICES.md.
"""

import argparse
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from dasna.detection.utils import (
    test_dasna, test_sage, load_data, load_mask, merge_standalone_train,
    normalize, pick_step, pos_neg_split,
)
from dasna.detection.model import DASNALayer
from dasna.detection.layers import InterAgg, IntraAgg, MambaIntraAgg
from dasna.detection.graphsage import (
    MeanAggregator, Encoder, GCNAggregator, GCNEncoder, GraphSage, GCN,
)


class ModelHandler:
    def __init__(self, config):
        args = argparse.Namespace(**config)
        if args.model not in ('DASNA', 'SAGE', 'GCN'):
            raise ValueError('model must be DASNA, SAGE, or GCN.')
        if args.multi_relation not in ('GNN', 'Mamba'):
            raise ValueError('multi_relation must be GNN or Mamba.')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.model == 'DASNA' and args.multi_relation == 'Mamba' and not args.cuda:
            raise RuntimeError('Mamba aggregation requires a compatible CUDA device and mamba-ssm. '
                               'For an explicit CPU baseline, select --aggregation GNN --cpu.')
        seed = args.seed[0] if isinstance(args.seed, list) else args.seed
        np.random.seed(seed)
        random.seed(seed)
        graphs, features, labels = load_data(
            args.data_name, prefix=args.raw_dir, processed_dir=args.processed_dir,
        )
        homo, relation1, relation2, relation3 = graphs
        fixed = load_mask(args.data_name, prefix=args.raw_dir)
        if fixed is not None:
            idx_train, idx_valid, idx_test, y_train, y_valid, y_test = fixed
        else:
            # All supported datasets share the fallback; retain the original
            # exclusion of Amazon's initially unlabeled node range.
            start = 3305 if args.data_name == 'amazon' else 0
            index = list(range(start, len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(
                index, labels[start:], stratify=labels[start:],
                train_size=args.train_ratio, random_state=2, shuffle=True,
            )
            idx_valid, idx_test, y_valid, y_test = train_test_split(
                idx_rest, y_rest, stratify=y_rest,
                test_size=args.test_ratio, random_state=2, shuffle=True,
            )
        if args.use_dm:
            extra_features, extra_labels, extra_homo = merge_standalone_train(
                args.generated_dir, args.data_name, args.data_suffix,
            )
            if extra_features.shape[1] != features.shape[1]:
                raise ValueError('Generated and original features have different dimensions.')
            original_n = features.shape[0]
            features = np.vstack([features, extra_features])
            extra_global = {k + original_n: {n + original_n for n in neighbors}
                            for k, neighbors in extra_homo.items()}
            for graph in (homo, relation1, relation2, relation3):
                graph.update({k: set(neighbors) for k, neighbors in extra_global.items()})
            labels = np.concatenate([labels, extra_labels])
            idx_train = idx_train + list(range(original_n, features.shape[0]))
            y_train = np.concatenate([y_train, extra_labels])
            print(f'Merged {len(extra_labels)} generated training nodes; validation/test masks stay fixed.')
        for name, split_labels in [('train', y_train), ('validation', y_valid), ('test', y_test)]:
            if set(np.unique(split_labels)) != {0, 1}:
                raise ValueError(f'{name} split must contain both binary classes.')
        train_pos, train_neg = pos_neg_split(idx_train, y_train)
        features = normalize(features)
        if args.feat_compress_dim > 0 and features.shape[1] > args.feat_compress_dim:
            if args.feat_compress_dim > min(features.shape):
                raise ValueError('feat_compress_dim exceeds the available PCA rank.')
            features = PCA(n_components=args.feat_compress_dim, random_state=seed).fit_transform(features).astype(np.float32)
        adjacency = homo if args.model in ('SAGE', 'GCN') else [relation1, relation2, relation3]
        self.args = args
        self.dataset = dict(
            feat_data=features, labels=labels, adj_lists=adjacency, homo=homo,
            idx_train=idx_train, idx_valid=idx_valid, idx_test=idx_test,
            y_train=y_train, y_valid=y_valid, y_test=y_test,
            train_pos=train_pos, train_neg=train_neg,
        )
        print(f'Dataset={args.data_name} train={len(y_train)} validation={len(y_valid)} '
              f'test={len(y_test)} features={features.shape[1]} device={"cuda" if args.cuda else "cpu"}')
        print(f'Model={args.model} aggregation={args.multi_relation} '
              f'neighbor_filter={args.use_neighbor_filter} undersampling={args.use_pick}')

    def train(self):
        args = self.args
        data = self.dataset
        features = nn.Embedding(*data['feat_data'].shape)
        features.weight = nn.Parameter(torch.tensor(data['feat_data'], dtype=torch.float32), requires_grad=False)
        if args.cuda:
            features = features.cuda()
        feature_dim = data['feat_data'].shape[1]
        if args.model == 'DASNA':
            use_mamba = args.multi_relation == 'Mamba'
            intra_class = MambaIntraAgg if use_mamba else IntraAgg
            kwargs = dict(choose_type=args.choose_type,
                          feat_residual_alpha=args.feat_residual_alpha,
                          use_neighbor_filter=args.use_neighbor_filter, use_pick=args.use_pick)
            if use_mamba:
                kwargs.update(d_state=args.mamba_d_state, d_conv=args.mamba_d_conv, expand=args.mamba_expand)
            intra = [intra_class(features, feature_dim, args.emb_size, data['train_pos'],
                                 args.rho, cuda=args.cuda, **kwargs) for _ in range(3)]
            inter = InterAgg(features, feature_dim, args.emb_size, data['train_pos'],
                             data['adj_lists'], intra, inter=args.multi_relation, cuda=args.cuda)
            model = DASNALayer(2, inter, args.alpha)
        elif args.model == 'SAGE':
            aggregator = MeanAggregator(features, cuda=args.cuda)
            encoder = Encoder(features, feature_dim, args.emb_size, data['adj_lists'],
                              aggregator, gcn=False, cuda=args.cuda)
            model = GraphSage(2, encoder)
        else:
            aggregator = GCNAggregator(features, cuda=args.cuda)
            encoder = GCNEncoder(features, feature_dim, args.emb_size, data['adj_lists'],
                                 aggregator, gcn=True, cuda=args.cuda)
            model = GCN(2, encoder)
        if args.cuda:
            model = model.cuda()
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                     lr=args.lr, weight_decay=args.weight_decay)
        directory = Path(args.save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / f'{args._config_name}.ckpt'
        best_auc = -1.0
        evaluate = test_dasna if args.model == 'DASNA' else test_sage
        for epoch in range(args.num_epochs):
            model.train()
            sampled = pick_step(data['idx_train'], data['y_train'], data['homo'],
                                size=len(data['train_pos']) * 2) if args.use_pick else list(data['idx_train'])
            random.shuffle(sampled)
            loss_sum = 0.0
            node_count = 0
            started = time.time()
            for start in range(0, len(sampled), args.batch_size):
                nodes = sampled[start:start + args.batch_size]
                labels = torch.as_tensor(data['labels'][nodes], dtype=torch.long,
                                         device='cuda' if args.cuda else 'cpu')
                optimizer.zero_grad()
                loss = model.loss(nodes, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(nodes)
                node_count += len(nodes)
            print(f'Epoch={epoch} loss={loss_sum / node_count:.6f} seconds={time.time() - started:.2f}')
            if epoch % args.valid_epochs == 0:
                print(f'Validation at epoch {epoch}')
                metrics = evaluate(data['idx_valid'], data['y_valid'], model, args.batch_size, args.thres)
                if metrics[3] > best_auc:
                    best_auc = metrics[3]
                    torch.save({'state_dict': model.state_dict(), 'config': vars(args)}, checkpoint)
                    print(f'Saved best validation AUC checkpoint: {checkpoint}')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['state_dict'])
        return evaluate(data['idx_test'], data['y_test'], model, args.batch_size, args.thres)
