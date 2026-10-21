"""Conditional variational graph autoencoder used by the augmentation stage.

Modified for DASNA: shared importable model class for portable state checkpoints.
See THIRD_PARTY_NOTICES.md for the inherited implementation's provenance.
"""
import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from dasna.augmentation.sage import SAGEConv


class VGAE(nn.Module):
    def __init__(self, in_dim, hid_dim, temporal=True, t_min=0, t_max=1024, etypes=1, threshold=0.5):
        super().__init__()
        self.hid_dim = hid_dim
        self.temporal = temporal
        self.t_min = t_min
        self.t_max = t_max
        self.time_len = self.t_max - self.t_min + 1 if temporal else None
        self.etypes = etypes
        self.threshold = threshold
        self.enc_shared = SAGEConv(in_dim, hid_dim, temporal=temporal,
                                   time_len=self.time_len, etypes=etypes)
        self.enc_mu = SAGEConv(hid_dim, hid_dim, temporal=temporal,
                               time_len=self.time_len, etypes=etypes)
        self.enc_sigma = SAGEConv(hid_dim, hid_dim, temporal=temporal,
                                  time_len=self.time_len, etypes=etypes)
        self.dec_attr = nn.Linear(hid_dim, in_dim)
        self.dec_time = nn.Linear(2 * hid_dim, 1)
        self.dec_type = nn.Linear(2 * hid_dim, self.etypes)
        self.dec_stru = nn.Linear(2 * hid_dim, 1)
        self.map_label_e = nn.Linear(1, in_dim, bias=False)
        self.map_label_d = nn.Linear(1, hid_dim, bias=False)

    def forward(self, x, pos_edge_index, neg_edge_index, label, edge_time=None, edge_type=None):
        self.mean, self.log_std = self.encode(x, pos_edge_index, label, edge_time, edge_type=edge_type)
        noise = torch.randn_like(self.mean)
        z = self.mean + noise * torch.exp(self.log_std)
        x_, edge_pred, t_, p_ = self.decode(z, pos_edge_index, neg_edge_index, label)
        return x_, edge_pred, t_, p_

    def encode(self, h, edge_index, label, edge_time=None, edge_type=None):
        h = h + self.map_label_e(label)
        h = self.enc_shared(h, edge_index, edge_time, edge_type)
        h = torch.relu(h)
        mean = self.enc_mu(h, edge_index, edge_time, edge_type)
        log_std = self.enc_sigma(h, edge_index, edge_time, edge_type)
        return mean, log_std

    def decode(self, z, pos_edge_index, neg_edge_index, label):
        z = z + self.map_label_d(label)
        x_ = self.dec_attr(z)
        pos_ze = torch.cat([z[pos_edge_index[0]], z[pos_edge_index[1]]], dim=1)
        neg_ze = torch.cat([z[neg_edge_index[0]], z[neg_edge_index[1]]], dim=1)
        pos_edge_pred = self.dec_stru(pos_ze).squeeze(-1)
        neg_edge_pred = self.dec_stru(neg_ze).squeeze(-1)
        edge_pred = torch.cat([pos_edge_pred, neg_edge_pred], dim=0)
        t_ = self.dec_time(pos_ze).squeeze(-1) if self.temporal else None
        p_ = self.dec_type(pos_ze) if self.etypes > 1 else None
        return x_, edge_pred, t_, p_

    def sample(self, z, label):
        z = z + self.map_label_d(label)
        x_ = self.dec_attr(z)
        n = z.size(0)

        max_mem_gb = 4
        avail_bytes = max_mem_gb * 1024**3
        chunk_mem = z.size(1) * 2 * z.element_size()
        safe_chunk = max(1, int(avail_bytes / chunk_mem / z.size(0)))
        chunk_size = min(getattr(self, '_adj_chunk_size', 512), safe_chunk, n)
        edges = []
        for i in range(0, n, chunk_size):
            j_end = min(i + chunk_size, n)
            zi = z[i:j_end].unsqueeze(1)
            zj = z.unsqueeze(0)
            ze_chunk = torch.cat((zi.expand(-1, n, -1), zj.expand(j_end - i, -1, -1)), dim=2)
            adj_chunk = torch.sigmoid(self.dec_stru(ze_chunk)).squeeze(-1)
            edges_chunk = (adj_chunk > self.threshold).nonzero()
            edges.append(edges_chunk + torch.tensor([[i, 0]], device=edges_chunk.device))
        edge_index = torch.cat(edges, dim=0).T
        edge_index = add_self_loops(edge_index, num_nodes=n)[0]
        pos_ze = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        if self.temporal:
            t_ = self.dec_time(pos_ze).squeeze(-1)
            t_ = torch.clamp(t_, min=0, max=1)
            t_ = t_ * (self.t_max - self.t_min)
        else:
            t_ = None
        p_ = self.dec_type(pos_ze).argmax(-1) if self.etypes > 1 else None
        return x_, edge_index, t_, p_
