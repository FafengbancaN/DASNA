# Modified for DASNA: neighbor-filter naming and explicit Mamba requirements.
# Source and component licenses: THIRD_PARTY_NOTICES.md.
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from operator import itemgetter
import math

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    MAMBA_IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    MAMBA_AVAILABLE = False
    MAMBA_IMPORT_ERROR = str(exc)

"""
	DASNA neighbor filtering and aggregation layers.
	For inherited component provenance, see THIRD_PARTY_NOTICES.md.
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class InterAgg(nn.Module):

	def __init__(self, features, feature_dim, embed_dim,
				 train_pos, adj_lists, intraggs, inter='GNN', cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param cuda: whether to use GPU
		"""
		super(InterAgg, self).__init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda
		self.train_pos = train_pos

		# initial filtering thresholds
		self.thresholds = [0.5, 0.5, 0.5]

		# parameter used to transform node embeddings before inter-relation aggregation
		self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim*len(intraggs)+self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

		# label predictor for similarity measure
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.relation_score_log = []

	def forward(self, nodes, labels, train_flag=True):
		"""
		:param nodes: a list of batch node ids
		:param labels: a list of batch node labels
		:param train_flag: indicates whether in training or testing mode
		:return combined: the embeddings of a batch of input node features
		:return center_scores: the label-aware scores of batch nodes
		"""

		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = []
		for adj_list in self.adj_lists:
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		# find unique nodes and their neighbors used in current batch
		unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
								 set.union(*to_neighs[2], set(nodes)))

		# calculate label-aware scores
		if self.cuda:
			batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
			pos_features = self.features(torch.cuda.LongTensor(list(self.train_pos)))
		else:
			batch_features = self.features(torch.LongTensor(list(unique_nodes)))
			pos_features = self.features(torch.LongTensor(list(self.train_pos)))
		batch_scores = self.label_clf(batch_features)
		pos_scores = self.label_clf(pos_features)
		id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

		# the label-aware scores for current batch of nodes
		center_scores = batch_scores[[id_mapping[node] for node in nodes], :]

		# get neighbor node id list for each batch node and relation
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

		# assign label-aware scores to neighbor nodes for each batch node and relation
		r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list]
		r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list]
		r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]

		# count the number of neighbors kept for aggregation for each batch node and relation
		r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
		r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
		r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

		# intra-aggregation steps for each relation
		# Eq. (8) in the paper
		r1_feats, r1_scores = self.intra_agg1.forward(nodes, labels, r1_list, center_scores, r1_scores, pos_scores, r1_sample_num_list, train_flag)
		r2_feats, r2_scores = self.intra_agg2.forward(nodes, labels, r2_list, center_scores, r2_scores, pos_scores, r2_sample_num_list, train_flag)
		r3_feats, r3_scores = self.intra_agg3.forward(nodes, labels, r3_list, center_scores, r3_scores, pos_scores, r3_sample_num_list, train_flag)

		# get features or embeddings for batch nodes
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		self_feats = self.features(index)

		# number of nodes in a batch
		n = len(nodes)

		# concat the intra-aggregated embeddings from each relation
		# Eq. (9) in the paper
		cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)

		combined = F.relu(cat_feats.mm(self.weight).t())

		return combined, center_scores


class AlphaGatingNet(nn.Module):
	"""
	Lightweight gating network that predicts per-node alpha for hybrid scoring.
	Input: center node features [batch, feat_dim]
	Output: alpha in (0, max_alpha] range per node
	"""
	def __init__(self, feat_dim, hidden_dim=8, max_alpha=0.5):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(feat_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
			nn.Sigmoid()
		)
		self.max_alpha = max_alpha

	def forward(self, feats):
		alpha = self.net(feats)
		return alpha.squeeze(-1) * self.max_alpha


class IntraAgg(nn.Module):

		def __init__(self, features, feat_dim, embed_dim, train_pos, rho, cuda=False,
					 choose_type='score', feat_residual_alpha=0.5, use_neighbor_filter=True, use_pick=True):
			"""
			:param features: the input node features or embeddings for all nodes
			:param feat_dim: the input dimension
			:param embed_dim: the embed dimension
			:param train_pos: positive samples in training set
			:param rho: the ratio of the oversample neighbors for the minority class
			:param cuda: whether to use GPU
			:param choose_type: choose method type: 'score' (original), 'residual' (feature residual)
			:param feat_residual_alpha: weight for feature residual when choose_type='residual' (0-1)
			:param use_neighbor_filter: whether to use Choose (neighbor filtering).
								 If False, uses no filtering (choose_step_none) for both train and test.
			:param use_pick: whether to use Pick (training set undersampling).
							  This is handled at training orchestration level; kept here for interface consistency.
			"""
			super(IntraAgg, self).__init__()

			self.features = features
			self.cuda = cuda
			self.feat_dim = feat_dim
			self.embed_dim = embed_dim
			self.train_pos = train_pos
			self.rho = rho
			self.weight = nn.Parameter(torch.FloatTensor(2*self.feat_dim, self.embed_dim))
			init.xavier_uniform_(self.weight)

			# Residual choose module
			self.choose_type = choose_type
			self.feat_residual_alpha = feat_residual_alpha
			self.use_neighbor_filter = use_neighbor_filter
			# Dynamic alpha gating: predict per-node alpha from center features
			self.alpha_gating = AlphaGatingNet(feat_dim, hidden_dim=8, max_alpha=0.5)

		def forward(self, nodes, batch_labels, to_neighs_list, batch_scores, neigh_scores, pos_scores, sample_list, train_flag):
			"""
			Code partially from https://github.com/williamleif/graphsage-simple/
			:param nodes: list of nodes in a batch
			:param to_neighs_list: neighbor node id list for each batch node in one relation
			:param batch_scores: the label-aware scores of batch nodes
			:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
			:param pos_scores: the label-aware scores 1-hop neighbors for the minority positive nodes
			:param train_flag: indicates whether in training or testing mode
			:param sample_list: the number of neighbors kept for each batch node in one relation
			:return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
			:return samp_scores: the average neighbor distances for each relation after filtering
			"""
			# Precompute center node features ONCE (used for both choose and aggregation)
			if self.cuda:
				self_feats = self.features(torch.LongTensor(nodes).cuda())
			else:
				self_feats = self.features(torch.LongTensor(nodes))

			# filer neighbors under given relation in the train mode
			if train_flag:
				if self.use_neighbor_filter and self.choose_type == 'residual':
					# --- Batch-extract all neighbor features ONCE (used for both choose and aggregation) ---
					all_neigh_ids = set()
					for to_neigh in to_neighs_list:
						all_neigh_ids.update(to_neigh)

					if all_neigh_ids:
						neigh_ids_list = list(all_neigh_ids)
						if self.cuda:
							neigh_feats_all = self.features(torch.LongTensor(neigh_ids_list).cuda())
						else:
							neigh_feats_all = self.features(torch.LongTensor(neigh_ids_list))
						neigh_id_to_idx = {nid: idx for idx, nid in enumerate(neigh_ids_list)}
					else:
						neigh_feats_all = None
						neigh_id_to_idx = {}

					neigh_feats_for_choose = []
					for to_neigh in to_neighs_list:
						if to_neigh and neigh_feats_all is not None:
							indices = [neigh_id_to_idx[nid] for nid in to_neigh]
							neigh_feats_for_choose.append(neigh_feats_all[indices])
						else:
							device = self_feats.device
							neigh_feats_for_choose.append(torch.tensor([], device=device))

					# Dynamic alpha per node: [batch_size], each in (0, 0.5]
					dyn_alpha = self.alpha_gating(self_feats)

					samp_neighs, samp_scores = choose_step_neighs_residual(
						self_feats, batch_labels, neigh_scores, to_neighs_list, pos_scores,
						self.train_pos, sample_list, self.rho, neigh_feats_for_choose,
						dyn_alpha
					)
				elif not self.use_neighbor_filter:
					# Choose disabled: skip all filtering, use all neighbors directly
					samp_neighs, samp_scores = choose_step_none(
						batch_labels, to_neighs_list, pos_scores, self.train_pos,
						sample_list, self.rho
					)
				else:
					# Choose enabled but choose_type is 'none' or 'score': use original logic
					if self.choose_type == 'none':
						samp_neighs, samp_scores = choose_step_none(
							batch_labels, to_neighs_list, pos_scores, self.train_pos,
							sample_list, self.rho
						)
					else:
						# Original score based choose (choose_type='score')
						samp_neighs, samp_scores = choose_step_neighs(
							batch_scores, batch_labels, neigh_scores, to_neighs_list,
							pos_scores, self.train_pos, sample_list, self.rho
						)
			else:
				if self.use_neighbor_filter and self.choose_type == 'residual':
					# --- Batch-extract all neighbor features ONCE (used for both choose and aggregation) ---
					all_neigh_ids = set()
					for to_neigh in to_neighs_list:
						all_neigh_ids.update(to_neigh)

					if all_neigh_ids:
						neigh_ids_list = list(all_neigh_ids)
						if self.cuda:
							neigh_feats_all = self.features(torch.LongTensor(neigh_ids_list).cuda())
						else:
							neigh_feats_all = self.features(torch.LongTensor(neigh_ids_list))
						neigh_id_to_idx = {nid: idx for idx, nid in enumerate(neigh_ids_list)}
					else:
						neigh_feats_all = None
						neigh_id_to_idx = {}

					neigh_feats_for_choose = []
					for to_neigh in to_neighs_list:
						if to_neigh and neigh_feats_all is not None:
							indices = [neigh_id_to_idx[nid] for nid in to_neigh]
							neigh_feats_for_choose.append(neigh_feats_all[indices])
						else:
							device = self_feats.device
							neigh_feats_for_choose.append(torch.tensor([], device=device))

					dyn_alpha = self.alpha_gating(self_feats)

					samp_neighs, samp_scores = choose_step_test_residual(
						self_feats, neigh_scores, to_neighs_list, sample_list,
						neigh_feats_for_choose, dyn_alpha
					)
				elif not self.use_neighbor_filter:
					# Choose disabled: skip all filtering, use all neighbors directly
					samp_neighs, samp_scores = choose_step_test_none(
						to_neighs_list, sample_list
					)
				elif self.choose_type == 'none':
					# No choose: use all neighbors without filtering
					samp_neighs, samp_scores = choose_step_test_none(
						to_neighs_list, sample_list
					)
				else:
					# Original score based choose (choose_type='score')
					samp_neighs, samp_scores = choose_step_test(
						batch_scores, neigh_scores, to_neighs_list, sample_list
					)

			# find the unique nodes among batch nodes and the filtered neighbors
			unique_nodes_list = list(set.union(*samp_neighs)) if samp_neighs else []
			unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

			# intra-relation aggregation only with sampled neighbors
			mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
			column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
			row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
			mask[row_indices, column_indices] = 1
			if self.cuda:
				mask = mask.cuda()
			num_neigh = mask.sum(1, keepdim=True)
			mask = mask.div(num_neigh)  # mean aggregator
			# Reuse precomputed self_feats; build embed_matrix from cached neigh_feats_all if available
			if self.cuda:
				embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda()) if unique_nodes_list else torch.empty(0, self.feat_dim, device='cuda')
			else:
				embed_matrix = self.features(torch.LongTensor(unique_nodes_list)) if unique_nodes_list else torch.empty(0, self.feat_dim)
			agg_feats = mask.mm(embed_matrix)  # single relation aggregator
			cat_feats = torch.cat((self_feats, agg_feats), dim=1)  # concat with last layer
			to_feats = F.relu(cat_feats.mm(self.weight))
			return to_feats, samp_scores


class MambaIntraAgg(nn.Module):

		def __init__(self, features, feat_dim, embed_dim, train_pos, rho, cuda=False,
					 d_state=16, d_conv=4, expand=2, choose_type='score', feat_residual_alpha=0.5,
					 use_neighbor_filter=True, use_pick=True):
			"""
			:param features: the input node features or embeddings for all nodes
			:param feat_dim: the input dimension
			:param embed_dim: the embed dimension
			:param train_pos: positive samples in training set
			:param rho: the ratio of the oversample neighbors for the minority class
			:param cuda: whether to use GPU
			:param d_state: Mamba state expansion factor
			:param d_conv: Mamba local convolution width
			:param expand: Mamba block expansion factor
			:param choose_type: choose method type: 'score' (original), 'residual' (feature residual)
			:param feat_residual_alpha: weight for feature residual when choose_type='residual' (0-1)
			:param use_neighbor_filter: whether to use Choose (neighbor filtering).
								  If False, uses no filtering (choose_step_none) for both train and test.
			:param use_pick: whether to use Pick (training set undersampling).
							  This is handled at training orchestration level; kept here for interface consistency.
			"""
			super(MambaIntraAgg, self).__init__()

			if not MAMBA_AVAILABLE:
				raise ImportError(
					"MambaIntraAgg requires mamba-ssm installed. "
					"Install the compatible mamba-ssm and causal-conv1d builds for your Torch/CUDA stack. "
					f"Import failure: {MAMBA_IMPORT_ERROR}"
				)

			self.features = features
			self.cuda = cuda
			self.feat_dim = feat_dim
			self.embed_dim = embed_dim
			self.train_pos = train_pos
			self.rho = rho

			# Mamba layer: processes neighbor sequences with selective state space model
			# d_model = feat_dim so output agg_feats has same dim as self_feats
			self.mamba = Mamba(
				d_model=feat_dim,
				d_state=d_state,
				d_conv=d_conv,
				expand=expand
			)

			# Position encoding for neighbor sequences (no natural order)
			self.pos_encoder = nn.Parameter(torch.randn(1, 100, feat_dim) * 0.02)

			# Original concat + linear transformation preserved from IntraAgg
			self.weight = nn.Parameter(torch.FloatTensor(2 * self.feat_dim, self.embed_dim))
			init.xavier_uniform_(self.weight)

			# Residual choose module
			self.choose_type = choose_type
			self.feat_residual_alpha = feat_residual_alpha
			self.use_neighbor_filter = use_neighbor_filter
			# Dynamic alpha gating: predict per-node alpha from center features
			self.alpha_gating = AlphaGatingNet(feat_dim, hidden_dim=8, max_alpha=0.5)

		def forward(self, nodes, batch_labels, to_neighs_list, batch_scores, neigh_scores,
					pos_scores, sample_list, train_flag):
			"""
			:param nodes: list of nodes in a batch
			:param to_neighs_list: neighbor node id list for each batch node in one relation
			:param batch_scores: the label-aware scores of batch nodes
			:param neigh_scores: the label-aware scores of 1-hop neighbors
			:param pos_scores: the label-aware scores for the minority positive nodes
			:param train_flag: indicates whether in training or testing mode
			:param sample_list: the number of neighbors kept for each batch node
			:return to_feats: the aggregated embeddings using Mamba
			:return samp_scores: the average neighbor distances after filtering
			"""

			# Get center features (computed ONCE, reused for both choose and aggregation)
			if self.cuda:
				self_feats = self.features(torch.LongTensor(nodes).cuda())
			else:
				self_feats = self.features(torch.LongTensor(nodes))

			# Filter neighbors (Pick-Choose step)
			if train_flag:
				if self.use_neighbor_filter and self.choose_type == 'residual':
					# --- Batch-extract all neighbor features ONCE (used for both choose and aggregation) ---
					all_neigh_ids = set()
					for to_neigh in to_neighs_list:
						all_neigh_ids.update(to_neigh)

					if all_neigh_ids:
						neigh_ids_list = list(all_neigh_ids)
						if self.cuda:
							neigh_feats_all = self.features(torch.LongTensor(neigh_ids_list).cuda())
						else:
							neigh_feats_all = self.features(torch.LongTensor(neigh_ids_list))
						neigh_id_to_idx = {nid: idx for idx, nid in enumerate(neigh_ids_list)}
					else:
						neigh_feats_all = None
						neigh_id_to_idx = {}

					neigh_feats_for_choose = []
					for to_neigh in to_neighs_list:
						if to_neigh and neigh_feats_all is not None:
							indices = [neigh_id_to_idx[nid] for nid in to_neigh]
							neigh_feats_for_choose.append(neigh_feats_all[indices])
						else:
							device = self_feats.device
							neigh_feats_for_choose.append(torch.tensor([], device=device))

					# Dynamic alpha per node: [batch_size], each in (0, 0.5]
					dyn_alpha = self.alpha_gating(self_feats)

					samp_neighs, samp_scores = choose_step_neighs_residual(
						self_feats, batch_labels, neigh_scores, to_neighs_list,
						pos_scores, self.train_pos, sample_list, self.rho,
						neigh_feats_for_choose, dyn_alpha
					)
				elif not self.use_neighbor_filter:
					# Choose disabled: skip all filtering, use all neighbors directly
					samp_neighs, samp_scores = choose_step_none(
						batch_labels, to_neighs_list, pos_scores, self.train_pos,
						sample_list, self.rho
					)
				else:
					if self.choose_type == 'none':
						samp_neighs, samp_scores = choose_step_none(
							batch_labels, to_neighs_list, pos_scores, self.train_pos,
							sample_list, self.rho
						)
					else:
						samp_neighs, samp_scores = choose_step_neighs(
							batch_scores, batch_labels, neigh_scores, to_neighs_list,
							pos_scores, self.train_pos, sample_list, self.rho
						)
			else:
				if self.use_neighbor_filter and self.choose_type == 'residual':
					# --- Batch-extract all neighbor features ONCE (used for both choose and aggregation) ---
					all_neigh_ids = set()
					for to_neigh in to_neighs_list:
						all_neigh_ids.update(to_neigh)

					if all_neigh_ids:
						neigh_ids_list = list(all_neigh_ids)
						if self.cuda:
							neigh_feats_all = self.features(torch.LongTensor(neigh_ids_list).cuda())
						else:
							neigh_feats_all = self.features(torch.LongTensor(neigh_ids_list))
						neigh_id_to_idx = {nid: idx for idx, nid in enumerate(neigh_ids_list)}
					else:
						neigh_feats_all = None
						neigh_id_to_idx = {}

					neigh_feats_for_choose = []
					for to_neigh in to_neighs_list:
						if to_neigh and neigh_feats_all is not None:
							indices = [neigh_id_to_idx[nid] for nid in to_neigh]
							neigh_feats_for_choose.append(neigh_feats_all[indices])
						else:
							device = self_feats.device
							neigh_feats_for_choose.append(torch.tensor([], device=device))

					dyn_alpha = self.alpha_gating(self_feats)

					samp_neighs, samp_scores = choose_step_test_residual(
						self_feats, neigh_scores, to_neighs_list, sample_list,
						neigh_feats_for_choose, dyn_alpha
					)
				elif not self.use_neighbor_filter:
					# neighbor filtering disabled: skip all filtering, use all neighbors directly
					samp_neighs, samp_scores = choose_step_test_none(
						to_neighs_list, sample_list
					)
				elif self.choose_type == 'none':
					samp_neighs, samp_scores = choose_step_test_none(
						to_neighs_list, sample_list
					)
				else:
					samp_neighs, samp_scores = choose_step_test(
						batch_scores, neigh_scores, to_neighs_list, sample_list
					)

			# Build unique node index
			unique_nodes_list = list(set.union(*samp_neighs)) if samp_neighs else []
			unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

			# Get feature matrix for unique nodes (reuse self_feats; build embed_matrix)
			if self.cuda:
				embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda()) if unique_nodes_list else torch.empty(0, self.feat_dim, device='cuda')
			else:
				embed_matrix = self.features(torch.LongTensor(unique_nodes_list)) if unique_nodes_list else torch.empty(0, self.feat_dim)

			# Build padded neighbor sequences for Mamba
			batch_size = len(samp_neighs)
			max_len = max((len(neighs) for neighs in samp_neighs), default=1)
			max_len = min(max_len, 100)  # Cap at position encoding size

			# Ensure max_len >= d_conv=4 for causal_conv1d (required by mamba-ssm)
			max_len = max(max_len, 4)

			# Initialize padded tensor and lengths
			padded = torch.zeros(batch_size, max_len, self.feat_dim, device=embed_matrix.device, dtype=torch.float32)
			lengths = []

			for i, neighs in enumerate(samp_neighs):
				neigh_indices = [unique_nodes[n] for n in neighs if n in unique_nodes][:max_len]
				length = len(neigh_indices)
				lengths.append(length)
				if length > 0:
					padded[i, :length] = embed_matrix[neigh_indices]

			# Add positional encoding
			padded = padded + self.pos_encoder[:, :max_len, :]

			# Ensure contiguous float32 for mamba-ssm (causal_conv1d requirement)
			padded = padded.contiguous().float()

			# Mamba processes neighbor sequences with selective aggregation
			mamba_out = self.mamba(padded)  # (batch, max_len, feat_dim)

			# Masked mean pooling: average over valid (non-padded) positions
			agg_feats_list = []
			for i in range(batch_size):
				length = lengths[i]
				if length > 0:
					agg_feats_list.append(mamba_out[i, :length].mean(dim=0))
				else:
					# Fallback to mean pooling from IntraAgg for isolated nodes
					mask = Variable(torch.zeros(1, len(unique_nodes_list)))
					row_indices = [0] * len(unique_nodes_list)
					col_indices = list(range(len(unique_nodes_list)))
					mask[row_indices, col_indices] = 1
					if self.cuda:
						mask = mask.cuda()
					num_neigh = mask.sum(1, keepdim=True).clamp(min=1)
					mask = mask.div(num_neigh)
					agg_feats_list.append(mask.mm(embed_matrix).squeeze(0))

			agg_feats = torch.stack(agg_feats_list)

			# Concat with self features + linear transformation (same as IntraAgg)
			cat_feats = torch.cat((self_feats, agg_feats), dim=1)
			to_feats = F.relu(cat_feats.mm(self.weight))

			return to_feats, samp_scores


def choose_step_none(center_labels, neighs_list, minor_scores, minor_list, sample_list, sample_rate):
	"""
	No-op choose step: return all neighbors without filtering.
	Used for ablation study -- disable neighbor selection entirely.

	:param center_labels: the label of batch nodes
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param minor_scores: the label-aware scores for nodes of minority class (not used)
	:param minor_list: minority node id list for each batch node in one relation (not used)
	:param sample_list: the number of neighbors kept for each batch node (not used)
	:param sample_rate: the ratio of the oversample neighbors for the minority class
	:return samp_neighs: list of all neighbor sets for each batch node
	:return samp_score_diff: list of empty score lists (placeholder)
	"""
	samp_neighs = []
	samp_score_diff = []
	for idx, center_label in enumerate(center_labels):
		neighs_indices = neighs_list[idx]
		if not isinstance(neighs_indices, list):
			neighs_indices = (
				neighs_indices.tolist()
				if isinstance(neighs_indices, (torch.Tensor, np.ndarray))
				else list(neighs_indices)
			)

		selected_neighs = list(neighs_indices)
		selected_score_diff = []

		if center_label == 1:
			num_oversample = int(sample_list[idx] * sample_rate)
			selected_neighs.extend(minor_list[:num_oversample])
			selected_score_diff.extend([0.0] * num_oversample)

		samp_neighs.append(set(selected_neighs))
		samp_score_diff.append(selected_score_diff)

	return samp_neighs, samp_score_diff


def choose_step_test_none(neighs_list, sample_list):
	"""
	No-op choose step for test mode: return all neighbors without filtering.

	:param neighs_list: neighbor node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node (not used)
	:return samp_neighs: list of all neighbor sets for each batch node
	:return samp_scores: list of empty score lists (placeholder)
	"""
	samp_neighs = []
	samp_scores = []
	for neighs_indices in neighs_list:
		if not isinstance(neighs_indices, list):
			neighs_indices = (
				neighs_indices.tolist()
				if isinstance(neighs_indices, (torch.Tensor, np.ndarray))
				else list(neighs_indices)
			)
		samp_neighs.append(set(neighs_indices))
		samp_scores.append([])
	return samp_neighs, samp_scores


def choose_step_neighs(center_scores, center_labels, neigh_scores, neighs_list, minor_scores, minor_list, sample_list, sample_rate):
    """
    Choose step for neighborhood sampling
    :param center_scores: the label-aware scores of batch nodes
    :param center_labels: the label of batch nodes
    :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
    :param neighs_list: neighbor node id list for each batch node in one relation
	:param minor_scores: the label-aware scores for nodes of minority class in one relation
    :param minor_list: minority node id list for each batch node in one relation
    :param sample_list: the number of neighbors kept for each batch node in one relation
	:para sample_rate: the ratio of the oversample neighbors for the minority class
    """
    samp_neighs = []
    samp_score_diff = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score_neigh = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        if not isinstance(neighs_indices, list):
            neighs_indices = (
                neighs_indices.tolist()
                if isinstance(neighs_indices, (torch.Tensor, np.ndarray))
                else list(neighs_indices)
            )
        num_sample = sample_list[idx]

        # compute the L1-distance of batch nodes and their neighbors
        score_diff_neigh = torch.abs(center_score_neigh - neigh_score).squeeze()
        sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(score_diff_neigh, dim=0, descending=False)
        selected_neigh_indices = sorted_neigh_indices.tolist()

        # top-p sampling according to distance ranking
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_neigh_indices[:num_sample]]
            selected_score_diff = sorted_score_diff_neigh.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            selected_score_diff = score_diff_neigh.tolist()
            if isinstance(selected_score_diff, float):
                selected_score_diff = [selected_score_diff]

        if center_labels[idx] == 1:
            num_oversample = int(num_sample * sample_rate)
            center_score_minor = center_score.repeat(minor_scores.size()[0], 1)
            score_diff_minor = torch.abs(center_score_minor - minor_scores[:, 0].view(-1, 1)).squeeze()
            sorted_score_diff_minor, sorted_minor_indices = torch.sort(score_diff_minor, dim=0, descending=False)
            selected_minor_indices = sorted_minor_indices.tolist()
            selected_neighs.extend([minor_list[n] for n in selected_minor_indices[:num_oversample]])
            selected_score_diff.extend(sorted_score_diff_minor.tolist()[:num_oversample])

        samp_neighs.append(set(selected_neighs))
        samp_score_diff.append(selected_score_diff)

    return samp_neighs, samp_score_diff


def choose_step_test(center_scores, neigh_scores, neighs_list, sample_list):
	"""
	Filter neighbors according label predictor result with adaptive thresholds
	:param center_scores: the label-aware scores of batch nodes
	:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node in one relation
	:return samp_neighs: the neighbor indices and neighbor simi scores
	:return samp_scores: the average neighbor distances for each relation after filtering
	"""

	samp_neighs = []
	samp_scores = []
	for idx, center_score in enumerate(center_scores):
		center_score = center_scores[idx][0]
		neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
		center_score = center_score.repeat(neigh_score.size()[0], 1)
		neighs_indices = neighs_list[idx]
		if not isinstance(neighs_indices, list):
			neighs_indices = (
				neighs_indices.tolist()
				if isinstance(neighs_indices, (torch.Tensor, np.ndarray))
				else list(neighs_indices)
			)
		num_sample = sample_list[idx]

		# compute the L1-distance of batch nodes and their neighbors
		score_diff = torch.abs(center_score - neigh_score).squeeze()
		sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
		selected_indices = sorted_indices.tolist()

		# top-p sampling according to distance ranking and thresholds
		if len(neigh_scores[idx]) > num_sample + 1:
			selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
			selected_scores = sorted_scores.tolist()[:num_sample]
		else:
			selected_neighs = neighs_indices
			selected_scores = score_diff.tolist()
			if isinstance(selected_scores, float):
				selected_scores = [selected_scores]

		samp_neighs.append(set(selected_neighs))
		samp_scores.append(selected_scores)

	return samp_neighs, samp_scores


# ============================================================================
# Residual-based Choose Functions (for ablation study)
# ============================================================================


def choose_step_neighs_residual(center_feats, center_labels, neigh_scores, neighs_list,
								 minor_scores, minor_list, sample_list, sample_rate,
								 neigh_feats_list, dyn_alpha):
	"""
	Choose step using hybrid scoring: combine label prediction score distance
	with feature residual magnitude for neighbor selection.
	Two-stage selection: Stage 1 picks 2k candidates by score distance,
	Stage 2 re-ranks by hybrid score (score distance + residual magnitude).

	:param center_feats: the features of batch nodes
	:param center_labels: the label of batch nodes
	:param neigh_scores: the label-aware scores of 1-hop neighbors each batch node in one relation
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param minor_scores: the label-aware scores for nodes of minority class in one relation
	:param minor_list: minority node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node in one relation
	:param sample_rate: the ratio of the oversample neighbors for the minority class
	:param neigh_feats_list: precomputed neighbor features (list of tensors, one per batch node)
	:param dyn_alpha: per-node alpha tensor [batch_size], each in (0, 0.5]; or scalar float
	:return samp_neighs: list of selected neighbor sets for each batch node
	:return samp_score_diff: list of score differences for each batch node
	"""
	samp_neighs = []
	samp_score_diff = []

	for idx in range(len(center_feats)):
		center_feat = center_feats[idx]
		neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
		center_score_neigh = neigh_scores[idx][:, 0]
		center_score_rep = center_score_neigh.view(-1)
		# Per-node alpha from gating network
		alpha = dyn_alpha[idx].item() if dyn_alpha.dim() > 0 else dyn_alpha

		neighs_indices = neighs_list[idx]
		if not isinstance(neighs_indices, list):
			neighs_indices = (
				neighs_indices.tolist()
				if isinstance(neighs_indices, (torch.Tensor, np.ndarray))
				else list(neighs_indices)
			)
		num_sample = sample_list[idx]
		neigh_len = len(neighs_indices)

		if neigh_len == 0:
			samp_neighs.append(set())
			samp_score_diff.append([])
			continue

		# L1-distance of batch nodes and their neighbors
		score_diff_neigh = torch.abs(center_score_rep - neigh_score.view(-1))
		sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(score_diff_neigh, dim=0, descending=False)

		# Stage 1: coarse selection -- pick 2x candidates by score distance
		if len(neigh_scores[idx]) > num_sample + 1:
			candidate_k = min(2 * num_sample, neigh_len)
			candidate_local = sorted_neigh_indices[:candidate_k]
			candidate_vals = sorted_score_diff_neigh[:candidate_k]
			candidate_local_list = candidate_local.tolist()

			# Stage 2: fine selection -- re-rank by hybrid (score distance + residual)
			neigh_feats = neigh_feats_list[idx] if idx < len(neigh_feats_list) else None
			if neigh_feats is not None and neigh_feats.numel() > 0 and neigh_feats.dim() == 2 and len(candidate_local_list) > 0:
				cf = center_feat
				if cf.dim() == 1:
					cf = cf.unsqueeze(0)
				cand_feats = neigh_feats[candidate_local_list, :]
				if cand_feats.dim() == 1:
					cand_feats = cand_feats.unsqueeze(0)
				residual = cand_feats - cf.expand_as(cand_feats)
				res_mag = torch.norm(residual, p=2, dim=1)
				residual_norm = (res_mag - res_mag.min()) / (res_mag.max() - res_mag.min() + 1e-8)
				cv = candidate_vals
				if cv.dim() == 0:
					cv = cv.unsqueeze(0)
				score_norm = (cv - cv.min()) / (cv.max() - cv.min() + 1e-8)
				hybrid = (1 - alpha) * score_norm + alpha * residual_norm
				final_k = min(num_sample, len(candidate_local_list))
				_, final_local_idx = torch.topk(hybrid, final_k, largest=False)
				final_local = [candidate_local_list[i] for i in final_local_idx.tolist()]
				selected_neigh_indices = final_local
				selected_score_diff = hybrid[final_local_idx].tolist()
			else:
				selected_neigh_indices = candidate_local_list[:num_sample]
				selected_score_diff = candidate_vals.tolist()[:num_sample]

			selected_neighs = [neighs_indices[n] for n in selected_neigh_indices]
			if isinstance(selected_score_diff, float):
				selected_score_diff = [selected_score_diff]
		else:
			selected_neighs = neighs_indices
			selected_score_diff = sorted_score_diff_neigh.tolist()
			if isinstance(selected_score_diff, float):
				selected_score_diff = [selected_score_diff]

		# oversampling for minority class (positive nodes)
		if center_labels[idx] == 1 and minor_scores.numel() > 0:
			center_score_minor = center_score_neigh[0].item() if center_score_neigh.numel() > 0 else 0.0
			minor_score_flat = minor_scores[:, 0]
			score_diff_minor = torch.abs(minor_score_flat - center_score_minor)
			sorted_score_diff_minor, sorted_minor_indices = torch.sort(score_diff_minor, dim=0, descending=False)
			num_oversample = min(int(num_sample * sample_rate), len(minor_list))
			selected_minor_indices = sorted_minor_indices.tolist()[:num_oversample]
			selected_neighs.extend([minor_list[n] for n in selected_minor_indices])
			selected_score_diff.extend(sorted_score_diff_minor.tolist()[:num_oversample])

		samp_neighs.append(set(selected_neighs))
		samp_score_diff.append(selected_score_diff)

	return samp_neighs, samp_score_diff


def choose_step_test_residual(center_feats, neigh_scores, neighs_list, sample_list,
							   neigh_feats_list, dyn_alpha):
	"""
	Filter neighbors in test mode using two-stage hybrid scoring
	(score distance + feature residual). No oversampling.

	:param center_feats: the features of batch nodes
	:param neigh_scores: the label-aware scores of 1-hop neighbors each batch node in one relation
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node in one relation
	:param neigh_feats_list: precomputed neighbor features (list of tensors, one per batch node)
	:param dyn_alpha: per-node alpha tensor [batch_size], each in (0, 0.5]; or scalar float
	:return samp_neighs: list of selected neighbor sets for each batch node
	:return samp_scores: list of hybrid scores for each batch node
	"""
	samp_neighs = []
	samp_scores = []

	for idx in range(len(center_feats)):
		center_feat = center_feats[idx]
		neigh_score = neigh_scores[idx][:, 0]
		center_score_val = neigh_score[0].item() if neigh_score.numel() > 0 else 0.0
		# Per-node alpha from gating network
		alpha = dyn_alpha[idx].item() if dyn_alpha.dim() > 0 else dyn_alpha

		neighs_indices = neighs_list[idx]
		if not isinstance(neighs_indices, list):
			neighs_indices = (
				neighs_indices.tolist()
				if isinstance(neighs_indices, (torch.Tensor, np.ndarray))
				else list(neighs_indices)
			)
		num_sample = sample_list[idx]
		neigh_len = len(neighs_indices)

		if neigh_len == 0:
			samp_neighs.append(set())
			samp_scores.append([])
			continue

		score_diff = torch.abs(neigh_score - center_score_val)
		sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)

		# Stage 1: coarse selection -- pick 2x candidates by score distance
		if len(neigh_scores[idx]) > num_sample + 1:
			candidate_k = min(2 * num_sample, neigh_len)
			candidate_local = sorted_indices[:candidate_k]
			candidate_vals = sorted_scores[:candidate_k]
			candidate_local_list = candidate_local.tolist()

			# Stage 2: fine selection -- re-rank by hybrid score
			neigh_feats = neigh_feats_list[idx] if idx < len(neigh_feats_list) else None
			if neigh_feats is not None and neigh_feats.numel() > 0 and neigh_feats.dim() == 2 and len(candidate_local_list) > 0:
				cf = center_feat
				if cf.dim() == 1:
					cf = cf.unsqueeze(0)
				cand_feats = neigh_feats[candidate_local_list, :]
				if cand_feats.dim() == 1:
					cand_feats = cand_feats.unsqueeze(0)
				residual = cand_feats - cf.expand_as(cand_feats)
				res_mag = torch.norm(residual, p=2, dim=1)
				residual_norm = (res_mag - res_mag.min()) / (res_mag.max() - res_mag.min() + 1e-8)
				cv = candidate_vals
				if cv.dim() == 0:
					cv = cv.unsqueeze(0)
				score_norm = (cv - cv.min()) / (cv.max() - cv.min() + 1e-8)
				hybrid = (1 - alpha) * score_norm + alpha * residual_norm
				final_k = min(num_sample, len(candidate_local_list))
				_, final_local_idx = torch.topk(hybrid, final_k, largest=False)
				final_local = [candidate_local_list[i] for i in final_local_idx.tolist()]
				selected_neighs = [neighs_indices[n] for n in final_local]
				selected_scores = hybrid[final_local_idx].tolist()
			else:
				selected_neighs = [neighs_indices[n] for n in candidate_local_list[:num_sample]]
				selected_scores = candidate_vals.tolist()[:num_sample]
		else:
			selected_neighs = neighs_indices
			selected_scores = sorted_scores.tolist()
			if isinstance(selected_scores, float):
				selected_scores = [selected_scores]

		samp_neighs.append(set(selected_neighs))
		samp_scores.append(selected_scores)

	return samp_neighs, samp_scores
