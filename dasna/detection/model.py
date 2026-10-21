# Modified for DASNA: project naming and package imports.
# Source and component licenses: THIRD_PARTY_NOTICES.md.
import torch
import torch.nn as nn
from torch.nn import init


"""
	DASNA classification layer.
	For inherited component provenance, see THIRD_PARTY_NOTICES.md.
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class DASNALayer(nn.Module):
	"""
	One Pick-Choose-Aggregate layer
	"""

	def __init__(self, num_classes, inter1, lambda_1):
		"""
		Initialize the DASNA model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(DASNALayer, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1
		self.epsilon = 0.1

	def forward(self, nodes, labels, train_flag=True):
		embeds1, label_scores = self.inter1(nodes, labels, train_flag)
		scores = self.weight.mm(embeds1)
		return scores.t(), label_scores

	def to_prob(self, nodes, labels, train_flag=True):
		gnn_logits, label_logits = self.forward(nodes, labels, train_flag)
		# logits shape: [batch, num_classes] (after .t() in forward) -> softmax over class dim
		gnn_scores = torch.softmax(gnn_logits, dim=1)
		label_scores = torch.softmax(label_logits, dim=1)
		return gnn_scores, label_scores

	def loss(self, nodes, labels, train_flag=True):
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
		# Simi loss, Eq. (7) in the paper
		label_loss = self.xent(label_scores, labels.reshape(-1))
		# GNN loss, Eq. (10) in the paper
		gnn_loss = self.xent(gnn_scores, labels.reshape(-1))
		# Joint classification and auxiliary label prediction loss.
		final_loss = gnn_loss + self.lambda_1 * label_loss
		return final_loss
