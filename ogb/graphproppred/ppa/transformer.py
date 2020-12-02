import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eig



class TopologicalEncoding():
    def __init__(self, encoding_size=50):
        self.encoding_size = encoding_size

    def encode(self, graph):
        A = graph.adjacency_matrix().to_dense()
        D = torch.diag(graph.in_degrees())
        L = (D - A).numpy()
        w, V = eig(L)
        momenta = np.dot(L, V)
        clipping = self.encoding_size - graph.num_nodes()

        if clipping > 0:
            momenta_ = []
            for momentum in momenta:
                momenta_.append(np.pad(momentum, (0, clipping)))
            momenta = np.array(momenta_)
        elif clipping < 0:
            momenta = momenta[:, :self.encoding_size]
        return torch.FloatTensor(np.real(momenta))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, scores = self.self_attn(src, src, src, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, scores


class Graphormer(nn.Module):
    def __init__(self, d_model=128, depth=5, nhead=8, expansion_factor=2,
                 alphabet_size=100, encoding_size=32, device='cpu', concatenate_encoding=True):

        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.nhead = nhead
        self.dim_feedforward = d_model * expansion_factor

        self.concatenate_encoding = concatenate_encoding

        if concatenate_encoding:
            self.encoder = TopologicalEncoding(encoding_size)
            self.node_embedder = nn.Embedding(alphabet_size, d_model - encoding_size)
        else:
            self.node_embedder = nn.Embedding(alphabet_size, d_model)
            self.encoder = TopologicalEncoding(d_model)

        self.blocks = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                             dim_feedforward=self.dim_feedforward)
                                     for _ in range(depth)])

    def forward(self, graphs, need_weights=False):
        encoding = torch.cat([self.encoder.encode(graph) for graph in dgl.unbatch(graphs)], dim=0)
        embedding = self.node_embedder(graphs.ndata['atomic'].type(torch.long))
        if self.concatenate_encoding:
            graphs.ndata['h'] = torch.cat((encoding, embedding.squeeze()), dim=-1)
        else:
            graphs.ndata['h'] = encoding + embedding.squeeze()

        batch = []
        for g in dgl.unbatch(graphs):
            batch.append(g.ndata['h'])
        h = torch.nn.utils.rnn.pad_sequence(batch)

        attentions_ = []
        for block in self.blocks:
            h, att_ = block(h)
            if need_weights: attentions_.append(att_)

        truncated = [h[:num_nodes, i, :] for i, num_nodes in enumerate(graphs.batch_num_nodes())]
        h = torch.cat(truncated, dim=0)

        if need_weights:
            return h, attentions_
        return h