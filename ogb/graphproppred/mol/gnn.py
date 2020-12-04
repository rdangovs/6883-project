import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean
import numpy as np

from models.transformers import ControllerTransformer
from torch.nn.utils.rnn import pad_sequence


class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=5, emb_dim=300,
                 gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 transformers=False, controller=False):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        if transformers:
            self.transformer_layers = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=emb_dim,
                                                                                            dim_feedforward=emb_dim * 4,
                                                                                            nhead=4) for _ in range(2)])
        if controller:
            self.controller_layers = torch.nn.ModuleList([ControllerTransformer(depth=2,
                                                                                expansion_ratio=4,
                                                                                n_heads=4,
                                                                                s2g_sharing=True,
                                                                                in_features=300,
                                                                                out_features=1,
                                                                                set_fn_feats=[256, 256, 256, 256, 5],
                                                                                method='lin2',
                                                                                hidden_mlp=[256],
                                                                                predict_diagonal=False,
                                                                                attention=True) for _ in range(2)])
        self.transformers = transformers
        self.controller = controller

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                                 gnn_type=gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                     gnn_type=gnn_type)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data, perturb=None):
        h_node = self.gnn_node(batched_data, perturb)

        sizes = batched_data.__num_nodes_list__
        cumsum = np.cumsum([0] + sizes[:-1])
        hs = [h_node[start:start + delta] for start, delta in
              zip(cumsum, sizes)]
        h_node = pad_sequence(hs)
        if self.controller:
            for layer in self.controller_layers:
                h_node = layer(h_node)
        elif self.transformers:
            for layer in self.transformer_layers:
                h_node = layer(h_node)
        h_node = h_node.transpose(0, 1)

        h_node = torch.cat([h_n[:size] for h_n, size in zip(h_node, sizes)], dim=0)
        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    GNN(num_tasks=10)
