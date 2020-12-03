import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F

import os
import sys

# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# building blocks to CONTROLLER
from models.set_to_graph import SetToGraph
from models.deep_sets import DeepSet


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class ControllerTransformer(nn.Module):
    def __init__(self, depth, expansion_ratio, n_heads, s2g_sharing, in_features,
                 out_features, set_fn_feats, method, hidden_mlp,
                 predict_diagonal, mask_scale, attention, cfg=None, device='cuda'):
        """
        ControllerTransformer model.
        :param depth: [int], depth of the transformer
        :param expansion_ratio: [int], expansion ratio for the ff layer in the Transformer Encoder
        :param n_heads: [int], number of heads
        :param s2g_sharing: [bool] whether to do s2g parameter sharing
        :param in_features: input set's number of features per data point
        :param out_features: number of output features.
        :param set_fn_feats: list of number of features for the output of each deepsets layer
        :param method: transformer method - quad, lin2 or lin5
        :param hidden_mlp: list[int], number of features in hidden layers mlp.
        :param predict_diagonal: Bool. True to predict the diagonal (diagonal needs a separate psi function).
        :param attention: Bool. Use attention in DeepSets
        :param cfg: configurations of using second bias in DeepSetLayer, normalization method and aggregation for lin5.
        """
        super(ControllerTransformer, self).__init__()
        self.depth = depth
        self.n_heads = n_heads
        self.s2g_sharing = s2g_sharing
        self.mask_scale = mask_scale
        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=in_features,
                                                                            dim_feedforward=in_features * expansion_ratio,
                                                                            nhead=self.n_heads) for _ in range(depth)])
        self.s2g_layers = nn.ModuleList([nn.ModuleList([SetToGraph(in_features=in_features,
                                                                   out_features=out_features,
                                                                   set_fn_feats=[256, 256, 5],
                                                                   method=method,
                                                                   hidden_mlp=hidden_mlp,
                                                                   predict_diagonal=predict_diagonal,
                                                                   attention=attention,
                                                                   cfg=cfg,
                                                                   ) for _ in range(n_heads)]) for _ in
                                         range(depth if not s2g_sharing else 1)])
        self.s2g_head = SetToGraph(in_features=in_features,
                                   out_features=out_features,
                                   set_fn_feats=set_fn_feats,
                                   method=method,
                                   hidden_mlp=hidden_mlp,
                                   predict_diagonal=predict_diagonal,
                                   attention=attention,
                                   cfg=cfg,
                                   )

    def forward(self, x: Tensor) -> Tensor:
        # going through the Transformer layers below
        for i in range(self.depth):
            mask_list = []
            for j in range(self.n_heads):
                # TODO: you don't have to loop. You can batch these and make it efficient!
                mask_logits = self.s2g_layers[i if not self.s2g_sharing else -1][j](x)
                mask = torch.tanh(mask_logits) * self.mask_scale  # TODO: tune this parameter well
                mask_list.append(mask)  # TODO: interpretation is that S2G predicts what to ignore
            mask = torch.cat(mask_list, dim=0)  # concatenating over the batch dimension
            # print(torch.min(mask), torch.max(mask))
            mask = mask.squeeze(dim=1)   # TODO: the mask is the information bottleneck: fix that!
            x = x.transpose(0, 1)
            x = self.transformer_layers[i](x, src_mask=mask)
            x = x.transpose(0, 1)
        # final S2G prediction
        return self.s2g_head(x)


if __name__ == '__main__':
    encoder_layer = ControllerTransformer(depth=2,
                                          expansion_ratio=2,
                                          n_heads=4,
                                          s2g_sharing=True,
                                          in_features=10,
                                          out_features=1,
                                          set_fn_feats=[256, 256, 256, 256, 5],
                                          method='lin2',
                                          hidden_mlp=[256],
                                          predict_diagonal=False,
                                          attention=True).cuda()
    print(encoder_layer)
    src = torch.rand(512, 32, 10).cuda()
    out = encoder_layer(src)
    print(out)
