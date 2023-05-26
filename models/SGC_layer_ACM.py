from typing import Optional
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
from Layers.gcn_conv import gcn_norm
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x
class SGConv(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}` on
            first execution, and will use the cached version for further
            executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, bn: bool = True, pn: bool = True, dropout: float = 0.,
                 lin_first: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SGConv, self).__init__(**kwargs)
        self.args = kwargs['args']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.bn = bn
        self.pn = pn
        self.dropout = dropout
        self.lin_first = lin_first
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(self.in_channels)
        if self.pn:
            self.pn = pair_norm()
        if self.lin_first:
            self.cached = False
        self.gcn_norm_type = self.args.gcn_norm_type
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, w_for_norm=None, **kwargs):

        if self.lin_first:
            x = self.lin(x)

        """"""
        # self.bn_ACM = kwargs["bn_ACM"]
        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype,mode=self.gcn_norm_type,)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype,mode=self.gcn_norm_type,)
            if w_for_norm != None:
                w_for_norm.data = w_for_norm.abs()
                print("w_for_norm", w_for_norm)
                # w_for_norm = torch.ones_like(w_for_norm)
                # https://zhuanlan.zhihu.com/p/433407462
                w_for_norm = w_for_norm.data
            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                if w_for_norm != None and k == 0:
                    w_for_norm.data = w_for_norm.abs()
                    print("w_for_norm", w_for_norm)
                    # x = x - torch.mean(x, dim=0)
                    # x = self.bn_ACM[k](x)
                    # x = x.data
                    x = RiemannAgg(x, w_for_norm)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                if k != self.K - 1 and w_for_norm != None:
                    # x = x - torch.mean(x, dim=0)
                    # x = self.bn_ACM[k](x)
                    # x = x.data
                    x = RiemannAgg(x, w_for_norm)

            if self.cached:
                self._cached_x = x
        else:
            x = cache

        if self.bn:
            x = self.bn(x)
        if self.pn:
            x = self.pn(x)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if not self.lin_first:
            x = self.lin(x)

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)


# def RiemannAgg(x, w):
#     squar_x = torch.square(x)
#     squar_x_w = torch.mul(squar_x, w)
#     sum_squar_x_w = torch.sum(squar_x_w, dim=1)
#     sqrt_x_w = torch.sqrt(sum_squar_x_w + 1e-6)
#     sqrt_x_w = torch.unsqueeze(sqrt_x_w, dim=1)
#     x = torch.div(x, sqrt_x_w)
#     return x

def RiemannAgg(x, w):

    squar_x = torch.square(x)
    squar_x_w = torch.mul(squar_x, w)
    sum_squar_x_w = torch.sum(squar_x_w, dim=1)
    sqrt_x_w = torch.sqrt(sum_squar_x_w + 1e-2)
    sqrt_x_w = torch.unsqueeze(sqrt_x_w, dim=1)
    x = torch.div(x, sqrt_x_w + 1e-2 ) # + 1

    return x