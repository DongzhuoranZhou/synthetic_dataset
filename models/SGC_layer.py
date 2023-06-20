from typing import Optional
from torch_geometric.typing import Adj, OptTensor

from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
import torch.nn.functional as F
from torch import nn
from models.common_blocks import batch_norm

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
                 bias: bool = True, bn: bool = True, pn: bool = True, gn: bool = True, dropout: float = 0.,normalize: bool = True,
                 lin_first: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SGConv, self).__init__(**kwargs)
        args = kwargs['args']
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.bn = bn
        self.pn = pn
        self.gn = gn
        self.layers_bn = nn.ModuleList([])
        self.dropout = dropout
        self.lin_first = lin_first
        if self.bn:
            # self.bn = torch.nn.BatchNorm1d(self.in_channels)
            for k in range(self.K):
                self.layers_bn.append(torch.nn.BatchNorm1d(self.in_channels))
            # self.bn1 = torch.nn.BatchNorm1d(self.in_channels)
            # self.bn2 = torch.nn.BatchNorm1d(self.in_channels)
        if self.pn:
            for k in range(self.K):
                self.layers_bn.append(pair_norm())
        if self.gn:
            for k in range(self.K):
                self.layers_bn.append(batch_norm(self.in_channels, self.type_norm, self.num_groups, self.skip_weight))
        if self.lin_first:
            self.cached = False
        self.cached = False
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.lin_first:
            x = self.lin(x)

        """"""
        cache = self._cached_x
        # print("cache: ", cache)
        if cache is None:
            # if self.normalize:
            if isinstance(edge_index, Tensor):
                if self.normalize:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                elif edge_weight is None:
                    edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype,
                                             device=edge_index.device)
            elif isinstance(edge_index, SparseTensor):
                if self.normalize:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                elif edge_weight is None:
                    edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype,
                                             device=edge_index.device)
            # elif edge_weight is None:
            #         edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype,
            #                                  device=edge_index.device)
            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                if self.bn:
                    # x = self.bn(x)
                    x = self.layers_bn[k](x)
                    # x = x.detach()
                    # if k == 0:
                    #     x = self.bn1(x)
                    # elif k == 1:
                    #     x = self.bn2(x)
                # x = self.bn1(x)
                elif self.pn:
                    x = self.layers_bn[k](x)
                elif self.gn:
                    x = self.layers_bn[k](x)

            if self.cached:
                self._cached_x = x
        else:
            x = cache

        # if self.bn:
        #     x = self.bn(x)
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
