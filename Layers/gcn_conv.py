from typing import Optional
import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value


@torch.jit._overload
def gcn_norm(edge_index, edge_weight, num_nodes, improved, add_self_loops,
             flow, dtype):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int],Optional[str],Optional[Tensor]) -> OptPairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight, num_nodes, improved, add_self_loops,
             flow, dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int],Optional[str],Optional[Tensor]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None,mode="sym",lambda_=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        # deg_inv_sqrt = deg.pow_(-0.5)
        # deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        # value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        if mode == "sym":
            # D^{-1/2}AD^{-1/2} symmetric normalization
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]
        elif mode == "rw":
            # D^{-1}A normalization
            deg_inv_sqrt = deg.pow_(-1)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            value = deg_inv_sqrt[col] * value
        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    # lambda_ = nn.Parameter(torch.FloatTensor(1)).to(edge_weight.device)
    # lambda_.data.fill_(1)
    # lambda_ = lambda_.data
    if mode == "sym":
        # D^{-1/2}AD^{-1/2} symmetric normalization
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif mode == "rw":
        # D^{-1}A normalization
        lambda_ = 1 # hyperparameter 1
        # lambda_ = lamb # hyperparameter 2
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[col] * edge_weight
        I = torch.ones_like(edge_weight).to(edge_weight.device)
        edge_weight = (1 - lambda_) * I + lambda_ *  edge_weight
    return edge_index, edge_weight


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.layer_index = kwargs['layer_index']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.gcn_norm_type = kwargs['gcn_norm_type']
        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        self.bn_ACM1 = torch.nn.BatchNorm1d(out_channels)
        self.bn_ACM2 = torch.nn.BatchNorm1d(out_channels)

        # self.lambda_ = nn.Parameter(torch.FloatTensor(1))
        # self.lambda_.data.fill_(1)
        # if self.with_ACM:

            # self.w_for_norm_first_layer = nn.Parameter(torch.FloatTensor(1, self.dim_hidden))
            # stdv_for_norm = 1. / np.sqrt(self.w_for_norm_first_layer.size(1))
            # self.w_for_norm_first_layer.data.uniform_(-stdv_for_norm, stdv_for_norm)

        self.w_for_norm = nn.Parameter(torch.FloatTensor(1, out_channels))
        stdv_for_norm = 1. / np.sqrt(self.w_for_norm.size(1))
        self.w_for_norm.data.uniform_(-stdv_for_norm, stdv_for_norm)

            # self.w_for_norm_last_layer = nn.Parameter(torch.FloatTensor(1, self.num_classes))
            # stdv_for_norm = 1. / np.sqrt(self.w_for_norm_last_layer.size(1))
            # self.w_for_norm_last_layer.data.uniform_(-stdv_for_norm, stdv_for_norm)

        # else:
        #     self.w_for_norm = None
        #     self.w_for_norm_last_layer = None

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, w_for_norm=None,**kwargs) -> Tensor:
        # if self.w_for_norm != None:
        #     w_for_norm = self.w_for_norm
        # lambda_ = kwargs['lambda_']
        lambda_ = 1
        # print('lambda_:',lambda_)
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype,mode=self.gcn_norm_type,lambda_=lambda_)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype,mode=self.gcn_norm_type,lambda_=lambda_)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if w_for_norm is None:
            x = self.lin(x)
            out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                 size=None)
            if self.bias is not None:
                out = out + self.bias
            return out

        else:
            w_for_norm.data = w_for_norm.abs()
            # w_for_norm = torch.ones_like(w_for_norm)
            if self.layer_index == 0:
                # p0_in, a = p0_generate(w_for_norm, self.in_channels)
                # x = push_forward(x, p0_in, a, b=0.9 * a)
                x = self.lin(x)
                # p0_out, a = p0_generate(w_for_norm, self.out_channels)
                # x = x - torch.mean(x, dim=0)
                # x = push_back(x, p0_out, w_for_norm)
            elif self.layer_index == -1:
                # p0_in, a = p0_generate(w_for_norm, self.in_channels)
                # x = push_forward(x, p0_in, a, b=0.9 * a)
                x = self.lin(x)
            else:
                # p0, a = p0_generate(w_for_norm, self.out_channels)
                # x = push_forward(x, p0, a, b=0.9 * a)
                x = self.lin(x)
                # x = push_back(x, p0, w_for_norm)
            if self.layer_index <=1:
                x = self.bn_ACM1(x)
                # x = x - torch.mean(x, dim=0)
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                 size=None)

            if self.bias is not None:
                out = out + self.bias
            if self.layer_index == -1:
                pass
            else:
                # out = out - torch.mean(out, dim=0)
                if self.layer_index <=1:
                    out = self.bn_ACM2(out)
                    # out = out - torch.mean(out, dim=0)
                out = RiemannAgg(out, w_for_norm)

            return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)




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

def push_back(x, p0, w_for_norm):
    f_p0_v_Numerator = -2 * (x - p0) * w_for_norm @ p0.t()
    f_p0_v_denominator = (x - p0) * w_for_norm @ (x - p0).t()
    f_p0_v_denominator = torch.diag(f_p0_v_denominator).unsqueeze(dim=1)
    f_p0_v = f_p0_v_Numerator / (f_p0_v_denominator + 1e-4)
    x_tmp = f_p0_v * (x - p0) + p0
    return x_tmp


def p0_generate(w_for_norm, out_channels):
    p0 = torch.zeros((1, out_channels)).cuda()
    a = 1 / torch.sqrt(w_for_norm[0][0] + 1e-4)
    p0[0][0] = a.item()
    return p0, a


def push_forward(x, p0, a, b=0):
    x_tmp = x
    w1 = x[:, 0]
    g_po_w = (b - a) / (w1 - a + 1e-4)
    g_po_w = g_po_w.unsqueeze(dim=1)
    Q_p0_w = g_po_w * (x_tmp - p0) + p0
    return Q_p0_w