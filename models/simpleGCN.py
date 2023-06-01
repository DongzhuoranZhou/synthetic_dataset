from torch import nn
import torch.nn.functional as F
from models.common_blocks import batch_norm
from torch_geometric.nn.inits import glorot
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import scatter
from torch.nn import Parameter
import numpy as np
class simpleGCN(nn.Module):
    def __init__(self, args):
        super(simpleGCN, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden

        self.dropout = args.dropout
        self.type_norm = args.type_norm
        self.skip_weight = args.skip_weight
        self.num_groups = args.num_groups
        self.norm_weight = None
        self.aggr = 'add'

        self.layers_activation = torch.nn.functional.relu
        self.layers_bn = nn.ModuleList([])
        self.weight = Parameter(torch.Tensor(self.num_feats, self.num_classes))
        glorot(self.weight)


        if self.with_ACM:
            self.w_for_norm = nn.Parameter(torch.FloatTensor(1, self.num_classes))
            stdv_for_norm = 1. / np.sqrt(self.w_for_norm.size(1))
            self.w_for_norm.data.uniform_(-stdv_for_norm, stdv_for_norm)

        for i in range(self.num_layers):
            # if self.type_norm in ['None', 'batch', 'pair']:
            #     skip_connect = False
            # else:
            #     skip_connect = True
            self.layers_bn.append(batch_norm(self.num_classes, self.type_norm, self.num_groups, self.skip_weight))
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)
    def norm(self, x, edge_index):
        edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index):

        if self.norm_weight is None:
            self.norm_weight = self.norm(x, edge_index)
            self.norm_weight = self.norm_weight.view(-1, 1)
        norm = self.norm_weight

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.mm(x, self.weight)

        for i in range(self.num_layers):
            if self.with_ACM:
                self.w_for_norm.data = self.w_for_norm.abs()
            x_j = x.index_select(0, edge_index[0])
            # x_conv = scatter_(self.aggr, norm * x_j, edge_index[1], 0, x.size(0))
            x_conv = scatter(norm * x_j, edge_index[1], 0, x.size(0), reduce=self.aggr)
            if self.with_ACM:
                x_conv = RiemannAgg(x_conv, self.w_for_norm)
            x = self.layers_bn[i](x_conv)

        return x





def RiemannAgg(x, w):

    squar_x = torch.square(x)
    squar_x_w = torch.mul(squar_x, w)
    sum_squar_x_w = torch.sum(squar_x_w, dim=1)
    sqrt_x_w = torch.sqrt(sum_squar_x_w + 1e-2)
    sqrt_x_w = torch.unsqueeze(sqrt_x_w, dim=1)
    x = torch.div(x, sqrt_x_w + 1e-2 ) # + 1

    return x
