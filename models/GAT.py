import torch
import torch.nn.functional as F
# from torch_geometric.nn import GATConv
from Layers.gat_conv import GATConv
import torch.nn as nn
import numpy as np
from torch.nn import Linear
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

class GAT(torch.nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = torch.nn.ModuleList([])
        self.layers_bn = torch.nn.ModuleList([])

        if self.with_ACM:
            self.lin = Linear(self.num_feats, self.num_classes, bias=True)
            # self.w_for_norm_first_layer = nn.Parameter(torch.FloatTensor(1, self.dim_hidden))
            # stdv_for_norm = 1. / np.sqrt(self.w_for_norm_first_layer.size(1))
            # self.w_for_norm_first_layer.data.uniform_(-stdv_for_norm, stdv_for_norm)

            self.w_for_norm = nn.Parameter(torch.FloatTensor(1, self.num_feats))
            stdv_for_norm = 1. / np.sqrt(self.w_for_norm.size(1))
            self.w_for_norm.data.uniform_(-stdv_for_norm, stdv_for_norm)

            # self.w_for_norm_last_layer = nn.Parameter(torch.FloatTensor(1, self.num_classes))
            # stdv_for_norm = 1. / np.sqrt(self.w_for_norm_last_layer.size(1))
            # self.w_for_norm_last_layer.data.uniform_(-stdv_for_norm, stdv_for_norm)

        else:
            self.w_for_norm = None
            self.w_for_norm_last_layer = None
        # space limit
        if self.dataset == 'obgn-arxiv':
            self.dim_hidden = 1

        self.layers_GCN.append(GATConv(self.num_feats, self.dim_hidden,
                                       bias=False, concat=False,num_feats=self.num_feats,with_ACM=self.with_ACM))
        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())

        for _ in range(self.num_layers - 2):
            self.layers_GCN.append(
                GATConv(self.dim_hidden, self.dim_hidden, bias=False, concat=False,num_feats=self.num_feats,with_ACM=self.with_ACM))
            if self.type_norm == 'batch':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
            elif self.type_norm == 'pair':
                self.layers_bn.append(pair_norm())
        self.layers_GCN.append(GATConv(self.dim_hidden, self.num_classes, bias=False, concat=False,num_feats=self.num_feats,with_ACM=self.with_ACM))
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)

        # for i in range(self.num_layers):
        #     dim_out = self.layers_GCN[i].out_channels
        #     # if self.type_norm in ['None', 'batch', 'pair']:
        #     #     skip_connect = False
        #     # else:
        #     #     skip_connect = True
        #     self.layers_bn.append(batch_norm(dim_out, self.type_norm, self.num_groups, self.skip_weight))

    def forward(self, x, edge_index):
        if self.with_ACM:

            for i in range(self.num_layers - 1):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.layers_GCN[i](x, edge_index,w_for_norm=self.w_for_norm)
                if self.type_norm == 'batch' or self.type_norm == 'pair':
                    x = self.layers_bn[i](x)
                x = F.tanh(x)
            x = self.layers_GCN[-1](x, edge_index,w_for_norm=self.w_for_norm)
            x = self.lin(x)
            return x

        else:
            for i in range(self.num_layers - 1):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.layers_GCN[i](x, edge_index)
                if self.type_norm == 'batch' or self.type_norm == 'pair':
                    x = self.layers_bn[i](x)
                x = F.relu(x)

            x = self.layers_GCN[-1](x, edge_index)
            return x
