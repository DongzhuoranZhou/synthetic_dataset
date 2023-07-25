import torch
import torch.nn.functional as F
from torch import nn
# from torch_geometric.nn import GCNConv
from Layers.gcn_conv import GCNConv
import numpy as np
from utils.train_utils import AcontainsB
from tricks.skipConnection import InitialConnection, DenseConnection, ResidualConnection
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


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.layers_res = nn.ModuleList([])
        # self.bn_ACM = nn.ModuleList([])
        # for i in range(self.num_layers - 1):
        #     self.bn_ACM.append(torch.nn.BatchNorm1d(self.in_channels))

        self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached,layer_index=0,gcn_norm_type=self.gcn_norm_type,normalize=self.normalize))
        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())
        elif self.type_norm == 'group':
            self.layers_bn.append(batch_norm(self.dim_hidden, self.type_norm, self.num_groups, self.skip_weight))

        if AcontainsB(self.type_trick, ['Residual']):
            self.layers_res.append(ResidualConnection(alpha=self.alpha))
        for i in range(self.num_layers - 2):
            self.layers_GCN.append(
                GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached,layer_index=0,gcn_norm_type=self.gcn_norm_type,normalize=self.normalize))

            if self.type_norm == 'batch':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
            elif self.type_norm == 'pair':
                self.layers_bn.append(pair_norm())
            elif self.type_norm == 'group':
                self.layers_bn.append(batch_norm(self.dim_hidden, self.type_norm, self.num_groups, self.skip_weight))

            if AcontainsB(self.type_trick, ['Residual']):
                self.layers_res.append(ResidualConnection(alpha=self.alpha))
        self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached,layer_index=-1,gcn_norm_type=self.gcn_norm_type,normalize=self.normalize))

        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())
        elif self.type_norm == 'group':
            self.layers_bn.append(batch_norm(self.dim_hidden, self.type_norm, self.num_groups, self.skip_weight))
        if AcontainsB(self.type_trick, ['Residual']):
            self.layers_res.append(ResidualConnection(alpha=self.alpha))
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)

        if self.with_ACM:

            self.w_for_norm_first_layer = nn.Parameter(torch.FloatTensor(1, self.dim_hidden))
            stdv_for_norm = 1. / np.sqrt(self.w_for_norm_first_layer.size(1))
            self.w_for_norm_first_layer.data.uniform_(-stdv_for_norm, stdv_for_norm)

            self.w_for_norm = nn.Parameter(torch.FloatTensor(1, self.dim_hidden))
            stdv_for_norm = 1. / np.sqrt(self.w_for_norm.size(1))
            self.w_for_norm.data.uniform_(-stdv_for_norm, stdv_for_norm)

            self.w_for_norm_last_layer = nn.Parameter(torch.FloatTensor(1, self.num_classes))
            stdv_for_norm = 1. / np.sqrt(self.w_for_norm_last_layer.size(1))
            self.w_for_norm_last_layer.data.uniform_(-stdv_for_norm, stdv_for_norm)

        else:
            self.w_for_norm = None
            self.w_for_norm_last_layer = None

        self.lambda_ = nn.Parameter(torch.FloatTensor(1))
        self.lambda_.data.fill_(1)
    def forward(self, x, edge_index):

        # implemented based on DeepGCN: https://github.com/LingxiaoShawn/PairNorm/blob/master/models.py
        x_list = []
        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index,w_for_norm=self.w_for_norm,args=self.args,layer=i,lambda_=self.lambda_)
            if self.type_norm in ['batch', 'pair', 'group']:
                x = self.layers_bn[i](x)
            x = F.relu(x)
            x_list.append(x)
            if AcontainsB(self.type_trick, ['Initial', 'Dense', 'Residual']):
                x = self.layers_res[i](x_list)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers_GCN[-1](x, edge_index,w_for_norm=self.w_for_norm_last_layer,args=self.args,layer=self.num_layers-1,lambda_=self.lambda_)
        return x
