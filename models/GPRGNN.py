#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn as nn
from Layers.gcn_conv import gcn_norm
class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None,w_for_norm=None, **kwargs):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        if w_for_norm != None:
            w_for_norm.data = w_for_norm.abs()
            # w_for_norm = torch.ones_like(w_for_norm)
            # w_for_norm = w_for_norm.data # / torch.sum(w_for_norm.data)

        if w_for_norm != None:
            # x = x.data
            # x = x - torch.mean(x, dim=0)
            x = RiemannAgg(x, w_for_norm)
        hidden = x * (self.temp[0])
        for k in range(self.K):

            x = self.propagate(edge_index, x=x, norm=norm)

            if w_for_norm != None:
                if torch.isnan(x).any():
                    print('x is nan')
                # x = x - torch.mean(x, dim=0)
                # x = x.data
                # w_for_norm.data = w_for_norm.abs()
                x = RiemannAgg(x, w_for_norm)
                if torch.isnan(x).any():
                    print('x is nan')
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, args):
        super(GPRGNN, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.Init = 'PPR'
        # layers
        self.lin1 = Linear(self.num_feats, self.dim_hidden)
        self.lin2 = Linear(self.dim_hidden, self.num_classes)
        self.prop1 = GPR_prop(self.num_layers, self.alpha, self.Init)

        self.dprate = args.dropout
        self.dropout = args.dropout
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)
        self.with_ACM = args.with_ACM
        if self.with_ACM:
            self.w_for_norm = nn.Parameter(torch.FloatTensor(1, self.num_classes))
            stdv_for_norm = 1. / np.sqrt(self.w_for_norm.size(1))
            self.w_for_norm.data.uniform_(-stdv_for_norm, stdv_for_norm)
        else:
            self.w_for_norm = None
    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.lin1(x))

        # TODO tanh
        # x = F.tanh(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin2(x)

        # TODO BN or - mean
        # x = x - torch.mean(x, dim=0)
        if self.dprate == 0.0:

            x = self.prop1(x, edge_index, w_for_norm=self.w_for_norm)

        else:

            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, w_for_norm=self.w_for_norm)


        return x

def RiemannAgg(x, w):

    squar_x = torch.square(x)
    squar_x_w = torch.mul(squar_x, w)
    sum_squar_x_w = torch.sum(squar_x_w, dim=1)
    sqrt_x_w = torch.sqrt(sum_squar_x_w + 1e-2)
    sqrt_x_w = torch.unsqueeze(sqrt_x_w, dim=1)
    x = torch.div(x, sqrt_x_w + 1e-2 ) # + 1

    return x



# def RiemannAgg(x, w):
#     squar_x = torch.square(x)
#     squar_x_w = torch.mul(squar_x, w)
#     sum_squar_x_w = torch.sum(squar_x_w, dim=1)
#     sqrt_x_w = torch.sqrt(sum_squar_x_w + 1e-6)
#     sqrt_x_w = torch.unsqueeze(sqrt_x_w, dim=1)
#     x = torch.div(x, sqrt_x_w+ 1e-6)
#     return x
