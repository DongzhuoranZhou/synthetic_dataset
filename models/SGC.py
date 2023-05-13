import torch
from torch import nn

# from torch_geometric.nn import SGConv
from models.SGC_layer import SGConv
import numpy as np


class SGC(nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        # bn used in dataset 'obgn-arxiv'
        # lin_first used in Coauthor_physics due to space limit
        self.bn = True if args.type_norm == 'batch' else False
        # self.bn_ACM = nn.ModuleList([])
        # for i in range(self.num_layers - 1):
        #     self.bn_ACM.append(torch.nn.BatchNorm1d(self.num_feats))
        self.lin_first = True if args.dataset == 'Coauthor_Physics' else False
        self.SGC = SGConv(self.num_feats, self.num_classes, K=self.num_layers,
                          cached=self.cached, bias=False, bn=self.bn, dropout=self.dropout,
                          lin_first=self.lin_first, args=args)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.with_ACM = args.with_ACM
        if self.with_ACM:
            if self.lin_first:
                self.w_for_norm = nn.Parameter(torch.FloatTensor(1, self.num_classes))
            else:
                self.w_for_norm = nn.Parameter(torch.FloatTensor(1, self.num_feats))
            stdv_for_norm = 1. / np.sqrt(self.w_for_norm.size(1))
            self.w_for_norm.data.uniform_(-stdv_for_norm, stdv_for_norm)
        else:
            self.w_for_norm = None

    def forward(self, x, edge_index):
        # implemented based on https://github.com/Tiiiger/SGC/blob/master/citation.py
        x = self.SGC(x, edge_index, w_for_norm=self.w_for_norm) # ,bn_ACM=self.bn_ACM
        return x




