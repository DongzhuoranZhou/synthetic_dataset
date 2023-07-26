import torch
import torch.nn.functional as F
# from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
# https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py
class SAGE(torch.nn.Module):
    def __init__(self, args):
        super(SAGE, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = torch.nn.ModuleList([])
        # self.layers_bn = torch.nn.ModuleList([])

        # space limit
        # if self.dataset == 'obgn-arxiv':
        #     self.dim_hidden = 1

        self.layers_GCN.append(SAGEConv(self.num_feats, self.dim_hidden))
        # if self.type_norm == 'batch':
        #     self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        for _ in range(self.num_layers - 2):
            self.layers_GCN.append(
                SAGEConv(self.dim_hidden, self.dim_hidden))
            # if self.type_norm == 'batch':
            #     self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        self.layers_GCN.append(SAGEConv(self.dim_hidden, self.num_classes))
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):

        for i in range(self.num_layers - 1):

            x = self.layers_GCN[i](x, edge_index)
            # if self.type_norm == 'batch':
            #     x = self.layers_bn[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(x)

        x = self.layers_GCN[-1](x, edge_index)
        return x