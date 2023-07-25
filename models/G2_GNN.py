import torch
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import SAGEConv, GATConv
from Layers.gcn_conv import GCNConv
import torch.optim as optim
class G2(nn.Module):
    def __init__(self, conv, p=2., conv_type='GraphSAGE', activation=nn.ReLU()):
        super(G2, self).__init__()
        self.conv = conv
        self.p = p
        self.activation = activation
        self.conv_type = conv_type

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        if self.conv_type == 'GAT':
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = self.activation(self.conv(X, edge_index))
        gg = torch.tanh(scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                                 edge_index[0], 0,dim_size=X.size(0), reduce='mean'))

        return gg

class G2_GNN(nn.Module):
    # def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='GraphSAGE', p=2., drop_in=0, drop=0, use_gg_conv=True):
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        super(G2_GNN, self).__init__()
        # self.conv_type = conv_type
        self.drop = self.drop_G2
        self.enc = nn.Linear(self.num_feats, self.dim_hidden)
        self.dec = nn.Linear(self.dim_hidden, self.num_classes)
        # self.drop_in = drop_in
        # self.drop = drop
        # self.nlayers = self.num_layers
        if self.conv_type == 'GCN':
            self.conv = GCNConv(self.dim_hidden, self.dim_hidden)
            # self.cached = True
            # self.conv = GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached,layer_index=0,gcn_norm_type=self.gcn_norm_type,normalize=self.normalize)
            if self.use_gg_conv == True:
                self.conv_gg = GCNConv(self.dim_hidden, self.dim_hidden)
                # self.conv = GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached, layer_index=0,
                #                     gcn_norm_type=self.gcn_norm_type, normalize=self.normalize)

        elif self.conv_type == 'GraphSAGE':
            self.conv = SAGEConv(self.dim_hidden, self.dim_hidden)
            if self.use_gg_conv == True:
                self.conv_gg = SAGEConv(self.dim_hidden, self.dim_hidden)
        elif self.conv_type == 'GAT':
            self.conv = GATConv(self.dim_hidden,self.dim_hidden,heads=4,concat=True)
            # self.conv =GATConv(self.dim_hidden, self.dim_hidden, heads=4,bias=False, concat=True, num_feats=self.num_feats,
            #         with_ACM=self.with_ACM)
            if self.use_gg_conv == True:
                self.conv_gg = GATConv(self.dim_hidden,self.dim_hidden,heads=4,concat=True)
                # self.conv_gg =GATConv(self.dim_hidden, self.dim_hidden, heads=4,bias=False, concat=True, num_feats=self.num_feats,
                #         with_ACM=self.with_ACM)
        else:
            print('specified graph conv not implemented')

        if self.use_gg_conv == True:
            self.G2 = G2(self.conv_gg,self.p,self.conv_type,activation=nn.ReLU())
        else:
            self.G2 = G2(self.conv,self.p,self.conv_type,activation=nn.ReLU())
        self.optimizer = optim.Adam(self.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    def forward(self, x, edge_index):
        X = x
        n_nodes = X.size(0)
        # edge_index = data.edge_index
        X = F.dropout(X, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))

        for i in range(self.num_layers):
            if self.conv_type == 'GAT':
                X_ = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                X_ = torch.relu(self.conv(X, edge_index))
            tau = self.G2(X, edge_index)
            X = (1 - tau) * X + tau * X_
        X = F.dropout(X, self.drop, training=self.training)

        return self.dec(X)
