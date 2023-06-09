import os

import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, WebKB, Actor, Amazon
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, to_networkx
from torch_geometric.utils import from_networkx


def load_data(dataset, which_run=0):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset)

    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = Planetoid(path, dataset, split='public', transform=T.NormalizeFeatures())[0]

    elif dataset in ["syn1", "syn2"]:
        G = torch.load("dataset/G_100_pairs_depth_8_width_1_hdim_16.pt")
        G = G.to_undirected()
        G_index = nx.get_node_attributes(G, "graph_index")
        data = from_networkx(G)
        # data.G_index = torch.Tensor(list(G_index.values()))
        data.G_index = G_index
        # data.x = data.x.to(torch.float32)
        data.x = data.x.to(torch.float64)
        data.num_classes = int(max(data.y) + 1)
        data = change_split(data, dataset, which_split=int(which_run // 10))
    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')
    num_nodes = data.x.size(0)
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
    if isinstance(edge_index, tuple):
        data.edge_index = edge_index[0]
    else:
        data.edge_index = edge_index
    return data


def change_split(data, dataset, which_split=0):
    # if dataset in ["CoauthorCS", "CoauthorPhysics"]:
    #     data = random_coauthor_amazon_splits(data)
    # elif dataset in ["AmazonComputers", "AmazonPhoto"]:
    #     data = random_coauthor_amazon_splits(data)
    if dataset in ["syn1", "syn2"]:
        data = random_syn_splits(data)
    # elif dataset in ["TEXAS", "WISCONSIN", "CORNELL"]:
    #     data = manual_split_WebKB_Actor(data, which_split)
    # elif dataset == "ACTOR":
    #     data = manual_split_WebKB_Actor(data, which_split)
    else:
        data = data
    data.y = data.y.long()
    return data


def random_syn_splits(data, split_rate=[0.6, 0.2, 0.2]):
    # https://github.com/mengliu1998/DeeperGNN/blob/da1f21c40ec535d8b7a6c8127e461a1cd9eadac1/DeeperGNN/train_eval.py#L17
    num_classes = data.num_classes
    num_nodes = data.num_nodes
    # if lcc:  # select largest connected component
    data_nx = to_networkx(data)
    data_nx = data_nx.to_undirected()
    print("Original #nodes:", data_nx.number_of_nodes())
    # data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
    # print("#Nodes after lcc:", data_nx.number_of_nodes())

    def index_to_mask(index, size):
        mask = torch.zeros(size, dtype=torch.bool, device=index.device)
        mask[index] = 1
        return mask

    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    # if lcc_mask is not None:
    #     for i in range(num_classes):
    #         index = (data.y[lcc_mask] == i).nonzero().view(-1)
    #         index = index[torch.randperm(index.size(0))]
    #         indices.append(index)
    # else:
    # data.G_index = data.G_index.to(torch.long)
    # for i in range(num_classes):
    #     index = (data.y == num_classes).nonzero().view(-1)
    #     index = index[torch.randperm(index.size(0))]
    #     indices.append(index)

    num_tuple = np.max(list(data.G_index.values())) + 1
    # root_index = torch.cat([i[:] for i in indices], dim=0).numpy()

    tuple_dict = {}
    for i in range(num_tuple):
        tuple_tree = list()
        for node_index, T_index in data.G_index.items():
            if T_index == i:
                tuple_tree.append(node_index)
        tuple_tree_set = set(tuple_tree)
        tuple_dict[i] = tuple_tree_set
        # if T_index == i:
        #     tuple_tree.append(node_index)
    data.tuple_dict = tuple_dict
    #
    #
    index_0 = (data.y == 0).nonzero().view(-1)
    index_0 = index_0[torch.randperm(index_0.size(0))]
    indices.append(index_0)
    index_1 = (data.y == 1).nonzero().view(-1)
    index_1 = index_1[torch.randperm(index_1.size(0))]
    index_1_list = index_1.numpy().tolist()
    index_1_set = set(index_1_list)
    index_1_list_right_order = list()
    for i in index_0:
        i = i.item()
        for T_index, tuple_tree_set in tuple_dict.items():
            if i in tuple_tree_set:
                T2_root_node_index = tuple_tree_set.intersection(index_1_set)
                T2_root_node_index = list(T2_root_node_index)[0]
                index_1_list_right_order.append(T2_root_node_index)
    index_1_tensor_right_order = torch.tensor(index_1_list_right_order)
    indices.append(index_1_tensor_right_order)
    # for i in range(1, num_tuple):
    #     tuple_tree = list()
    #     tuple_dict[i] = tuple_tree
    #     # T1_index = (data.G_index == i).nonzero().view(-1)
    #     T1_index = data.G_index[i]
    #     tuple_dict[i].append(T1_index)
    #     index = (data.G_index == i).nonzero().view(-1)
    #     index = index[torch.randperm(index.size(0))]
    #     indices.append(index)

    # for i in range(num_classes):
    #     index = (data.y == i).nonzero().view(-1)
    #     index = index[torch.randperm(index.size(0))]
    #     indices.append(index)
    # for i in range(root_index):
    #     data.G_index[i]
    #     # make sure that the trees in the same tuple are sampled.

    train_index = torch.cat([i[:int(split_rate[0] * len(i))] for i in indices], dim=0)
    val_index = torch.cat(
        [i[int(split_rate[0] * len(i)):int((split_rate[0] + split_rate[1]) * len(i))] for i in indices], dim=0)
    test_index = torch.cat([i[int((split_rate[0] + split_rate[1]) * len(i)):] for i in indices], dim=0)
    # train_index = torch.cat([i[:20] for i in indices], dim=0)
    # val_index = torch.cat([i[20:50] for i in indices], dim=0)

    # rest_index = torch.cat([i[50:] for i in indices], dim=0)
    # rest_index = rest_index[torch.randperm(rest_index.size(0))]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data
