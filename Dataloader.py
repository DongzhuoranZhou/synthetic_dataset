import os

import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, WebKB, Actor, Amazon
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, to_networkx
# from torch_geometric.utils import from_networkx
from collections import defaultdict
from utils.convert import from_networkx
import copy
def load_data(dataset, type_split="pair",dataset_name=None,precisition="float32",direction="directed",noise = True):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset)

    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = Planetoid(path, dataset, split='public', transform=T.NormalizeFeatures())[0]

    elif dataset in ["syn1", "syn2","syn3", "syn4"]:
        G = torch.load(dataset_name) # "dataset/G_1000_pairs_depth_32_width_1_hdim_16_gap_True.pt"
        G_index = nx.get_node_attributes(G, "graph_index")

        # TODO explicit undirected
        if direction == "undirected":
            G = G.to_undirected()


        # G = G.to_undirected()
        # data = defaultdict(list)
        # if G.number_of_nodes() > 0:
        #     node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        # else:
        #     node_attrs = {}



        # for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        #     print(i)
        #     if set(feat_dict.keys()) != set(node_attrs):
        #         # print(i, G.nodes[i])
        #         print(set(feat_dict.keys()), set(node_attrs))
        #         raise ValueError('Not all nodes contain the same attributes')
        #     for key, value in feat_dict.items():
        #         data[str(key)].append(value)
        data = from_networkx(G)
        # data.x = data.x.to(torch.float64)
        if precisition == "float32":
            data.x = data.x.to(torch.float32)
        elif precisition == "float64":
            data.x = data.x.to(torch.float64)
        elif precisition == "float16":
            data.x = data.x.to(torch.float16)
        # data.x = data.x.to(torch.float32)
        data.num_classes = int(max(data.y) + 1)
        data.G_index = G_index

        # TODO add two types of noise. Fiirst for data.x choose 10% of the nodes and add noise to them. Second for data.y choose 2% of them and flip.
        # TODO add noise to data.x
        if noise:
            embedding_noise = True # No big effect, acc keep 1
            label_noise = True # Has effect, acc from 1 to 0.99
        else:
            embedding_noise = False
            label_noise = False

        # TODO add noise to data.y

        data = change_split(data, dataset, type_split=type_split)
        data.y_original = copy.deepcopy(data.y)
        if embedding_noise:
            # nodex_random_indices = np.random.choice(data.x[data.train_mask], int(data.x.size(0) * 0.1), replace=False)
            train_indices = np.where(data.train_mask)[0]
            num_samples = int(len(train_indices) * 0.1)
            random_indices = np.random.choice(train_indices, num_samples, replace=False)
            data.x[random_indices] = torch.rand(data.x[random_indices].size()) + data.x[random_indices]
        if label_noise:
            # sample_root_indices = np.array(data.y)
            # data.y_original = copy.deepcopy(data.y)
            noise_rate = 0.02
            train_indices = np.where(data.train_mask)[0]
            data.y = np.array(data.y)
            # num_samples = int(len(train_indices) * 0.02)
            positive_indices = np.where(data.y[train_indices] == 1)[0] # TODO bug
            ramdom_positive_indices = np.random.choice(train_indices[positive_indices], int(positive_indices.size * noise_rate), replace=False)


            negative_indices = np.where(data.y[train_indices] == 0)[0]
            ramdom_negative_indices = np.random.choice(train_indices[negative_indices], int(negative_indices.size * noise_rate), replace=False)

            data.y[ramdom_positive_indices] = 0
            data.y[ramdom_negative_indices] = 1
            data.y = torch.from_numpy(data.y)
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


def change_split(data, dataset, type_split):
    if dataset in ["CoauthorCS", "CoauthorPhysics"]:
        data = random_coauthor_amazon_splits(data)
    elif dataset in ["AmazonComputers", "AmazonPhoto"]:
        data = random_coauthor_amazon_splits(data)
    elif dataset in ["syn1", "syn2","syn3", "syn4"]:
        data = random_coauthor_amazon_splits(data, type_split=type_split)
    # elif dataset in ["TEXAS", "WISCONSIN", "CORNELL"]:
    #     data = manual_split_WebKB_Actor(data, which_split)
    # elif dataset == "ACTOR":
    #     data = manual_split_WebKB_Actor(data, which_split)
    else:
        data = data
    data.y = data.y.long()
    return data


def random_coauthor_amazon_splits(data, split_rate=[0.6, 0.2, 0.2],type_split="pair"):
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
    # if lcc_mask is not None:
    #     for i in range(num_classes):
    #         index = (data.y[lcc_mask] == i).nonzero().view(-1)
    #         index = index[torch.randperm(index.size(0))]
    #         indices.append(index)
    # else:

    # TODO random split for single tree
    if type_split == "single":
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    elif type_split == "pair":
        # TODO random split for pair of trees
        index_0 = (data.y == 0).nonzero().view(-1)
        index_0 = index_0[torch.randperm(index_0.size(0))]
        indices.append(index_0)

        index_1 = (data.y == 1).nonzero().view(-1)
        index_1 = index_1[torch.randperm(index_1.size(0))]
        index_1_list = index_1.numpy().tolist()
        index_1_set = set(index_1_list)
        index_1_list_right_order = list()

        num_tuple = np.max(list(data.G_index.values())) + 1
        tuple_dict = {}
        for i in range(num_tuple):
            tuple_tree = list()
            for node_index, T_index in data.G_index.items():
                if T_index == i:
                    tuple_tree.append(node_index)
            tuple_tree_set = set(tuple_tree)
            tuple_dict[i] = tuple_tree_set

        for i in index_0:
            i = i.item()
            for T_index, tuple_tree_set in tuple_dict.items():
                if i in tuple_tree_set:
                    T2_root_node_index = tuple_tree_set.intersection(index_1_set)
                    T2_root_node_index = list(T2_root_node_index)[0]
                    index_1_list_right_order.append(T2_root_node_index)
        index_1_tensor_right_order = torch.tensor(index_1_list_right_order)
        indices.append(index_1_tensor_right_order)

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
    data.train_index = train_index
    data.val_index = val_index
    data.test_index = test_index

    return data