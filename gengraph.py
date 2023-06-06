from matplotlib import pyplot as plt
import os
from utils import synthetic_structsim
from utils import featgen, labelgen
import numpy as np
import networkx as nx
import copy
import torch
import importlib
from os.path import exists
from collections import defaultdict

####################################
#
# Experiment utilities
#
####################################
def preprocess_input_graph(G, labels, normalize_adj=False):
    """ Load an existing graph to be converted for the experiments.
    Args:
        G: Networkx graph to be loaded.
        labels: Associated node labels.
        normalize_adj: Should the method return a normalized adjacency matrix.
    Returns:
        A dictionary containing adjacency, node features and labels
    """
    adj = np.array(nx.to_numpy_array(G))
    if normalize_adj:
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["x"].shape[0]
    f = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        f[i, :] = G.nodes[u]["x"]

    # add batch dim
    adj = np.expand_dims(adj, axis=0)
    f = np.expand_dims(f, axis=0)
    labels = np.array(list(labels.values()))
    labels = np.expand_dims(labels, axis=0)
    return {"adj": adj, "x": f, "y": labels}


####################################
#
# Generating synthetic graphs
#
###################################
def perturb(graph_list, p):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def gen_syn3(height=8, feature_generator=None, max_width=1, max_nodes=20, embedding_dim=16, high_gap=True):
    """ generate pairs of trees with different p. Topology difference

    l = k - 2
    p(T1): the depth of the T1 is k
    p(T2): the depth of the T1 is k - 2
    "minimal change from p(T1) to p(T2)":  delete the last two levels of T1
    """
    basis_type = "tree"

    G_list = dict()

    # T1 generation
    T1, role_id_T1 = synthetic_structsim.build_graph(
        height=height, basis_type=basis_type, start=0, max_width=max_width, max_nodes=max_nodes
    )

    # add node embedding to graph
    if feature_generator is None:
        feature_generator = featgen.GaussianFeatureGen(embedding_dim=embedding_dim)
    feature_generator.gen_node_features(T1)
    # add node label to graph
    label_generator = labelgen.ConstLabelGen()
    label_generator.gen_node_labels(T1, role_id_T1)

    # add depth information to graph
    depth_generator = featgen.DepthGen()
    depth_generator.gen_node_depths(T1)

    # add label to T1 root node
    T1.nodes[0]["y"] = np.array([1], dtype=int)[0]
    # minimal possible change, to make sure T1 follows the rule of p(T1)
    depth_T1 = max(nx.shortest_path_length(T1, target=0).values())
    if depth_T1 != height:
        print("depth is too small")
        return None, None
    depth_node = nx.shortest_path_length(T1, target=0)
    # l = height - 2
    # l_depth_nodes = [node for node, depth in depth_node.items() if depth == l]
    l_list = [height, height - 1]
    dict_l_depth_nodes = {l: [node for node, depth in depth_node.items() if depth == l] for l in l_list}

    T2 = copy.deepcopy(T1)
    T2.nodes[0]["y"] = np.array([0], dtype=int)[0]

    # minimal possible change, to make sure T2 follows the rule of p(T2)
    for l in l_list:
        for node in dict_l_depth_nodes[l]:
            neighbors = list(T2.neighbors(node))
            # delete edges
            for neighbor in neighbors:
                T2.remove_edge(node, neighbor)
            # delete node
            T2.remove_node(node)

    # add node embedding to graph
    print("T2", "num: ", T2.number_of_nodes(), "depth: ", max(nx.shortest_path_length(T2, target=0).values()))
    G_list[0] = (T1, T2)
    # name = basis_type + "_" + str(height)

    # path = os.path.join("log/syn4_base_h20_o20")
    # writer = SummaryWriter(path)
    # io_utils.log_graph(writer, G, "graph/full",args=args)

    return G_list


def gen_syn4(height=8, feature_generator=None, max_width=2, max_nodes=20, embedding_dim=16, num_pairs=10,
             high_gap=True):
    G_list = list()
    for _ in range(num_pairs):
        G_list_item = gen_syn3(height=height, feature_generator=feature_generator, max_width=max_width,
                               max_nodes=max_nodes, embedding_dim=embedding_dim, high_gap=high_gap)
        G_list.append(G_list_item[0])
    graph_index_generator = featgen.GraphIndexGen()
    graph_index_generator.gen_node_graph_index(G_list)
    return G_list


def gen_syn1(height=8, feature_generator=None, max_width=2, max_nodes=20, embedding_dim=16, high_gap=True):
    """ generate pairs of trees with different p.

    l = k - 2
    p(T1): all first elements on the level ℓ are greater than or equal to 1
    p(T2): all first elements on the level ℓ are smaller than  1
    "minimal change from p(T1) to p(T2)":  change all first elements greater than 1 on the level ℓ to a random number smaller than 1
    """
    basis_type = "tree"

    G_list = dict()

    # T1 generation
    T1, role_id_T1 = synthetic_structsim.build_graph(
        height=height, basis_type=basis_type, start=0, max_width=max_width, max_nodes=max_nodes
    )
    # T2 = T1.copy()
    # role_id_T2 = role_id_T1.copy()
    #
    # role_id_T1[0] = 1
    # role_id_T2[0] = 0

    # add node embedding to graph
    if feature_generator is None:
        feature_generator = featgen.GaussianFeatureGen(embedding_dim=embedding_dim)
    feature_generator.gen_node_features(T1)
    # feature_generator.gen_node_features(T2)
    # add node label to graph
    label_generator = labelgen.ConstLabelGen()
    label_generator.gen_node_labels(T1, role_id_T1)
    # label_generator.gen_node_labels(T2, role_id_T2)

    # add depth information to graph
    depth_generator = featgen.DepthGen()
    depth_generator.gen_node_depths(T1)

    # add label to T1 root node
    T1.nodes[0]["y"] = np.array([1], dtype=int)[0]
    # minimal possible change, to make sure T1 follows the rule of p(T1)
    depth_T1 = max(nx.shortest_path_length(T1, target=0).values())
    if depth_T1 != height:
        print("depth is too small")
        return None, None
    depth_node = nx.shortest_path_length(T1, target=0)
    l = height - 2
    l_depth_nodes = [node for node, depth in depth_node.items() if depth == l]

    # property of p(T1)
    if not high_gap:
        T1_upper_limit, T1_lower_limit = 10, 1
        T2_upper_limit, T2_lower_limit = 1, -8
    else:
        T1_upper_limit, T1_lower_limit = 5, 10
        T2_upper_limit, T2_lower_limit = -5, -10
    for node in l_depth_nodes:
        if T1.nodes[node]["x"][0] < 1:
            T1.nodes[node]["x"][0] = np.random.uniform(T1_lower_limit,
                                                       T1_upper_limit)  # TODO: change to a random number larger than 1
            # T1.nodes[node]["x"][0] = 1 + (1  - T1.nodes[node]["x"][0])
    # create T2 by changing the first element on the level l
    # property of p(T2)
    T2 = copy.deepcopy(T1)
    T2.nodes[0]["y"] = np.array([0], dtype=int)[0]
    # minimal possible change
    # depth_T2 = max(nx.shortest_path_length(T2, target=0).values())
    # if depth_T2 != height:
    #     print("depth is too small")
    #     return None, None
    # depth_node = nx.shortest_path_length(T2, target=0)
    # l = height - 2
    # l_depth_nodes = [node for node, depth in depth_node.items() if depth == l]
    for node in l_depth_nodes:
        if T2.nodes[node]["x"][0] >= 1:
            T2.nodes[node]["x"][0] = np.random.uniform(T2_lower_limit,
                                                       T2_upper_limit)  # TODO: change to a random number smaller than 1
            # T2.nodes[node]["feat"][0] = 1 - (T2.nodes[node]["feat"][0] - 1)

    for node in l_depth_nodes:
        T1.nodes[node]["y"] = np.array([-2], dtype=int)[0]
        T2.nodes[node]["y"] = np.array([-2], dtype=int)[0]
    # T2 = T1.copy()
    # role_id_T2 = role_id_T1.copy()

    G_list[0] = (T1, T2)
    # name = basis_type + "_" + str(height)

    # path = os.path.join("log/syn4_base_h20_o20")
    # writer = SummaryWriter(path)
    # io_utils.log_graph(writer, G, "graph/full",args=args)

    return G_list


def gen_syn2(height=8, feature_generator=None, max_width=2, max_nodes=20, embedding_dim=16, num_pairs=10,
             high_gap=True):
    G_list = list()
    for _ in range(num_pairs):
        G_list_item = gen_syn1(height=height, feature_generator=feature_generator, max_width=max_width,
                               max_nodes=max_nodes, embedding_dim=embedding_dim, high_gap=high_gap)
        G_list.append(G_list_item[0])
    graph_index_generator = featgen.GraphIndexGen()
    graph_index_generator.gen_node_graph_index(G_list)
    return G_list


if __name__ == "__main__":
    # single depth
    # embedding_dim = 64
    # num_pairs = 1000
    # depth = 4
    # width = 1
    # high_gap = True
    # # G_list = gen_syn1(height=3, feature_generator=featgen.GaussianFeatureGen(embedding_dim=16), max_width=2,
    # #                         max_nodes=50)
    # G_list = gen_syn2(height=depth, feature_generator=featgen.GaussianFeatureGen(embedding_dim=embedding_dim), max_width=width,
    #                         max_nodes=1000,num_pairs=num_pairs,high_gap=high_gap)
    # # for tup in G_list:
    # #     (T1, T2)= tup
    # #     G_list.append(T1)
    # #     G_list.append(T2)
    # G_list = [T for tup in G_list for T in tup]
    # G = nx.disjoint_union_all(G_list)
    # # G = G.to_undirected()
    # print("dataset/G_{}_pairs_depth_{}_width_{}_hdim_{}_high_gap_{}.pt".format(num_pairs,depth,width,embedding_dim,high_gap))
    # torch.save(G, "dataset/G_{}_pairs_depth_{}_width_{}_hdim_{}_gap_{}.pt".format(num_pairs,depth,width,embedding_dim,high_gap))
    #

    # multiple depth for syn2
    # depth_list =  [4,6,8,10,11,12,13,14,15,16,32]
    # depth_list = [7,9,]
    # # depth_list = [18, 20, 22, 24, 26, 28, 30]
    # for depth in depth_list:
    #     embedding_dim = 16
    #     num_pairs = 1000
    #     # depth = 4
    #     width = 1
    #     high_gap = True
    #     # G_list = gen_syn1(height=3, feature_generator=featgen.GaussianFeatureGen(embedding_dim=16), max_width=2,
    #     #                         max_nodes=50)
    #     G_list = gen_syn2(height=depth, feature_generator=featgen.GaussianFeatureGen(embedding_dim=embedding_dim),
    #                       max_width=width,
    #                       max_nodes=1000, num_pairs=num_pairs, high_gap=high_gap)
    #     # for tup in G_list:
    #     #     (T1, T2)= tup
    #     #     G_list.append(T1)
    #     #     G_list.append(T2)
    #     G_list = [T for tup in G_list for T in tup]
    #     G = nx.disjoint_union_all(G_list)
    #     # G = G.to_undirected()
    #     print(
    #         "dataset/G_{}_pairs_depth_{}_width_{}_hdim_{}_high_gap_{}.pt".format(num_pairs, depth, width, embedding_dim,
    #                                                                              high_gap))
    #     torch.save(G, "dataset/G_{}_pairs_depth_{}_width_{}_hdim_{}_gap_{}.pt".format(num_pairs, depth, width,
    #                                                                                   embedding_dim, high_gap))

    # multiple depth for syn3
    # G_list = gen_syn3(height=4, feature_generator=featgen.GaussianFeatureGen(embedding_dim=16), max_width=2,max_nodes=1000)
    # nx.draw(G_list[0][0], with_labels=True)
    # plt.show()
    # nx.draw(G_list[0][1], with_labels=True)
    # plt.show()
    depth_list = [6, 8, 10]
    # depth_list = [7,9,]
    # depth_list = [18, 20, 22, 24, 26, 28, 30]
    dataset_to_generate = "syn2"

    if dataset_to_generate == "syn4":
        gen_function = "gen_syn4"
        gen_function = getattr(importlib.import_module("gengraph"), gen_function)
    elif dataset_to_generate == "syn2":
        gen_function = "gen_syn2"
        gen_function = getattr(importlib.import_module("gengraph"), gen_function)
    for depth in depth_list:
        embedding_dim = 16
        num_pairs = 1000
        # depth = 4
        width = 2
        high_gap = True
        # G_list = gen_syn1(height=3, feature_generator=featgen.GaussianFeatureGen(embedding_dim=16), max_width=2,
        #                         max_nodes=50)

        G_list = gen_function(height=depth, feature_generator=featgen.GaussianFeatureGen(embedding_dim=embedding_dim),
                              max_width=width,
                              max_nodes=100000000, num_pairs=num_pairs, high_gap=high_gap)

        # nx.draw(G_list[0][0], with_labels=True)
        # plt.show()
        # nx.draw(G_list[0][1], with_labels=True)
        # plt.show()
        # for tup in G_list:
        #     (T1, T2)= tup
        #     G_list.append(T1)
        #     G_list.append(T2)
        G_list = [T for tup in G_list for T in tup]

        # for G in G_list:
        #     data = defaultdict(list)
        #     if G.number_of_nodes() > 0:
        #         node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        #     else:
        #         node_attrs = {}



            # for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            #     print(i)
            #     if set(feat_dict.keys()) != set(node_attrs):
            #         # print(i, G.nodes[i])
            #         nx.draw(G, with_labels=True)
            #         plt.show()
            #         print(set(feat_dict.keys()), set(node_attrs))
            #         raise ValueError('Not all nodes contain the same attributes')
            #     for key, value in feat_dict.items():
            #         data[str(key)].append(value)

        G = nx.disjoint_union_all(G_list)
        G = G.to_undirected()
        root_dir = "dataset/{}/width_{}".format(dataset_to_generate,width)
        if not exists(root_dir):
            os.makedirs(root_dir)
        print(
            "{}/G_{}_pairs_depth_{}_width_{}_hdim_{}_high_gap_{}.pt".format(root_dir, num_pairs, depth, width,
                                                                            embedding_dim,
                                                                            high_gap))
        torch.save(G, "{}/G_{}_pairs_depth_{}_width_{}_hdim_{}_high_gap_{}.pt".format(root_dir, num_pairs, depth, width,
                                                                                      embedding_dim,
                                                                                      high_gap))
