from matplotlib import pyplot as plt
import os
from utils import synthetic_structsim
from utils import featgen, labelgen
import numpy as np
import networkx as nx
import copy
import torch
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


def gen_syn1(height=8, feature_generator=None, max_width=2, max_nodes=20, embedding_dim=16):
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
    for node in l_depth_nodes:
        if T1.nodes[node]["x"][0] < 1:
            T1.nodes[node]["x"][0] = np.random.uniform(1, 10) / np.sqrt(
                embedding_dim)  # TODO: change to a random number larger than 1
            T1.nodes[node]["x"][0] = 1 + (1  - T1.nodes[node]["x"][0])
    # create T2 by changing the first element on the level l
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
            T2.nodes[node]["x"][0] = np.random.uniform(-10, 1) / np.sqrt(
                embedding_dim)  # TODO: change to a random number smaller than 1
            # T2.nodes[node]["feat"][0] = 1 - (T2.nodes[node]["feat"][0] - 1)

    # T2 = T1.copy()
    # role_id_T2 = role_id_T1.copy()

    G_list[0] = (T1, T2)
    # name = basis_type + "_" + str(height)

    # path = os.path.join("log/syn4_base_h20_o20")
    # writer = SummaryWriter(path)
    # io_utils.log_graph(writer, G, "graph/full",args=args)

    return G_list

def gen_syn2(height=8, feature_generator=None, max_width=2, max_nodes=20, embedding_dim=16,num_pairs=10):
    G_list = list()
    for _ in range(num_pairs):
        G_list_item = gen_syn1(height=height, feature_generator=feature_generator, max_width=max_width, max_nodes=max_nodes, embedding_dim=embedding_dim)
        G_list.append(G_list_item[0])
    return G_list
if __name__ == "__main__":
    embedding_dim = 16
    # G_list = gen_syn1(height=3, feature_generator=featgen.GaussianFeatureGen(embedding_dim=16), max_width=2,
    #                         max_nodes=50)
    G_list = gen_syn2(height=3, feature_generator=featgen.GaussianFeatureGen(embedding_dim=16), max_width=4,
                            max_nodes=50)
    # for tup in G_list:
    #     (T1, T2)= tup
    #     G_list.append(T1)
    #     G_list.append(T2)
    G_list = [T for tup in G_list for T in tup]
    G = nx.disjoint_union_all(G_list)
    torch.save(G, "dataset/G_10_pairs_depth_3.pt")
    pass