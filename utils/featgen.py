import networkx as nx
import numpy as np
import random

import abc




class FeatureGen(metaclass=abc.ABCMeta):
    """Feature Generator base class."""
    @abc.abstractmethod
    def gen_node_features(self, G):
        pass


# class DepthGen(metaclass=abc.ABCMeta):
#     """Feature Generator base class."""
#     @abc.abstractmethod
#     def gen_node_depths(self, G):
#         pass

class ConstFeatureGen(FeatureGen):
    """Constant Feature class."""
    def __init__(self, val):
        self.val = val

    def gen_node_features(self, G):
        feat_dict = {i:{'feat': np.array(self.val, dtype=np.float32)} for i in G.nodes()}
        print ('feat_dict[0]["x"]:', feat_dict[0]['x'].dtype)
        nx.set_node_attributes(G, feat_dict)
        print ('G.nodes[0]["x"]:', G.nodes[0]['x'].dtype)


class GaussianFeatureGen(FeatureGen):
    """Gaussian Feature class."""
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        # self.mu = mu
        # if sigma.ndim < 2:
        #     self.sigma = np.diag(sigma)
        # else:
        #     self.sigma = sigma
    @staticmethod
    def initialize_node_embeddings(num_nodes, embedding_dim):
        embeddings = np.random.randn(num_nodes, embedding_dim) / np.sqrt(embedding_dim)
        return embeddings

    def gen_node_features(self, G):
        # feat = np.random.multivariate_normal(self.mu, self.sigma, G.number_of_nodes())
        num_nodes = G.number_of_nodes()
        embeddings = GaussianFeatureGen.initialize_node_embeddings(num_nodes, self.embedding_dim)
        feat_dict = {
                i: {"x": embeddings[i]} for i in range(embeddings.shape[0])
            }
        nx.set_node_attributes(G, feat_dict)


class DepthGen():
    """Feature Generator base class."""

    def __init__(self):
        pass
        # self.max_depth = max_depth
    def gen_node_depths(self, G):
        depths = nx.shortest_path_length(G, target=0)
        depth_dict = {
            i: {"depth": depths[i]} for i in range(len(depths))
        }
        nx.set_node_attributes(G, depth_dict)


class GraphIndexGen():
    def __init__(self):
        pass
    def gen_node_graph_index(self, G_list):
        # tree_index_dict = {
        #     i: {"tree_index": i} for G in G_list for i in range(G.number_of_nodes() )
        # }
        # G_tree_index_dict = {}
        # G_index_dict = {}
        for G_index, G in enumerate(G_list):
            (T1,T2) = G
            for T in (T1,T2):
                num_nodes = T.number_of_nodes()
                G_index_dict = {
                    i: {"graph_index": G_index} for i in range(num_nodes)
                }
                nx.set_node_attributes(T, G_index_dict)
                # for i in range(T.number_of_nodes()):
                #     G_index_dict[G_index] = G_index
                # G_tree_index_dict[G_index] = G_index_dict
        # for G_index, G in enumerate(G_list):
        #     (T1, T2) = G
        #     for T in (T1, T2):
        # nx.set_node_attributes(T, G_tree_index_dict[G_index])
