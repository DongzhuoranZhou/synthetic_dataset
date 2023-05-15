import networkx as nx
import numpy as np
import random

import abc


class LabelGen(metaclass=abc.ABCMeta):
    """Feature Generator base class."""

    @abc.abstractmethod
    def gen_node_labels(self, G, role_id):
        pass


class ConstLabelGen(LabelGen):
    """Constant Feature class."""

    def __init__(self):
        # self.val = val
        pass

    def gen_node_labels(self, G, role_id):
        role_id = np.array(role_id, dtype=np.int)
        # feat_dict = {i:{'label': np.array(self.val, dtype=np.float32)} for i in G.nodes()}
        label_dict = {i: {'y': role_id[i]} for i in G.nodes()}
        # print('label_dict[0]["label"]:', label_dict[0]['label'].dtype)
        nx.set_node_attributes(G, label_dict)
        # print('G.nodes[0]["label"]:', G.nodes[0]['label'].dtype)
