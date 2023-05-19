import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def T1_generation(height, max_nodes, max_witdh=3,start=0,save_path=None):
    # basic graph
    # max_depth = 6
    # max_nodes = 300 # just for safe
    T1 = nx.DiGraph()
    # G = nx.Graph()
    T1.add_node(start)
    current_depth = max(nx.shortest_path_length(T1, 0).values())
    iter = 0
    while current_depth <= height - 1 and T1.number_of_nodes() <= max_nodes - 1:
        iter += 1
        # print("iter", iter)
        current_depth = max(nx.shortest_path_length(T1, target=0).values())
        potential_target_nodes = [x for x in T1.nodes()]  # if G.degree(x)==1
        # available_nodes = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
        if potential_target_nodes == []:  # if no target_node, add root as target_node
            potential_target_nodes = [0]
        # print("nodes", T1.nodes())
        # print("current_depth", current_depth)
        # print("leaves_nodes", potential_target_nodes)
        higher_than_max_witdh = True
        while higher_than_max_witdh:
            target_node = random.sample(potential_target_nodes, 1)[0]
            current_in_degree = T1.in_degree(target_node)
            # if current_in_degree + 1 < max_witdh:
            #     higher_than_max_witdh = False
            if current_in_degree  < max_witdh:
                higher_than_max_witdh = False
        # print("sampled leaf_node", target_node)
        Y = np.random.poisson(1)

        if Y:  # Y == 0:
            for i in range(min(Y+1,max_witdh)):
                # print("add node", T1.number_of_nodes(), "to node", target_node)
                new_node = T1.number_of_nodes()
                T1.add_node(new_node)
                T1.add_edge(new_node, target_node)  # add edge from leaf to new node
                # print(T1.nodes())
                current_depth = max(nx.shortest_path_length(T1, target=0).values())
                if T1.number_of_nodes() > max_nodes - 1:
                    break
                current_in_degree = T1.in_degree(target_node)
                if current_in_degree  >= max_witdh:
                    break


    if save_path:
        fig = plt.figure()
        nx.draw(T1, with_labels=True)
        fig.savefig(save_path)
        plt.show(block=False)
    print("T1", "num: ",T1.number_of_nodes(), "depth: ", max(nx.shortest_path_length(T1, target=0).values()))
    return T1


def T2_generation(G, save_path=None):
    pathes_length = nx.shortest_path_length(G, target=0)
    max_depth = max(pathes_length.values())
    # print("max_depth", max_depth)
    L = max_depth - 2
    source_indices_of_depth_path = list()
    for source, length in pathes_length.items():
        if length >= L:
            source_indices_of_depth_path.append(source)
    T2 = G.copy()

    if type(source_indices_of_depth_path) != list:
        source_indices_of_depth_path = [source_indices_of_depth_path]
    for source in source_indices_of_depth_path:
        # print(source)
        # print(nx.shortest_path(G, source=source, target=0))
        shortest_path = nx.shortest_path(G, source=source, target=0)
        # print(nx.shortest_path_length(G, source=source, target=0))
        # G.remove_node(shortest_path[:max_depth-L])
        # length_of_shortest_path = len(shortest_path) - 1
        length_of_shortest_path = nx.shortest_path_length(G, source=source, target=0)
        trim_path_length = length_of_shortest_path - (L - 1)
        trim_path = shortest_path[:trim_path_length]
        # print("trim_path", trim_path)
        T2.remove_nodes_from(trim_path)  # also remove the edge
        # G.remove_node()
        # print(nx.shortest_path(T2,source=source,target=0))
        # print(max(nx.shortest_path_length(T2, target=0).values()))

    if save_path:
        fig = plt.figure()
        nx.draw(T2, with_labels=True)

        fig.savefig(save_path)
        plt.show(block=False)
    print("T2", "num: ",T2.number_of_nodes(), "depth: ", max(nx.shortest_path_length(T2, target=0).values()))
    return T2


if __name__ == "__main__":
    T1 = T1_generation(10, 5000, save_path="T1.png")
    T2 = T2_generation(T1, save_path="T2.png")
