import networkx as nx
import numpy as np
import math
from tree_generation import T1_generation, T2_generation


def tree(height, max_nodes, max_width, start, ):
    """Builds a balanced r-tree of height h
    INPUT:
    -------------
    start       :    starting index for the shape
    height      :    int height of the tree
    r           :    int number of branches per node
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a tree shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at role_start)
    """
    # graph = nx.balanced_tree(r, height)
    graph = T1_generation(height=height, max_nodes=max_nodes, max_witdh=max_width, start=start, save_path="T1.png")  #
    roles = [-1] * graph.number_of_nodes()  # paddings for roles
    return graph, roles


def build_graph(
        height,
        basis_type="tree",
        start=0,
        max_width=20,
        max_nodes=300,
        add_random_edges=0,
):
    """This function creates a basis (scale-free, path, or cycle)
    and attaches elements of the type in the list randomly along the basis.
    Possibility to add random edges afterwards.
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape,
                            next args:args for building the shape,
                            except for the start)
    start            :      initial nb for the first node
    rdm_basis_plugins:      boolean. Should the shapes be randomly placed
                            along the basis (True) or regularly (False)?
    add_random_edges :      nb of edges to randomly add on the structure
    m                :      number of edges to attach to existing node (for BA graph)
    OUTPUT:
    --------------------------------------------------------------------------------------
    basis            :      a nx graph with the particular shape
    role_ids         :      labels for each role
    plugins          :      node ids with the attached shapes
    """
    # if basis_type == "ba":
    #     basis, role_id = eval(basis_type)(start, width_basis, m=m)
    # else:
    basis, role_id = eval(basis_type)(height=height, start=start, max_nodes=max_nodes, max_width=max_width, )

    # n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    # n_basis = nx.number_of_nodes(basis)
    # start += n_basis  # indicator of the id of the next node

    if add_random_edges > 0:
        # add random edges between nodes:
        for p in range(add_random_edges):
            src, dest = np.random.choice(nx.number_of_nodes(basis), 2, replace=False)
            print(src, dest)
            basis.add_edges_from([(src, dest)])

    return basis, role_id
