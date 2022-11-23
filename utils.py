from scipy.io import mmread
import scipy
import networkx as nx
import numpy as np


def bfs(path):
    A = mmread(path)
    if type(A) == scipy.sparse.coo.coo_matrix:
        A = A.A

    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    edges = nx.bfs_edges(G, 0)

    print(edges)
    node_map = {}
    count = 0
    for e in edges:
        n1, n2 = e
        if n1 not in node_map:
            node_map[n1] = count
            count += 1
        if n2 not in node_map:
            node_map[n2] = count
            count += 1

    for i in range(len(A)):
        if i not in node_map:
            node_map[i] = count
            count += 1

    copy = nx.relabel_nodes(G, node_map, copy=True)
    a = nx.to_numpy_array(copy)


def mtx_to_graph(path):
    A = mmread(path)

    if type(A) == scipy.sparse.coo.coo_matrix:
        A = A.A

    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)

    scc = [
        list(c)
        for c in nx.strongly_connected_components(G)
    ]

    return scc
    # print("G is strogly connected: ", nx.is_strongly_connected(G))
    # print(len(scc))

    # G = nx.cycle_graph(4, create_using=nx.DiGraph())
    # nx.add_cycle(G, [10, 11, 12])
    # print([
    #     c
    #     for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    # ])

    # print(nx.node_connected_component(G, 0))
    # print(len(scc[0]))


if __name__ == "__main__":
    bfs("data/graph-100-0.040-small-world-p-0.5.mtx")
