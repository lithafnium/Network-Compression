import argparse
import numpy as np
import networkx as nx
import scipy as sp
import scipy.io
import csv


def generate_graph(nodes: int, density: float):
    print(nodes, density)
    G = nx.fast_gnp_random_graph(nodes, density)
    adj = nx.to_numpy_array(G)
    return adj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generates graphs given number of nodes and density"
    )
    parser.add_argument("-n", "--nodes", type=int, default=1000)
    parser.add_argument("-d", "--density", type=float, default=0.05)
    args = parser.parse_args()

    n = args.nodes
    d = args.density

    adj = generate_graph(n, d)
    sp.io.mmwrite(f"graph-{n}-{d}.mtx", adj)
