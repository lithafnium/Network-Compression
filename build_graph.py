import argparse
import networkx as nx
import scipy.io as io
import numpy as np

from math import ceil
from train import SMALL_WORLD, ERDOS_RENYI
from PIL import Image
from skimage.filters import gaussian


def generate_erdos_renyi(nodes: int, density: float):
    print(f"Erdos Renyi ---- Nodes: ({nodes}), Density: ({density})")
    G = nx.fast_gnp_random_graph(nodes, density)
    adj = nx.to_numpy_array(G)
    return adj, G


# TODO(Leonard): figure out how to control for density in this setting
def generate_non_random_graph(nodes: int):
    deg = [(1, 0), (1, 0), (1, 0), (2, 0), (1, 0), (2, 1), (0, 1), (0, 1)]
    G = nx.random_clustered_graph(deg)

# TODO(leonard): deprecate this


def generate_newman_watts_strogatz_graph(nodes: int, density: float):
    print(
        f"Newmann Watts Strogatz Small World ---- Nodes: ({nodes}), Density: ({density})")
    G = nx.newman_watts_strogatz_graph(nodes, int(0.2 * nodes), density)
    adj = nx.to_numpy_array(G)
    return adj, G


def generate_watts_strogatz_graph(nodes: int, density: float, switch_prob=0.3):
    print(
        f"Watts Strogatz Small World ---- Nodes: ({nodes}), Density: ({density})")
    # Derived from the fact that (nk / 2) / (nC2) = density
    k = ceil(density * (nodes - 1))
    G = nx.watts_strogatz_graph(nodes, k, p=switch_prob)
    adj = nx.to_numpy_array(G)
    return adj, G


def reorder_bfs(adj, G):
    edges = nx.bfs_edges(G)

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

    # i think this is okay but this is just to
    # label disconnected nodes in the graph
    for i in range(len(adj)):
        if i not in node_map:
            node_map[i] = count
            count += 1

    copy = nx.relabel_nodes(G, node_map, copy=True)
    adj = nx.to_numpy_array(copy)

    return adj, copy


def dag_util(G, v, visited, s):
    visited[v] = True
    for i in G.neighbors(v):
        if visited[i] == False:
            dag_util(G, i, visited, s)

    s.append(v)


def reorder_dag(adj, G):

    n = G.number_of_nodes()
    visited = [False]*n
    s = []

    for i in range(n):
        if visited[i] == False:
            dag_util(G, i, visited, s)

    # reversing list gives topological order
    s = s[::-1]
    node_map = {}
    count = 0
    for e in s:
        node_map[e] = count
        count += 1

    copy = nx.relabel_nodes(G, node_map, copy=True)
    adj = nx.to_numpy_array(copy)

    return adj, copy


def generate_image(adj, path):
    gaussian_adj = gaussian(adj, preserve_range=True, sigma=0.7)
    image_adj = (adj * 255).astype('uint8')

    Image.fromarray(image_adj, mode="L").save(
        f"graph_images/{path}-original.png")

    gaussian_adj = (gaussian_adj * 255).astype('uint8')
    Image.fromarray(gaussian_adj,
                    mode="L").save(f"graph_images/{path}-blurred.png")


def main():
    parser = argparse.ArgumentParser(
        "Generates graphs given number of nodes and density"
    )
    parser.add_argument("-n", "--nodes", type=int, default=1000, required=True)
    parser.add_argument("-d", "--density", type=float,
                        default=0.05, required=True)
    parser.add_argument("-g", "--graph-type", type=str, default=ERDOS_RENYI,
                        choices=[ERDOS_RENYI, SMALL_WORLD], required=True)
    parser.add_argument("-r", "--reorder", type=str,
                        default="none", choices=["bfs", "dag", "none"])
    args = parser.parse_args()

    n = args.nodes
    d = args.density
    r = args.reorder

    if args.graph_type == SMALL_WORLD:
        adj, G = generate_watts_strogatz_graph(n, d, switch_prob=0.5)
        if r == "bfs":
            adj, G = reorder_bfs(adj, G)
        if r == "dag":
            adj, G = reorder_dag(adj, G)
        # Save the actual density as the target density does not necessarily match
        path = f"data/graph-{n}-{nx.density(G):.3f}-small-world-p-0.5.mtx"

        generate_image(adj, path.strip("data/"))

    elif args.graph_type == ERDOS_RENYI:
        adj, G = generate_erdos_renyi(n, d)
        if r == "bfs":
            adj, G = reorder_bfs(adj, G)

        path = f"data/graph-{n}-{nx.density(G):.3f}-Erdos-Renyi.mtx"

    print("Saving to", path)
    io.mmwrite(path, adj)

    print("Sanity Check Graph Density:", nx.density(G))


if __name__ == "__main__":
    main()
