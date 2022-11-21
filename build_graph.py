import argparse
import networkx as nx
import scipy.io as io
from math import ceil
from train import SMALL_WORLD, ERDOS_RENYI

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
    print(f"Newmann Watts Strogatz Small World ---- Nodes: ({nodes}), Density: ({density})")
    G = nx.newman_watts_strogatz_graph(nodes, int(0.2 * nodes), density)
    adj = nx.to_numpy_array(G)
    return adj, G

def generate_watts_strogatz_graph(nodes: int, density: float, switch_prob=0.3):
    print(f"Watts Strogatz Small World ---- Nodes: ({nodes}), Density: ({density})")
    # Derived from the fact that (nk / 2) / (nC2) = density
    k = ceil(density * (nodes - 1))
    G = nx.watts_strogatz_graph(nodes, k, p=switch_prob)
    adj = nx.to_numpy_array(G)
    return adj, G

def main():
    parser = argparse.ArgumentParser(
        "Generates graphs given number of nodes and density"
    )
    parser.add_argument("-n", "--nodes", type=int, default=1000, required=True)
    parser.add_argument("-d", "--density", type=float, default=0.05, required=True)
    parser.add_argument("-g", "--graph-type", type=str, default=ERDOS_RENYI, choices=[ERDOS_RENYI, SMALL_WORLD], required=True)
    args = parser.parse_args()

    n = args.nodes
    d = args.density

    if args.graph_type == SMALL_WORLD:
        adj, G = generate_watts_strogatz_graph(n, d, switch_prob=0.5)
        # Save the actual density as the target density does not necessarily match
        io.mmwrite(f"data/graph-{n}-{nx.density(G):.3f}-small-world-p-0.5.mtx", adj)
    elif args.graph_type == ERDOS_RENYI:
        adj, G = generate_erdos_renyi(n, d)
        io.mmwrite(f"data/graph-{n}-{nx.density(G):.3f}-Erdos-Renyi.mtx", adj)

    print("Sanity Check Graph Density:", nx.density(G))

if __name__ == "__main__":
    main()
    
