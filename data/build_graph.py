import argparse
import networkx as nx
import scipy.io as io


def generate_erdos_renyi(nodes: int, density: float):
    print(f"Erdos Renyi ---- Nodes: ({nodes}), Density: ({density})")
    G = nx.fast_gnp_random_graph(nodes, density)
    adj = nx.to_numpy_array(G)
    return adj


# TODO(Leonard): figure out how to control for density in this setting
def generate_non_random_graph(nodes: int):

    deg = [(1, 0), (1, 0), (1, 0), (2, 0), (1, 0), (2, 1), (0, 1), (0, 1)]
    G = nx.random_clustered_graph(deg)

def generate_small_world_graph(nodes: int, density: float):
    G = nx.newman_watts_strogatz_graph(nodes, int(0.2 * nodes), density)
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

    adj = generate_erdos_renyi(n, d)
    # adj = generate_small_world_graph(n, d)
    io.mmwrite(f"graph-{n}-{d}-Erdos-Renyi.mtx", adj)
    # io.mmwrite(f"graph-{n}-{d}-small-world.mtx", adj)
