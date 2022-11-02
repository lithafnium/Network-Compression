import argparse
import numpy as np
import networkx as nx
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

    with open(f"graph-{n}-{d}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id_1", "id_2", "label"])
        for r in range(n):
            for c in range(n):
                if adj[r][c] == 1:
                    writer.writerow([r, c, 1])
                else: 
                    writer.writerow([r, c, 0])
    


