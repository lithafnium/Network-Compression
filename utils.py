from scipy.io import mmread
import scipy
import networkx as nx
import numpy as np

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
  mtx_to_graph("data/mtx_graphs/socfb-Harvard1.mtx")