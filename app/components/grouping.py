import numpy as np, networkx as nx
from sklearn.neighbors import NearestNeighbors

def block_by_cosine(X: np.ndarray, k=8, sim_thresh=0.82):
    nbrs = NearestNeighbors(n_neighbors=min(k,len(X)), metric="cosine").fit(X)
    dists, idxs = nbrs.kneighbors(X)
    G = nx.Graph(); G.add_nodes_from(range(len(X)))
    for i,(dr,ir) in enumerate(zip(dists,idxs)):
        for d,j in zip(dr,ir):
            if i==j: continue
            sim = 1-d
            if sim>=sim_thresh: G.add_edge(i,j,weight=sim)
    return [sorted(list(c)) for c in nx.connected_components(G)]
