from main import *

import os, glob, pickle
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from scipy.spatial import cKDTree


from torch_geometric.nn import knn_graph


AA1_VOCAB = list("ARNDCQEGHILKMFPSTWYV")
AA1_TO_IDX = {a:i for i,a in enumerate(AA1_VOCAB)}


def one_hot_aa1(a1 :str):
    "20 dim one-hot encoding for amino acids"
    x = torch.zeros(len(AA1_VOCAB), dtype = torch.float32)
    i = AA1_TO_IDX.get(a1, None)
    if i is not None:
        x[i] = 1.0
    
    return x



def graph_to_data(G, coordinates = True, knn= 25):
    "converts the networkx graph to torch_geometruc data"

    nodes = sorted(G.nodes())

    xs, poss, resid, chain_idx = [], [], [], []

    # map chain strings to integers
    chains = {}
    for n in nodes:
        nd = G.nodes[n]
        a1 = nd.get("resname1", "X")
        xs.append(one_hot_aa1(a1))

        coord = np.asarray(nd.get("coord"), dtype = np.float32)
        poss.append(coord)

        resid.append(int(nd.get("resid", -1)))
        ch = str(nd.get("chain", ""))
        if ch not in chains:
            chains[ch] = len(chains)

        chain_idx.append(chains[ch])

    
    x = torch.stack(xs, dim = 0)
    pos = torch.tensor(np.stack(poss, axis = 0))
    resid = torch.tensor(resid, dtype = torch.long)
    chain = torch.tensor(chain_idx, dtype = torch.long)
    node_ids = torch.tensor(nodes, dtype = torch.long)

    edge_idx = knn_graph(pos, k=knn, loop = False)
    edge_idx = to_undirected(edge_idx)    

    if coordinates:
        # normalize the coordinates 
        pos = pos - pos.mean(axis=0)     
        pos = pos / pos.std(axis=0) 

        # concatenate the coords with residues 
        x = torch.cat([x, pos], dim = -1)

    print(f"X.shape : {x.shape}")


    data = Data(x=x, pos=pos, edge_index=edge_idx)
    data.resid = resid
    data.chain = chain
    data.node_ids = node_ids
    data.num_chains = torch.tensor(len(chains))

    print(data)
    return data















if __name__ == "__main__":

    slide = 5
    k = 25
    coordinates = True
    
    data_dir = "/Users/berfininal/Documents/ML-proteins/implicit_graphon/MD_data/MDR_00004293"
    frame_directories = []
    dataset = []
    for i in range(0,10000,slide):
        frame_dir = f"{data_dir}/frames/frame_{i:06d}.pdb"
        frame_directories.append(frame_dir)
        G = generate_and_plot_graph(frame_dir,
                                save_dir= "/Users/berfininal/Documents/ML-proteins/implicit_graphon/MD_data/MDR_00004293/graphs",
                                k = k, residues = True, random_edges=False)
        data = graph_to_data(G, coordinates= coordinates, knn = k)
        print("..c")
        dataset.append(data)   
        
    out_path = f"/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/residues_protein_dataset_k_{k}_frames_{slide}_coordinates_{coordinates}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    print(dataset)
    print(f"Generated {len(dataset)} datasets with {len(dataset[0])} trials each")

