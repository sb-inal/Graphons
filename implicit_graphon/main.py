import os
import h5py
import numpy as np
import pandas as pd

import mdtraj as md
from tqdm import tqdm

import torch 
import plotly.graph_objects as go

from Bio.PDB import PDBParser
import numpy as npƒ 
import networkx as nx
from scipy.spatial import cKDTree
import numpy as np
import networkx as nx
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import random 





import pickle

import re
from collections import defaultdict

from utils import *
import copy

import wandb

""""
wandb.init(
    project="Graphons",   
    entity="s-berfininal"     
)
"""
wandb.init(mode="disabled")


def extract_trajectory_frames(data_dir, stride = 5):

    topology = os.path.join(data_dir, "structure.pdb")
    trajectory = os.path.join(data_dir, "trajectory.xtc")
    out_dir = os.path.join(data_dir, "frames")
    os.makedirs(out_dir, exist_ok=True)

    traj = md.load(trajectory, top = topology)
    n_frames = traj.n_frames

    # Save a subset of frames
    for i in range(0, n_frames, stride):
        single = traj[i]
        out_pdb = os.path.join(out_dir, f"frame_{i:06d}.pdb")

        single.save_pdb(out_pdb)

        if i%100 == 0:
            print(f" Saved {out_pdb}")

AA3_TO_AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    # common variants
    "HID":"H","HIE":"H","HIP":"H","ASH":"D","GLH":"E","LYN":"K","CYX":"C","CYM":"C"
}


def generate_and_plot_graph(frame_path, save_dir, k=5,residues = False, random_edges = False, random_ratio = 0.1):
    parser = PDBParser(QUIET = True)
    structure = parser.get_structure('frame', frame_path)

    # Exract C(alpha) coordinates
    ca_coords = []
    node_metadata = []
    for model in structure:
        for chain in model:
            for res in chain:
                hetflag, resseq, icode = res.get_id()             
                if hetflag != ' ':
                    continue
                if 'CA' not in res:
                    continue
                else:
                    ca_coords.append(res['CA'].get_coord())

                    if residues:
                        res3 = res.get_resname().strip().upper()
                        node_metadata.append({
                            'resname3' : res3,
                            'resname1' : AA3_TO_AA1.get(res3, 'X'),
                            'resid' : int(resseq),
                            "chain" : str(chain.id).strip(),
                            "icode" : (icode or "").strip()
                        })

    coords = np.array(ca_coords)

    # KNN-graph
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k = k+1)
    G = nx.Graph()

    for i, coord in enumerate(coords):
        G.add_node(i, coord = coord)

    for i, (neighbors, _) in enumerate(zip(idxs, dists)):
        for j in neighbors[1:]:
            if not G.has_edge(i,j):
                G.add_edge(i,j)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(frame_path))[0]

    pkl_path = os.path.join(save_dir, f"{basename}_graph_k_{k}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(G,f)

    if random_edges:
        n_edges = int(random_ratio * G.number_of_nodes())
        rng = random.Random(42)
        candidates = list(nx.non_edges(G))
        chosen = rng.sample(candidates, n_edges)
        G.add_edges_from(chosen)


    # Generate adjacency matrix and save
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    

    adj_path = os.path.join(save_dir, f"adj_matrix/{basename}_adj_k_{k}.npy")
    np.save(adj_path, adj_matrix)

    return G

    # print()
    # print(f"Graph saved to {pkl_path} and adjacency matrix saved to {adj_path}")
    # print()




def generate_graphs(frame_path, k = 5):
    frame = np.load(frame_path, allow_pickle=True)

    pos = torch.from_numpy(frame).float()
    edge_idx = knn_graph(pos, k=k, loop = False)
    edge_idx = to_undirected(edge_idx)

    data = Data(pos = pos, edge_index = edge_idx)
    print(data)

    # Interactive plot
    node_trace = go.Scatter3d(
        x = frame[:,0], y = frame[:,1], z = frame[:, 2],
        mode = "markers",
        marker = dict(size =4, color = "blue",
                      name = "C nodes"
                      ))

    edge_x, edge_y, edge_z = [], [], []
    for u, v in edge_idx.t().numpy():
        x0, y0, z0 = frame[u]
        x1,y1,z1 = frame[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x = edge_x, y = edge_y, z = edge_z,
        mode = "lines",
        line = dict(color = "black", width = 0.5),
        hoverinfo = "none"
    )

    fig = go.Figure(edge_trace, node_trace)    

    fig.show()



def generate_dataset(data_dir, frames, k=5, n_trials = 10, ignr = False):
    """
    Generate a dataset of graphs from different time frames of a protein trajectory.
    Graphs are generated using kNN with different k values.
    
    Currently, it is one graphon (protein-kinase), multiple graphs 
    graphon_1 (kinase protein) -> frame0 -> same graph with different k values
    graphon_1 (kinase protein) -> frame1 -> same graph with different k values
    """
    graphs = []
    # Load the once the per-frame adjacencies
    frame_mats = []
    for f in frames:
        fname = f"frame_{f}_adj_k_{k}.npy"
        path = os.path.join(data_dir, fname)
        # print(f"Path : {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        frame_mats.append(np.load(path))

    if ignr:
        return [copy.deepcopy(frame_mats) for _ in range(n_trials)]
    else:
        return frame_mats
    




def full_pipeline(data_dir, frames, ks, k, n_trials, extract_trajectory = False, generate_graphs = False, dataset_gen = True, slide = 5, random_edges = False, random_ratio = 0.1):
    # Extract trajectory frames from a protein structure, saved in frames folder in the given directory
    if extract_trajectory:
        extract_trajectory_frames(data_dir)

    # Frames of interest 
    frame_directories = []
    for i in range(0,10000,slide):
        frame_directories.append(f"{data_dir}/frames/frame_{i:06d}.pdb")

    # Generate the graphs and their corresponding adjacency matrices using the k values and save them in the corresponding directories.
    if generate_graphs:
        for dir in frame_directories:
            # Generate the graph and save the graphs in the given frames
            for k in ks:
                output_dir = f"{data_dir}/graphs"
                generate_and_plot_graph(dir, save_dir = f"{data_dir}/graphs", k = k, random_edges=random_edges, random_ratio = random_ratio)

            print()
            print("Generate graph is done!!!")
            print()

    # Create a dataset
    data_dir_ = f"{data_dir}/graphs/adj_matrix"
    if dataset_gen:
        #if not os.path.exists(data_dir):
        dataset = generate_dataset(f"{data_dir_}", frames=frames, k=k, n_trials=n_trials)

        print()
        print("Generate dataset is done!!!")
        print()

    return dataset

# Each directory contains a different type of kinase protein, and 
# each directory contrains multiple frames extracted from the trajectory of the protein
# k is set to 5 

if __name__=='__main__':
    directories = ["/Users/berfininal/Documents/ML-proteins/implicit_graphon/MD_data/MDR_00004293", 
                   "/Users/berfininal/Documents/ML-proteins/implicit_graphon/MD_data/MDR_00004022"
                   #"/Users/berfininal/Documents/ML-proteins/implicit_graphon/MD_data/MDR_00002630",
                    #"/Users/berfininal/Documents/ML-proteins/implicit_graphon/MD_data/MDR_00002771",
                    #"/Users/berfininal/Documents/ML-proteins/implicit_graphon/MD_data/MDR_00002806",
                    # "/Users/berfininal/Documents/ML-proteins/implicit_graphon/MD_data/MDR_00003013",
                   ]
    

    # folder_path = '/Users/berfininal/Documents/ML-proteins/implicit_graphon/MD_data'
    # directories = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    print(directories)
    
    graphs = []
    slide = 5
    frames = [str(f"{i:06d}") for i in range(0, 10000, slide)]
    ks = [25] #, 10, 15, 20, 25]
    k = 25
    random_ratio = 0.2
    for dir in directories:
        graphs_ = full_pipeline(dir, frames, ks=ks, k=k, n_trials=10, 
                                extract_trajectory=False, 
                                generate_graphs=True, 
                                dataset_gen=True, 
                                slide = slide,
                                random_edges=True, 
                                random_ratio=random_ratio)
        print(f"GRAPHS : {graphs_}")
        graphs.append(graphs_)

    out_path = f"/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/protein_dataset_k_{k}_frames_{slide}_random_edges_ratio_{random_ratio}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"Generated {len(graphs)} datasets with {len(graphs[0])} trials each, and {len((graphs[0][0]))} graphs/frames, with {len(graphs[0][0][0])} and {len(graphs[1][0][0])} atoms in each graphons.")

    print([len(graphs[i][0][0]) for i in range(len(graphs))])




    #### This one is for 
    # Graphon (kinase) -> trials (10) -> [frame0, frame1, frame2, frame3...]
    # in this case, k is fixed for each frame 

    """
    Steps followed above:
    1. Extract trajectory frames from a protein structure.
    2. Generate kNN graphs for each frame, with different k values.
    3. Save the graphs and their adjacency matrices.
    4. Generate a dataset of graphs from different time frames for each protein/graphon with multiple trials.
    """