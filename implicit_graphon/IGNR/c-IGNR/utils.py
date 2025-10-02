import torch
from torch_geometric.loader import DataLoader

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

from model_cIGNR_two_heads import cIGNR_
from siren_pytorch import SirenNet

import os
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

import pickle 

print(f"Device : {device}")
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_undirected




from torch_to_pdb import save_ca_pdb



def get_dataset(dataset, batch_size = 10):

    print(f"Device : {device}")

    ppath = os.getcwd()

    save_path=ppath+'/Result/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('saving path is:'+save_path)

    if dataset == 'd2r_knn_4':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/d2r_knn_4_dataset.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 3
        # First 2000 samples only for faster experiments

        print(f"data.shape = {len(data)}")

        # data = data[:2000]

        n_sample = len(data)

        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]       


        print(f"n_sample test set = {len(test_dataset)}") 

        return train_dataset, test_dataset
    


def save_ca_pdb(coords: torch.Tensor,
    path: str,
    chain_id: str = "A",
    start_resi: int = 1,
    residue_names = None,  # optional list of length N
):
    """
    Write Cα-only coordinates to a PDB file PyMOL will read.
    coords: [N,3] tensor (Å). Each row is (x,y,z) of a residue's CA atom.
    """
    
    coords = coords.detach().cpu().float()
    assert coords.ndim == 2 and coords.shape[1] == 3, "coords must be [N,3]"
    N = coords.shape[0]
    if residue_names is None:
        residue_names = ["GLY"] * N  # placeholder residue type

    with open(path, "w") as f:
        serial = 1
        for i in range(N):
            x, y, z = coords[i].tolist()
            resn = residue_names[i][:3]  # 3-char residue name
            resi = start_resi + i

            # PDB fixed-width columns; element 'C' (carbon) in cols 77–78
            line = (
                f"{'ATOM':<6}"            # 1-6  Record name
                f"{serial:5d} "           # 7-11 Serial, 12 space
                f"{'CA':^4}"              # 13-16 Atom name (centered)
                f"{'':1}"                 # 17    altLoc
                f"{resn:>3} "             # 18-20 resName, 21 space
                f"{chain_id:1}"           # 22    chainID
                f"{resi:4d}"              # 23-26 resSeq
                f"{'':1}"                 # 27    iCode
                f"{'':3}"                 # 28-30 spaces
                f"{x:8.3f}{y:8.3f}{z:8.3f}"  # 31-54 x,y,z
                f"{1.00:6.2f}{0.00:6.2f}"    # 55-60 occ, 61-66 temp
                f"{'':10}"                # 67-76 spaces
                f"{'C':>2}"               # 77-78 element
                f"{'':2}\n"               # 79-80
            )
            f.write(line)
            serial += 1
        f.write("END\n")






if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset('d2r_knn_4', batch_size=10)
    N = 273
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False) # Data is pre-shuffled by fixed seed
    test_loader  = DataLoader(test_dataset, batch_size=10, shuffle=False) # For evaluating and saving all embeddings

    train_sample = next(iter(train_loader))
    print(train_sample.x[:N].shape)

    test_sample = next(iter(test_loader))
    print(test_sample.x[:N].shape)


    save_ca_pdb(train_sample.x[:N].cpu(), "train_sample_.pdb", chain_id = "A", start_resi = 1)
