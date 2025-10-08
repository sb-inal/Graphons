import torch
from torch_geometric.loader import DataLoader

import argparse

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

from model_cIGNR_two_heads import cIGNR_
from siren_pytorch import SirenNet

import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pickle 

print(f"Device : {device}")
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_undirected


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphonAE arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    ### Optimization parameter
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--step_size', dest='step_size', type=float,
            help='Learning rate scheduler step size')
    parser.add_argument('--gamma', dest='gamma', type=float,
            help='Learning rate scheduler gamma')

    ### Training specific
    parser.add_argument('--n_epoch', dest='n_epoch', type=int,
            help='Number of training epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--cuda', dest='cuda', type=int,
            help='cuda device number')

    parser.add_argument('--feature', dest='feature',
            help='Feature used for encoder.')
    parser.add_argument('--save_dir', dest='save_dir',
            help='name of the saving directory')
    parser.add_argument('--flag_eval',dest='flag_eval',type=int,help='whether to compute graphon recon error') 

    # General model param
    parser.add_argument('--flag_emb',dest='flag_emb',type=int) 
    parser.add_argument('--gnn_num_layer', dest='gnn_num_layer', type=int)

    ### Model selection and sampling reconstruction number
    #parser.add_argument('--add_perm',dest='add_perm',type=bool)
    #parser.add_argument('--model_ind',dest='model_ind',type=int) #model number to select
    parser.add_argument('--M',dest='M',type=int) #sampling number for graph reconstruction if needed

    ### SIREN-MLP-model specific
    parser.add_argument('--mlp_dim_hidden', dest='mlp_dim_hidden') #hidden dim (number of neurons) for f_theta
    parser.add_argument('--emb_dim', dest='emb_dim', type=int)
    parser.add_argument('--latent_dim', dest='latent_dim', type=int) #from graph embedding to latent embedding, reducing graph embedding dimension
    parser.add_argument('--mlp_act', dest='mlp_act') # whether to use sine activation for the mlps

    parser.add_argument('--gnn_type', dest='gnn_type') # to specify the type of encoder used
    parser.add_argument('--gnn_layers', nargs='+', type=int, default=[8,8,8])
    parser.add_argument('--gnn_layers_coords', nargs='+', type=int, default=[8,8,8])

    parser.add_argument('--knn', type=int, default=5)

    parser.set_defaults(dataset='2ratio_rand',
                        feature='row_id',
                        lr=0.01,
                        n_epoch=12,
                        batch_size=10,
                        cuda=0,
                        save_dir='00',
                        step_size=4,
                        gamma=0.1,
                        gnn_num_layer=3,
                        latent_dim=16,
                        emb_dim=16,
                        mlp_dim_hidden='48,36,24',
                        mlp_act = 'sine',
                        flag_emb=1,
                        flag_eval=0,
                        M=0)
    return parser.parse_args()




def save_ca_pdb(coords,
    path: str,
    chain_id: str = "A",
    start_resi: int = 1,
    residue_names = None,  # optional list of length N
):
    """
    Write Cα-only coordinates to a PDB file PyMOL will read.
    coords: [N,3] tensor (Å). Each row is (x,y,z) of a residue's CA atom.
    """
    N = 273

    coords = coords[:N]
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



def lddt(predicted_coords, 
    true_coords,
    cutoff = torch.tensor(15.0).to(device),
    thresholds: list = [0.5, 1.0, 2.0, 4.0]
) -> float:
    """ LDDT is a measure of the difference between the true distance matrix
    and the distance matrix of the predicted points. The difference is computed only on 
    points closer than cutoff **in the true structure**.
    

    It is an approximate score. 
    Code is extracted and adapted from 
    AlphaFold 2: https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/lddt.py
    
    Thresholds are hardcoded to be [0.5, 1.0, 2.0, 4.0] as in the original paper.
    
    true_points_mask: (batch, length, 1) binary-valued float array.  This mask
      should be 1 for points that exist in the true points.
    
    I set the true_points_mask to 1 in the code. 
    
    """

    assert len(predicted_coords.shape) ==3
    assert predicted_coords.shape[-1] ==3

    dmat_true = torch.cdist(true_coords, true_coords).to(device)  # [batch_size, N, N] , where N = 273 
    dmat_predicted = torch.cdist(predicted_coords, predicted_coords).to(device)  # [batch_size, N, N] , where N = 273 

    ### distances to score
    true_points_mask = torch.ones(true_coords.shape[0], true_coords.shape[1], 1).to(device)


    dist_to_score = (
        (dmat_true <cutoff) 
        * true_points_mask 
        * true_points_mask.permute(0, 2, 1)
        * (1 - torch.eye(dmat_true.shape[1]).to(device)))

    # shift unscores distances to be far away
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    score = 0.25* (
            (dist_l1 < 0.5).to(torch.float32) 
            + (dist_l1 < 1.0).to(torch.float32)
            + (dist_l1 < 2.0).to(torch.float32)
            + (dist_l1 < 4.0).to(torch.float32)
            )

    # Normalize over the approapriate axes (normalizing over batches)
    reduce_axes = (-2,-1)
    norm = 1.0 / (1e-10 + torch.sum(dist_to_score, axis = reduce_axes))
    score = norm * (1e-10 + torch.sum(dist_to_score * score, axis = reduce_axes))

    return round(torch.mean(torch.round(score, decimals = 3)).item(),3)


def tm_score(true_coords, predicted_coords):

    N = int(true_coords.shape[0])

    d = torch.linalg.vector_norm(true_coords - predicted_coords, dim = -1)

    d0 = 1.24 * (max(N-15, 1)) ** (1/3) - 1.8 
    d0 = max(d0, 0.5)

    score = (1.0/N) * torch.sum(1.0 / (1.0 + (d /d0)) **2)
    score = torch.round(torch.tensor([score]), decimals = 3)

    return round(score.item(),3)


