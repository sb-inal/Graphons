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
    ppath = os.getcwd()

    save_path=ppath+'/Result/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('saving path is:'+save_path)

    with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/d2r_knn_4_dataset.pkl",'rb') as f:  
        data = pickle.load(f)

    n_card = 3

    print(f"data.shape = {len(data)}")

    # data = data[:2000]

    n_sample = len(data)

    n_train = round(n_sample*.9)
    print(f"n_sample = {n_sample}") 

    train_dataset = data[:n_train]
    test_dataset = data[n_train:]       


    print(f"n_sample test set = {len(test_dataset)}") 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # Data is pre-shuffled by fixed seed
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # For evaluating and saving all embeddings

    return train_loader, test_loader



def load_checkpoint_(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    prog_args = checkpoint['configs']   # if you want to reuse configs

    snet_adj = SirenNet(
        dim_in = 2, # input [x,y] coordinate
        dim_hidden = prog_args.mlp_dim_hidden,
        dim_out = 1, # output graphon (edge) probability 
        num_layers = prog_args.mlp_num_layer, # f_theta number of layers
        final_activation = 'sigmoid',
        w0_initial = 30.,
        activation = prog_args.mlp_act
        )
    

    snet_coords = SirenNet(
        dim_in = 2, # input [x,y] coordinate
        dim_hidden = prog_args.mlp_dim_hidden,
        dim_out = 3, # atom coordinates in R^3
        num_layers = prog_args.mlp_num_layer, # f_theta number of layers
        final_activation = 'sigmoid',
        w0_initial = 30.,
        activation = prog_args.mlp_act
        )
    
    input_card = 3

    model = cIGNR_(snet_adj, snet_coords, input_card, prog_args.emb_dim, prog_args.latent_dim, prog_args.gnn_num_layer, prog_args.gnn_layers, 
                   device=prog_args.device, flag_emb=prog_args.flag_emb, gnn_type = prog_args.gnn_type).to(device)
   
    # --- 3. Load weights ---
    model.load_state_dict(checkpoint['model_state_dict'])


    return model






def check(dataset, model):

    train_loader, test_loader = get_dataset(dataset, batch_size=10)

    # Load model
    # model = load_checkpoint_(path)

    # Evaluate model
    model.eval()
    with torch.no_grad():

        for batch_idx, data in enumerate(train_loader):
            print(f"Data : {data}")

            print(f"Data.shape : {data.x.shape}")
            x = data.x.float().to(device)

            print(f"x.shape: {x.shape}")

            edge_index = data.edge_index.to(torch.int64).to(device)

            edge_index = edge_index.long()

            batch = data.batch.to(torch.int64).to(device)
            C_input = to_dense_adj(edge_index, batch=batch)

            graph_rep, node_rep = model.encode(x, edge_index, batch)
            M = 273

            print(node_rep[0])
            print(node_rep[274])

            print(f"node_rep.shape : {node_rep.shape}")
            pred_coords = model.mlp_coords(node_rep.to(device))

            print(f"pred_coords.shape : {pred_coords.shape}")
            print(f"pred_coords.shape : {pred_coords[0]}")
            print(f"pred_coords.shape : {pred_coords[274]}")

            save_ca_pdb(pred_coords.cpu(), "predicted.pdb", chain_id = "A", start_resi = 1)


            break






if __name__ == "__main__":
    dataset = 'd2r_knn_20_unaligned'
    path = "/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/c-IGNR/Result/checkpoints/two_head_checkpoint_dataset_d2r_knn_20_unaligned_gin_dim_16_knn_4.pt"

    check(dataset, path)