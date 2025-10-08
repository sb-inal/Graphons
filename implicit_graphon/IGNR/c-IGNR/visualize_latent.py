import torch
import os

import numpy as np

from model_cIGNR_two_heads import *
import pickle

from data import *
from data_ import get_dataset
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import umap


def plot_latents(latent_codes, save_plot_path):
    num_frames = latent_codes.shape[0]
    frame_indices = np.arange(num_frames)
    cmap = cm.get_cmap('plasma', num_frames)


    plt.figure(figsize=(8, 6))
    plt.scatter(
        latent_codes[:, 0], latent_codes[:, 1],
        c=frame_indices,
        cmap=cmap,
        s=40,
        edgecolor='k')

    plt.colorbar(label='Frame index (time)')
    plt.title('Latent codes over time')
    plt.xlabel('Latent dim 1')
    plt.ylabel('Latent dim 2')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_plot_path, dpi=300) 




    
def main_visualize(prog_args):
    
    train_loader, test_loader, n_card = get_dataset(prog_args, num_samples = 4000)

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
    
    model = cIGNR_(net_adj=snet_adj, net_coords = snet_coords, input_card=n_card, emb_dim = prog_args.emb_dim, latent_dim = prog_args.latent_dim, 
                  num_layer=prog_args.gnn_num_layer, 
                  gnn_layers= prog_args.gnn_layers,
                  device=prog_args.device, flag_emb=prog_args.flag_emb, gnn_type = prog_args.gnn_type, JK = 'last')

    checkpoint = torch.load(path, map_location = 'cpu', weights_only = False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(torch.device(device))

    all_zs = []

    ## Get the latent encodings...
    for batch_idx, data in enumerate(train_loader):
        x = data.x.float().to(device)
        
        edge_index = data.edge_index.to(torch.int64).to(device)
        batch = data.batch.to(torch.int64)

        z, node_representation = model.encode(x, edge_index, batch.to(device))
        all_zs.append(z.detach().cpu())

    latent_codes = torch.cat(all_zs, dim=0)

    latent_codes = latent_codes.detach().cpu().numpy().astype(np.float32)

    print("Has NaNs?", np.isnan(latent_codes).any())
    print("Has Infs?", np.isinf(latent_codes).any())
    print("Max / Min values:", latent_codes.max(), latent_codes.min())


    latent_codes  = StandardScaler().fit_transform(latent_codes) 

    ################## PCA #################
    pca = PCA(n_components=2)
    latent_codes_pca = pca.fit_transform(latent_codes)

    explained_var = pca.explained_variance_
    explained_var_ratio = pca.explained_variance_ratio_

    print("Explained variance:", explained_var)
    print("Explained variance ratio:", explained_var_ratio)
    print("Cumulative explained variance ratio:", np.cumsum(explained_var_ratio))

    ################## TSNE #####################
    latent_codes_tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=0).fit_transform(latent_codes)
    
    ################### UMAP ###################
    reducer = umap.UMAP()
    latent_codes_umap = reducer.fit_transform(latent_codes)

    plot_path = f"/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/Result/plots/" 

    plot_latents(latent_codes_pca, plot_path + f"pca/two_head_plot_{gnn_type}_dim_{latent_dim}_{dataset}_pca.png" )
    plot_latents(latent_codes_tsne, plot_path + f"tsne/two_head_plot_{gnn_type}_dim_{latent_dim}_{dataset}_tsne.png")
    plot_latents(latent_codes_umap, plot_path + f"umap/two_head_plot_{gnn_type}_dim_{latent_dim}_{dataset}_umap.png")

