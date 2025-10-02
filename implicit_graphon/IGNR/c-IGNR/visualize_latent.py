import torch
import os

import numpy as np

from model_cIGNR_two_heads import *
import pickle

from data import *
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

def get_protein_data(dataset):
    ppath = os.getcwd()

    feature = 'row_id'
    batch_size = 20

    if dataset == 'protein_dataset_10':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/protein_dataset_k_5_cignr_2630.pkl", 'rb') as f:  
            data = pickle.load(f)
    elif dataset == 'protein_dataset_5':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/protein_dataset_k_5_frames_5.pkl", 'rb') as f:  
            data = pickle.load(f)
    elif dataset == 'protein_dataset_k_25_frames_5':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/protein_dataset_k_25_frames_5.pkl", 'rb') as f:  
            data = pickle.load(f)
    elif dataset == 'protein_dataset_k_25_frames_5_random_connections':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/protein_dataset_k_25_frames_5_random_edges.pkl", 'rb') as f:  
            data = pickle.load(f)
    elif dataset == 'protein_dataset_k_25_frames_5_random_connections_ratio_0.2':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/protein_dataset_k_25_frames_5_random_edges_ratio_0.2.pkl", 'rb') as f:  
            data = pickle.load(f)





    print(f"data.shape = {len(data)}")
    # print(data)

    G_data = data[0]
    print(len(data))  # 6
    n_sample = len(G_data) # 100
    print(f"n_sample = {n_sample}") # 10
    ################### TODO ##############
    # Implement the rest of the dataloader for protein dataset.
    # print(f"data = {data}")
    n_train = round(n_sample*.9)
    if np.mod(n_train,2)==1:
        n_train = n_train +1

    G_train = G_data[:n_train]
    G_test  = G_data[n_train:]


    print('total samples:'+str(n_sample)+' train samples:'+str(n_train))
    train_dataset, _, n_card = GraphData_tg(G_train, features = feature, add_perm=False, proteins=True)
    # print(f"train_dataset : {train_dataset}")
    test_dataset, _, n_card = GraphData_tg(G_test, features = feature, add_perm=False, proteins=True)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
    test_loader  = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False) 


    return train_loader, test_loader, n_card
    


    
def main_visualize(gnn_type, latent_dim, path, gnn_layers, dim_reduction = 'pca', dataset = 'protein_dataset_5' ):
    mlp_dim_hidden= [48,36,24]
    mlp_num_layer = len(mlp_dim_hidden)
    mlp_act = 'sine'
    emb_dim = 2
    gnn_num_layer = 3
    flag_emb = 0

    if dataset == 'protein_dataset_k_5_frames_5_RESIDUES':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/residues_protein_dataset_k_5_frames_5.pkl",'rb') as f:  
            data = pickle.load(f)
        n_card = 20
        n_sample = len(data)
        n_train = round(n_sample*.9)
        train_dataset = data[:n_train]
        batch_size = 20
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
    elif dataset == 'protein_dataset_k_5_frames_5_RESIDUES_COORDINATES':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/residues_protein_dataset_k_5_frames_5_coordinates_True.pkl",'rb') as f:  
            data = pickle.load(f)
        n_card = 23
        n_sample = len(data)
        n_train = round(n_sample*.9)
        train_dataset = data[:n_train]
        batch_size = 20
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 

    elif dataset == 'protein_dataset_k_25_frames_5_RESIDUES_COORDINATES':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/residues_protein_dataset_k_25_frames_5_coordinates_True.pkl",'rb') as f:  
            data = pickle.load(f)
        n_card = 23
        n_sample = len(data)
        n_train = round(n_sample*.9)
        train_dataset = data[:n_train]
        batch_size = 20
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)     
    
    elif dataset == 'protein_dataset_k_25_frames_5_RESIDUES':      
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/residues_protein_dataset_k_25_frames_5_coordinates_False.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 20
        n_sample = len(data)
        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]  
        batch_size = 20
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)   
    
    elif dataset == 'd2r_knn_4':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/d2r_knn_4_dataset.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 3
        # First 2000 samples only for faster experiments
        data = data[:4000]

        n_sample = len(data)

        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]   

        batch_size = 20
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)      
    
    elif dataset == 'd2r_knn_20':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/d2r_knn_20_dataset.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 3
        # First 2000 samples only for faster experiments
        data = data[:4000]

        n_sample = len(data)

        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]   

        batch_size = 20
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)    

    elif dataset == 'd2r_knn_4_unaligned':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/d2r_knn_4_dataset_unaligned.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 3
        # First 2000 samples only for faster experiments
        data = data[:2000]

        n_sample = len(data)

        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]   

        batch_size = 20
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)    

    elif dataset == 'd2r_knn_20_unaligned':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/d2r_knn_20_dataset_unaligned.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 3
        # First 2000 samples only for faster experiments
        data = data[:2000]

        n_sample = len(data)

        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]   

        batch_size = 20
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)    
    
    #else:

    #    train_loader, test_loader, n_card = get_protein_data(dataset = dataset)

    snet = SirenNet(
        dim_in = 2, # input [x,y] coordinate
        dim_hidden = mlp_dim_hidden,
        dim_out = 1, # output graphon (edge) probability 
        num_layers = mlp_num_layer, # f_theta number of layers
        final_activation = 'sigmoid',
        w0_initial = 30.,
        activation = mlp_act)
    

    snet_coords = SirenNet(
        dim_in = 2, # input [x,y] coordinate
        dim_hidden = mlp_dim_hidden,
        dim_out = 3, # atom coordinates in R^3
        num_layers = mlp_num_layer, # f_theta number of layers
        final_activation = 'sigmoid',
        w0_initial = 30.,
        activation = mlp_act
        )
    
    model = cIGNR_(net_adj=snet, net_coords=snet_coords, input_card=n_card, emb_dim=emb_dim, latent_dim=latent_dim, num_layer=gnn_num_layer, device=device, flag_emb=flag_emb, gnn_type= gnn_type, gnn_layers=gnn_layers)

    checkpoint = torch.load(path, map_location = 'cpu', weights_only = False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(torch.device(device))

    all_zs = []

    ## Get the latent encodings...
    for batch_idx, data in enumerate(train_loader):

        # print(f"Data.shape : {data.x.shape}")

        if dataset == 'protein_dataset_k_5_frames_5_RESIDUES' or dataset == 'protein_dataset_k_5_frames_5_RESIDUES_COORDINATES' or dataset == 'protein_dataset_k_25_frames_5_RESIDUES_COORDINATES' or dataset == 'protein_dataset_k_25_frames_5_RESIDUES':
            x = data.x.float().to(device)
        elif dataset == "d2r_knn_4" or dataset == "d2r_knn_20" or dataset == "d2r_knn_4_unaligned" or dataset == "d2r_knn_20_unaligned":
            x = data.x.float().to(device)
        else:
            x = data.x.to(torch.int64).to(device)             

        edge_index = data.edge_index.to(torch.int64).to(device)


        batch = data.batch.to(torch.int64)

        z, node_representation = model.encode(x, edge_index, batch.to(device))

        all_zs.append(z.detach().cpu())

    latent_codes = torch.cat(all_zs, dim=0)

    print(f"Latents.shape = {latent_codes.shape}")


    num_frames = latent_codes.shape[0]
    frame_indices = np.arange(num_frames)

    cmap = cm.get_cmap('plasma', num_frames)

    latent_codes = latent_codes.detach().cpu().numpy().astype(np.float32)

    print("Has NaNs?", np.isnan(latent_codes).any())
    print("Has Infs?", np.isinf(latent_codes).any())
    print("Max / Min values:", latent_codes.max(), latent_codes.min())

    print(f"latentcode.shape : {latent_codes.shape}")

    if dim_reduction=='pca':

        latent_codes  = StandardScaler().fit_transform(latent_codes) 
        pca = PCA(n_components=2)
        # pca = TruncatedSVD(n_components = 2)

        latent_codes = pca.fit_transform(latent_codes)
        
        explained_var = pca.explained_variance_
        explained_var_ratio = pca.explained_variance_ratio_

        print("Explained variance:", explained_var)
        print("Explained variance ratio:", explained_var_ratio)
        print("Cumulative explained variance ratio:", np.cumsum(explained_var_ratio))

    elif dim_reduction=='tsne':
        latent_codes  = StandardScaler().fit_transform(latent_codes) 
        var = np.var(latent_codes, axis=0)
        # latent_codes= latent_codes[:, var > 1e-8]
        latent_codes = TSNE(n_components=2, perplexity=30, init='pca', random_state=0).fit_transform(latent_codes)
    
    print(f"Latents.shape after PCA= {latent_codes.shape}")

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

    save_plot_path = f"/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/c-IGNR/Result/plots/two_head_plot_{gnn_type}_dim_{latent_dim}_{dataset}_{dim_reduction}.png"

    plt.savefig(save_plot_path, dpi=300) 




if __name__=='__main__':
    gnn_type =  'gin'
    latent_dim = 2
    path = f"/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/c-IGNR/Result/checkpoints/two_head_checkpoint_dataset_d2r_knn_4_unaligned_chebnet_dim_2_knn_4.pt"
    # path = "/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/c-IGNR/Result/00/checkpoint_latent_dim_2_gnn_gconvlstm.pt"
    gnn_layers = [23,2,2,latent_dim]
    dim_reduction = 'pca'


    gnn_type = 'chebnet'
    latent_dim = 2

    # prog_args.dataset = 'protein_dataset_k_25_frames_5_random_connections_ratio_0.2'
    dataset = 'd2r_knn_4_unaligned'
    #'protein_dataset_k_25_frames_5_random_connections' # protein_dataset_k_25_frames_5' # protein_dataset_k_25_frames_5_random_connections
    n_epoch = 1
    emb_dim = 2
    batch_size = 10
    gnn_num_layer = 3

    if dataset == "d2r_knn_4" or dataset == "d2r_knn_20" or dataset == "d2r_knn_4_unaligned" or dataset == "d2r_knn_20_unaligned":
        gnn_layers = [3, 2, 2, latent_dim]
    

    flag_emb = 0
    knn = 4


    lr = 0.01 # 0.01

    main_visualize(gnn_type, latent_dim=latent_dim, path=path, gnn_layers=gnn_layers,
                   dim_reduction=dim_reduction, dataset=dataset)
    
