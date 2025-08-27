import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import copy

import torch

from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.datasets import MoleculeNet

import ot
from ot.gromov import gromov_wasserstein2
import cv2

# --- Model ---
from model_cIGNR import cIGNR

from siren_pytorch import *

from evaluation import *
from visualize_latent import *

from data import rGraphData_tg2,GraphData_tg
from visualize_latent import *

from sklearn.metrics import rand_score
from sklearn.cluster import KMeans
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

import wandb
wandb.login()
project = "Graphons"





'''
Traning Function
'''


################## Added by me (sbi) ####################
def test(args, test_loader, model):

    model.eval()
    loss_list = []

    # with wandb.init(project = project, entity = "_thesis_", config = args) as run:
        
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            # print(f"Data.shape : {data.x.shape}")
            
            if args.dataset == 'protein_dataset_k_5_frames_5_RESIDUES' or args.dataset == 'protein_dataset_k_5_frames_5_RESIDUES_COORDINATES' or args.dataset == 'protein_dataset_k_25_frames_5_RESIDUES_COORDINATES' or args.dataset == 'protein_dataset_k_25_frames_5_RESIDUES':
                x = data.x.float().to(args.device)
            else:
                x = data.x.to(torch.int64).to(args.device)      

            edge_index = data.edge_index.to(torch.int64).to(args.device)
            batch = data.batch.to(torch.int64).to(args.device)
            C_input = to_dense_adj(edge_index, batch=batch)

            # siren mlp
            loss, z, C_recon_list = model(x, edge_index, batch, C_input, args.M)

            loss_list.append(loss.item())

            # run.log({"Test GW2 loss": loss_item})
            
            print(f'Test set, Batch: {batch_idx:03d}, Loss:{loss.item():.4f}')

        loss_batch = np.mean(loss_list) # this is not loss on test set, this is average training loss across batches
        print()
        print(f"Reconstruction loss in test set : {(np.round(loss_batch),3)}")
        # z = get_emb(args,model,test_loader)

    return np.round(loss_batch, 3)




def train(args, train_loader, model, test_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.step_size),
        gamma=float(args.gamma))

    loss_list = []
    loss_list_batch = []


    tmp_list = []


    best_loss_batch  = np.inf
    best_model = None
    #best_C_recon_test = []

    best_acc = 0.
    acc_list = []
    
    since = time.time()
    with wandb.init(project = project, entity = "_thesis_", config = args) as run:
        
        for epoch in range(args.n_epoch):

            model.train()
            
            loss_epoch = []

            for batch_idx, data in enumerate(train_loader):
                print(f"Data : {data}")

                print(f"Data.shape : {data.x.shape}")
                
                if args.dataset == 'protein_dataset_k_5_frames_5_RESIDUES' or args.dataset == 'protein_dataset_k_5_frames_5_RESIDUES_COORDINATES' or args.dataset == 'protein_dataset_k_25_frames_5_RESIDUES_COORDINATES' or args.dataset == 'protein_dataset_k_25_frames_5_RESIDUES' :
                    x = data.x.float().to(args.device)
                else:
                    x = data.x.to(torch.int64).to(args.device)
                edge_index = data.edge_index.to(torch.int64).to(args.device)
                batch = data.batch.to(torch.int64).to(args.device)
                C_input = to_dense_adj(edge_index, batch=batch)


                # siren mlp
                if args.gnn_type =='srgnn':
                    loss, z, C_recon_list  = model(x = data, edge_index = edge_index, batch = batch, C_input = C_input, M = args.M)
                    # print(f"loss = {loss}")
                else:
                    loss, z, C_recon_list = model(x, edge_index, batch, C_input, args.M)

                loss_list.append(loss.item())

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                loss_item = loss.item()
                run.log({"Training GW2 loss": loss_item})
                loss_epoch.append(loss_item)
                
                print(f'Epoch: {epoch:03d}, Batch: {batch_idx:03d}, Loss:{loss.item():.4f}')

            loss_batch = np.mean(loss_epoch) # this is not loss on test set, this is average training loss across batches
            loss_list_batch.append(np.round(loss_batch,3))
            # z = get_emb(args,model,test_loader)

            if loss_batch<best_loss_batch:
                best_loss_batch=loss_batch
                best_model=copy.deepcopy(model)
                # best_z = get_emb(args,model,test_loader)
        ########### ... ###############
        test_loss = test(args, test_loader, model)
        run.log({"Test GW2 loss" : test_loss})
   
    finish = time.time()
    print('time used:'+str(finish-since))

    print('loss on per epoch:')
    print(loss_list_batch)

    # print('acc per epoch:')
    # print(acc_list)

    # save trained model and loss here
    print('Finished Training')
    saved_path = "/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/c-IGNR/Result/checkpoints/"+ f'_checkpoint_dataset_{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}_knn_{args.knn}.pt'
    torch.save({'epoch': epoch, 
        'batch': batch_idx, 
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'configs': args},
        saved_path
       )
    

    np.savetxt(args.save_path + f'_loss_dataset_{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}.out', loss_list, fmt='%.4f', delimiter=',')
    np.savetxt(args.save_path + f'_loss_batch_dataset_{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}.out', loss_list_batch, fmt='%.4f', delimiter=',')
    #np.savetxt(args.save_path + 'acc_list.out',acc_list,fmt='%.4f',delimiter=',')
    #np.save(args.save_path + f'_z_best_dataset_{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}.npy', best_z)

    # plot loss here
    fig1 = plt.figure(figsize=(8,5))
    plt.plot(loss_list)
    fig1.savefig(args.save_path+f'_loss_dataset_{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}.png')
    plt.close(fig1)

    return saved_path



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



def main(prog_args):

    prog_args.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(prog_args.device)

    ppath = os.getcwd()

    prog_args.save_path=ppath+'/Result/'+prog_args.save_dir+'/'
    if not os.path.exists(prog_args.save_path):
        os.makedirs(prog_args.save_path)
    print('saving path is:'+prog_args.save_path)

    # Load Specific Dataset

    if prog_args.dataset in ['IMDB-B','IMDB-M']:   

        with open(ppath+'/Data/'+prog_args.dataset+'.pkl','rb') as f:  
            data = pickle.load(f)

        G_data = data
        n_sample = len(G_data[0])
        print('total samples:'+str(n_sample))
        G_train = G_data #G_data[:n_train] training on all data for unsupervised embedding
        #G_test  = G_data
        gtlabel = G_train[2]
        train_dataset,_,n_card = rGraphData_tg2(G_train,prog_args.dataset,prog_args.feature) 
        test_dataset = train_dataset # obtain unsupervised embedding on whole dataset

    elif prog_args.dataset == 'protein_dataset_10':

        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/protein_dataset_k_5_cignr_2630.pkl",'rb') as f:  
            data = pickle.load(f)

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

    
        print('total samples:'+str(n_sample)+'  train samples:'+str(n_train))
        train_dataset, _, n_card = GraphData_tg(G_train, features = prog_args.feature, add_perm=False, proteins=True)
        # print(f"train_dataset : {train_dataset}")
        test_dataset, _, n_card = GraphData_tg(G_test, features = prog_args.feature, add_perm=False, proteins=True)
        
    elif prog_args.dataset == 'protein_dataset_5':        
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/c-IGNR/Data/protein_dataset_k_5_frames_5.pkl",'rb') as f:  
            data = pickle.load(f)

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
    
        print('total samples:'+str(n_sample)+'  train samples:'+str(n_train))
        train_dataset, _, n_card = GraphData_tg(G_train, features = prog_args.feature, add_perm=False, proteins=True)
        # print(f"train_dataset : {train_dataset}")
        test_dataset, _, n_card = GraphData_tg(G_test, features = prog_args.feature, add_perm=False, proteins=True)

    elif prog_args.dataset == 'protein_dataset_k_25_frames_5':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/protein_dataset_k_25_frames_5.pkl",'rb') as f:  
            data = pickle.load(f)

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
    
        print('total samples:'+str(n_sample)+'  train samples:'+str(n_train))
        train_dataset, _, n_card = GraphData_tg(G_train, features = prog_args.feature, add_perm=False, proteins=True)
        test_dataset, _, n_card = GraphData_tg(G_test, features = prog_args.feature, add_perm=False, proteins=True)

    elif prog_args.dataset == 'protein_dataset_k_25_frames_5_random_connections':
            with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/protein_dataset_k_25_frames_5_random_edges.pkl",'rb') as f:  
                data = pickle.load(f)

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
        
            print('total samples:'+str(n_sample)+'  train samples:'+str(n_train))
            train_dataset, _, n_card = GraphData_tg(G_train, features = prog_args.feature, add_perm=False, proteins=True)
            test_dataset, _, n_card = GraphData_tg(G_test, features = prog_args.feature, add_perm=False, proteins=True)

    elif prog_args.dataset == 'protein_dataset_k_25_frames_5_random_connections_ratio_0.2':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/protein_dataset_k_25_frames_5_random_edges_ratio_0.2.pkl",'rb') as f:  
            data = pickle.load(f)

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
    
        print('total samples:'+str(n_sample)+'  train samples:'+str(n_train))
        train_dataset, _, n_card = GraphData_tg(G_train, features = prog_args.feature, add_perm=False, proteins=True)
        test_dataset, _, n_card = GraphData_tg(G_test, features = prog_args.feature, add_perm=False, proteins=True)

    elif prog_args.dataset == 'protein_dataset_k_5_frames_5_RESIDUES':      
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/residues_protein_dataset_k_5_frames_5.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 20
        n_sample = len(data)
        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]        
        
    elif prog_args.dataset == 'protein_dataset_k_25_frames_5_RESIDUES':      
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/residues_protein_dataset_k_25_frames_5_coordinates_False.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 20
        n_sample = len(data)
        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]    
    
    elif prog_args.dataset == 'protein_dataset_k_5_frames_5_RESIDUES_COORDINATES':      
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/residues_protein_dataset_k_5_frames_5_coordinates_True.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 23
        n_sample = len(data)
        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]    

    elif prog_args.dataset == 'protein_dataset_k_25_frames_5_RESIDUES_COORDINATES':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/residues_protein_dataset_k_25_frames_5_coordinates_True.pkl",'rb') as f:
            data = pickle.load(f)

        n_card = 23
        n_sample = len(data)
        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]


    elif prog_args.dataset == 'bbbp':
        dataset = MoleculeNet(root='data/', name='BBBP')

        train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]

        #train_loader = DataLoader(train_set, batch_size=prog_args.batch_size)
        #test_loader = DataLoader(test_set, batch_size=prog_args.batch_size)

        atom_counts = [data.x.size(0) for data in dataset]

        # Basic stats
        max_atoms = max(atom_counts)

        n_card = max_atoms
            
    else: # For learning parameterized graphon on synthetic data

        with open(ppath+'/Data/'+prog_args.dataset+'.pkl','rb') as f:  
            data = pickle.load(f)

        G_data = data[0]
        labels = data[1]

        n_sample = len(G_data)
        n_train  = round(n_sample*.9)
        if np.mod(n_train,2) == 1:
            n_train = n_train+1
        print('total samples:'+str(n_sample)+'  train samples:'+str(n_train))
        G_train = G_data[:n_train]
        G_test  = G_data[n_train:]
        tlabel  = labels[n_train:] 

        train_dataset,_,n_card = GraphData_tg(G_train, features=prog_args.feature, add_perm=True)
        print(f"train_dataset : {train_dataset}")
        test_dataset,_,n_card = GraphData_tg(G_test, features=prog_args.feature, add_perm=True)


    print(prog_args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=prog_args.batch_size, shuffle=False, drop_last=True) # Data is pre-shuffled by fixed seed
    test_loader  = DataLoader(test_dataset, batch_size=prog_args.batch_size, shuffle=False) # For evaluating and saving all embeddings


    prog_args.step_size = len(train_loader)*prog_args.step_size     
    prog_args.mlp_dim_hidden = [int(x) for x in prog_args.mlp_dim_hidden.split(',')]
    prog_args.mlp_num_layer = len(prog_args.mlp_dim_hidden)

    snet = SirenNet(
        dim_in = 2, # input [x,y] coordinate
        dim_hidden = prog_args.mlp_dim_hidden,
        dim_out = 1, # output graphon (edge) probability 
        num_layers = prog_args.mlp_num_layer, # f_theta number of layers
        final_activation = 'sigmoid',
        w0_initial = 30.,
        activation = prog_args.mlp_act
        )
    
    model = cIGNR(net=snet, input_card=n_card, emb_dim = prog_args.emb_dim, latent_dim = prog_args.latent_dim, 
                  num_layer=prog_args.gnn_num_layer, 
                  gnn_layers= prog_args.gnn_layers,
                  device=prog_args.device, flag_emb=prog_args.flag_emb, gnn_type = prog_args.gnn_type)

    model = model.to(torch.device(prog_args.device))
    saved_path = train(prog_args, train_loader, model, test_loader)

    # plot_eval(prog_args,model,train_dataset,9,'sample_train')

    if prog_args.flag_eval==1 and prog_args.dataset in ['gCircle', '2ratio_rand']:
        print('Evaluating Graphon Matching Loss')
        # Note: evaluating Graphon Reconstruction can be slow due to comparing large resolution matrices
        compute_graphon_loss(prog_args,model,test_loader,tlabel)

    return saved_path


if __name__ == '__main__':

    gnn_types = ['gconvgru', 'gconvlstm', 'chebnet'] #'gin',
    latent_dims = [2, 4, 8, 16, 128]

    for gnn_type in gnn_types:
        for dim in latent_dims:
            prog_args = arg_parse()
            prog_args.gnn_type = gnn_type
            prog_args.latent_dim = dim

            # prog_args.dataset = 'protein_dataset_k_25_frames_5_random_connections_ratio_0.2'
            prog_args.dataset = 'protein_dataset_k_25_frames_5_RESIDUES'
            #'protein_dataset_k_25_frames_5_random_connections' # protein_dataset_k_25_frames_5' # protein_dataset_k_25_frames_5_random_connections
            prog_args.n_epoch = 3
            prog_args.emb_dim = 2
            prog_args.batch_size = 10
            prog_args.gnn_num_layers = 3

            if prog_args.dataset == 'protein_dataset_k_5_frames_5_RESIDUES' or prog_args.dataset == 'protein_dataset_k_25_frames_5_RESIDUES':
                prog_args.gnn_layers = [20, 2, 2, dim]
            elif prog_args.dataset == "protein_dataset_k_5_frames_5_RESIDUES_COORDINATES" or prog_args.dataset == "protein_dataset_k_25_frames_5_RESIDUES_COORDINATES":
               prog_args.gnn_layers = [23, 2, 2, dim]

            prog_args.flag_emb = 0
            prog_args.knn = 25


            prog_args.lr = 0.001 # 0.01
            print()
            print(f"Dataset : {prog_args.dataset}")
            print()

            saved_path = main(prog_args)
            
            for dim_red in ['pca', 'tsne']:
                main_visualize(gnn_type, latent_dim=dim, path=saved_path, 
                               gnn_layers=prog_args.gnn_layers, dim_reduction=dim_red,
                               dataset=prog_args.dataset)
            
