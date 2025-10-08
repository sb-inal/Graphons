import argparse
from xml.parsers.expat import model
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
from model_cIGNR_two_heads import cIGNR_
from siren_pytorch import *

from evaluation import *
from visualize_latent import *
from data_ import get_dataset

from sklearn.metrics import rand_score
from sklearn.cluster import KMeans
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import wandb
wandb.login()
project = "Graphons"



### multiple loss training
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad


from utils import save_ca_pdb, lddt, tm_score, arg_parse


'''
Traning Function
'''
################## Added by me (sbi) ####################
def test(args, test_loader, model):

    model.eval()
    loss_list = []
    loss_coords_ = []

    # with wandb.init(project = project, entity = "_thesis_", config = args) as run:
        
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            # print(f"Data.shape : {data.x.shape}")
            x = data.x.float().to(args.device)

            edge_index = data.edge_index.to(torch.int64).to(args.device)
            batch = data.batch.to(torch.int64).to(args.device)
            C_input = to_dense_adj(edge_index, batch=batch)

            loss, coords_pred = model(x, edge_index, batch, C_input, args.M)        
            loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))
            
            print(f"Loss GW : {loss.item():.4f}, Loss coords : {loss_coords.item():.4f}")
            total_loss = loss + loss_coords

            loss_list.append(total_loss.item())
            loss_coords_.append(loss_coords.item())


            if batch_idx%20== 0 or batch_idx==76 :
                path_pdb = f"/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/pdbs/true/test/mse_train_{prog_args.dataset}_{prog_args.gnn_type}_{prog_args.latent_dim}_{batch_idx}.pdb"
                save_ca_pdb(coords_pred, path_pdb)
                path_pdb_true = f"/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/pdbs/true/test/mse_train_{prog_args.dataset}_{prog_args.gnn_type}_{prog_args.latent_dim}_{batch_idx}.pdb"
                save_ca_pdb(x, path_pdb_true)

            # run.log({"Test GW2 loss": loss_item})
            
            print(f'Test set, Batch: {batch_idx:03d}, Loss:{loss.item():.4f}')

        loss_batch = np.mean(loss_list) # this is not loss on test set, this is average training loss across batches
        loss_batch_coords = np.mean(loss_coords_)
        print()
        print(f"Reconstruction loss in test set : {(np.round(loss_batch),3)}")
        print(f"Reconstruction loss in test set coords: {(np.round(loss_batch_coords),3)}")
        # z = get_emb(args,model,test_loader)

    return np.round(loss_batch, 3)



def train(args, train_loader, model, test_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #aggregator = UPGrad()
    batch_size = args.batch_size

    ids_in_optim = {id(p) for g in optimizer.param_groups for p in g['params']}
    missing = [n for n,p in model.mlp_coords.named_parameters() if id(p) not in ids_in_optim]
    print("Missing mlp_coords params in optimizer:", missing)  # should be []


    #lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #    optimizer,
    #    step_size=int(args.step_size),
    #    gamma=float(args.gamma))
    

    for i, param_group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in param_group['params'])
        print(f"Group {i}: {num_params:,} parameters")
        print(" lr:", param_group['lr'])
        print(" weight_decay:", param_group['weight_decay'])

    loss_list = []
    loss_list_batch = []
    tmp_list = []

    best_loss_batch  = np.inf
    best_model = None

    best_acc = 0.0
    acc_list = []
    
    since = time.time()
    with wandb.init(project = project, entity = "_thesis_", config = args) as run:
        for epoch in range(args.n_epoch):
            model.train()
            
            loss_epoch = []

            for batch_idx, data in enumerate(train_loader):

                x = data.x.float().to(args.device)
                edge_index = data.edge_index.to(torch.int64).to(args.device)

                edge_index = edge_index.long()

                batch = data.batch.to(torch.int64).to(args.device)
                C_input = to_dense_adj(edge_index, batch=batch)

                # siren mlp
                if args.gnn_type =='srgnn':
                    loss, z, C_recon_list  = model(x = data, edge_index = edge_index, batch = batch, C_input = C_input, M = args.M)
                else:
                    loss, coords_pred = model(x, edge_index, batch, C_input, args.M)        
                    loss_coords = torch.nn.functional.mse_loss(coords_pred, x.to(device))
                
                    total_loss = loss + loss_coords

                loss_list.append(total_loss)
                total_loss.backward()

                #g = [p.grad.abs().mean().item() for p in model.mlp_coords.parameters() if p.grad is not None]
                #print("mlp_coords grad means:", g)  # should be non-zero


                if batch_idx%100==0 or batch_idx==688:
                    path_pdb = f"/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/pdbs/predicted/train/mse_train_{prog_args.dataset}_{prog_args.gnn_type}_{prog_args.latent_dim}_{batch_idx}.pdb"
                    save_ca_pdb(coords_pred, path_pdb)
                    path_pdb_true = f"/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/pdbs/true/train/mse_train_{prog_args.dataset}_{prog_args.gnn_type}_{prog_args.latent_dim}_{batch_idx}.pdb"
                    save_ca_pdb(x, path_pdb_true)

                #### LDDT score 
                N = 273
                lddt_ = lddt(coords_pred.view(batch_size, N, 3), x.view(batch_size, N, 3))
                tm_score_ = tm_score(x, coords_pred)


                # mtl_backward(losses=loss, features = z, aggregator=aggregator)
                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()
                
                #print("decoder2 grad mean:",
                #sum((p.grad.abs().mean() for p in model.mlp_coords.parameters() if p.grad is not None)))

                #loss_item = sum(loss).item()
                run.log({"Training GW2 loss": loss.item()})
                run.log({"Training MSE loss": loss_coords.item()})
                run.log({"Training Total loss": total_loss.item()})

                run.log({"Training LDDT score": lddt_})
                run.log({"Training TM-Score ": tm_score_})

                print(f'Epoch: {epoch:03d}, Batch: {batch_idx:03d}, Loss:{total_loss.item():.4f}, Loss GW : {loss.item():.4f}, Loss coords : {loss_coords.item():.4f}')

            loss_batch = np.mean(loss_epoch) # this is not loss on test set, this is average training loss across batches
            loss_list_batch.append(np.round(loss_batch,3))


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

    # save trained model and loss here
    print('Finished Training')
    saved_path = "/home/binal1/Graphons/implicit_graphon/IGNR/c-IGNR/Result/checkpoints/"+ f'two_head_checkpoint_dataset_{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}_knn_{args.knn}.pt'
    torch.save({'epoch': epoch, 
        'batch': batch_idx, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'configs': args},
        saved_path
       )
    
    #np.savetxt(args.save_path + f'_loss_dataset_{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}.out', loss_list, fmt='%.4f', delimiter=',')
    #np.savetxt(args.save_path + f'_loss_batch_dataset_{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}.out', loss_list_batch, fmt='%.4f', delimiter=',')
    #np.savetxt(args.save_path + 'acc_list.out',acc_list,fmt='%.4f',delimiter=',')
    #np.save(args.save_path + f'_z_best_dataset_{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}.npy', best_z)

    # plot loss here
    # fig1 = plt.figure(figsize=(8,5))
    # plt.plot(loss_list)
    #fig1.savefig(args.save_path+f'_loss_dataset_{args.dataset}_{args.gnn_type}_dim_{args.latent_dim}.png')
    #plt.close(fig1)

    return saved_path





def main(prog_args):

    prog_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(prog_args.device)

    ppath = os.getcwd()

    prog_args.save_path=ppath+'/Result/'+prog_args.save_dir+'/'
    if not os.path.exists(prog_args.save_path):
        os.makedirs(prog_args.save_path)
    print('saving path is:'+prog_args.save_path)

    # Load Specific Dataset
    train_loader, test_loader, n_card = get_dataset(prog_args)

    print(f"Train set length : {len(train_loader.dataset)}")

    prog_args.step_size = len(train_loader)*prog_args.step_size     
    prog_args.mlp_dim_hidden = [int(x) for x in prog_args.mlp_dim_hidden.split(',')]
    prog_args.mlp_num_layer = len(prog_args.mlp_dim_hidden)

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

    model = model.to(torch.device(prog_args.device))

    saved_path = train(prog_args, train_loader, model, test_loader)

    if prog_args.flag_eval==1 and prog_args.dataset in ['gCircle', '2ratio_rand']:
        print('Evaluating Graphon Matching Loss')
        # Note: evaluating Graphon Reconstruction can be slow due to comparing large resolution matrices
        compute_graphon_loss(prog_args,model,test_loader,tlabel)

    return saved_path, train_loader, test_loader, model


if __name__ == '__main__':

    gnn_types = ['chebnet', 'gin', 'gconvgru'] # ['chebnet', 'gin', 'gconvgru'] # ['gconvgru', 'gconvlstm', 'chebnet'] #'gin',
    latent_dims = [2, 4, 8, 16, 128]

    for gnn_type in gnn_types:
        for dim in latent_dims:
            prog_args = arg_parse()
            prog_args.gnn_type = gnn_type
            prog_args.latent_dim = dim

            prog_args.dataset = 'd2r_knn_20'
            prog_args.n_epoch = 1
            prog_args.emb_dim = 2
            prog_args.batch_size = 16
            prog_args.gnn_num_layer = 3

            if prog_args.dataset == 'protein_dataset_k_5_frames_5_RESIDUES' or prog_args.dataset == 'protein_dataset_k_25_frames_5_RESIDUES':
                prog_args.gnn_layers = [20, 2, 2, dim]
            elif prog_args.dataset == "protein_dataset_k_5_frames_5_RESIDUES_COORDINATES" or prog_args.dataset == "protein_dataset_k_25_frames_5_RESIDUES_COORDINATES":
               prog_args.gnn_layers = [23, 2, 2, dim]
            elif prog_args.dataset == "d2r_knn_4" or prog_args.dataset == "d2r_knn_20" or prog_args.dataset == "d2r_knn_4_unaligned" or prog_args.dataset == "d2r_knn_20_unaligned":
                prog_args.gnn_layers = [3, 8, 8, dim]
            
            prog_args.flag_emb = 0
            prog_args.knn = 20

            prog_args.lr = 1e-4
            print()
            print(f"Dataset : {prog_args.dataset}")
            print()

            saved_path, train_loader, test_loader, model = main(prog_args)
            print(f"Saved path : {saved_path}")

            ############## Torch to PDB ############### 

            # save_ca_pdb(train_loader, model, f"_predicted_pdbs/_mse_train_sample_dataset_{prog_args.dataset}_{prog_args.gnn_type}_{prog_args.latent_dim}.pdb", chain_id = "A", start_resi = 1)
            # save_ca_pdb(test_loader, model, f"_predicted_pdbs/_mse_test_sample_dataset_{prog_args.dataset}_{prog_args.gnn_type}_{prog_args.latent_dim}.pdb", chain_id = "A", start_resi = 1)

            ###############################################
            main_visualize(prog_args)
            
            
