import torch

from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


import argparse
import pickle



from data import *
from data import rGraphData_tg2,GraphData_tg


def get_dataset(prog_args, num_samples = None):

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

    elif prog_args.dataset == 'd2r_knn_4':
        with open("/home/binal1/Graphons/implicit_graphon/IGNR/IGNR/Data/Data/d2r_knn_4_dataset.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 3
        # First 2000 samples only for faster experiments
        if num_samples is not None:
            data = data[:num_samples]

        n_sample = len(data)

        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]       
        
    elif prog_args.dataset == 'd2r_knn_20':
        with open("/home/binal1/Graphons/implicit_graphon/IGNR/IGNR/Data/Data/d2r_knn_20_dataset.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 3
        # First 2000 samples only for faster experiments
        if num_samples is not None:
            data = data[:num_samples]

        # data = data[:200]

        n_sample = len(data)

        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]      
    
    elif prog_args.dataset == 'd2r_knn_4_unaligned':
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/d2r_knn_4_dataset_unaligned.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 3

        if num_samples is not None:
            data = data[:num_samples]

        n_sample = len(data)

        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]  
    
    elif prog_args.dataset == "d2r_knn_20_unaligned":
        with open("/Users/berfininal/Documents/ML-proteins/implicit_graphon/IGNR/IGNR/Data/d2r_knn_20_dataset_unaligned.pkl",'rb') as f:  
            data = pickle.load(f)

        n_card = 3
        if num_samples is not None:
            data = data[:num_samples]

        n_sample = len(data)

        n_train = round(n_sample*.9)
        print(f"n_sample = {n_sample}") 

        train_dataset = data[:n_train]
        test_dataset = data[n_train:]  

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

    print()
    print(prog_args.dataset)
    print(f"Number of samples - train : {len(train_dataset)}")
    print(f"Number of samples - test : {len(test_dataset)}")
    print()

    train_loader = DataLoader(train_dataset, batch_size=prog_args.batch_size, shuffle=False, drop_last=True) # Data is pre-shuffled by fixed seed
    test_loader  = DataLoader(test_dataset, batch_size=prog_args.batch_size, shuffle=False, drop_last=True) # For evaluating and saving all embeddings

    return train_loader, test_loader, n_card