import os
import h5py
import numpy as np
import pandas as pd

import mdtraj as md
from tqdm import tqdm

if __name__=='__main__':

    ## Load the full trajectory
    traj = md.load('MDR_00004293\\traj_first100.xtc', top = 'MDR_00004293\\structure.pdb')
    print(f"Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")

    print(traj)
    
    
    
    
    
    """
    N = 100
    subtraj = traj[:100]
    subtraj.save(f'traj_first{N}.xtc')
    """


