import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# import mdtraj as md
import pytraj as pt
from tqdm import tqdm


def process_pdb(pdb_id, data_dir, h5_file, output_dir):
    pdb_id_lower = pdb_id.lower()
    nc_file = os.path.join(data_dir, f"{pdb_id_lower}.nc")
    top_file = os.path.join(data_dir, f"{pdb_id_lower}.top")
    if not os.path.exists(nc_file) or not os.path.exists(top_file):
        return None, f"{pdb_id}: Missing NC or Top file."
    # traj = pt.load(nc_file, top_file)
    traj = md.load(nc_file, top_file)
    
    # Mask out hydrogen atoms
    traj_filtered = traj["!@H*"]
    with h5py.File(h5_file, "r") as h5f:
        if pdb_id not in h5f:
            return None, f"{pdb_id}: HDF5 dataset not found."
        pitem = h5f[pdb_id]
        cutoff = int(pitem["molecules_begin_atom_index"][:][-1])
        # Get the number of atoms for each molecule after filtering
        atom_counts = [mol.n_atoms for mol in traj_filtered.topology.mols]
        expected_counts = [
            int(pitem["molecules_begin_atom_index"][:][i + 1])
            - int(pitem["molecules_begin_atom_index"][:][i])
            for i in range(len(pitem["molecules_begin_atom_index"][:]) - 1)
        ]
        expected_counts.append(len(pitem["atoms_coordinates_ref"][:]) - cutoff)
        if atom_counts != expected_counts:
            return None, f"{pdb_id}: Atom counts do not match expected values."
        # Compute protein and ligand atom ranges
        protein_atom_count = sum(atom_counts[:-1])  # All except the last molecule
        ligand_atom_count = atom_counts[-1]  # Last molecule
        total_atom_count = traj_filtered.top.n_atoms  # Total number of atoms
        # Verify with HDF5 file
        assert (
            protein_atom_count == cutoff
        ), f"Protein atom count ({protein_atom_count}) does not match cutoff ({cutoff})"
        # Compute disgtance for all atoms excluding hydrogen
        #dist_all = np.float32(pt.matrix.dist(traj_filtered))
                
        n = traj_filtered.n_atoms
        pairs = np.array(np.triu_indices(n, k=1)).T
        dist_all = md.compute_distances(traj_filtered, pairs).astype(np.float32)

        
        assert dist_all.shape == (
            traj_filtered.top.n_atoms,
            traj_filtered.top.n_atoms,
        ), f"All atoms correlation matrix shape mismatch: {dist_all.shape}"
        
        # Save the correlation matrix and metadata in a single .npz file
        np.savez(
            os.path.join(output_dir, f"{pdb_id_lower}_distance_data.npz"),
            dist_all=dist_all,
            molecules_begin_atom_index=pitem["molecules_begin_atom_index"][:],
            atom_count=atom_counts,
            protein_atom_count=protein_atom_count,
            ligand_atom_count=ligand_atom_count,
            total_atom_count=total_atom_count,
        )

        # Collect metadata for CSV
        metadata = {
            "pdb_id": pdb_id,
            "molecules_begin_atom_index": pitem["molecules_begin_atom_index"][:].tolist(),
            "atom_count": atom_counts,
            "protein_atom_count": protein_atom_count,
            "ligand_atom_count": ligand_atom_count,
            "total_atom_count": total_atom_count,
        }
        return metadata, None


def compute_and_save_correlation_matrices(ids_file, data_dir, h5_file, output_dir):
    with open(ids_file, "r") as f:
        ids = f.read().splitlines()
    metadata_list = []
    # Open the HDF5 file once for all IDs
    h5f = h5py.File(Path(h5_file).absolute(), "r")
    for pdb_id in tqdm(ids, total=len(ids), desc="Processing PDB files"):
        metadata, error = process_pdb(pdb_id, data_dir, h5_file, output_dir)
        if metadata:
            metadata_list.append(metadata)
        if error:
            print(error)
    # Close the HDF5 file
    h5f.close()
    # Ensure the metadata folder exists
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    # Save metadata to CSV with unique name
    csv_file = os.path.join(metadata_dir, f"{Path(ids_file).stem}_no_hydrogen_metadata.csv")
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ids_file",
        type=str,
        required=True,
        help="Path to the file containing PDB IDs",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing NC and TOP files",
    )
    parser.add_argument("--h5_file", type=str, required=True, help="Path to the HDF5 file")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output correlation matrices and metadata",
    )
    args = parser.parse_args()
    compute_and_save_correlation_matrices(
        args.ids_file, args.data_dir, args.h5_file, args.output_dir
    )






