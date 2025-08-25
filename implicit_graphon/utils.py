import os
import numpy as np
import plotly.graph_objects as go
from Bio.PDB import PDBParser



def plot_graph(coords, idxs, dists, k):
    
    scatter = go.Scatter3d(
        x=coords[:,0], y=coords[:,1], z=coords[:,2],
        mode='markers',
        marker=dict(size=1, color='black')
    )

    ## Plot
    edge_x, edge_y, edge_z, edge_color = [], [], [], []
    for i, (p, neighs, ds) in enumerate(zip(coords, idxs, dists)):
        for nbr_idx, dist in zip(neighs[1:], ds[1:]):
            q = coords[nbr_idx]
            edge_x += [p[0], q[0], None]
            edge_y += [p[1], q[1], None]
            edge_z += [p[2], q[2], None]
            edge_color.append(dist)


    lines = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(
            width=1,
            color=np.repeat(edge_color, 3), 
            colorscale='Viridis',
            colorbar=dict(title=''),
        ),
        hoverinfo='none'
    )

    fig = go.Figure([scatter, lines])
    fig.update_layout(
        title=f"kNN = {k}",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
        ),
        width=800, height=800
    )
    fig.show()
