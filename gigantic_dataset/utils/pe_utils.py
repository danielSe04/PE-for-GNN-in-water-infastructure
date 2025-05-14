import sklearn
import matplotlib.pyplot as plt
import os
import torch
from torch import Tensor
from torch_geometric.utils import degree, subgraph
from abc import ABC, abstractmethod
import wandb
import io

def plot_pe_wandb(pe: Tensor, plot_name: str) -> None:
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=42) #TODO perhaps change perplexity and random_state parameter
    z_2d = tsne.fit_transform(pe.cpu())
    fig, ax = plt.subplots(figsize=(16,12))
    
    ax.scatter(z_2d[:, 0], z_2d[:, 1], s=10)
    ax.set_title("t-SNE visualization of positional encoding")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    wandb.log({plot_name: wandb.Image(fig)})

def sparse_diagonal(G: Tensor):
        G = G.coalesce()
        dest_nodes, source_nodes = G.indices()
        values = G.values()
        mask = dest_nodes == source_nodes
        diag_values = torch.zeros(G.size(0), device=G.device)
        diag_values[dest_nodes[mask]] = values[mask]
        return diag_values

class PE_Initializer(ABC):
    @abstractmethod
    def __call__(self, num_nodes: int, pe_dim: int, edge_index: Tensor | None = None, batch: Tensor | None = None, geo_coords: Tensor | None = None):
        pass

class RWPE_Initializer(PE_Initializer):
    def __call__(self, num_nodes: int, pe_dim: int, edge_index: Tensor | None = None, batch: Tensor | None = None, geo_coords: Tensor | None = None):
        if batch is None or len(batch.unique()) == 1:
            return self.calculate_rwpe(edge_index=edge_index, num_nodes=num_nodes, pe_dim=pe_dim)
        # if there are multiple graphs in the data, call the function per data, store the node order, and rearrange the pe to be in the correct position
        pe = torch.zeros((num_nodes, self.pe_dim), device=edge_index.device)
        for graph_id in batch.unique(sorted=True):
            # Get the indices of the nodes in this subgraph
            node_mask = (batch == graph_id)
            node_indices = node_mask.nonzero()[0]
            # Generate a subgraph, calculate its rwpe, and store it at the correct indices
            subgraph_edge_index, _ = subgraph(node_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes)
            sub_pe = self.calculate_rwpe(subgraph_edge_index, len(node_indices), pe_dim)
            pe[node_indices] = sub_pe
        print(pe)
        return pe        
    
    def calculate_rwpe(self, edge_index: torch.Tensor, num_nodes: int, pe_dim) -> torch.Tensor:
        source_nodes = edge_index[1]
        node_degrees = degree(source_nodes, num_nodes=num_nodes, dtype=torch.float)
        # Calculate and invert the degrees
        degrees_inv = 1.0 / node_degrees
        degrees_inv[node_degrees==0] = 0.0
        edge_weights = degrees_inv[source_nodes]
        
        # Calculate RWPE by calculating powers of the random walk matrix, and storing the diagonals of the powers
        rw_base = torch.sparse_coo_tensor(edge_index, edge_weights, (num_nodes, num_nodes))
        rw_power = rw_base.clone()
        pe = [sparse_diagonal(rw_base)]
        for i in range(pe_dim-1):
            rw_power = torch.sparse.mm(rw_power, rw_base)
            pe.append(sparse_diagonal(rw_power))
        # rearrange the dimensions to have the pe of a node in a row, not in a column
        return torch.stack(pe, dim=-1) # Shape: (node_amt, pe_dim)
    
class GeoPE_Initializer(PE_Initializer):
    def __init__(self, coord_dim: int = 2, emb_hidden_dim=128, emb_dim: int = 20):
        self.coord_dim = coord_dim
        self.emb_hidden_dim = emb_hidden_dim
        self.emb_dim = emb_dim
    def __call__(self, num_nodes: int, pe_dim: int, edge_index: Tensor | None = None, batch: Tensor | None = None, geo_coords: Tensor | None = None):
        pass