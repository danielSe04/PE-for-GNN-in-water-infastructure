import torch
from torch import Tensor, nn
from torch_geometric.utils import degree, subgraph
from abc import ABC, abstractmethod
from gigantic_dataset.utils.pe_utils import sparse_diagonal, GridCellSpatialRelationEncoder
import numpy as np

class PE_Initializer(ABC):
    @abstractmethod
    def __call__(self, num_nodes: int, pe_dim: int, edge_index: Tensor | None = None, batch: Tensor | None = None, geo_coords: Tensor | None = None) -> Tensor:
        pass

class RWPE_Initializer(PE_Initializer):
    def __call__(self, num_nodes: int, pe_dim: int, edge_index: Tensor | None = None, batch: Tensor | None = None, geo_coords: Tensor | None = None) -> Tensor:
        assert edge_index is not None, "Trying to initialize random walk positional encoding, but edge indices are None"
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

# This class is adopted from the initialization of the PE-GCN example in https://github.com/konstantinklemmer/pe-gnn.  
class GeoPE_Initializer(PE_Initializer):
    def __init__(self, coord_dim: int = 2, pe_hidden_dim=128, pe_dim: int = 20):
        self.coord_dim = coord_dim
        self.emb_hidden_dim = pe_hidden_dim
        self.emb_dim = pe_dim
        self.spatial_encoder = GridCellSpatialRelationEncoder(spa_embed_dim=pe_hidden_dim, ffn=True, min_radius=1e-06, max_radius=360)
        self.spatial_encoder.to(self.spatial_encoder.device)
        self.decoder = nn.Sequential(
            nn.Linear(pe_hidden_dim, pe_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 2, pe_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 4, pe_dim)
        )
        self.decoder.to(self.spatial_encoder.device)
    def __call__(self, num_nodes: int, pe_dim: int, geo_coords: Tensor | None = None, **kwargs) -> Tensor:
        assert geo_coords is not None, "Trying to initialize geo-located positional encoding, but coordinates are None"
        geo_coords = geo_coords.reshape(1, geo_coords.shape[0], geo_coords.shape[1])
        emb = self.spatial_encoder(geo_coords.detach().cpu().numpy())
        emb = emb.reshape(emb.shape[1], emb.shape[2])
        return self.decoder(emb).float()