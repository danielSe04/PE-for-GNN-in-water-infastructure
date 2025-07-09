from typing import Any
import torch
from torch import Tensor, nn
from torch_geometric.utils import degree, subgraph, get_laplacian
from torch_geometric.data import Data
from abc import abstractmethod
from gigantic_dataset.utils.train_protos import ConfigRef
from gigantic_dataset.utils.pe_utils import sparse_diagonal
from gigantic_dataset.model.pe_gnn.grid_spatial_relation_encoder import GridCellSpatialRelationEncoder
from gigantic_dataset.model.SPE.gin import GINRho
from gigantic_dataset.model.SPE.mlp import MLP
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


class PE_Initializer(nn.Module):
    '''
    This superclass inherits from Torch modules so it can be registered in the main model. 
    Then, it can be loaded together with the model, without adding additional loading utility.
    '''
    pe_dim: int
    def __init__(self, pe_dim: int = 20):
        super(PE_Initializer, self).__init__()
        self.pe_dim = pe_dim

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        pass

class RWPE_Initializer(PE_Initializer):
    def forward(self, data: Data) -> Tensor:
        '''
        Computes a random walk positional encoding of pe_dim dimensions with the formula PE_ii = [RW_{ii}^1, ..., RW_{ii}^{pe_dim}]
        for all graphs in the passed data batch. Here, RW = AD^{-1}.

        Args:
            data (Data): The data object containing the graphs for which the random walks are calculated.
        Returns: Tensor, which is a positional encoding aligned with the node indexing dicated by node features and indices used by edge_index.
        '''
        assert data.edge_index is not None, "Trying to initialize random walk positional encoding, but edge indices are None"
        assert data.x is not None
        num_nodes = data.x.shape[0]
        if data.batch is None or len(data.batch.unique()) == 1:
            return self.calculate_rwpe(edge_index=data.edge_index, num_nodes=num_nodes, pe_dim=self.pe_dim)
        # if there are multiple graphs in the data, call the function per graph, store the node order, and rearrange the pe to be in the correct position.
        pe = torch.zeros((num_nodes, self.pe_dim), device=data.edge_index.device)
        for graph_id in data.batch.unique(sorted=True):
            # Get the indices of the nodes in this subgraph
            node_mask = (data.batch == graph_id)
            node_indices = node_mask.nonzero()[0]
            # Generate a subgraph, calculate its rwpe, and store it at the correct indices
            subgraph_edge_index, _ = subgraph(node_indices, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
            sub_pe = self.calculate_rwpe(subgraph_edge_index, len(node_indices), self.pe_dim)
            pe[node_indices] = sub_pe
        return pe        
    
    def calculate_rwpe(self, edge_index: torch.Tensor, num_nodes: int, pe_dim: int) -> torch.Tensor:
        '''
        Calculates the random walk positional encoding for a single graph. 
        '''
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
        for _ in range(pe_dim-1):
            rw_power = torch.sparse.mm(rw_power, rw_base)
            pe.append(sparse_diagonal(rw_power))
        # rearrange the dimensions to have the pe of a node in a row, not in a column
        return torch.stack(pe, dim=-1) # Shape: (node_amt, pe_dim)

class GeoPE_Initializer(PE_Initializer):
    '''
    This class is adopted from the initialization of the PE-GCN example in https://github.com/konstantinklemmer/pe-gnn.  
    '''
    def __init__(self, coord_dim: int = 2, pe_hidden_dim: int =128, pe_dim: int = 20):
        '''
        Initializes the models that calculate the PE-GNN positional encoding (Klemmer et al. (2023)).
        
        Args:
            coord_dim (int): The dimensionality of the coordinates.
            pe_dim (int) The dimension of the final positional encoding.
            pe_hidden_dim (int): The hidden dimensions used in the encoder.
        '''
        super(GeoPE_Initializer, self).__init__(pe_dim)
        self.pe_dim = pe_dim
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
    
    def forward(self, data: Data) -> Tensor:
        '''
        Calculates the frequency based positional encoding that Klemmer et al. (2023) propose.
        '''
        assert "coordinates" in data, "Trying to initialize geo-located positional encoding, but coordinates are not in Data object"
        geo_coords = data.coordinates.reshape(1, data.coordinates.shape[0], data.coordinates.shape[1])
        emb = self.spatial_encoder(geo_coords.detach().cpu().numpy())
        emb = emb.reshape(emb.shape[1], emb.shape[2])
        return self.decoder(emb).float()
    

class SPE_Initializer(PE_Initializer):
    '''
    This class is adapted from the original spe implementation, see https://github.com/Graph-COM/SPE.
    '''
    rho: nn.Module
    phi_list: nn.ModuleList
    def __init__(self, pe_dim: int = 20, pe_hidden_dim: int = 32) -> None:
        '''
        Initializes the models that calculate the SPE positional encoding. We use GIN as rho, and MLP as phi (see Huang et al. (2024)).
        
        Args:
            pe_dim (int) The dimension of the final positional encoding.
            pe_hidden_dim (int): The hidden dimensions used by the GIN network (rho). This is important for the amount of memory used by SPE.
        '''
        super(SPE_Initializer, self).__init__(pe_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eig_amt = 8
        m = 4
        self.rho = GINRho(n_layers=2, in_dims=m, hidden_dims=pe_hidden_dim, out_dims=pe_dim, bn=False).to(self.device) # These values are adopted from the config in the SPE codebase, changed n_layers from 8 to 2
        self.phi_list = nn.ModuleList()
        for _ in range(m):
            self.phi_list.append(MLP(1, 1, hidden_dims=16, n_layers=3).to(self.device))

    def calculate_eigs(self, edge_index: Tensor, num_nodes: int) -> tuple[Any, Any]:
        lap_edge_index, lap_edge_weight = get_laplacian(edge_index=edge_index, normalization="sym") # Get the normalized laplacian
        laplacian = coo_matrix((lap_edge_weight.cpu(), (lap_edge_index[0].cpu().numpy(), lap_edge_index[1].cpu().numpy())), shape=(num_nodes, num_nodes))
        k = min(num_nodes, self.eig_amt)
        return eigsh(laplacian, k=k, which="SM")  # type: ignore

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: Data object containing edge_index [2, E_sum] and batch [N_sum] Tensors
        Returns Tensor: Positional encoding matrix. [N_sum, D_pe]
        """
        assert data.x is not None
        assert data.edge_index is not None
        device = data.x.device

        Lambda = [] # [B, eig_amt]
        V = [] # [N_sum, eig_amt]
        V_list = [] # [N_i, eig_amt] * B
        if data.batch == None: # if there is only one graph in the data
            num_nodes = len(data.x)
            eigenvalues, V = self.calculate_eigs(data.edge_index, num_nodes)
            Lambda = [eigenvalues]
            V_list = [torch.tensor(V, device=device)]
            data.batch = torch.zeros(num_nodes, dtype=torch.int64, device=device)
        else: # if there are multiple graphs in the data, treat each graph seperately.
            for graph_id in data.batch.unique(sorted=False):
                node_mask = (data.batch == graph_id)
                node_indices = node_mask.nonzero()[0]
                num_nodes = int(node_mask.sum())
                # Generate a subgraph, calculate its eigenvalues and eigenvectors
                subgraph_edge_index, _ = subgraph(node_indices, data.edge_index, relabel_nodes=True)
                
                eigenvalues, eigenvectors = self.calculate_eigs(subgraph_edge_index, num_nodes)
                Lambda.append(eigenvalues)
                V_list.append(torch.tensor(eigenvectors, device=device))
        
        Lambda = torch.tensor(np.array(Lambda), device=device)
        
        # Mask invalid eigenvalues
        a = torch.arange(0, Lambda.size(1)).unsqueeze(0).to(Lambda.device)
        mask = torch.cat([a < torch.sum(data.batch == i) for i in range(data.batch[-1]+1)], dim=0) # [B, D_pe, 1]
        Lambda = Lambda * mask
        del mask
        Lambda = Lambda.unsqueeze(dim=2)   # [B, eig_amt, 1]

        # Pass the eigenvalues through the MLP networks.
        Z = torch.stack([
            phi(Lambda).squeeze(dim=2)     # [B, eig_amt]
            for phi in self.phi_list
        ], dim=2)                          # [B, eig_amt, M]

        # Pre-calculations before passing the data through the GIN network
        Z_list = list(Z)                    # [eig_amt, M] * B
        W_list = []                        # [N_i, N_i, M] * B
        for V, Z in zip(V_list, Z_list):   # [N_i, eig_amt] and [eig_amt, M]
            V = V.unsqueeze(dim=0)         # [1, N_i, eig_amt]
            Z = Z.permute(1, 0)            # [M, eig_amt]
            Z = Z.diag_embed()             # [M, eig_amt, eig_amt]
            V_T = V.mT                     # [1, eig_amt, N_i]
            W = V.matmul(Z).matmul(V_T)    # [M, N_i, N_i]
            W = W.permute(1, 2, 0)         # [N_i, N_i, M]
            W_list.append(W)
        # Pass the data through the GIN network.
        rho = self.rho(W_list, data.edge_index) # [N_sum, d_pe]
        # I have had issues with running out of memory, so I clean up after the initialization is complete
        del W_list, V_list, Z_list
        torch.cuda.empty_cache()
        return rho

    @property
    def out_dims(self) -> int:
        return self.rho.out_dims
    
def get_pe_initializer(pe_dim: int = 20) -> PE_Initializer | None:
    '''
    Helper function that initializes the correct PE_Initializer based on the configuration files.
    '''
    pe_init = ConfigRef.config.pe_config.pe_init
    if pe_init == "rw":
        return RWPE_Initializer(pe_dim=pe_dim)
    elif pe_init == "geo":
        return GeoPE_Initializer(pe_dim=pe_dim)
    elif pe_init == "spe":
        return SPE_Initializer(pe_dim=pe_dim)
    return None