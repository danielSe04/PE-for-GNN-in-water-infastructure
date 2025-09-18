from typing import Any
import sklearn
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, subgraph, degree
from torch_scatter import scatter_mean, scatter_std, scatter_min, scatter_max, scatter_add
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
import wandb

def plot_pe_wandb(pe: Tensor | list[Tensor], edge_index: Tensor, plot_name: str, plot_edges: bool = True, save_wandb: bool = True) -> None:
    '''
    This function plots positional encodings using a t-SNE dimension reduction. It uploads them to wandb.

    Args:
        pe (Tensor | list[Tensor]): The positional encoding to be plotted. At most PE instances will be plotted.
        edge_index (Tensor): The edge indices of the graph that the PE belongs to.
        plot_name (str): The name the plot will be given.
        plot_edges (bool): Indicates whether the edges will be visualized in the plots. This is only done for the plot of at most one instance.
        save_wandb (bool): Indicates whether the plot should be stored on wandb (True), or locally (False).
    '''
    # Create plots
    if isinstance(pe, Tensor):
        pe = [pe]
    if len(pe) > 2:
        print("Warning, more than two positional encodings to be visualized.", pe)
        fig, axs = plt.subplots(1, 2, figsize=(2*8,6))
    else:
        fig, axs = plt.subplots(1, len(pe), figsize=(len(pe)*8,6))

    for i, pe_snapshot in enumerate(pe):
        if i >= 2: # Only plot at most two positional encodings.
            break
        ax = axs[i] if len(pe) > 1 else axs
        if pe_snapshot.std() == 0: # For t-SNE to work, the std must be nonzero.
            print("Warning: Positional encoding has zero standard deviation. Adding a small amount of noise.")
            pe_snapshot += 1e-4 * torch.randn_like(pe_snapshot)
        if torch.isnan(pe_snapshot).any():
            print("Warning: Positional encoding contains NaNs. They are converted to zeros.")
            pe_snapshot = torch.nan_to_num(pe_snapshot)

        # Apply t-SNE
        pe_snapshot = pe_snapshot.cpu().numpy()
        perplexity = 30.0 if len(pe_snapshot) > 30.0 else len(pe_snapshot) - 1.0
        tsne = sklearn.manifold.TSNE(n_components=2, random_state=42, perplexity=perplexity) # type: ignore
        z_2d = tsne.fit_transform(pe_snapshot)
        
        # Plot the t-SNE reduced points
        ax.scatter(z_2d[:, 0], z_2d[:, 1], s=10, color='darkblue')
        ax.set_title("t-SNE visualization of positional encoding")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

        # Plot the edges if selected
        if plot_edges and i == len(pe) - 1:
            segments = [
                [z_2d[edge_index[0][i]], z_2d[edge_index[1][i]]]
                for i in range(edge_index.shape[1])
            ]
            lc = LineCollection(segments, linewidths=0.2, alpha=0.5, color='darkorange')
            ax.add_collection(lc)
    # Save the plots
    if save_wandb:
        wandb.log({plot_name: wandb.Image(fig)})
    else:
        fig.savefig(os.path.join("plots" + plot_name))
    plt.close(fig)

def sparse_diagonal(G: Tensor) -> Tensor:
        '''
        This function returns the diagonal of the sparse input matrix, without converting it to a dense one.
        
        Args:
            G (Tensor): Input graph in COO format.
        Returns: 
            Tensor: The values in the diagonal.
        '''
        G = G.coalesce()
        dest_nodes, source_nodes = G.indices()
        values = G.values()
        mask = dest_nodes == source_nodes
        diag_values = torch.zeros(G.size(0), device=G.device)
        diag_values[dest_nodes[mask]] = values[mask]
        return diag_values

def calculate_rwpe(edge_index: torch.Tensor | np.ndarray, num_nodes: int, pe_dim: int) -> torch.Tensor:
        '''
        Calculates the random walk positional encoding for a single graph. 
        '''
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if pe_dim < 1:
            return torch.empty(num_nodes, 0, dtype=torch.float, device=device)
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
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

def calculate_eigs(edge_index: Tensor | np.ndarray, eig_amt: int, num_nodes: int) -> tuple[Any, Any]:
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    lap_edge_index, lap_edge_weight = get_laplacian(edge_index=edge_index, normalization="sym") # Get the normalized laplacian
    laplacian = coo_matrix((lap_edge_weight.cpu(), (lap_edge_index[0].cpu().numpy(), lap_edge_index[1].cpu().numpy())), shape=(num_nodes, num_nodes))
    k = min(num_nodes, eig_amt)
    return eigsh(laplacian, k=k, which="SM")  # type: ignore

def laplacian_eigenvector_loss(pe: Tensor, laplacian: Tensor, lambda_ortho: float = 0.1) -> float:
    '''
    Computes the laplacian eigenvector loss, which is defined as Trace(Y^T L Y) where Y is the model output 
    for the positional encoding (pe), and L is the graph laplacian. Additionally, Dwivedi et al. (2022) do normalization.
    '''
    k = pe.shape[1]
    # Normalization of the PE:
    pe = pe - pe.mean(dim=0, keepdim=True)
    pe = F.normalize(pe, dim=0)

    # Calculate the trace term:
    pe_t = pe.transpose(0, 1)
    pe_laplacian = torch.sparse.mm(laplacian, pe) # Matrix multiplication
    trace = torch.trace(pe_t @ pe_laplacian) / k

    # Calculate orthogonality term:
    identity_matrix = torch.eye(k, device=pe.device)
    ortho = F.mse_loss(pe_t @ pe, identity_matrix) # Since pe is normalized, frobenius norm is equal to k * mse, and the k cancels out
    loss = trace + lambda_ortho * ortho
    return loss.item()

def calculate_auxiliary_loss(loss_criterion: str, aux_true: Tensor, aux_pred: Tensor, edge_index: Tensor | None = None, batch: Tensor | None = None) -> float:
    '''
    This function computes the auxiliary loss based on the pe configuration, the true values, and output values.

    Args:
        aux_true (Tensor): The true values of the variable that the loss is calculated on.
        aux_pred (Tensor): The predicted values of the variable that the loss is calculated on.
        edge_index (Tensor): The edge_index of the batched graph.
        batch (Tensor): The graph batch.
    Returns:
        float: The auxiliary loss.
    '''

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if loss_criterion == "mse" or loss_criterion == "morans-i":
        assert aux_true.shape == aux_pred.shape, f"Cannot compute MSE: Shape of target is {aux_true.shape}, but shape of output is {aux_pred.shape}"
        return torch.nn.MSELoss(reduction="mean").to(device)(aux_pred, aux_true)
    elif loss_criterion == "laplacian":
        assert edge_index is not None, f"Edge indices are required for calculation of {loss_criterion} loss, but edge_index is None."
        assert batch is not None, f"Batch is required for calculation of {loss_criterion} loss, but batch is None."
        total_loss = 0.0
        num_graphs = int(batch.max()) + 1
        # first compute the graph laplacian
        for i in batch.unique(sorted=False):
            node_mask = (batch == i)
            aux_i = aux_pred[node_mask]
            node_indices = node_mask.nonzero(as_tuple=False).view(-1)
            subgraph_edge_index, _ = subgraph(node_indices, edge_index, relabel_nodes=True)
            lap_edge_index, lap_edge_weight = get_laplacian(edge_index=subgraph_edge_index, normalization=None)
            laplacian = torch.sparse_coo_tensor(indices=lap_edge_index, values=lap_edge_weight, size=(len(aux_i), len(aux_i)))
            laplacian.coalesce()
            total_loss += laplacian_eigenvector_loss(aux_i, laplacian)
        return total_loss / num_graphs
    else:
        raise NotImplementedError

# The following functions are copied from the original PE-GNN implementation. See https://github.com/konstantinklemmer/pe-gnn
# I adapted them to work for batched graphs.
def normal_torch(tensor, batch, min_val=0):
    '''
    This function computes a normalization on Moran's I.
    '''
    t_min = scatter_min(tensor, batch, dim=0)[0]
    t_max = scatter_max(tensor, batch, dim=0)[0]

    den = t_max - t_min
    den[den == 0] = 1

    if min_val == -1:
        tensor_norm = 2 * ((tensor - t_min[batch]) / den[batch]) - 1
    if min_val== 0:
        tensor_norm = ((tensor - t_min[batch]) / den[batch])
    return tensor_norm # type: ignore
    
def lw_tensor_local_moran(y, edge_index, batch=None, na_to_zero=True, norm=True, norm_min_val=0):
    '''
    This function calculates the Moran's I metric for the output variable(s) adhering to the graph batching.
    Moran's I is defined as I_i = (n-1) frac{y_i - bar y}{sum_{j=1}^n (y_j - bar y)^2} sum_{j=1, j neq i}^n a_{i,j} (y_j - bar y)
    where i is a node, and bar y is the average of y.
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch == None:
        batch = torch.zeros(len(y)).to(device).long()
    n_1_per_graph = torch.bincount(batch) - 1
    y_mean_per_graph = scatter_mean(y, batch, dim=0)
    y_sd_per_graph = scatter_std(y, batch, dim=0)
    y_sd_per_graph[y_sd_per_graph==0] = 1
    z = (y - y_mean_per_graph[batch]) / y_sd_per_graph[batch]
    den = scatter_add(z * z, batch, dim=0)
    w_sparse = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1], device=device), size=(len(y), len(y)), device=device).coalesce()
    zl = torch.sparse.mm(w_sparse, z).to(device)
    # zl / den[batch]
    mi = n_1_per_graph[batch].unsqueeze(1) * z * zl / den[batch]
    if na_to_zero==True:
        mi = torch.nan_to_num(mi, nan=0.0)
    if norm==True:
        mi = normal_torch(mi, batch, min_val=norm_min_val)
    return mi # TODO: This is intentional apparently. It detaches the Moran's I from torch's computation tree,
                            # however without this yields worse results. It seems that being detached, Moran's I doesn't really do anything for the performance.