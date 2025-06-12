from typing import Callable, List
import sklearn
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_mean, scatter_std, scatter_min, scatter_max, scatter_add
import wandb
import numpy as np
import math
from collections import defaultdict
from typing import Any
from gigantic_dataset.utils.train_protos import ConfigRef

def plot_pe_wandb(pe: Tensor | list[Tensor], edge_index: Tensor, plot_name: str, plot_edges: bool = False) -> None:
    if isinstance(pe, Tensor):
        pe = [pe]
    print("Number of snapshots: ", len(pe))
    if len(pe) > 2:
        print("Warning, more than two positional encodings to be visualized.", pe)
        fig, axs = plt.subplots(1, 2, figsize=(2*8,6))
    else:
        fig, axs = plt.subplots(1, len(pe), figsize=(len(pe)*8,6))

    for i, pe_snapshot in enumerate(pe):
        if i >= 2:
            break
        ax = axs[i] if len(pe) > 1 else axs
        if pe_snapshot.std() == 0:
            print("Warning: Positional encoding has zero standard deviation. Adding a small amount of noise.")
            pe_snapshot += 1e-4 * torch.randn_like(pe_snapshot)
        if torch.isnan(pe_snapshot).any():
            print("Warning: Positional encoding contains NaNs. They are converted to zeros.")
            pe_snapshot = torch.nan_to_num(pe_snapshot)

        pe_snapshot = pe_snapshot.cpu().numpy()
        perplexity = 30.0 if len(pe_snapshot) > 30.0 else len(pe_snapshot) - 1.0
        tsne = sklearn.manifold.TSNE(n_components=2, random_state=42, perplexity=perplexity) #TODO perhaps change perplexity and random_state parameter
        z_2d = tsne.fit_transform(pe_snapshot)
        
        ax.scatter(z_2d[:, 0], z_2d[:, 1], s=10, color='darkblue')
        ax.set_title("t-SNE visualization of positional encoding")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

        if plot_edges:
            segments = [
                [z_2d[edge_index[0][i]], z_2d[edge_index[1][i]]]
                for i in range(edge_index.shape[1])
            ]
            lc = LineCollection(segments, linewidths=0.2, alpha=0.5, color='blue')
            ax.add_collection(lc)

    wandb.log({plot_name: wandb.Image(fig)})
    plt.close(fig)

def sparse_diagonal(G: Tensor):
        G = G.coalesce()
        dest_nodes, source_nodes = G.indices()
        values = G.values()
        mask = dest_nodes == source_nodes
        diag_values = torch.zeros(G.size(0), device=G.device)
        diag_values[dest_nodes[mask]] = values[mask]
        return diag_values

def laplacian_eigenvector_loss(pe: Tensor, laplacian: Tensor, lambda_ortho: int = 0.1):
    '''
    Computes the laplacian eigenvector loss, which is defined as Trace(Y^T L Y) where Y is the model output 
    for the positional encoding, and L is the graph laplacian. Additionally, Dwivedi et al. (2022) do normalization
    '''
    k = pe.shape[1]
    # Normalization:
    pe = pe - pe.mean(dim=0, keepdim=True)
    pe = F.normalize(pe, dim=0)

    # Calculate the trace term:
    pe_t = pe.transpose(0, 1)
    pe_laplacian = torch.sparse.mm(laplacian, pe) # Matrix multiplication
    trace = torch.trace(pe_t @ pe_laplacian) / k

    # Calculate orthogonality term:
    identity_matrix = torch.eye(k, device=pe.device)
    ortho = F.mse_loss(pe_t @ pe, identity_matrix) # Since pe is normalized, frobenius norm is equal to k * mse, and the k cancels out
    return trace + lambda_ortho * ortho

def calculate_auxiliary_loss(pe: Tensor, pe_out: Tensor, edge_index: Tensor | None = None, edge_weight: Tensor | None = None):
    loss_criterion = ConfigRef.config.pe_criterion
    if loss_criterion == "mse" or loss_criterion == "morans-i":
        assert pe.shape == pe_out.shape, f"Cannot compute MSE: Shape of target is {pe.shape}, but shape of output is {pe_out.shape}"
        return torch.nn.MSELoss(reduction="mean").to(ConfigRef.config.device)(pe_out, pe)
    elif loss_criterion == "laplacian":
        assert edge_index is not None, f"Edge indices are required for calculation of {loss_criterion} loss, but edge_index is None."
        # first compute the graph laplacian
        lap_edge_index, lap_edge_weight = get_laplacian(edge_index=edge_index, edge_weight=edge_weight, normalization=None)
        laplacian = torch.sparse_coo_tensor(indices=lap_edge_index, values=lap_edge_weight, size=(len(pe_out), len(pe_out)))
        laplacian.coalesce()
        return laplacian_eigenvector_loss(pe_out, laplacian)
    else:
        raise NotImplementedError
    
def calculate_subgraph(data: Data, k: int = 5, batch_size: int = 1024) -> Data:
    assert "coordinates" in data, "Coordinates are needed for knn-based subgraphing, but are not included in data"
    config = ConfigRef.config
    device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
    batch: torch.Tensor = data.batch
    x_subgraph = []
    y_subgraph = [] if "y" in data else None
    coordinates_subgraph = []
    batch_subgraph = []
    # For all graphs in the batch, 
    for graph_id in batch.unique(sorted=True):
        # Get the indices of the nodes in this subgraph
        node_mask = (batch == graph_id)
        node_indices = node_mask.nonzero(as_tuple=True)[0]
        num_nodes = node_indices.size(0)
        if num_nodes > batch_size:
            selection = torch.randperm(num_nodes)[:batch_size]
            selected_indices = node_indices[selection]
        else:
            selected_indices = node_indices
        x_subgraph.append(data.x[selected_indices])
        coordinates_subgraph.append(data.coordinates[selected_indices])
        batch_subgraph.extend([graph_id for i in len(selected_indices)])
        if y_subgraph is not None:
            y_subgraph.append(data.y[selected_indices])

    # Construct the knn subgraph
    batch_subgraph = torch.tensor(batch_subgraph, device=device)
    subgraph_edge_index = knn_graph(coordinates_subgraph, k=k, batch=batch_subgraph)
    # Create a data object for the subgraph
    data_dict: dict[str, Any] = defaultdict(list)
    data_dict["edge_index"] = subgraph_edge_index
    data_dict["x"] = torch.cat(x_subgraph, dim=0)
    data_dict["coordinates"] = torch.cat(x_subgraph, dim=0)
    if y_subgraph is not None:
        data_dict["y"] = torch.cat(y_subgraph, dim=0)
    data_subgraph = Data.from_dict(data_dict)
    return data_subgraph


# The following functions are copied from the original PE-GNN implementation. See https://github.com/konstantinklemmer/pe-gnn
def get_activation_function(activation, context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str))
    
def normal_torch(tensor, batch, min_val=0):
    t_min = scatter_min(tensor, batch, dim=0)[0]
    t_max = scatter_max(tensor, batch, dim=0)[0]

    den = t_max - t_min
    den[den == 0] = 1

    if min_val == -1:
        tensor_norm = 2 * ((tensor - t_min[batch]) / den[batch]) - 1
    if min_val== 0:
        tensor_norm = ((tensor - t_min[batch]) / den[batch])
    return tensor_norm
    
def lw_tensor_local_moran(y, edge_index, batch=None, na_to_zero=True, norm=True, norm_min_val=0):
    device = ConfigRef.config.device
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
    mi = n_1_per_graph[batch].unsqueeze(1) * z * zl / den[batch]
    if na_to_zero==True:
        mi = torch.nan_to_num(mi, nan=0.0)
    if norm==True:
        mi = normal_torch(mi, batch, min_val=norm_min_val)
    print("moran's I shape: ", mi.shape)
    return torch.tensor(mi) # This is intentional

# This class is copied from the original PE-GNN implementation. See https://github.com/konstantinklemmer/pe-gnn
class SingleFeedForwardNN(nn.Module):
    """
        Creates a single layer fully connected feed forward neural network.
        this will use non-linearity, layer normalization, dropout
        this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(self, input_dim,
                    output_dim,
                    dropout_rate=None,
                    activation="sigmoid",
                    use_layernormalize=False,
                    skip_connection = False,
                    context_str = ''):
        '''
        Args:
            input_dim (int32): the input embedding dim
            output_dim (int32): dimension of the output of the network.
            dropout_rate (scalar tensor or float): Dropout keep prob.
            activation (string): tanh or relu or leakyrelu or sigmoid
            use_layernormalize (bool): do layer normalization or not
            skip_connection (bool): do skip connection or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            # the layer normalization is only used in the hidden layer, not the last layer
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # the skip connection is only possible, if the input and out dimention is the same
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False
        
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform(self.linear.weight)

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size,..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        # Linear layer
        output = self.linear(input_tensor)
        # non-linearity
        output = self.act(output)
        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # skip connection
        if self.skip_connection:
            output = output + input_tensor

        # layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output

# This class is copied from the original PE-GNN implementation. See https://github.com/konstantinklemmer/pe-gnn
class MultiLayerFeedForwardNN(nn.Module):
    """
        Creates a fully connected feed forward neural network.
        N fully connected feed forward NN, each hidden layer will use non-linearity, layer normalization, dropout
        The last layer do not have any of these
    """

    def __init__(self, input_dim,
                    output_dim,
                    num_hidden_layers=0,
                    dropout_rate=0.5,
                    hidden_dim=-1,
                    activation="relu",
                    use_layernormalize=True,
                    skip_connection = False,
                    context_str = None):
        '''
        Args:
            input_dim (int32): the input embedding dim
            num_hidden_layers (int32): number of hidden layers in the network, set to 0 for a linear network.
            output_dim (int32): dimension of the output of the network.
            dropout (scalar tensor or float): Dropout keep prob.
            hidden_dim (int32): size of the hidden layers
            activation (string): tanh or relu
            use_layernormalize (bool): do layer normalization or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(MultiLayerFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append(SingleFeedForwardNN(input_dim = self.input_dim,
                                                    output_dim = self.output_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = False,
                                                    skip_connection = False,
                                                    context_str = self.context_str))
        else:
            self.layers.append(SingleFeedForwardNN(input_dim = self.input_dim,
                                                    output_dim = self.hidden_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = self.use_layernormalize,
                                                    skip_connection = self.skip_connection,
                                                    context_str = self.context_str))

            for i in range(self.num_hidden_layers-1):
                self.layers.append(SingleFeedForwardNN(input_dim = self.hidden_dim,
                                                    output_dim = self.hidden_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = self.use_layernormalize,
                                                    skip_connection = self.skip_connection,
                                                    context_str = self.context_str))

            self.layers.append(SingleFeedForwardNN(input_dim = self.hidden_dim,
                                                    output_dim = self.output_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = False,
                                                    skip_connection = False,
                                                    context_str = self.context_str))

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size, ..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        output = input_tensor
        for i in range(len(self.layers)):
            output = self.layers[i](output)

        return output

# This method is copied from the original PE-GNN implementation. See https://github.com/konstantinklemmer/pe-gnn
def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) / (frequency_num*1.0 - 1))
        timescales = min_radius * np.exp(np.arange(frequency_num).astype(float) * log_timescale_increment)
        freq_list = 1.0/timescales
    return freq_list

# This class is copied from the original PE-GNN implementation. See https://github.com/konstantinklemmer/pe-gnn
class GridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
            max_radius =0.01, min_radius = 0.00001,
            freq_init = "geometric",
            ffn=None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridCellSpatialRelationEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim 
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.ffn = ffn
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()
        self.input_embed_dim = self.cal_input_dim()

        if self.ffn is not None:
          self.ffn = MultiLayerFeedForwardNN(2 * frequency_num * 2, spa_embed_dim)
          self.ffn.to(self.device)

    def cal_elementwise_angle(self, coord, cur_freq):
        '''
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        '''
        return coord/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (input_embed_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)


    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis = 1)

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")
        
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis = 4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat
        # make sinuniod function................................................................
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1
        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: lllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll...batch_size, num_context_pt, spa_embed_dim)
        """   
        spr_embeds = self.make_input_embeds(coords)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds