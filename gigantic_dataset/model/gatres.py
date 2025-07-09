#
# Created on Mon Nov 25 2024
# Copyright (c) 2024 Huy Truong & Andrés Tello
# ------------------------------
# Purpose: a simple GATRes
# ------------------------------
#


from typing import Any, Callable, Literal, Union
from functools import partial
from torch import clone, Tensor
import torch.nn.functional as F
import torch
from torch.nn import Module, ModuleList
from torch_geometric.nn import GATConv, SimpleConv, Linear
from torch_geometric.data import Data
from gigantic_dataset.utils.configs import ModelConfig, PEConfig
from gigantic_dataset.utils.train_protos import ConfigRef, LoadModelProto
from gigantic_dataset.utils.train_utils import load_weights
from gigantic_dataset.model.equiformer.graph_attention_transformer import GraphAttentionWrapper, get_standard_irreps
from gigantic_dataset.model.equiformer.tensor_product_rescale import LinearRS
from .pe_initializers import get_pe_initializer, PE_Initializer
from e3nn import o3
import os
import torch

from torch.nn import Module

class GResBlockMeanConv(Module):
    '''
    The standard GATRes block.
    '''
    def __init__(self, in_dim, out_dim, hc, gat_conv: Module = GATConv, activation_func: Callable[[Tensor], Tensor] = F.relu): #type: ignore
        super(GResBlockMeanConv, self).__init__()

        # self.norm1 = BatchNorm(in_channels=in_dim)
        self.conv1 = gat_conv(in_dim, hc, 2, concat=True)
        self.conv2 = gat_conv(hc * 2, out_dim, 1, concat=False)
        self.mean_conv = SimpleConv(aggr="mean")
        self.activation_func = activation_func
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None, coordinates: Tensor | None = None) -> Tensor:
        # x = self.norm1(x)
        x_0 = clone(x) # Take only the non-pe part
        x_0 = x_0[:, : self.out_dim]
        x = self.activation_func(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        x = self.mean_conv(x, edge_index) + x_0
        x = self.activation_func(x)
        return x

class EquiformerResBlockMeanConv(GResBlockMeanConv):
    '''
    An Equiformer block. It emulates the structure of a GATRes by using two Equiformer attention blocks and adding a residual.
    '''
    def __init__(self, in_dim, out_dim, hc, activation_func = F.relu):
        super().__init__(in_dim, out_dim, hc, GraphAttentionWrapper, activation_func) #type: ignore
    
    def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None, coordinates: Tensor | None = None) -> Tensor:
        assert coordinates is not None, "Equiformer requires coordinates, but coordinates are None."
        x_0 = clone(x)
        
        x = self.activation_func(self.conv1(x, edge_index, coordinates, edge_attr, batch))
        x = self.conv2(x, edge_index, coordinates, edge_attr, batch)
        x = self.mean_conv(x, edge_index) + x_0
        x = self.activation_func(x)
        return x

class MeanConvBase(Module):
    '''
    A base definition of a complete layer of the models.
    By default, this class has a PE initializer, but if the models does not use PE, this will be None.
    '''
    pe_initializer: PE_Initializer | None

    def __init__(self, in_dim: int, out_dim: int, name: str = "EquiformerMeanConv", num_blocks: int = 5, nc: int = 8, pe_dim: int = 0):
        super(MeanConvBase, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_blocks = num_blocks
        self.nc = nc
        self.num_blocks = num_blocks
        self.pe_dim = pe_dim
        self.name = name
        self.pe_initializer = get_pe_initializer(pe_dim)

    def get_pe_dim(self) -> int:
        return self.pe_dim
    
    def get_pe(self, data: Data) -> Tensor | None:
        return self.pe_initializer(data) if self.pe_initializer is not None else None


class GATResMeanConv(MeanConvBase):
    '''
    A standard GATRes layer.
    '''
    def __init__(self, in_dim: int, out_dim: int, name: str = "GATResMeanConv", num_blocks: int = 5, nc: int = 32, pe_dim: int = 0, concat_pe_per_layer: bool = False):
        super(GATResMeanConv, self).__init__(in_dim, out_dim, name, num_blocks, nc, pe_dim)
        self.lin0 = Linear(in_dim, nc)
        self.blocks = ModuleList()
        if concat_pe_per_layer:
            gat_out_dim = nc
        else: 
            gat_out_dim = nc+pe_dim
        for _ in range(self.num_blocks):
            block = GResBlockMeanConv(in_dim=nc+pe_dim, out_dim=gat_out_dim, hc=nc+pe_dim)
            self.blocks.append(block)

        self.lin1 = Linear(gat_out_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, pe: Tensor | None = None, batch: Tensor | None = None, edge_attr: Tensor | None = None) -> Union[Tensor, tuple[Tensor, Tensor]]:
        x = self.lin0(x)
        for i in range(self.num_blocks):
            x = self.blocks[i](x, edge_index, edge_attr, batch)
        x = self.lin1(x)
        return x
    
    
class GATResMeanConvLSPE(GATResMeanConv):
    '''
    An LSPE-enhanced GATRes layer (Dwivedi et al. (2022)).
    '''
    def __init__(self, in_dim: int, out_dim: int, name: str = "GATResMeanConvLSPE", num_blocks: int = 5, nc: int = 32, pe_dim: int = 20):
        super(GATResMeanConvLSPE, self).__init__(in_dim=in_dim, out_dim=out_dim, name=name, num_blocks=num_blocks, nc=nc, pe_dim=pe_dim, concat_pe_per_layer=True)
        self.blocks_pe = ModuleList()
        for _ in range(self.num_blocks):
            # The positional encoding uses tanh and not relu to allow negative coordinates
            block = GResBlockMeanConv(in_dim=self.pe_dim, out_dim=self.pe_dim, hc=self.pe_dim, activation_func=torch.tanh)
            self.blocks_pe.append(block)

    def forward(self, x: Tensor, edge_index: Tensor, pe: Tensor | None = None, batch: Tensor | None = None, edge_attr: Tensor | None = None) -> tuple[Tensor, Tensor]:
        assert pe is not None, "LSPE requires a positional encoding."
        x = self.lin0(x)
        for i in range(self.num_blocks):
            x_pe = torch.cat((x,pe), -1) #type: ignore
            x = self.blocks[i](x_pe, edge_index, edge_attr, batch)
            pe = self.blocks_pe[i](pe, edge_index, edge_attr, batch)
        
        x = self.lin1(x)
        return x, pe #type: ignore

class PE_concat_GATResMeanConv(GATResMeanConv):
    '''
    A layer that adds a positional encoding by simple concatenation. E.g. this is used in SPE and PE-GNN.
    '''
    def __init__(self, in_dim, out_dim, name = "GATResMeanConv", num_blocks = 5, nc = 32, pe_dim = 0, concat_pe_per_layer = False):
        super(PE_concat_GATResMeanConv, self).__init__(in_dim, out_dim, name, num_blocks, nc, pe_dim, concat_pe_per_layer)

    def forward(self, x: Tensor, edge_index: Tensor, pe: Tensor | None = None, batch: Tensor | None = None, edge_attr: Tensor | None = None) -> tuple[Tensor, Tensor]:
        assert pe is not None, "LSPE requires a positional encoding."
        x = self.lin0(x)
        x = torch.cat((x,pe), -1) # type: ignore
        for i in range(self.num_blocks):
            x = self.blocks[i](x, edge_index, edge_attr, batch)
        x = self.lin1(x)
        return x, pe # type: ignore
    

class EquiformerMeanConv(MeanConvBase):
    '''
    A complete equiformer layer.
    '''
    def __init__(self, in_dim: int, out_dim: int, name: str = "EquiformerMeanConv", num_blocks: int = 5, nc: int = 8, concat_pe_per_layer: bool = False, *args, **kwargs):
        super(EquiformerMeanConv, self).__init__(in_dim, out_dim, name, num_blocks, nc, 0)
        self.input_irreps = o3.Irreps('{}x0e'.format(self.in_dim))

        self.hidden_irreps = get_standard_irreps(self.nc)

        self.lin0 = LinearRS(self.input_irreps, self.hidden_irreps)
        self.blocks = ModuleList()
        for _ in range(self.num_blocks):
            block = EquiformerResBlockMeanConv(in_dim=nc, out_dim=nc, hc=nc, activation_func=partial(F.leaky_relu, negative_slope=0.2))
            self.blocks.append(block)

        self.lin1 = LinearRS(self.hidden_irreps, o3.Irreps('{}x0e'.format(self.out_dim)))
    
    def forward(self, x: Tensor, edge_index: Tensor, coordinates: Tensor, batch: Tensor | None = None, edge_attr: Tensor | None = None):
        x = self.lin0(x)
        for i in range(self.num_blocks):
            x = self.blocks[i](x, edge_index, edge_attr, batch, coordinates)
        x = self.lin1(x)
        return x

class LoadModel(LoadModelProto):
    def __call__(
        self,
        in_dims: list[int],
        out_dims: list[int],
        model_class: Module = GATResMeanConv, # type: ignore
        load_weights_from: Literal["model_config", "train_config"] = "train_config",
        do_load_best: bool = True,
        **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        pe_config: PEConfig = train_config.pe_config
        in_dim = in_dims[0]
        out_node_dim = out_dims[0]
        modules = []
        # TODO make pe_config a list, and choose the model class based on the pe_config. This requires some other refactorings
        for model_config in model_configs:
            model = model_class(
                nc=model_config.nc,
                num_blocks=model_config.num_layers,
                in_dim=in_dim,
                out_dim=out_node_dim,
                pe_dim=pe_config.pe_dim
            )
            setattr(model, "name", model_config.name)
            if load_weights_from == "model_config" and model_config.weight_path != "" and os.path.exists(model_config.weight_path):
                models, _ = load_weights(path=model_config.weight_path, models=[model], load_keys=[model_config.name])
                model = models[0]
            modules.append(model)
        if load_weights_from == "train_config" and train_config.load_path != "" and os.path.exists(train_config.load_path):
            filter_word = "best" if do_load_best else "last"
            model_weight_paths = [entry for entry in os.listdir(train_config.load_path) if "training_log" not in entry and filter_word in entry]
            if len(model_weight_paths) > 0:
                assert len(model_weight_paths) == len(modules)
                for i in range(len(modules)):
                    matching_order_paths = [path for path in model_weight_paths if str(i) in path]
                    matching_order_path = matching_order_paths[0]
                    tmps, _ = load_weights(
                        path=os.path.join(train_config.load_path, matching_order_path), models=[modules[i]], load_keys=[model_configs[i].name]
                    )
                    modules[i] = tmps[0].to(train_config.device)

        return modules

class MLP(Module):
    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None, edge_attr: Tensor | None = None) -> Tensor:
        assert edge_attr is not None
        # take node id pairs
        src, dst = edge_index[0], edge_index[1]

        # get their feature and concat with edge feature (assume shape is perfectly fit)
        cat_x = torch.cat([x[src], x[dst], edge_attr], dim=-1)

        # your mlp architecture
        out = self.actual_mlp(cat_x)

        return out