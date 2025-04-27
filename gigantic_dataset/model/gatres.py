#
# Created on Mon Nov 25 2024
# Copyright (c) 2024 Huy Truong & Andrés Tello
# ------------------------------
# Purpose: a simple GATRes
# ------------------------------
#


from typing import Any, Literal

from torch import clone, Tensor
import torch.nn.functional as F
import torch
from torch.nn import Module, ModuleList
from torch_geometric.nn import GATConv, SimpleConv, Linear, BatchNorm, GraphNorm
from gigantic_dataset.utils.configs import ModelConfig
from gigantic_dataset.utils.train_protos import ConfigRef, LoadModelProto
from gigantic_dataset.utils.train_utils import load_weights
import os
import torch

from torch.nn import Module


class LoadGATRes(LoadModelProto):
    def __call__(
        self,
        in_dims: list[int],
        out_dims: list[int],
        load_weights_from: Literal["model_config", "train_config"] = "train_config",
        do_load_best: bool = True,
        **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_node_dim = out_dims[0]
        modules = []
        for model_config in model_configs:
            model = GATResMeanConv(
                nc=model_config.nc,
                num_blocks=model_config.num_layers,
                in_dim=in_dim,
                out_dim=out_node_dim,
            )
            setattr(model, "name", model_config.name)
            if load_weights_from == "model_config" and model_config.weight_path != "" and os.path.exists(model_config.weight_path):
                models, _ = load_weights(path=model_config.weight_path, models=[model], load_keys=[model_config.name])
                model = models[0]
            modules.append(model)
        if load_weights_from == "train_config" and train_config.save_path != "" and os.path.exists(train_config.save_path):
            filter_word = "best" if do_load_best else "last"
            model_weight_paths = [entry for entry in os.listdir(train_config.save_path) if "training_log" not in entry and filter_word in entry]
            if len(model_weight_paths) > 0:
                assert len(model_weight_paths) == len(modules)
                for i in range(len(modules)):
                    matching_order_paths = [path for path in model_weight_paths if str(i) in path]
                    matching_order_path = matching_order_paths[0]
                    tmps, _ = load_weights(
                        path=os.path.join(train_config.save_path, matching_order_path), models=[modules[i]], load_keys=[model_configs[i].name]
                    )
                    modules[i] = tmps[0].to(train_config.device)

        return modules


class GResBlockMeanConv(Module):
    def __init__(self, in_dim, out_dim, hc):
        super(GResBlockMeanConv, self).__init__()

        # self.norm1 = BatchNorm(in_channels=in_dim)
        self.conv1 = GATConv(in_dim, hc, 2, concat=True)
        self.conv2 = GATConv(hc * 2, out_dim, 1, concat=False)
        self.mean_conv = SimpleConv(aggr="mean")

    def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None) -> Tensor:
        # x = self.norm1(x)
        x_0 = clone(x)
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = self.mean_conv(x, edge_index) + x_0
        x = F.relu(x)
        return x


class GATResMeanConv(Module):
    def __init__(self, in_dim: int, out_dim: int, name: str = "GATResMeanConv", num_blocks: int = 5, nc: int = 32):
        super(GATResMeanConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_blocks = num_blocks
        self.nc = nc
        self.num_blocks = num_blocks

        self.lin0 = Linear(in_dim, nc)
        self.blocks = ModuleList()
        self.name = name
        for _ in range(self.num_blocks):
            block = GResBlockMeanConv(nc, nc, nc)
            self.blocks.append(block)

        self.lin1 = Linear(nc, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None, edge_attr: Tensor | None = None) -> Tensor:
        x = self.lin0(x)

        for i in range(self.num_blocks):
            x = self.blocks[i](x, edge_index, edge_attr, batch)

        x = self.lin1(x)
        return x


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
