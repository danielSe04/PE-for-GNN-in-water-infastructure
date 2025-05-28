#
# Created on Wed Nov 20 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Training code for task Pressure Estimation
# ------------------------------
#
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable
import time
import numpy as np
import os
import math

from torch.nn.modules import Module
from torch.optim import Optimizer
from gigantic_dataset.model.pe_initializers import PE_Initializer
from gigantic_dataset.utils.auxil_v8 import pretty_print
from gigantic_dataset.utils.train_utils import (
    generate_unique_name_from_config,
    print_metrics,
    save_checkpoint,
    log_metrics_on_wandb,
    apply_masks,
    print_single_metrics,
    wrapper_data_loader,
)
from gigantic_dataset.utils.gen_random_mask_v8 import generate_batch_mask
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch
import wandb
from gigantic_dataset.utils.train_protos import StartProfilerProto, ForwardProto, TrainOneEpochProto, TestOneEpochProto, ConfigRef
from gigantic_dataset.utils.pe_utils import plot_pe_wandb, calculate_pe_loss, lw_tensor_local_moran, calculate_subgraph


class WandbStartProfiler(StartProfilerProto):
    def __call__(self, dataset_name: str = "", overriden_project_name: str = "", **kwargs: Any) -> tuple[str, str]:
        config = ConfigRef.config

        run_name, suffix, union_model_name = generate_unique_name_from_config(config, dataset_name)

        my_dict = config.as_dict()
        my_dict["run_name"] = run_name
        
        # start a new wandb run to track this script
        if config.log_method == "wandb":
            if wandb.run is not None:  # check if wandb is inititized
                wandb.finish(0, quiet=True)
            wandb.init(
                # set the wandb project where this run will be logged
                project=overriden_project_name if overriden_project_name != "" else config.project_name,
                name=run_name,
                # track hyperparameters and run metadata
                config=my_dict,
                settings=None,
            )
        if config.save_path != "":
            os.makedirs(config.save_path, exist_ok=True)

        pretty_print(my_dict)
        return run_name, suffix


class SupervisedSingleForward(ForwardProto):
    def __call__(
        self, models: list[Module], data: Data, batch_mask: torch.Tensor, take_first_channel: bool = True, **kwargs: Any
    ) -> tuple[Any, Any, Any | None]:
        assert data.x is not None and data.y is not None and isinstance(data.y, torch.Tensor)

        out = models[0](x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        y_pred = out
        y_true = data.y
        # exclude non-judging channels
        if y_true.shape[-1] > 1 and take_first_channel:
            y_true = y_true[..., 0]
            y_true.unsqueeze_(dim=-1)
            y_pred = y_pred[..., 0]
            y_pred.unsqueeze_(dim=-1)

        return (y_true, y_pred, out)


class SemiSingleForwardPE(ForwardProto):
    def __init__(self, pe_initializer: PE_Initializer, pe_supervised: bool = False):
        self.pe_initializer = pe_initializer
        self.pe_supervised = pe_supervised

    def __call__(
        self, models: list[Module], data: Data, batch_mask: torch.Tensor, take_first_channel: bool = True, **kwargs: Any
    ) -> tuple[Any, Any, Any | None, Any, Any, Any]:
        assert data.x is not None
        pe_dim = models[0].get_pe_dim()
        pe = self.pe_initializer(edge_index=data.edge_index, num_nodes=len(data.x), pe_dim=pe_dim, batch=data.batch, geo_coords=data.coordinates)
        data.y = deepcopy(data.x)

        data.x[batch_mask, 0] = 0
        if not self.pe_supervised:
            print("Positional encoding is semi-supervised")
            pe[batch_mask] = 0
        out, pe_out = models[0](x=data.x, edge_index=data.edge_index, pe=pe, batch=data.batch, edge_attr=data.edge_attr)
        y_pred = apply_masks(out, [batch_mask])  # out[batch_mask] #type:ignore
        y_true = data.y[batch_mask]  # type:ignore
        pe_pred = pe_out
        pe_true = pe if not ConfigRef.config.pe_criterion == "morans-i" else lw_tensor_local_moran(y=pe, w_sparse=data.edge_index).to(pe.device)
        if not self.pe_supervised:
            pe_pred = apply_masks(pe_out, [batch_mask])
            pe_true = pe[batch_mask]

        if y_true.shape[-1] > 1 and take_first_channel:
            y_true = y_true[..., 0]  # exclude non-judging channels
            y_true.unsqueeze_(dim=-1)
        
        return (y_true, y_pred, out, pe_true, pe_pred, pe_out)

class SemiSingleForward(ForwardProto):
    def __call__(
        self, models: list[Module], data: Data, batch_mask: torch.Tensor, take_first_channel: bool = True, **kwargs: Any
    ) -> tuple[Any, Any, Any | None]:
        assert data.x is not None
        data.y = deepcopy(data.x)

        data.x[batch_mask, 0] = 0
        out = models[0](x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr)

        y_pred = apply_masks(out, [batch_mask])  # out[batch_mask] #type:ignore
        y_true = data.y[batch_mask]  # type:ignore

        if y_true.shape[-1] > 1 and take_first_channel:
            y_true = y_true[..., 0]  # exclude non-judging channels
            y_true.unsqueeze_(dim=-1)
        return (y_true, y_pred, out)


class TrainOneEpoch(TrainOneEpochProto):
    def __call__(
        self,
        models: list[Module],
        optimizers: list[Optimizer],
        loader: DataLoader,
        criterion: Callable[..., Any],
        metric_fn_dict: dict[str, Callable[..., Any]],
        **kwargs: Any,
    ) -> tuple[float, dict, Any, float, Any | None]:
        config = ConfigRef.config
        func_ref = ConfigRef.ref
        device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        mask_rate = config.mask_rate
        use_data_batch = config.use_data_batch
        for pt in models:
            pt.train()
            pt.to(device)
        len_loader_dataset = len(loader.dataset)  # type:ignore
        total_loss = 0
        total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
        out = None

        for data in loader:  # Iterate in batches over the training dataset.
            [optimizer.zero_grad() for optimizer in optimizers]  # Clear gradients.
            if config.subgraphing:
                data = calculate_subgraph(data, k=config.k, batch_size=config.batch_size_subgraph)

            num_nodes = torch.unique(data.batch, return_counts=True)[1]
            batch_mask = generate_batch_mask(num_nodes=num_nodes, edge_index=data.edge_index, mask_rate=mask_rate, required_mask=None)

            non_blocking = False
            data.x = data.x.to(device, non_blocking=non_blocking)
            data.y = data.y.to(device, non_blocking=non_blocking) if "y" in data else None
            data.edge_y = data.edge_y.to(device, non_blocking=non_blocking) if "edge_y" in data else None

            data.edge_attr = data.edge_attr.to(device, non_blocking=non_blocking) if "edge_attr" in data else None
            data.batch = data.batch.to(device, non_blocking=non_blocking) if use_data_batch else None
            data.edge_index = data.edge_index.to(device, non_blocking=non_blocking)
            data.coordinates = data.coordinates.to(device, non_blocking=non_blocking) if "coordinates" in data else None

            y_true, y_pred, out, *pe = func_ref.forward_fn(models=models, data=data, batch_mask=batch_mask, **kwargs)
            
            # Calculate loss
            tr_loss = criterion(y_pred, y_true)
            loss = tr_loss
            pe_loss = float('nan')
            pe_out = None
            if len(pe) == 3:
                pe_true, pe_pred, pe_out = pe
                pe_loss = calculate_pe_loss(pe_true, pe_pred, edge_index=data.edge_index)
                loss = tr_loss + config.pe_loss_alpha*pe_loss
            loss.backward()  # Derive gradients.
            [optimizer.step() for optimizer in optimizers]  # Update parameters based on gradients.

            with torch.no_grad():
                total_loss += float(tr_loss) * data.num_graphs
                y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(y_pred, y_true)

                for k, fn in metric_fn_dict.items():
                    computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                    total_metric_dict[k] += computed_metric * data.num_graphs

        with torch.no_grad():
            dividend = max(1, len_loader_dataset)
            metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
            return total_loss / dividend, metric_dict, out, pe_loss, pe_out


class TestOneEpoch(TestOneEpochProto):
    def __call__(
        self,
        models: list[Module],
        loader: DataLoader,
        criterion: Callable[..., Any],
        metric_fn_dict: dict[str, Callable[..., Any]],
        plot_pe: bool = False,
        dataset_names: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[float, dict, Any, float, Any | None]:
        config = ConfigRef.config
        func_ref = ConfigRef.ref
        device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        mask_rate = config.mask_rate
        use_data_batch = config.use_data_batch

        out = None
        for pt in models:
            pt.eval()
            pt.to(device)
        with torch.no_grad():
            total_loss = 0
            total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
            len_loader_dataset = len(loader.dataset)  # type:ignore
            for i, data in enumerate(loader):
                
                # assert data.edge_index.max() < data.num_nodes

                batch_mask = generate_batch_mask(
                    num_nodes=torch.unique(data.batch, return_counts=True)[1], edge_index=data.edge_index, mask_rate=mask_rate, required_mask=None
                )

                non_blocking = False
                data.x = data.x.to(device, non_blocking=non_blocking)
                data.y = data.y.to(device, non_blocking=non_blocking) if "y" in data else None
                data.edge_y = data.edge_y.to(device, non_blocking=non_blocking) if "edge_y" in data else None

                data.edge_attr = data.edge_attr.to(device, non_blocking=non_blocking) if "edge_attr" in data else None
                data.batch = data.batch.to(device, non_blocking=non_blocking) if use_data_batch else None
                data.edge_index = data.edge_index.to(device, non_blocking=non_blocking)
                data.coordinates = data.coordinates.to(device, non_blocking=non_blocking) if "coordinates" in data else None

                y_true, y_pred, out, *pe = func_ref.forward_fn(models=models, data=data, batch_mask=batch_mask, **kwargs)
                pe_out = None
                pe_loss = float('nan')
                if len(pe) == 3:
                    pe_true, pe_pred, pe_out = pe
                    pe_loss = criterion(pe_true, pe_pred)

                val_loss = criterion(y_pred, y_true)
                # update metrics
                y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(y_pred, y_true)
                total_loss += float(val_loss) * data.num_graphs
                for k, fn in metric_fn_dict.items():
                    computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                    total_metric_dict[k] += computed_metric * data.num_graphs
                if plot_pe and pe_out is not None:
                    if i < len(dataset_names):
                        plot_pe_wandb(pe=pe_out, edge_index=data.edge_index, plot_name=dataset_names[i])
                    else:
                        print("Error: More datasets than dataset names. Defaulting to standard dataset name.")
                        plot_pe_wandb(pe=pe_out, edge_index=data.edge_index, plot_name=f"Default_{i}")

            dividend = max(1, len_loader_dataset)
            metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
            return total_loss / dividend, metric_dict, out, pe_loss, pe_out


def train(
    models: list[torch.nn.Module],
    datasets: list[Dataset],
    train_metric_fn_dict: dict[str, Callable],
    val_metric_fn_dict: dict[str, Callable],
    dataset_names: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    sampling_strategy = kwargs.get("sampling_strategy", "batch")
    train_shuffle = kwargs.get("train_shuffle", True)
    config = ConfigRef.config
    func_ref = ConfigRef.ref

    # get default loaders
    train_loader = wrapper_data_loader(
        datasets[0], sampling_strategy=sampling_strategy, batch_size=config.batch_size, shuffle=train_shuffle, pin_memory=False
    )
    val_loader = wrapper_data_loader(datasets[1], sampling_strategy=sampling_strategy, batch_size=config.batch_size, shuffle=False, pin_memory=False)

    criterion = func_ref.load_criterion(**kwargs)
    optimizers = func_ref.load_optimizers(models=models, **kwargs)
    scheduler = func_ref.load_scheduler(**kwargs)

    # intial records
    best_loss = np.inf
    best_metric_dict = {}
    best_epoch = 0
    best_save_path = ""
    last_save_path = ""

    start_time = time.time()
    print("Start time:", datetime.fromtimestamp(start_time))
    print("*" * 80)

    for epoch in range(1, config.epochs + 1):
        # print(f"Training @epoch {epoch}...")
        tr_loss, tr_metric_dict, out, tr_loss_pe, _ = func_ref.train_one_epoch_fn(
            models=models,
            optimizers=optimizers,
            loader=train_loader,
            criterion=criterion,
            metric_fn_dict=train_metric_fn_dict,
            **kwargs,
        )

        plot_pe = (epoch == 1 or epoch == config.epochs)
        val_loss, val_metric_dict, _, val_loss_pe, _ = func_ref.test_one_epoch_fn(
            models=models,
            loader=val_loader,
            criterion=criterion,
            metric_fn_dict=val_metric_fn_dict,
            plot_pe=plot_pe,
            dataset_names=dataset_names if not plot_pe else [name + f"-{epoch}" for name in dataset_names],
            **kwargs,
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_metric_dict = val_metric_dict
            best_epoch = epoch
            # save training_checkpoint

            save_kwargs = dict(  # noqa: C408
                optimizers_state_dict={i: optimizer.state_dict() if optimizer else None for i, optimizer in enumerate(optimizers)},
                epoch=best_epoch,
                loss=best_loss,
                pe_loss=val_loss_pe,
                val_metric_dict=best_metric_dict,
                norm_type=config.norm_type,
            )

            best_save_path = save_checkpoint(path=config.save_path, models=models, prefix="best", **save_kwargs)  # type:ignore

        if (epoch == 1 or (epoch % config.log_per_epoch) == 0) and not math.isnan(tr_loss):
            print_metrics(
                epoch=epoch,
                tr_loss=tr_loss,
                tr_loss_pe=tr_loss_pe,
                val_loss=val_loss,
                val_loss_pe=val_loss_pe,
                tr_metric_dict=tr_metric_dict,
                val_metric_dict=val_metric_dict,
            )
            save_kwargs = dict(  # noqa: C408
                optimizers_state_dict={i: optimizer.state_dict() if optimizer else None for i, optimizer in enumerate(optimizers)},  # type:ignore
                epoch=epoch,  # type:ignore
                loss=tr_loss,  # type:ignore
                val_metric_dict=val_metric_dict,
                norm_type=config.norm_type,
            )
            last_save_path = save_checkpoint(path=config.save_path, models=models, prefix="last", **save_kwargs)  # type:ignore

        if config.log_method == "wandb":
            log_metrics_on_wandb(
                epoch=epoch,
                commit=True,
                train_loss=tr_loss,
                train_loss_pe=tr_loss_pe,
                val_loss=val_loss,
                val_loss_pe=val_loss_pe,
                best_loss=best_loss,
                best_epoch=best_epoch,
                tr_metric_dict=tr_metric_dict,
                val_metric_dict=val_metric_dict,
            )

        if scheduler is not None:
            scheduler.step(val_loss)

    return {"best_save_path": best_save_path, "last_save_path": last_save_path}


def eval(
    models: list[torch.nn.Module],
    datasets: list[Dataset],
    test_metric_fn_dict: dict[str, Callable],
    plot_pe: bool = False,
    dataset_names: list[str] | None = None,
    **kwargs: Any,
) -> Any | None:
    config = ConfigRef.config
    func_ref = ConfigRef.ref

    test_loader = wrapper_data_loader(datasets[-1], sampling_strategy="batch", batch_size=config.batch_size, shuffle=False, pin_memory=False)

    assert isinstance(test_loader, DataLoader)

    # load reference
    criterion = func_ref.load_criterion(**kwargs)

    best_epoch = 0

    start_time = time.time()
    dt1 = datetime.fromtimestamp(start_time)
    print("Start time:", dt1)
    print("*" * 80)

    test_loss, test_metric_dict, _, pe_loss, _ = func_ref.test_one_epoch_fn(
        models=models,
        loader=test_loader,
        criterion=criterion,
        metric_fn_dict=test_metric_fn_dict,
        config=config,
        plot_pe=plot_pe,
        dataset_names=dataset_names if not plot_pe else [name + "-test" for name in dataset_names],
        **kwargs,
    )

    print_single_metrics(
        epoch=0,
        test_loss=test_loss,
        pe_loss=pe_loss,
        test_metric_dict=test_metric_dict,
    )

    if config.log_method == "wandb":
        log_metrics_on_wandb(
            epoch=0,
            commit=True,
            test_loss=test_loss,
            pe_loss=pe_loss,
            best_epoch=best_epoch,
            test_metric_dict=test_metric_dict,
        )

    end_time = time.time()
    dt2 = datetime.fromtimestamp(end_time)
    print("*" * 80)
    print("End time:", dt2)
    print("Executation time: ", dt2 - dt1)

    if config.log_method == "wandb":
        wandb.finish()
