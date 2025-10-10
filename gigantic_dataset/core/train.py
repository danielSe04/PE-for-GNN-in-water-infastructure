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
from gigantic_dataset.utils.auxil_v8 import pretty_print
from gigantic_dataset.utils.train_utils import (
    generate_unique_name_from_config,
    print_metrics,
    save_checkpoint,
    log_metrics_on_wandb,
    apply_masks,
    print_single_metrics,
    wrapper_data_loader,
    get_data_loader_per_network
)
from gigantic_dataset.core.datasets_large import get_dataset_name_from_zip_file_path
from gigantic_dataset.utils.gen_random_mask_v8 import generate_batch_mask
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import subgraph
from torch_geometric.loader import DataLoader
import torch
import wandb
from gigantic_dataset.utils.train_protos import StartProfilerProto, ForwardProto, TrainOneEpochProto, TestOneEpochProto, ConfigRef
from gigantic_dataset.utils.pe_utils import plot_pe_wandb, calculate_auxiliary_loss, lw_tensor_local_moran


class WandbStartProfiler(StartProfilerProto):
    def __call__(self, dataset_name: str = "", overriden_project_name: str = "", print_config: bool = True, **kwargs: Any) -> tuple[str, str]:
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
        if print_config:
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
    '''
    This class implements the same functionality as SemiSingleForward, but adds and returns the positional encoding in the forward function.
    '''
    def __init__(self, pe_supervised: bool = False):
        self.pe_supervised = pe_supervised

    def __call__(
        self, models: list[Module], data: Data, batch_mask: torch.Tensor, take_first_channel: bool = True, **kwargs: Any
    ) -> tuple[Any, Any, Any | None, Any, Any, Any]:
        assert data.x is not None
        assert models[0].pe_initializer is not None
        # calculate the positional encoding
        pe = models[0].get_pe(data=data)
        data.y = deepcopy(data.x)

        # mask node features and possibly pe
        data.x[batch_mask, 0] = 0
        if not self.pe_supervised:
            pe[batch_mask] = 0

        result = models[0](x=data.x, edge_index=data.edge_index, pe=pe, batch=data.batch, edge_attr=data.edge_attr)
        if isinstance(result, tuple):
            out, pe_out = result
        else:
            out = result
            pe_out = pe
        y_pred = apply_masks(out, [batch_mask])  # out[batch_mask] #type:ignore
        y_true = data.y[batch_mask]  # type:ignore
        pe_pred = pe_out
        pe_true = pe

        if not self.pe_supervised:
            pe_pred = apply_masks(pe_out, [batch_mask])
            pe_true = pe[batch_mask]

        if y_true.shape[-1] > 1 and take_first_channel:
            y_true = y_true[..., 0]  # exclude non-judging channels
            y_true.unsqueeze_(dim=-1)
        
        return (y_true, y_pred, out, pe_true, pe_pred, pe_out)

class SemiSingleForward(ForwardProto):
    def __call__(
        self, models: list[Module], data: Data, batch_mask: torch.Tensor, take_first_channel: bool = True, pass_coordinates: bool = False, **kwargs: Any
    ) -> tuple[Any, Any, Any | None]:
        assert data.x is not None
        data.y = deepcopy(data.x)

        data.x[batch_mask, 0] = 0
        if not pass_coordinates:
            out = models[0](x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        else:
            assert "coordinates" in data, "Coordinates are supposed to be passed, but are not included in Data object"
            out = models[0](x=data.x, edge_index=data.edge_index, coordinates=data.coordinates, batch=data.batch, edge_attr=data.edge_attr)

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
        total_loss = total_loss_aux = 0.0
        aux_criterion = ConfigRef.config.pe_config.aux_criterion
        total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
        out = pe_out = None

        for data in loader:  # Iterate in batches over the training dataset.
            [optimizer.zero_grad() for optimizer in optimizers]  # Clear gradients.

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
            data.rw_pe = data.rw_pe.to(device, non_blocking=non_blocking) if "rw_pe" in data else None
            data.eigenvalues = data.eigenvalues.to(device, non_blocking=non_blocking) if "eigenvalues" in data else None
            data.eigenvectors = data.eigenvectors.to(device, non_blocking=non_blocking) if "eigenvectors" in data else None

            y_true, y_pred, out, *pe = func_ref.forward_fn(models=models, data=data, batch_mask=batch_mask, **kwargs)
            
            # Calculate loss
            tr_loss = criterion(y_pred, y_true)
            loss = tr_loss
            pe_out = None
            if not aux_criterion == "": # if there is an auxiliary learning task, calculate its loss
                pe_true = pe_pred = pe_out = None
                if len(pe) == 3:
                    pe_true, pe_pred, pe_out = pe
                if aux_criterion == "morans-i":
                    aux_true = lw_tensor_local_moran(y=data.y, edge_index=data.edge_index, batch=data.batch).to(device)
                    aux_true = aux_true[:, 0] # take only the pressure estimation, as in the regular criterion
                    aux_pred = data.y[:, 0] # type: ignore
                else:
                    assert pe_true is not None and pe_pred is not None, "Calculating auxiliary loss based on positional encoding, but PE is None"
                    aux_true = pe_true
                    aux_pred = pe_pred
                aux_loss = calculate_auxiliary_loss(aux_criterion, aux_true, aux_pred, edge_index=data.edge_index, batch=data.batch)
                total_loss_aux += aux_loss
                loss = tr_loss + config.pe_config.aux_loss_alpha*aux_loss

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
            return total_loss / dividend, metric_dict, out, total_loss_aux / dividend, pe_out


class TestOneEpoch(TestOneEpochProto):
    def __call__(
        self,
        models: list[Module],
        loader: DataLoader,
        criterion: Callable[..., Any],
        metric_fn_dict: dict[str, Callable[..., Any]],
        plot_pe: bool = False, # Plotting pe assumes that data loader only loads one type of graph topology
        topology_name: str = "",
        **kwargs: Any,
    ) -> tuple[float, dict, Any, float]:
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
            total_loss_aux = 0
            aux_criterion = ConfigRef.config.pe_config.aux_criterion
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
                data.rw_pe = data.rw_pe.to(device, non_blocking=non_blocking) if "rw_pe" in data else None
                data.eigenvalues = data.eigenvalues.to(device, non_blocking=non_blocking) if "eigenvalues" in data else None
                data.eigenvectors = data.eigenvectors.to(device, non_blocking=non_blocking) if "eigenvectors" in data else None

                y_true, y_pred, out, *pe = func_ref.forward_fn(models=models, data=data, batch_mask=batch_mask, **kwargs)
                val_loss = criterion(y_pred, y_true)
                pe_true = pe_pred = pe_out = None
                if len(pe) == 3:
                    pe_true, pe_pred, pe_out = pe
                if not aux_criterion == "":
                    if aux_criterion == "morans-i":
                        aux_true = lw_tensor_local_moran(y=data.y, edge_index=data.edge_index, batch=data.batch).to(device)
                        aux_true = aux_true[:, 0] # take only the pressure estimation, as in the regular criterion
                        aux_pred = data.y[:, 0] # type: ignore
                    else:
                        assert pe_true is not None and pe_pred is not None, "Calculating auxiliary loss based on positional encoding, but PE is None"
                        aux_true = pe_true
                        aux_pred = pe_pred
                    aux_loss = calculate_auxiliary_loss(aux_criterion, aux_true, aux_pred, edge_index=data.edge_index, batch=data.batch)
                    total_loss_aux += aux_loss

                # update metrics
                y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(y_pred, y_true)
                total_loss += float(val_loss) * data.num_graphs
                for k, fn in metric_fn_dict.items():
                    computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                    total_metric_dict[k] += computed_metric * data.num_graphs
                # Plot the positional encoding
                if i == 0 and plot_pe and pe_out is not None: # Plotting is only done once, and if a pe is actually available
                    if data.batch is None or len(data.batch.unique()) == 1:
                        plot_pe_wandb(pe=pe_out, edge_index=data.edge_index, plot_name=topology_name)
                        continue
                    snapshot_ids = data.batch.unique(sorted=True)
                    num_plots = 2
                    chosen_snapshots = np.random.choice(snapshot_ids.to('cpu'), size=num_plots, replace=False)
                    pe_plot = []
                    for id in chosen_snapshots:
                        pe_plot.append(pe_out[data.batch == id])
                    node_idx = (data.batch == chosen_snapshots[0]).nonzero(as_tuple=False).view(-1) # Take the indices of one graph example
                    edge_index, _ = subgraph(node_idx, data.edge_index, edge_attr=None, relabel_nodes=True)
                    plot_pe_wandb(pe=pe_plot, edge_index=edge_index, plot_name=topology_name, save_wandb=(config.log_method == "wandb"))
                    

            dividend = max(1, len_loader_dataset)
            metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
            return total_loss / dividend, metric_dict, out, total_loss_aux / dividend


def train(
    models: list[torch.nn.Module],
    datasets: list[Dataset],
    train_metric_fn_dict: dict[str, Callable],
    val_metric_fn_dict: dict[str, Callable],
    train_per_network: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    sampling_strategy = kwargs.get("sampling_strategy", "batch")
    train_shuffle = kwargs.get("train_shuffle", True)
    config = ConfigRef.config
    func_ref = ConfigRef.ref

    # get default loaders
    if not train_per_network:
        train_loader = wrapper_data_loader(
            datasets[0], sampling_strategy=sampling_strategy, batch_size=config.batch_size, shuffle=train_shuffle, pin_memory=False
        )
        val_loader = wrapper_data_loader(datasets[1], sampling_strategy=sampling_strategy, batch_size=config.batch_size, shuffle=False, pin_memory=False)
        train_loaders = [train_loader]
        val_loaders = [val_loader]
    else: # If we train per network, then get data loaders for the individual networks
        loaders = get_data_loader_per_network(datasets[:2], config.batch_size)
        train_loaders = loaders[0]
        val_loaders = loaders[1]
    zip_file_paths = datasets[0].zip_file_paths # type: ignore
    topology_names = [get_dataset_name_from_zip_file_path(z) for z in zip_file_paths]
    criterion = func_ref.load_criterion(**kwargs)
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
    best_save_paths = []
    last_save_paths = []
    models_list = []
    base_models = models
    for i in range(len(train_loaders)):
        # Get the correct data loader
        train_loader = train_loaders[i]
        val_loader = val_loaders[i]
        save_path = config.save_path
        if train_per_network:
            # Start the profiler to log the statistics for this dataset.
            run_name, _ = func_ref.start_profiler_fn(dataset_name=topology_names[i], print_config=False)
            save_path = os.path.join(config.save_path, run_name)
            print(f"Training on {topology_names[i]} dataset.")
        if not i == (len(train_loaders)-1):
            # make a copy of the models
            models = deepcopy(base_models)
        else:
            models = base_models
        optimizers = func_ref.load_optimizers(models=models, **kwargs)
        for epoch in range(1, config.epochs + 1):
            tr_loss, tr_metric_dict, out, tr_loss_pe, _ = func_ref.train_one_epoch_fn(
                models=models,
                optimizers=optimizers,
                loader=train_loader,
                criterion=criterion,
                metric_fn_dict=train_metric_fn_dict,
                **kwargs,
            )

            val_loss, val_metric_dict, _, val_loss_pe = func_ref.test_one_epoch_fn(
                models=models,
                loader=val_loader,
                criterion=criterion,
                metric_fn_dict=val_metric_fn_dict,
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

                best_save_path = save_checkpoint(path=save_path, models=models, prefix="best", **save_kwargs)  # type:ignore

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
                last_save_path = save_checkpoint(path=save_path, models=models, prefix="last", **save_kwargs)  # type:ignore

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
        best_save_paths.append(best_save_path)
        last_save_paths.append(last_save_path)
        models_list.append(models)
    # All models are returned in the return dictionary so that we can test individually per network in the evaluation.
    return {"best_save_paths": best_save_paths, "last_save_paths": last_save_paths, "models": models_list}


def eval(
    models: list[torch.nn.Module] | list[list[torch.nn.Module]],
    datasets: list[Dataset],
    test_metric_fn_dict: dict[str, Callable],
    plot_pe: bool = False,
    match_models_to_networks = False,
    **kwargs: Any,
) -> Any | None:
    config = ConfigRef.config
    func_ref = ConfigRef.ref
    if isinstance(models[0], torch.nn.Module):
        models_list: list[list[torch.nn.Module]] = [models] # type: ignore
    else:
        models_list = models # type: ignore
    test_dataset = datasets[-1]

    # Create data loaders for every topology individually
    topology_loaders = topology_names = []
    if test_dataset.get_number_of_networks() > 1 or match_models_to_networks: #type: ignore
        topology_loaders = get_data_loader_per_network(test_dataset, config.batch_size)[0]
    zip_file_paths = test_dataset.zip_file_paths # type: ignore
    topology_names = [get_dataset_name_from_zip_file_path(z) for z in zip_file_paths]

    # load reference
    criterion = func_ref.load_criterion(**kwargs)

    best_epoch = 0

    start_time = time.time()
    dt1 = datetime.fromtimestamp(start_time)
    print("Start time:", dt1)
    print("*" * 80)

    if not match_models_to_networks:
        test_loader = wrapper_data_loader(test_dataset, sampling_strategy="batch", batch_size=config.batch_size, shuffle=False, pin_memory=False)
        assert isinstance(test_loader, DataLoader)
        test_loss, test_metric_dict, _, pe_loss = func_ref.test_one_epoch_fn(
            models=models_list[0],
            loader=test_loader,
            criterion=criterion,
            metric_fn_dict=test_metric_fn_dict,
            config=config,
            plot_pe=plot_pe,
            topology_name=topology_names[0] if plot_pe else "",
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
    print("First pass through time:", dt2)
    print("Executation time: ", dt2 - dt1)
     
    for i, loader in enumerate(topology_loaders):
        models_used = models_list[0] if not match_models_to_networks else models_list[i]
        test_loss, test_metric_dict, _, pe_loss = func_ref.test_one_epoch_fn(
            models=models_used,
            loader=loader,
            criterion=criterion,
            metric_fn_dict=test_metric_fn_dict,
            config=config,
            plot_pe=plot_pe,
            topology_name=topology_names[i],
            **kwargs,
        )

        print_single_metrics(
            epoch=0,
            dataset_name=topology_names[i],
            test_loss=test_loss,
            pe_loss=pe_loss,
            test_metric_dict=test_metric_dict,
        )

    end_time = time.time()
    dt2 = datetime.fromtimestamp(end_time)
    print("*" * 80)
    print("End time:", dt2)
    print("Executation time: ", dt2 - dt1)

    if config.log_method == "wandb":
        wandb.finish()
