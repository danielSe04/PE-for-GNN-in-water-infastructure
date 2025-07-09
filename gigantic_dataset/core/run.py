#
# Created on Sat Nov 23 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Run code
# ------------------------------
#

from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable
from torch_geometric.data import Dataset
from gigantic_dataset.utils.configs import TrainConfig, GidaConfig
from gigantic_dataset.core.train import train, eval, TrainOneEpoch, TestOneEpoch, WandbStartProfiler, SemiSingleForward, SemiSingleForwardPE

from gigantic_dataset.utils.train_protos import (
    load_gida_datasets,
    default_post_foward_transform,
    default_load_criterion,
    default_load_optimizers,
    default_load_scheduler,
)
from gigantic_dataset.utils.train_utils import (
    MinMaxNormalize,
    ZNormalize,
    get_default_metric_fn_collection,
    find_latest_files,
)
from gigantic_dataset.model.gatres import LoadModel, GATResMeanConvLSPE, PE_concat_GATResMeanConv, EquiformerMeanConv


from torch_geometric.data import Data
from gigantic_dataset.utils.train_protos import ConfigRef

from gigantic_dataset.utils.func_ref import FuncRef
from functools import partial
import os.path as osp


def extract_dataset_name(gida_config: GidaConfig) -> str:
    """
    Helper function to export a nice name for folder and wandb run.
    """
    # names: list[str] = []
    # for p in gida_config.zip_file_paths:
    #     basename = osp.basename(p)
    #     segments = basename.split("_")
    #     if len(segments) >= 2:
    #         names.append(segments[1])
    #     else:
    #         names.append(basename)
    # dataset_name = "-".join(names)

    if len(gida_config.zip_file_paths) <= 0:
        return ""

    num_networks = len(gida_config.zip_file_paths)
    if num_networks == 1:
        p = gida_config.zip_file_paths[0]
        basename = osp.basename(p)
        segments = basename.split("_")
        if len(segments) >= 2:
            dataset_name = segments[1]
        else:
            dataset_name = basename
    else:
        dataset_name = f"{num_networks}wdns"
    return dataset_name

def add_pe_to_func_ref(func_ref, train_config: TrainConfig):
    '''
    Helper function that modifies the func_ref object based on the configuration of the positional encoding.
    '''
    pe_technique = train_config.pe_config.pe_technique
    if pe_technique == "":
        return func_ref
    if pe_technique == "equiformer":
        func_ref.load_models = partial(LoadModel(), model_class=EquiformerMeanConv)
        func_ref.forward_fn = partial(SemiSingleForward(), pass_coordinates=True)
    else:
        if pe_technique == "lspe":
            func_ref.load_models = partial(LoadModel(), model_class=GATResMeanConvLSPE)
        elif pe_technique == "pe-gnn" or pe_technique == "concat":
            func_ref.load_models = partial(LoadModel(), model_class=PE_concat_GATResMeanConv)
        else:
            raise NotImplementedError()
        func_ref.forward_fn = SemiSingleForwardPE(pe_supervised=(train_config.pe_config.pe_task == "supervised"))
    return func_ref
        
def pressure_estimation(
    gida_yaml_path: str,
    train_yaml_path: str,
    save_path: str = "",
    load_path: str = "",
    custom_stats_tuple_pt_path: str = "",
    custom_subset_shuffle_pt_path: str = "",
    train_per_network: bool = False,
) -> Any:
    """prepare for the pressure estimation task on gida

    Args:
        gida_yaml_path (str): gida path, corresponding to parameter set of GiDa Interface
        train_yaml_path (str): training config, parameter set for training stuff
        save_path (str, optional): to override train_config.save_path. Leave blank to auto-gen save path (and folder). Defaults to "".
        load_path (str, optional): to override train_config.load_path. If both are blank, model weights are initialized randomly. Defaults to "".
        custom_stats_tuple_pt_path (str, optional): Custom .pt file to LOAD (READ-ONLY) stats tuple. If empty, we load stats from the default dataset log in `train_config.load_path`. Defaults to "".
        custom_subset_shuffle_pt_path (str, optional):Custom .pt file to LOAD (READ-ONLY) subset shuffle ids. If empty, we load ids from the default dataset log in `train_config.load_path`. Defaults to "".
        train_per_network (bool, optional): If this is set to true, we train one separate model for each network.
    Returns:
        Any: return dict if possible
    """
    # load dataset config
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)

    # load train config
    train_config = TrainConfig()
    train_config._parsed = True
    train_config._from_yaml(train_yaml_path, unsafe_load=True)
    train_config.save_path = save_path
    if not load_path == "":
        train_config.load_path = load_path

    # to flush to terminal
    setattr(train_config, "data", gida_config.as_dict())

    # add function references, where we mix and match training code.
    dataset_name = extract_dataset_name(gida_config)
    func_ref = FuncRef(
        start_profiler_fn=partial(WandbStartProfiler(), dataset_name=dataset_name),
        forward_fn=SemiSingleForward(),
        train_one_epoch_fn=TrainOneEpoch(),
        test_one_epoch_fn=TestOneEpoch(),
        train_fn=train,
        eval_fn=eval,
        post_forward_tf_fn=default_post_foward_transform,
        load_criterion=default_load_criterion,
        load_datasets=partial(
            load_gida_datasets, custom_stats_tuple_pt_path=custom_stats_tuple_pt_path, custom_subset_shuffle_pt_path=custom_subset_shuffle_pt_path, is_training=True
        ),
        load_models=LoadModel(),
        load_optimizers=default_load_optimizers,
        load_scheduler=default_load_scheduler,
    )

    # Modify function references based on the positional encoding specified, if pe is used
    func_ref = add_pe_to_func_ref(func_ref, train_config)

    # initialize the ConfigRef which we can call from anywhere
    ConfigRef.initialize_and_start_profiler(config=train_config, ref=func_ref)

    # we first load dataset
    datasets: list[Dataset] = func_ref.load_datasets(gida_config)
    print(f"len={datasets[0].len()}")

    # gather in_dim and out_dim for loading models
    sample: Data = datasets[0][0]  # type: ignore
    in_node_dim = sample.x.shape[-1]  # type:ignore
    if sample.y is not None:
        out_node_dim = sample.y.shape[-1]  # type:ignore
    elif hasattr(sample, "edge_y") and sample.edge_y is not None:
        out_node_dim = sample.edge_y.shape[-1]  # type:ignore
    else:
        if in_node_dim > 1:  # extra attrs
            out_node_dim = 1
        else:
            out_node_dim = in_node_dim

    # take transform from gida to perform inverse normalization
    if train_config.norm_type != "unused":
        train_tf: ZNormalize | MinMaxNormalize | None = datasets[0].transform  # type:ignore
        assert train_tf is not None
        reduced_train_tf = deepcopy(train_tf)
        if in_node_dim > 1 and out_node_dim == 1:
            if isinstance(reduced_train_tf, ZNormalize):
                reduced_train_tf.mean = reduced_train_tf.mean[..., 0]
                reduced_train_tf.std = reduced_train_tf.std[..., 0]
            elif isinstance(reduced_train_tf, MinMaxNormalize):
                reduced_train_tf.min_val = reduced_train_tf.min_val[..., 0]
                reduced_train_tf.max_val = reduced_train_tf.max_val[..., 0]

        reduced_train_tf.to(device=ConfigRef.config.device)
        inverse_apply_fn: Callable = partial(reduced_train_tf.transform, denormalize=True)
        func_ref.post_forward_tf_fn = partial(func_ref.post_forward_tf_fn, tf=inverse_apply_fn)

    # start model
    models = func_ref.load_models(in_dims=[in_node_dim], out_dims=[out_node_dim])

    # run train
    ret_dict = func_ref.train_fn(
        datasets=datasets,
        models=models,
        train_metric_fn_dict=get_default_metric_fn_collection(prefix="train", task="semi"),
        val_metric_fn_dict=get_default_metric_fn_collection(prefix="val", task="semi"),
        train_per_network=train_per_network,
    )
    # temporarily comment for fast check
    # TODO: If you wish to switch wandb project, we must re-call start profiler fn and override the project name
    func_ref.start_profiler_fn(dataset_name=dataset_name, overriden_project_name=train_config.project_name.replace("train", "test"))
    models = ret_dict["models"] # type: ignore
    # test
    func_ref.eval_fn(
        datasets=datasets,
        models=models,
        test_metric_fn_dict=get_default_metric_fn_collection(prefix="test", task="semi"),
        plot_pe=True,
        match_models_to_networks=train_per_network
    )

    return ret_dict


def pressure_estimation_inference(
    gida_yaml_path: str,
    train_yaml_path: str,
    save_path: str = "",
    load_path: str = "",
    custom_stats_tuple_pt_path: str = "",
    custom_subset_shuffle_pt_path: str = "",
) -> None:
    """for inference only <br />

    Args:
        gida_yaml_path (str): gida path, corresponding to parameter set of GiDa Interface
        train_yaml_path (str): training config, parameter set for training stuff
        save_path (str, optional): to override train_config.save_path. Leave blank to auto-gen save path (and folder). Defaults to "".
        load_path (str, optional): to override train_config.load_path. If both are blank, model weights are initialized randomly. Defaults to "".
        custom_stats_tuple_pt_path (str, optional): Custom .pt file to LOAD (READ-ONLY) stats tuple. If empty, we load stats from the default dataset log in `train_config.load_path`. Defaults to "".
        custom_subset_shuffle_pt_path (str, optional):Custom .pt file to LOAD (READ-ONLY) subset shuffle ids. If empty, we load ids from the default dataset log in `train_config.load_path`. Defaults to "".
    Returns:
        list[str]: return save_path where storing model weights and training stuff.
    """  # noqa: E501
    # load dataset config
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)

    # load train config
    train_config = TrainConfig()
    train_config._parsed = True
    train_config._from_yaml(train_yaml_path, unsafe_load=True)
    if save_path != "":
        train_config.save_path = save_path

    if load_path != "":
        train_config.load_path = load_path

    # to flush to terminal
    setattr(train_config, "data", gida_config.as_dict())

    # add function references, where we mix and match training code.
    dataset_name = extract_dataset_name(gida_config)
    func_ref = FuncRef(
        start_profiler_fn=partial(WandbStartProfiler(), dataset_name=dataset_name),
        forward_fn=SemiSingleForward(),
        train_one_epoch_fn=TrainOneEpoch(),
        test_one_epoch_fn=TestOneEpoch(),
        train_fn=train,
        eval_fn=eval,
        post_forward_tf_fn=default_post_foward_transform,
        load_criterion=default_load_criterion,
        load_datasets=partial(
            load_gida_datasets, custom_stats_tuple_pt_path=custom_stats_tuple_pt_path, custom_subset_shuffle_pt_path=custom_subset_shuffle_pt_path, is_training=False
        ),
        # load_datasets=load_dask_datasets,
        load_models=LoadModel(),
        load_optimizers=default_load_optimizers,
        load_scheduler=default_load_scheduler,
    )

    # Modify function references based on the positional encoding specified, if pe is used
    func_ref = add_pe_to_func_ref(func_ref, train_config)
    
    # initialize the ConfigRef which we can call from anywhere
    ConfigRef.initialize_and_start_profiler(config=train_config, ref=func_ref)
    # we first load dataset
    datasets: list[Dataset] = func_ref.load_datasets(gida_config)
    print(f"len={datasets[0].len()}")

    # gather in_dim and out_dim for loading models
    sample: Data = datasets[0][0]  # type: ignore
    print(f"sample.x = {sample.x}")
    in_node_dim = sample.x.shape[-1]  # type:ignore
    if sample.y is not None:
        out_node_dim = sample.y.shape[-1]  # type:ignore
    elif hasattr(sample, "edge_y") and sample.edge_y is not None:
        out_node_dim = sample.edge_y.shape[-1]  # type:ignore
    else:
        if in_node_dim > 1:  # extra attrs
            out_node_dim = 1
        else:
            out_node_dim = in_node_dim

    # take transform from gida to perform inverse normalization
    if train_config.norm_type != "unused":
        train_tf: ZNormalize | MinMaxNormalize | None = datasets[0].transform  # type:ignore
        assert train_tf is not None
        train_tf = deepcopy(train_tf)
        train_tf.to(device=ConfigRef.config.device)
        inverse_apply_fn: Callable = partial(train_tf.transform, denormalize=True)
        func_ref.post_forward_tf_fn = partial(func_ref.post_forward_tf_fn, tf=inverse_apply_fn)

    # start model
    models = func_ref.load_models(model_configs=train_config.model_configs, in_dims=[in_node_dim], out_dims=[out_node_dim])

    # test
    func_ref.eval_fn(
        datasets=[datasets[0]],
        models=models,
        test_metric_fn_dict=get_default_metric_fn_collection(prefix="test", task="semi"),
        plot_pe=True,
    )