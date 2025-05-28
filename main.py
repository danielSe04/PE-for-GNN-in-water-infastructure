#
# Created on Sat Apr 26 2025
# Copyright (c) 2025 Andrés Tello
# ------------------------------
# Purpose: main code
# ------------------------------
import argparse
import os

from gigantic_dataset.utils.configs import *
from gigantic_dataset.core.run import pressure_estimation, pressure_estimation_inference


if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = '07891121c333b9108aadfe41b1b5720f631c7cad'

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="", type=str)
    parser.add_argument('--data_config', default="", type=str)
    parser.add_argument('--model_config', default="", type=str)
    parser.add_argument('--save_path', default="", type=str)
    parser.add_argument('--custom_stats_tuple_pt_path', default="", type=str)
    args = parser.parse_args()

    assert args.task in ['train', 'inference'], "'--task' argument is required."
    assert os.path.exists(args.data_config), "'--data_config' must be valid path to yaml file."
    assert os.path.exists(args.model_config), "'--model_config' must be valid path to yaml file."

TASK = args.task
DATA_CONFIG = args.data_config
MODEL_CONFIG = args.model_config
SAVE_PATH = args.save_path
CUSTOM_STATS_TUPLE_PT_PATH = args.custom_stats_tuple_pt_path

if TASK == "train":
    pressure_estimation(
        DATA_CONFIG,
        MODEL_CONFIG,
    )
elif TASK == "inference":
    if not SAVE_PATH == "":
        assert os.path.exists(SAVE_PATH), "--save_path must be a valid path or empty."
    if not CUSTOM_STATS_TUPLE_PT_PATH == "":
        assert os.path.exists(CUSTOM_STATS_TUPLE_PT_PATH), "--custom_stats_tuple_pt_path must be a valid path or empty."
    pressure_estimation_inference(
        DATA_CONFIG,
        MODEL_CONFIG,
        save_path=SAVE_PATH,
        custom_stats_tuple_pt_path=CUSTOM_STATS_TUPLE_PT_PATH,
        custom_subset_shuffle_pt_path=r""
        # custom_subset_shuffle_pt_path=r"/scratch/p303753/GDS_OUTPUTS/experiments_logs/single-10k+ZJ+gatres+20250302_1646/gida_dataset_log.pt"
    )