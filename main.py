#
# Created on Sat Apr 26 2025
# Copyright (c) 2025 Andrés Tello
# ------------------------------
# Purpose: main code
# ------------------------------
import argparse

from gigantic_dataset.utils.configs import *
from gigantic_dataset.core.run import pressure_estimation, pressure_estimation_inference


if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = 'YOUR WANDB API KEY HERE'

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="", type=str)
    args = parser.parse_args()

    assert args.task in ['train', 'inference'], "'--task' argument is required."

TASK = args.task

if TASK == "train":
    pressure_estimation(
        "gigantic_dataset/arguments/train/data.yaml",
        "gigantic_dataset/arguments/train/model.yaml",
    )
elif TASK == "inference":
    pressure_estimation_inference(
        "gigantic_dataset/arguments/train/data.yaml",
        "gigantic_dataset/arguments/train/model.yaml",
        save_path=r"/home/andres/Dropbox/PhD Smart Environments - "
                  r"RUG/ExternalProjects/WDN_datasets/gigantic_dataset/experiments_logs/multi-29wdn-290k+19-Anytown"
                  r"-CTOWN-d-town-EPANET-EXN-FFCL-1-foss-hanoi-Jilin-KL-ky10-ky1-ky13-ky14-ky16-ky18-ky2-ky24-ky4-ky5"
                  r"-ky6-ky7-modena-new-NPCL-1-OBCL-1-RuralNetwork-WA1+gatres+20250302_1506",
        custom_stats_tuple_pt_path=r"/home/andres/Dropbox/PhD Smart Environments - "
                                   r"RUG/ExternalProjects/WDN_datasets/gigantic_dataset/experiments_logs/multi-29wdn"
                                   r"-290k+19-Anytown-CTOWN-d-town-EPANET-EXN-FFCL-1-foss-hanoi-Jilin-KL-ky10-ky1"
                                   r"-ky13-ky14-ky16-ky18-ky2-ky24-ky4-ky5-ky6-ky7-modena-new-NPCL-1-OBCL-1"
                                   r"-RuralNetwork-WA1+gatres+20250302_1506/gida_dataset_log.pt",
        custom_subset_shuffle_pt_path=r""
        # custom_subset_shuffle_pt_path=r"/scratch/p303753/GDS_OUTPUTS/experiments_logs/single-10k+ZJ+gatres+20250302_1646/gida_dataset_log.pt"
    )