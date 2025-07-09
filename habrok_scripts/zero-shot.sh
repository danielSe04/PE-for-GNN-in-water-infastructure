#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=train-wdn-pe-zero-shot
#SBATCH --mem=32000
#SBATCH --gpus-per-node=a100:1

module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/12.4.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/torch/bin/activate

tar -czf $TMPDIR/code.tar.gz ./*
cd $TMPDIR
tar -xzf code.tar.gz

export CUDA_LAUNCH_BLOCKING=1
python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive.yaml" --model_config "gigantic_dataset/arguments/train/model_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/baseline_transductive+6wdns+gatres+20250616_2205/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/baseline_transductive+6wdns+gatres+20250616_2205/gida_dataset_log.pt"

python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive.yaml" --model_config "gigantic_dataset/arguments/train/model_pe-gnn_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/pe-gnn_deductive+6wdns+gatres+20250622_0624/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/pe-gnn_deductive+6wdns+gatres+20250622_0624/gida_dataset_log.pt"

python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive.yaml" --model_config "gigantic_dataset/arguments/train/model_lspe_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/lspe-transductive+6wdns+gatres+20250622_0548/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/lspe-transductive+6wdns+gatres+20250622_0548/gida_dataset_log.pt"

python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive.yaml" --model_config "gigantic_dataset/arguments/train/model_equiformer_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/equiformer_transductive+6wdns+gatres+20250618_1013/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/equiformer_transductive+6wdns+gatres+20250618_1013/gida_dataset_log.pt"

python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive.yaml" --model_config "gigantic_dataset/arguments/train/model_spe_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/spe-transductive+6wdns+gatres+20250622_2348/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/spe-transductive+6wdns+gatres+20250622_2348/gida_dataset_log.pt"


tar -czf $HOME/2025-BSC-s5202841-Daniel-Seidel/gigantic_dataset/experiments_logs/logs.tar.gz ./gigantic_dataset/experiments_logs/*
cd $HOME/2025-BSC-s5202841-Daniel-Seidel

tar -xzf gigantic_dataset/experiments_logs/logs.tar.gz
rm gigantic_dataset/experiments_logs/logs.tar.gz