#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=train-wdn-pe-baseline-transductive
#SBATCH --mem=8000
#SBATCH --gpus-per-node=a100:1

module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/12.4.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/torch/bin/activate

tar -czf $TMPDIR/code.tar.gz ./*
cd $TMPDIR
tar -xzf code.tar.gz

python main.py --task train --data_config "gigantic_dataset/arguments/train/data_transductive.yaml" --model_config "gigantic_dataset/arguments/train/model_transductive.yaml" --custom_subset_shuffle_pt_path "gigantic_dataset/experiments_logs/baseline_transductive+6wdns+gatres+20250904_1001/gida_dataset_log.pt"

tar -czf $HOME/2025-BSC-s5202841-Daniel-Seidel/gigantic_dataset/experiments_logs/logs.tar.gz ./gigantic_dataset/experiments_logs/*
cd $HOME/2025-BSC-s5202841-Daniel-Seidel

tar -xzf gigantic_dataset/experiments_logs/logs.tar.gz
rm gigantic_dataset/experiments_logs/logs.tar.gz