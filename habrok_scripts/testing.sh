#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=testing
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


# baseline
python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive_EXN.yaml" --model_config "gigantic_dataset/arguments/train/model_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/baseline_inductive+3wdns+gatres+20250914_1359/baseline_inductive+EXN+gatres+20250914_1407/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/baseline_inductive+3wdns+gatres+20250914_1359/gida_dataset_log.pt"

python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive_ky5.yaml" --model_config "gigantic_dataset/arguments/train/model_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/baseline_inductive+3wdns+gatres+20250914_1359/baseline_inductive+ky5+gatres+20250914_1359/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/baseline_inductive+3wdns+gatres+20250914_1359/gida_dataset_log.pt"

#pe-gnn
python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive_EXN.yaml" --model_config "gigantic_dataset/arguments/train/model_pe-gnn_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/pe-gnn_inductive-new+3wdns+gatres+20250914_1456/pe-gnn_inductive-new+EXN+gatres+20250914_1507/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/pe-gnn_inductive-new+3wdns+gatres+20250914_1456/gida_dataset_log.pt"

python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive_ky5.yaml" --model_config "gigantic_dataset/arguments/train/model_pe-gnn_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/pe-gnn_inductive-new+3wdns+gatres+20250914_1456/pe-gnn_inductive-new+ky5+gatres+20250914_1456/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/pe-gnn_inductive-new+3wdns+gatres+20250914_1456/gida_dataset_log.pt"

# lspe
python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive_EXN.yaml" --model_config "gigantic_dataset/arguments/train/model_lspe_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/lspe_inductive-new+3wdns+gatres+20250914_1440/lspe_inductive-new+EXN+gatres+20250914_1447/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/lspe_inductive-new+3wdns+gatres+20250914_1440/gida_dataset_log.pt"

python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive_ky5.yaml" --model_config "gigantic_dataset/arguments/train/model_lspe_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/lspe_inductive-new+3wdns+gatres+20250914_1440/lspe_inductive-new+ky5+gatres+20250914_1440/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/lspe_inductive-new+3wdns+gatres+20250914_1440/gida_dataset_log.pt"

# spe
python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive_EXN.yaml" --model_config "gigantic_dataset/arguments/train/model_spe_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/spe-inductive-new+3wdns+gatres+20250914_1604/spe-inductive-new+EXN+gatres+20250914_1700/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/spe-inductive-new+3wdns+gatres+20250914_1604/gida_dataset_log.pt"

python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive_ky5.yaml" --model_config "gigantic_dataset/arguments/train/model_spe_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/spe-inductive-new+3wdns+gatres+20250914_1604/spe-inductive-new+ky5+gatres+20250914_1604/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/spe-inductive-new+3wdns+gatres+20250914_1604/gida_dataset_log.pt"

# equiformer
python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive_EXN.yaml" --model_config "gigantic_dataset/arguments/train/model_equiformer_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/equiformer_inductive+3wdns+gatres+20250918_1419/equiformer_inductive+EXN+gatres+20250918_1719/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/equiformer_inductive+3wdns+gatres+20250918_1419/gida_dataset_log.pt"

python main.py --task inference --data_config "gigantic_dataset/arguments/train/data_inductive_ky5.yaml" --model_config "gigantic_dataset/arguments/train/model_equiformer_inductive.yaml" \
    --load_path "gigantic_dataset/experiments_logs/equiformer_inductive+3wdns+gatres+20250918_1419/equiformer_inductive+ky5+gatres+20250918_1419/" --custom_stats_tuple_pt_path "gigantic_dataset/experiments_logs/equiformer_inductive+3wdns+gatres+20250918_1419/gida_dataset_log.pt"

# tar -czf $HOME/2025-BSC-s5202841-Daniel-Seidel/gigantic_dataset/experiments_logs/logs.tar.gz ./gigantic_dataset/experiments_logs/*
# cd $HOME/2025-BSC-s5202841-Daniel-Seidel

# tar -xzf gigantic_dataset/experiments_logs/logs.tar.gz
# rm gigantic_dataset/experiments_logs/logs.tar.gz