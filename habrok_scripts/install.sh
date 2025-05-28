#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=installs
#SBATCH --mem=8000

module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/torch/bin/activate

pip install -r requirements.txt