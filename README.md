# 2025-BSC-s5202841-Daniel-Seidel
Bachelor thesis Daniel Seidel


## Setup
First create a virtual environment using python 3.11.3: 
```
python3.11 -m venv .env #You can specify a path instead of .env
source .env/bin/activate
```
Then install all the required dependencies. Note that the torch version is specific to the GPU used. First install torch, torch_geometric, and the auxiliary packages needed for torch_geometric. Then install the rest of the required packages. The list in requirements.txt includes all packages required and more.
```
pip install --upgrade pip setuptools wheel
pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torch_cluster==1.6.3+pt24cu124 torch_scatter==2.1.2+pt24cu124 torch_sparse==0.6.18+pt24cu124 torch_spline_conv==1.2.2+pt24cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install -r requirements.txt
```
Lastly, login to wandb. To do so, change the environment variable in main.py to your API key.