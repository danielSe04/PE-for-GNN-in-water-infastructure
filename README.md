# 2025-BSC-s5202841-Daniel-Seidel
Bachelor thesis Daniel Seidel


## Setup
First create a virtual environment using python 3.9.6 (this is the latest version supported on Habrok): 
```
python3 -m venv .env
source .env/bin/activate
```
Then install all the required dependencies. Note that the versions specified in the requirements are based on python 3.9.6, so if you use another python version some of these might be incompatible. Additionally, check the torch version as it is specific to the GPU used.
```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
Lastly, login to wandb