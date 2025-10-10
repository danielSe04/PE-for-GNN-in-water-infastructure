# 2025-BSC-s5202841-Daniel-Seidel
Bachelor thesis Daniel Seidel


## Setup
First create a virtual environment using python 3.11: 
```
python3.11 -m venv .env #You can specify a path instead of .env where the virtual environment will be stored.
source .env/bin/activate
```
Then install all the required dependencies. Note that the `torch` version is specific to the GPU used. First install `torch`, `torch_geometric`, and the auxiliary packages needed for `torch_geometric`. Then install the rest of the required packages. The list in `` includes all packages required and more.
```
pip install --upgrade pip setuptools wheel
pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torch_cluster==1.6.3+pt24cu124 torch_scatter==2.1.2+pt24cu124 torch_sparse==0.6.18+pt24cu124 torch_spline_conv==1.2.2+pt24cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install -r requirements.txt
```
Lastly, login to wandb. To do so, change the environment variable in main.py to your API key.

## Experiments
There are two modes of experiments implemented in the code, namely experiments where we train models before evaluating them (`pressure_estimation` in `run.py`) and experiments where we merely evaluate an existing model (`pressure_estimation_inference` in `run.py`). Using these two modes, the scripts in the `habrok_scripts` folder implement three types of experiments: 
1. Transductive experiments, where a model is trained and evaluated on the same networks
2. Zero-shot induction, where a model is evaluated on networks that it has never seen during training.
3. Transfer learning induction, where a base model is finetuned and evaluated on a specific network it has not seen during the prior training.

Both types of inductive experiments require a model that has been trained before. The path to that model is specified in the `load_path` flag of the program (see the scripts in the `habrok_scripts` folder). Additionally, there is the `custom_stats_tuple_pt_path` argument, which provides data for the normalization in the program. Both of these arguments must point to a model that has been trained before. In the transfer learning task, we train models individually for each network in the data configuaration file, which is specified in the `data_config` argument. This individual training is indicated by the `train_per_network` flag. Lastly, there is the `model_config` argument, which gives the specifics of the model.  The model and data configurations we used can be found in the `gigantic_dataset/arguments` folder.

## Code
The program flows straightforwardly from the `main.py`. The most important files are contained in the `core` folder: `datasets_large` contains the code for all dataset functionality, `run` and `train` contain the code that trains and evaluates the model. The model itself (GATRes) is contained in the `model` folder. This folder also contains most of the code for the positional encodings.

## Positional Encodings
The PE initializers (SPE, LSPE, PE-GNN), as well as the Equiformer model are contained in the `model` folder. The instances of the initializers themselves are stored in the main GATRes model so that they are loaded together with the model automatically. There are several variations of the regular GATRes model (the `GATResMeanConv` class) depending on which positional encoding is chosen. The positional encoding consists of a number of components, which are specified in the pe_config field of the model configuration file.
1. `pe_technique` determines how the PE is added to the model. It is either "lspe", meaning that the PE is learned and added to the node embedding in every layer, "concat" meaning that the PE is concatenated with the node embedding before the node embedding is passed through the model, or "equiformer" meaning that the equiformer is used.
2. `pe_init` specifies the type of PE initialization. This is either "spe", "rw" (for the random walk initialization of LSPE), or "geo" (for the geo-located initialization of PE-GNN).
3. `pe_dim` specifies the dimensionality of the positional encoding.
4. `pe_task` determines whether the PE is added to the node embedding regularly (then it is set to "supervised") or whether the PE is masked for a number of nodes (95%), like the variable predicted (then it is set to "semi", meaning the PE is learned in semi-supervised fashion).
5. `aux_criterion` indicates whether there is an auxiliary learning task associated to the PE, and what criterion it uses ("mse", "laplacian", "morans-i", or "" to not use an auxiliary learning task).
6. `aux_loss_alpha` is a hyperparameter that the auxiliary loss is multiplied with, that is, it determines the weight of the auxiliary learning task. The total loss then has the formula $\text{total\_loss} = \text{loss} + \text{aux\_loss\_alpha} * \text{aux\_loss}$

Any utility functions used in connection to the positional encodings are contained in the `pe_utils` file.

## Coordinates
Other than the datasets, which are on Habrok, the coordinates are stored directly in the code base. The only exception is L-TOWN, since its coordinate file is too large to upload to github. If L-TOWN is to be used, its coordinate file must be added manually. All coordinates are normalized to a range between 0 and 1 upon loading.