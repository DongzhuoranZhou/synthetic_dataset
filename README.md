Please follow the instructions below to run the code.

## Instructions
1. Clone the repository
2. Install the required packages
3. Run the code
4. View the results
5. View the report
6. View the presentation

### 1. Clone the repository
Clone the repository to your local machine using the following command:
```
git clone https://github.com/DongzhuoranZhou/synthetic_dataset.git
```

### 2. Install the required packages
Install the required packages using the following command:
```
conda create -n deep_gcn_benchmark python=3.8
conda activate deep_gcn_benchmark
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.0+cu117.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.0+cu117.html
pip install torch-geometric
pip install networkx
pip install matplotlib
``` 

### 3. Run the code
Run the code using the following command:
```
python main.py --dataset syn2 --epochs 100 --N_exp 5 --num_layers 16 --type_model SGC --type_norm pair --type_split single --logdir logs/SGC_pairnorm
```

### 4. View the results
The results will be saved in the `results` folder.

### 5. View the log
The logs will be saved in the `logs` folder.

### 6. View the acc curves
```
python plot_acc_curve.py
```

The acc curve will be saved in the `acc_curve.png`.
