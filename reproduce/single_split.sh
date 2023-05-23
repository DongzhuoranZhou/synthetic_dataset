#!/bin/bash

#dataset=$1
dataset="syn2"
#hidden_channels_lst=(16 32 64 128 256)
#lr_lst=(0.01 0.05 0.002)

num_layers_lst=(4 6 8 10 11 12 13 14 15 16 32)

for num_layers in "${num_layers_lst[@]}"; do
		python main.py --gpu --epochs 1000 --N_exp 5 --compare_model 1 --dataset $dataset --num_layers $num_layers --type_model GAT  --dim_hidden 16 --type_split single
done

#--dataset syn2 --gpu --epochs 1000 --N_exp 2 --num_layers 32 --type_model GCN --type_split pair --type_norm pair
