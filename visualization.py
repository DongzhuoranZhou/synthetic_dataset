import os
import matplotlib
# matplotlib.use("Qt5Agg")
from bhtsne import tsne
import matplotlib.pyplot as plt
import os
import argparse
import yaml
import torch
from Dataloader import load_data
import torch.nn.functional as F
from trainer import evaluate
from ogb.nodeproppred import Evaluator
import importlib

# from options.base_options import BaseOptions
import numpy as np
import configs
class visualization(object):
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.args = args
        self.saved_model_name = args.saved_model_name
        # self.saved_model_name = self.saved_model_name + '_' + str(self.dataset) + '_' + str(
        #     self.type_model) + '_' + str(self.args.num_layers) + '_' + 'with_ACM_' + str(self.args.with_ACM) + '.pth'
        self.saved_model_name = self.saved_model_name + '_' + str(self.dataset) + '_' + str(
            self.type_model) + '_' + str(self.args.num_layers) + '.pth'
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        Model = getattr(importlib.import_module("models"), self.type_model)
        self.model = Model(self.args)
        self.which_run = 0
        # if self.dataset == 'ogbn-arxiv':
        #     self.data, self.split_idx = load_ogbn(self.dataset)
        #     self.data.to(self.device)
        #     self.train_idx = self.split_idx['train'].to(self.device)
        #     self.evaluator = Evaluator(name='ogbn-arxiv')
        #     self.loss_fn = torch.nn.functional.nll_loss
        # else:
        self.data = load_data(self.dataset, self.which_run)
        neighbour_indices = dict()
        for i in range(self.data.num_nodes):
            # index_neighbours = torch.where(self.data.edge_index[0] == i)[0]
            index_node = i
            index_neighbours_src = torch.where(self.data.edge_index[0] == i)[0]
            index_neighbours_tar = torch.where(self.data.edge_index[1] == i)[0]
            index_neighbours = torch.cat((index_neighbours_src, index_neighbours_tar), dim=0)
            index_neighbours = torch.unique(index_neighbours)
            neighbour_indices_tar = self.data.edge_index[:, index_neighbours][1]
            neighbour_indices_src = self.data.edge_index[:, index_neighbours][0]
            neighbour_indices[i] = torch.cat((neighbour_indices_tar, neighbour_indices_src), dim=0)
            neighbour_indices[i] = torch.unique(neighbour_indices[i])
            neighbour_indices[i] = neighbour_indices[i][~np.isin(neighbour_indices[i], index_node)]
        self.data.neighbour_indices = neighbour_indices
        self.data.to(self.device)

    def plot_points(self, logits, before_train=False, projection="tsne"):
        colors = [
            '#008080', '#420420', '#7fe5f0', '#065535',
            '#ffd700', '#ffc0cb', '#bada55',
        ]
        if projection == "tsne":
            z = tsne(logits.astype('float64'))
        else:
            z = logits
        self.data.to("cpu")
        y = self.data.y.numpy()

        # plt.figure(figsize=(6, 6))
        list_s = [40,5]
        for i in range(self.args.num_classes):
            plt.scatter(z[y == i, 0], z[y == i, 1], s=list_s[i], color=colors[i])
            print("class",i)
        plt.axis('off')
        if before_train:
            before_train = "before_train"
        else:
            before_train = "after_train"
        plt.title("model: {} {} ".format(self.saved_model_name, before_train))
        plt.show()

    def test(self, before_train=False):
        self.data.to(self.device)
        if not before_train:
            state_dict_path = os.path.join(self.saved_model_name)
            state_dict = torch.load(state_dict_path)
            self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        # self.model.double()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)

        acc_test = evaluate(logits, self.data.y, self.data.test_mask)
        print("Test Accuracy: {:.4f}".format(acc_test))
        return logits

    def energy_calculation_for_output(self, before_train=False):
        self.data.to(self.device)
        if not before_train:
            state_dict_path = os.path.join(self.saved_model_name)
            state_dict = torch.load(state_dict_path)
            self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        features_out = self.model(self.data.x, self.data.edge_index)
        # E_energy = torch.zeros(1).to("cpu")
        # features_out = features_out.to("cpu")
        # data = self.data.to("cpu")
        # for i in range(data.num_nodes):
        #     node = features_out[i]
        #     neighbour_indice = data.neighbour_indices[i]
        #     neighbour_indice = neighbour_indice[~np.isin(neighbour_indice, i)]
        #     distance = node.unsqueeze(dim=0) - features_out[neighbour_indice, :]
        #     # E_energy.append(torch.mean(torch.norm(distance,p=2,dim=1),dim=0))
        #     # E_energy += torch.mean(torch.norm(distance, p=2, dim=1), dim=0, dtype=torch.float64)
        #     E_energy += torch.sum(torch.norm(distance, p=2, dim=1), dim=0, dtype=torch.float64)
        # E_energy_mean = E_energy / data.num_nodes
        E_energy_mean = self.energy_calculation(features_out)
        return E_energy_mean

    def energy_calculation(self, features):
        E_energy = torch.zeros(1).to("cpu")
        features_out = features.to("cpu")
        data = self.data.to("cpu")
        for i in range(data.num_nodes):
            node = features_out[i]
            neighbour_indice = data.neighbour_indices[i]
            neighbour_indice = neighbour_indice[~np.isin(neighbour_indice, i)]
            distance = node.unsqueeze(dim=0) - features_out[neighbour_indice, :]
            # E_energy.append(torch.mean(torch.norm(distance,p=2,dim=1),dim=0))
            # E_energy += torch.mean(torch.norm(distance, p=2, dim=1), dim=0, dtype=torch.float64)
            E_energy += torch.sum(torch.norm(distance, p=2, dim=1), dim=0, dtype=torch.float64)
        E_energy_mean = E_energy / data.num_nodes
        return E_energy_mean

    def energy_for_each_layer(self):
        pass

    def plot_energy(self):
        pass

if __name__ == '__main__':
    # load model, test, save logits, plot

    # args = BaseOptions().initialize()
    args = configs.arg_parse()
    visualizationer = visualization(args)
    projection = None # "tsne"
    # input
    y = visualizationer.data.y.cpu().numpy()
    input_embedding = visualizationer.data.x.cpu().numpy()
    T1_embedding_input = input_embedding[y == 1, :]
    T2_embedding_input = input_embedding[y == 0, :]
    dis_T1_T2_root = np.linalg.norm(T1_embedding_input - T2_embedding_input, axis=1)
    print("Distance between T1 and T2 root input: {:.9f}".format(np.mean(dis_T1_T2_root)))

    # visualizationer.data.to(visualizationer.device)
    # visualizationer.plot_points(input_embedding, before_train=True, projection="tsne")


    # E_energy_mean = visualizationer.energy_calculation(visualizationer.data.x)
    # print("Energy of initial nodes embedding: {:.4f}".format(E_energy_mean.item()))
    # output
    # output with untrained model
    logits_before_train = visualizationer.test(before_train=True)
    visualizationer.plot_points(logits_before_train.cpu().numpy(), before_train=True, projection=projection)
    E_energy_mean = visualizationer.energy_calculation_for_output(before_train=True)
    print("Energy of nodes embedding with untrained model: {:.9f}".format(E_energy_mean.item()))
    T_root_embedding_output = logits_before_train.cpu().numpy()
    T1_embedding_output = T_root_embedding_output[y == 1, :]
    T2_embedding_output = T_root_embedding_output[y == 0, :]
    dis_T1_T2_root = np.linalg.norm(T1_embedding_output - T2_embedding_output, axis=1)
    print("Distance between T1 and T2 root with untrained model: {:.9f}".format(np.mean(dis_T1_T2_root)))

    # output with trained model
    logits_after_train = visualizationer.test(before_train=False)
    visualizationer.plot_points(logits_after_train.cpu().numpy(), before_train=False, projection=projection)
    E_energy_mean = visualizationer.energy_calculation_for_output(before_train=False)
    print("Energy of nodes embedding with trained model: {:.9f}".format(E_energy_mean.item()))
    T_root_embedding_output = logits_after_train.cpu().numpy()
    T1_embedding_output = T_root_embedding_output[y == 1, :]
    T2_embedding_output = T_root_embedding_output[y == 0, :]
    dis_T1_T2_root = np.linalg.norm(T1_embedding_output - T2_embedding_output, axis=1)
    print("Distance between T1 and T2 root: {:.9f}".format(np.mean(dis_T1_T2_root)))
    pass