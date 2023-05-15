import networkx as nx

import configs
import os
import gengraph
import utils.featgen as featgen
import numpy as np
from models.GCN import GCN
import importlib
import torch
import time
import torch.nn as nn
import sklearn.metrics as metrics
import utils.train_utils as train_utils
import torch
from torch_geometric.utils import from_networkx
import torch_geometric
from Dataloader import load_data
def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test

def train_node_classifier(G, labels, args, writer=None):
    # train/test split only for nodes
    device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
    Model = getattr(importlib.import_module("models"), args.type_model)
    model = Model(args)
    model.to(device)

    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    pyg_G = from_networkx(G)
    # data = gengraph.preprocess_input_graph(G, labels)
    data = load_data(args.dataset, which_run=0)
    loader = torch_geometric.data.DataLoader([pyg_G], batch_size=1)
    labels_train = torch.tensor(data["y"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["x"], requires_grad=True, dtype=torch.float)
    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    ypred = None
    for epoch in range(args.epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred, adj_att = model(x.cuda(), adj.cuda())
        else:
            ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]
        if args.gpu:
            loss = model.loss(ypred_train, labels_train.cuda())
        else:
            loss = model.loss(ypred_train, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.clip)

        optimizer.step()
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(
            ypred.cpu(), data["labels"], train_idx, test_idx
        )
        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss, epoch)
            writer.add_scalars(
                "prec",
                {"train": result_train["prec"], "test": result_test["prec"]},
                epoch,
            )
            writer.add_scalars(
                "recall",
                {"train": result_train["recall"], "test": result_test["recall"]},
                epoch,
            )
            writer.add_scalars(
                "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            )

        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

        if scheduler is not None:
            scheduler.step()
    print(result_train["conf_mat"])
    print(result_test["conf_mat"])

    # computation graph
    model.eval()
    if args.gpu:
        ypred, _ = model(x.cuda(), adj.cuda())
    else:
        ypred, _ = model(x, adj)
    cg_data = {
        "adj": data["adj"],
        "feat": data["feat"],
        "label": data["labels"],
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx,
    }
    # import pdb
    # pdb.set_trace()
    # io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)

def syn_task1(args, writer=None,type_model="GCN"):
    """
    For single tree1 and tree2, the node features are randomly generated. p(tree1)!=p(tree2)
    :param args:
    :param writer:
    :return:
    """
    # data
    G_list = gengraph.gen_syn1(height=3, feature_generator=featgen.GaussianFeatureGen(embedding_dim=16), max_width=2,
                            max_nodes=50)
    G_list = [T for tup in G_list for T in tup]
    G = nx.disjoint_union_all(G_list)
    labels = G.nodes[G.root].data["label"]
    num_classes = max(labels) + 1
    Model = getattr(importlib.import_module("models"), type_model)

    print("Method:", args.method)
    model = Model(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        num_classes,
        args.num_gc_layers,
        bn=args.bn,
        args=args,
    )

    if args.gpu:
        model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task2(args, writer=None,type_model="GCN"):
    """
    For tree1 and tree2, the node features are randomly generated. p(tree1)!=p(tree2)
    :param args:
    :param writer:
    :return:
    """
    # data
    # G, labels, name = gengraph.gen_syn1(
    #     feature_generator=featgen.GaussianFeatureGen(embedding_dim=args.input_dim)
    # )
    # embedding_dim = 16
    # G_list = gen_syn1(height=3, feature_generator=featgen.GaussianFeatureGen(embedding_dim=16), max_width=2,
    #                         max_nodes=50)
    # G_list = gengraph.gen_syn2(height=3, feature_generator=featgen.GaussianFeatureGen(embedding_dim=16), max_width=4,
    #                         max_nodes=50)
    # G_list = [T for tup in G_list for T in tup]
    # G = nx.disjoint_union_all(G_list)

    G = torch.load("dataset/G_10_pairs_depth_3.pt")

    labels = nx.get_node_attributes(G, "label")
    # num_classes = 2
    # Model = getattr(importlib.import_module("models"), type_model)
    #
    # # print("Method:", args.method)
    # # model = Model(
    # #     args.input_dim,
    # #     args.hidden_dim,
    # #     args.output_dim,
    # #     num_classes,
    # #     args.num_gc_layers,
    # #     bn=args.bn,
    # #     args=args,
    # # )
    #
    # model = Model(args)
    #
    # device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
    # model = model.to

    train_node_classifier(G, labels, args, writer=writer)

def main():
    prog_args = configs.arg_parse()
    if prog_args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(prog_args.cuda_num)
        print("CUDA", prog_args.cuda_num)
    else:
        print("Using CPU")
    # torch.device(f'cuda:{prog_args.cuda_num}' if prog_args.cuda else 'cpu')
    if prog_args.dataset is not None:
        if prog_args.dataset == "syn1":
            syn_task1(prog_args)
        if prog_args.dataset == "syn2":
            syn_task2(prog_args)

if __name__ == "__main__":
    main()