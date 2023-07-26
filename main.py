import gc

import numpy as np
import torch
from os.path import exists
import os

# from options.base_options import BaseOptions
from trainer import trainer
from utils.general_utils import set_seed, print_args, overwrite_with_yaml
import logging
import configs


def main(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    #
    # fh = logging.FileHandler('logs/log.txt')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    logging.basicConfig(level=logging.INFO,
                        )

    # logdir = args.logdir
    if not exists(args.logdir):
        os.makedirs(args.logdir)

    logfilename = "{}/{}_{}layers_{}.log".format(args.logdir, args.type_model, args.num_layers, args.dataset)
    fh = logging.FileHandler(logfilename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    list_test_acc = []
    list_valid_acc = []
    list_train_acc = []
    list_train_loss = []
    # if args.compare_model:
    #     args = overwrite_with_yaml(args, args.type_model, args.dataset)
    print_args(args, logger)
    for seed in range(args.N_exp):
        logging.info(f'seed (which_run) = <{seed}>')
        logging.info(f'seed (which_run) = <{seed}>')
        args.random_seed = seed
        set_seed(args)
        torch.cuda.empty_cache()
        trnr = trainer(args, seed)
        train_loss, valid_acc, test_acc, train_acc = trnr.train_and_test()

        list_test_acc.append(test_acc)
        list_valid_acc.append(valid_acc)
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)

        del trnr
        torch.cuda.empty_cache()
        gc.collect()

        # record training data
        logging.info('mean and std of train and test acc: train {:.9f}±{:.9f} test {:.9f}±{:.9f}'.format(
            np.mean(list_train_acc), np.std(list_train_acc),
            np.mean(list_test_acc), np.std(list_test_acc)))

    logging.info(
        'final mean and std of train and test acc with <{}> runs: train {:.9f}±{:.9f} test {:.9f}±{:.9f}'.format(
            args.N_exp, np.mean(list_train_acc), np.std(list_train_acc), np.mean(list_test_acc), np.std(list_test_acc)))

    for handler in logger.handlers:
        # 判断处理程序是否是指定的FileHandler
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == logfilename:
            # 移除该处理程序
            logger.removeHandler(handler)
    logger.handlers.clear()
    return np.mean(list_train_acc), np.std(list_train_acc), np.mean(list_test_acc), np.std(list_test_acc)


def run_all(args,num_layers_lst,logdir_root='logs'):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    width = 1
    logging.basicConfig(level=logging.INFO,
                        )
    args.logdir = '{}/dataset_{}/width_{}/hdim{}/direction_{}/Model_{}_Norm_{}_Trick_{}'.format(logdir_root,args.dataset,width,str(args.num_feats),args.direction,
                                                                            args.type_model, str(
                                                                                args.type_norm),str(args.type_trick))

    print("logdir: {}".format(args.logdir))
    num_feature = args.num_feats
    num_pairs = args.num_pairs
    acc_lst = []
    acc_dict = {}
    # summary log
    logfilename = "{}/{}_{}layers_{}_summary.log".format(args.logdir, args.type_model,
                                                         "_".join([str(num_layers_lst[0]), str(num_layers_lst[-1])]),
                                                         args.dataset)
    # logging.info("logfilename:", str(logfilename))
    # print("logfilename:", logfilename)

    if not exists(args.logdir):
        os.makedirs(args.logdir)

    for num_layers in num_layers_lst:

        dataset_name = "dataset/{}/width_{}/G_{}_pairs_depth_{}_width_{}_hdim_{}_high_gap_True.pt".format(args.dataset,width,num_pairs,num_layers,width,
                                                                                          num_feature)


        # dataset_name = "dataset/{}/width_{}/G_{}_pairs_depth_{}_width_{}_hdim_{}_high_gap_True_backup.pt".format(args.dataset,width,num_pairs,16,width,
        #                                                                                   num_feature)
        args.num_layers = num_layers *2
        args.dataset_name = dataset_name
        mean_train_acc, std_train_acc, mean_test_acc, std_test_acc = main(args)
        acc_lst.append((mean_train_acc, std_train_acc, mean_test_acc, std_test_acc))
        acc_dict[num_layers] = (mean_train_acc, std_train_acc, mean_test_acc, std_test_acc)

    fh = logging.FileHandler(logfilename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # for num_layers,acc in zip(num_layers_lst,acc_lst):
    for num_layer in num_layers_lst:
        acc = acc_dict[num_layer]
        mean_train_acc, std_train_acc, mean_test_acc, std_test_acc = acc
        logging.info(
            "num_layers: {}, mean_train_acc: {}, std_train_acc: {}, mean_test_acc: {}, std_test_acc: {}".format(
                num_layer, mean_train_acc, std_train_acc, mean_test_acc, std_test_acc))
    logging.info("acc_lst: {}".format(acc_lst))
    logging.info("acc_dict: {}".format(acc_dict))
    logging.info("num_layers_lst: {}".format(num_layers_lst))
    torch.save(acc_dict, "{}/{}_{}layers_{}_summary.pt".format(args.logdir, args.type_model,
                                                               "_".join([str(num_layers_lst[0]), str(num_layers_lst[-1])]),
                                                               args.dataset))
    for handler in logger.handlers:
        # 判断处理程序是否是指定的FileHandler
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == logfilename:
            # 移除该处理程序
            logger.removeHandler(handler)
    logger.handlers.clear()

if __name__ == "__main__":

    args = configs.arg_parse()

    # Combination of hyperparameters
    type_models_list = ['GCNII','APPNP','DAGNN','JKNet','GPRGNN','GCN','GAT','SGC',]
    type_models_list = ["G2_GNN"]
    type_models_list = ["GCN"]
    type_models_list = ["GAT"]
    type_models_list = ["SAGE"]
    # type_models_list = ["SAGE"]
    # type_models_list = ["GAT"]
    # type_models_list = ['GPRGNN']
    # type_models_list = ['GAT']
    # type_models_list = ['SGC']
    # type_models_list = ['JKNet']
    # type_models_list = ['GPRGNN']
    # type_models_list = ['DAGNN']
    # type_models_list = ['GPRGNN']
    # type_models_list = ['APPNP']
    # type_norm_list = ['pair', 'None', 'batch','group',]
    type_norm_list = ['None']

    # type_norm_list = ['group']
    # type_norm_list = ['group', 'None']
    # type_norm_list = ['None']
    # type_trick_list = ['Residual', 'None']
    type_trick_list = ['None']
    type_trick_list = ['None', 'max']
    # type_models_list = ['simpleGCN']
    # type_norm_list = ['GPRGNN']
    # type_norm_list = ['None']
    # type_trick_list = ['None']
    # num_layers_lst = [3,4,6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    num_layers_lst = [3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    num_layers_lst = [3]
    # num_layers_lst   = [24]
    # num_layers_lst = list(range(15, 33, 2))
    # num_layers_lst = [20]
    # num_layers_lst = [14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # num_layers_lst = [10]
    # num_layers_lst = [11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # num_layers_lst = [4,6]
    # num_layers_lst = [4, 6]
    # num_layers_lst = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    param_combinations = []
    for model_type in type_models_list:
        for norm_type in type_norm_list:
            for trick_type in type_trick_list:
                params = {
                    'type_model': model_type,
                    'type_norm': norm_type,
                    'type_trick': trick_type
                }
                if norm_type != 'None' and model_type not in ['GCN', 'GAT', 'SGC']:
                    continue
                # if trick_type != 'None' and norm_type != 'None' and model_type != 'GCN':
                #     continue
                if trick_type == 'Residual':
                    if model_type != 'GCN' or norm_type != 'None':
                        continue
                if trick_type == 'mean' or 'max':
                    if model_type != 'SAGE':
                        continue
                    else:
                        args.aggr = trick_type

                param_combinations.append(params)
    # Run
    for param_combination in param_combinations:
        print(param_combination)
        for key, value in param_combination.items():
            setattr(args, key, value)
        run_all(args,num_layers_lst,logdir_root=args.logdir_root)
