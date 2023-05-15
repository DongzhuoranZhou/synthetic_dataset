import gc

import numpy as np
import torch

# from options.base_options import BaseOptions
from trainer import trainer
from utils.general_utils import set_seed, print_args, overwrite_with_yaml
import logging
import configs

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

def main(args):
    logfilename = "logs/{}_{}layers_{}.log".format(args.type_model, args.num_layers, args.dataset)
    fh = logging.FileHandler(logfilename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    list_test_acc = []
    list_valid_acc = []
    list_train_loss = []
    if args.compare_model:
        args = overwrite_with_yaml(args, args.type_model, args.dataset)
    print_args(args, logger)
    for seed in range(args.N_exp):
        logging.info(f'seed (which_run) = <{seed}>')
        logging.info(f'seed (which_run) = <{seed}>')
        args.random_seed = seed
        set_seed(args)
        torch.cuda.empty_cache()
        trnr = trainer(args, seed)
        train_loss, valid_acc, test_acc = trnr.train_and_test()
        list_test_acc.append(test_acc)
        list_valid_acc.append(valid_acc)
        list_train_loss.append(train_loss)

        del trnr
        torch.cuda.empty_cache()
        gc.collect()

        # record training data
        logging.info('mean and std of test acc: {:.4f}±{:.4f}'.format(
            np.mean(list_test_acc), np.std(list_test_acc)))

    logging.info('final mean and std of test acc with <{}> runs: {:.4f}±{:.4f}'.format(
        args.N_exp, np.mean(list_test_acc), np.std(list_test_acc)))


if __name__ == "__main__":
    # args = BaseOptions().initialize()
    args = configs.arg_parse()
    main(args)
