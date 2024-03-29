import argparse
import utils.parser_utils as parser_utils

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1','True'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value: {}'.format(value))

def reset_dataset_dependent_parameters(args):


    if args.dataset == 'syn2':
        args.num_feats =16
        args.num_classes = 2
        args.dropout = 0.5  # 0.5
        args.lr = 0.008  # 0.005 0.008
        args.weight_decay = 5e-4
        args.epochs = 1000
        args.patience = 200  # 100, 200
        args.dim_hidden = 16
        args.activation = 'relu'
        # args.num_layers = 2
    if args.dataset == 'syn4':
        args.num_feats = 16
        args.num_classes = 2
        args.dropout = 0.5  # 0.5
        args.lr = 0.005  # 0.005
        args.weight_decay = 5e-4
        args.epochs = 1000
        args.patience = 200  # 100
        args.dim_hidden = 16
        args.activation = 'relu'
        # args.num_layers = 2
    if args.dataset == 'Cora':
        args.num_feats = 1433
        args.num_classes = 7
        args.dropout = 0.5  # 0.5
        args.lr = 0.005  # 0.005
        args.weight_decay = 5e-4
        args.epochs = 1000
        args.patience = 100  # 100
        args.dim_hidden = 64
        args.activation = 'relu'
    return args

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--logdir_root', dest='logdir_root',type=str,
                        help='Tensorboard log directory')
    parser.add_argument('--num_feats', dest='num_feats', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--num_classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--bn', dest='bn', action='store_const',
                        const=True, default=False,
                        help='Whether batch normalization is used')
    parser.add_argument('--name-suffix', dest='name_suffix',
                        help='suffix added to the output filename')

    parser.set_defaults(
                        logdir='log',
                        # input_dim=10,
                        # hidden_dim=20,
                        num_classes=2,
                        )

    # from Deep_GCN_Benchmarking
    parser.add_argument("--dataset", type=str, default="Cora", required=False,
                        help="The input dataset.",
                        choices=['Cora', 'Citeseer', 'Pubmed', 'ogbn-arxiv',
                                 'CoauthorCS', 'CoauthorPhysics', 'AmazonComputers', 'AmazonPhoto',
                                 'TEXAS', 'WISCONSIN', 'ACTOR', 'CORNELL',"syn1","syn2","syn3","syn4"])
    parser.add_argument('--direction', type=str, default="undirected",help='directed or undirected graph',
                        choices=['directed', 'undirected'])
    parser.add_argument("--type_split", type=str, default="Cora", required=False,
                        help="The type of dataset split.",
                        choices=["pair", "single"])
    parser.add_argument("--dataset_name", type=str, default="dataset/G_1000_pairs_depth_32_width_1_hdim_16_gap_True.pt", required=False,
                        help="The type of dataset split.")
    parser.add_argument('--saved_model_name', type=str, default='results/best_model', help='best model name')
    # build up the common parameter
    parser.add_argument('--random_seed', type=int, default=100)
    parser.add_argument('--N_exp', type=int, default=100)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument("--cuda", type=str_to_bool, default=True,
                        help="reproduce in cuda mode")
    parser.add_argument('--cuda_num', type=int, default=0, help="GPU number")
    # parser.add_argument('--log_file_name', type=str, default='time_and_memory.log')

    parser.add_argument('--compare_model', type=int, default=0,
                        help="0: test tricks, 1: test models")

    parser.add_argument('--type_model', type=str, default="GCN",
                        choices=['GCN', 'GAT', 'SGC', 'GCNII', 'DAGNN', 'GPRGNN', 'APPNP', 'JKNet', 'DeeperGCN',"EdgeDrop",'simpleGCN','G2_GNN'])
    parser.add_argument('--type_trick', type=str, default="None")
    # parser.add_argument('--layer_agg', type=str, default='concat',
    #                     choices=['concat', 'maxpool', 'attention', 'mean'],
    #                     help='aggregation function for skip connections')

    parser.add_argument('--num_layers', type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training the one shot model")
    parser.add_argument("--num_pairs", type=int, default=1000,
                        help="number of training the one shot model")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--dropout", type=float, default=0.6,
                        help="dropout for GCN")
    parser.add_argument('--embedding_dropout', type=float, default=0.6,
                        help='dropout for embeddings')
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="weight decay")  # 5e-4
    parser.add_argument('--dim_hidden', type=int, default=64)
    parser.add_argument('--transductive', type=bool, default=True,
                        help='transductive or inductive setting')
    parser.add_argument('--activation', type=str, default="relu", required=False)

    # Hyperparameters for specific model, such as GCNII, EdgeDropping, APPNNP, PairNorm
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="residual weight for input embedding")
    parser.add_argument('--lamda', type=float, default=0.5,
                        help="used in identity_mapping and GCNII")
    parser.add_argument('--weight_decay1', type=float, default=0.01, help='weight decay in some models')
    parser.add_argument('--weight_decay2', type=float, default=5e-4, help='weight decay in some models')
    parser.add_argument('--type_norm', type=str, default="None")
    parser.add_argument('--adj_dropout', type=float, default=0.5,
                        help="dropout rate in APPNP")  # 5e-4
    parser.add_argument('--edge_dropout', type=float, default=0.2,
                        help="dropout rate in EdgeDrop")  # 5e-4

    parser.add_argument('--node_norm_type', type=str, default="n", choices=['n', 'v', 'm', 'srv', 'pr'])
    parser.add_argument('--skip_weight', type=float, default=0.005)
    parser.add_argument('--num_groups', type=int, default=1)
    parser.add_argument('--has_residual_MLP', type=bool, default=False)
    parser.add_argument("--noise", type=str_to_bool,default=False,help="if add noise to the input features and labels")
    # Hyperparameters for random dropout
    parser.add_argument('--precision', type=str, default="float32",choices=['float32', 'float16', 'float64'])
    parser.add_argument('--graph_dropout', type=float, default=0.2,
                        help="graph dropout rate (for dropout tricks)")  # 5e-4
    parser.add_argument('--layerwise_dropout', action='store_true', default=False)
    parser.add_argument('--with_ACM', action='store_true', default=False)
    parser.add_argument('--normalize', type=str_to_bool, default=True,
                        help='normalize the adjacency matrix or not')
    parser.add_argument('--gcn_norm_type', type=str, default='sym', choices=['sym', 'rw'])



    # Hyperparameters for G2-GNN
    parser.add_argument('--p', type=int, default=2.5,help='exponent p in G^2')
    parser.add_argument('--drop_in', type=float, default=0.3,
                        help='input dropout rate')
    parser.add_argument('--use_gg_conv', type=str_to_bool, default=True,
                        help='if use the graph gradient on plain GNN')
    parser.add_argument('--conv_type', type=str, default='GraphSAGE', choices=['GraphSAGE', 'GCN', 'GAT'])
    parser.add_argument('--drop_G2', type=float, default=0.2,
                        help='dropout rate for G2 output layer')


    # Hyperparameters for GrahpSAGE
    parser.add_argument('--aggr', type=str, default='mean', choices=['mean', 'max'])
    args = parser.parse_args()
    args = reset_dataset_dependent_parameters(args)
    return args

