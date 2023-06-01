import matplotlib.pyplot as plt
import numpy as np

# def acc_curve(acc_list,layer_list, save_path=None):
#     plt.figure()
#     plt.plot(acc_list,layer_list)
#     plt.xlabel("#layer")
#     plt.ylabel("acc")
#     plt.title("acc curve")
#     if save_path:
#         plt.savefig(save_path)
#     plt.show(block=False)
#     return plt
#
#
#
#
# def acc_curve_with_covariance(accuracy, covariance):
#     x = np.linspace(0, 1, 100)  # 创建一个包含100个点的线性空间
#     y = accuracy * np.exp(-0.5 * ((x - 0.5) / covariance) ** 2)  # 根据准确率和协方差计算曲线上的y值
#
#     plt.plot(x, y)  # 绘制准确率曲线
#     plt.xlabel('Threshold')  # 设置x轴标签
#     plt.ylabel('Accuracy')  # 设置y轴标签
#     plt.title('Accuracy Curve with Covariance')  # 设置图表标题
#     plt.grid(True)  # 添加网格线
#     plt.show()  # 显示图表
#
# # 示例用法
# accuracy = 0.8
# covariance = 0.2
# acc_curve_with_covariance(accuracy, covariance)
import torch


def plot_accuracy_curve_with_covariance(list_container, save_path=None, acc_type='test'):
    colors = [
        '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700', '#ffc0cb', '#bada55',
    ]
    colors = [('blue', 'lightblue'), ('green', 'lightgreen'), ('red', 'pink'), ('black', 'gray'),
              ('yellow', 'lightyellow'), ('purple', 'violet'), ('orange', 'yellow'), ('brown', 'pink'),
              ('gray', 'lightgray'), ('pink', 'lightpink'), ('violet', 'brown')]
    for index, container in enumerate(list_container):
        mean_color, std_color = colors[index]
        accuracy = container.accuracy
        covariance = container.covariance
        num_layer = container.num_layer
        x = np.array(num_layer)
        mean = np.array(accuracy)
        std = np.sqrt(np.array(covariance))

        plt.plot(x, mean, color=mean_color, label=container.label + '_Accuracy')
        plt.fill_between(x, mean - std, mean + std, color=std_color, alpha=0.3, label=container.label + '_Covariance')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve of GCN with Covariance with single split')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show(block=False)
    return plt


def plot_train_test_accuracy_curve_with_covariance(list_container, save_path=None, acc_type_list=None, title=None):
    # colors = [
    #     '#008080', '#420420', '#7fe5f0', '#065535',
    #     '#ffd700', '#ffc0cb', '#bada55',
    # ]
    colors = [('blue', 'lightblue'), ('green', 'lightgreen'), ('red', 'pink'), ('black', 'gray'),
              ('yellow', 'purple'), ('purple', 'violet'), ('orange', 'yellow'), ('brown', 'pink'),
              ('gray', 'lightgray'), ('pink', 'lightpink'), ('violet', 'brown')]
    if type(acc_type_list) == str:
        acc_type_list = [acc_type_list]
    if len(acc_type_list) == 1:
        single_acc_type = True
    else:
        single_acc_type = False
    for index, container in enumerate(list_container):
        # mean_color, std_color = colors[index]
        if single_acc_type:
            mean_color, std_color = colors[index]
            accuracy = getattr(container, "{}_accuracy".format(acc_type_list[0]))
            covariance = getattr(container, "{}_covariance".format(acc_type_list[0]))
            num_layer = container.num_layer
            x = np.array(num_layer)
            mean = np.array(accuracy)
            std = np.sqrt(np.array(covariance))
            plt.plot(x, mean, color=mean_color, label=container.label + '_' + acc_type_list[0] + '_acc')
            plt.fill_between(x, mean - std, mean + std, color=std_color, alpha=0.3,
                             label=container.label + '_std')
        else:
            for index_acc, acc_type in enumerate(acc_type_list):
                print("index", index)
                print("index_acc", index_acc)
                print("index*2+index_acc", index * 2 + index_acc)
                mean_color, std_color = colors[index * 2 + index_acc]
                accuracy = getattr(container, "{}_accuracy".format(acc_type))
                covariance = getattr(container, "{}_covariance".format(acc_type))
                num_layer = container.num_layer
                x = np.array(num_layer)
                mean = np.array(accuracy)
                std = np.sqrt(np.array(covariance))
                plt.plot(x, mean, color=mean_color, label=container.label + '_' + acc_type + '_acc')
                plt.fill_between(x, mean - std, mean + std, color=std_color, alpha=0.3,
                                 label=container.label + '_std')
    plt.xlabel('#Layers')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='upper right',fontsize='x-small')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show(block=False)
    return plt


class DataContainer:
    def __init__(self, num_layer, accuracy, covariance, label=None):
        self.num_layer = num_layer
        self.accuracy = accuracy
        self.covariance = covariance
        self.label = label


class TrainTestDataContainer:
    def __init__(self, num_layer, train_accuracy, train_covariance, test_accuracy, test_covariance, label=None):
        self.num_layer = num_layer
        self.train_accuracy = train_accuracy
        self.train_covariance = train_covariance
        self.test_accuracy = test_accuracy
        self.test_covariance = test_covariance
        self.label = label


def plot_data_preprocess(acc_list, layer_list, label=None):
    mean_train_acc_list = []
    std_train_acc_list = []
    mean_test_acc_list = []
    std_test_acc_list = []
    for num_layer in layer_list:
        (mean_train_acc, std_train_acc, mean_test_acc, std_test_acc) = acc_list[num_layer]
        mean_train_acc_list.append(mean_train_acc)
        std_train_acc_list.append(std_train_acc)
        mean_test_acc_list.append(mean_test_acc)
        std_test_acc_list.append(std_test_acc)
    num_layer = layer_list
    # container = DataContainer(num_layer, mean_test_acc_list, std_test_acc_list, label=label)
    container = TrainTestDataContainer(num_layer, mean_train_acc_list, std_train_acc_list, mean_test_acc_list,
                                       std_test_acc_list, label=label)
    return container


if __name__ == '__main__':
    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # root = 'logs/reproduce/reproduce/cluster/hdim16'
    # num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    hdim = 64
    root = 'logs/reproduce/reproduce/cluster/hdim64'
    num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'


    # num_layers_lst = [2,3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # root = 'logs/reproduce/reproduce/cluster/fixedDepth/hdim16'
    # num_layers = '_2_3_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'
    #### hdim 16
    # GCN
    GCN_hdim16_acc_dict = torch.load(
        '{}/Model_GCN_Norm_None_Trick_None/GCN{}layers_syn2_summary.pt'.format(root,num_layers))
    GCN_hdim16_TrainTestDataContainer = plot_data_preprocess(GCN_hdim16_acc_dict, num_layers_lst, label='GCN_hdim{}'.format(hdim))
    GCN_hdim16_batch_acc_dict = torch.load(
        '{}/Model_GCN_Norm_batch_Trick_None/GCN{}layers_syn2_summary.pt'.format(root,num_layers))
    GCN_hdim16_batch_TrainTestDataContainer = plot_data_preprocess(GCN_hdim16_batch_acc_dict, num_layers_lst,
                                                                   label='GCN_hdim{}_batch'.format(hdim))
    GCN_hdim16_pair_acc_dict = torch.load(
        '{}/Model_GCN_Norm_pair_Trick_None/GCN{}layers_syn2_summary.pt'.format(root,num_layers))
    GCN_hdim16_pair_TrainTestDataContainer = plot_data_preprocess(GCN_hdim16_pair_acc_dict, num_layers_lst,
                                                                  label='GCN_hdim{}_pair'.format(hdim))
    # TODO train with 5 grounds
    # GCN_hdim16_ground_acc_dict = torch.load(
    #     '{}/Model_GCN_Norm_ground_Trick_None/GCN{}layers_syn2_summary.pt')
    # GCN_hdim16_ground_TrainTestDataContainer = plot_data_preprocess(GCN_hdim16_ground_acc_dict, num_layers_lst,
    #                                                               label='GCN_hdim16_ground')

    GCN_hdim16_residual_acc_dict = torch.load(
        '{}/Model_GCN_Norm_None_Trick_Residual/GCN{}layers_syn2_summary.pt'.format(root,num_layers))
    GCN_hdim16_residual_TrainTestDataContainer = plot_data_preprocess(GCN_hdim16_residual_acc_dict, num_layers_lst,
                                                                        label='GCN_hdim{}_residual'.format(hdim))



    # GAT
    GAT_hdim16_acc_dict = torch.load(
        '{}/Model_GAT_Norm_None_Trick_None/GAT{}layers_syn2_summary.pt'.format(root,num_layers))
    GAT_hdim16_TrainTestDataContainer = plot_data_preprocess(GAT_hdim16_acc_dict, num_layers_lst, label='GAT_hdim{}'.format(hdim))

    GAT_hdim16_batch_acc_dict = torch.load(
        '{}/Model_GAT_Norm_batch_Trick_None/GAT{}layers_syn2_summary.pt'.format(root,num_layers)  )
    GAT_hdim16_batch_TrainTestDataContainer = plot_data_preprocess(GAT_hdim16_batch_acc_dict, num_layers_lst,
                                                                     label='GAT_hdim{}_batch'.format(hdim))
    GAT_hdim16_pair_acc_dict = torch.load(
        '{}/Model_GAT_Norm_pair_Trick_None/GAT{}layers_syn2_summary.pt'.format(root,num_layers)  )
    GAT_hdim16_pair_TrainTestDataContainer = plot_data_preprocess(GAT_hdim16_pair_acc_dict, num_layers_lst,
                                                                    label='GAT_hdim{}_pair'.format(hdim))


    # SGC
    SGC_hdim16_acc_dict = torch.load(
        '{}/Model_SGC_Norm_None_Trick_None/SGC{}layers_syn2_summary.pt'.format(root,num_layers))
    SGC_hdim16_TrainTestDataContainer = plot_data_preprocess(SGC_hdim16_acc_dict, num_layers_lst, label='SGC_hdim{}'.format(hdim))

    SGC_hdim16_batch_acc_dict = torch.load(
        '{}/Model_SGC_Norm_batch_Trick_None/SGC{}layers_syn2_summary.pt'.format(root,num_layers))
    SGC_hdim16_batch_TrainTestDataContainer = plot_data_preprocess(SGC_hdim16_batch_acc_dict, num_layers_lst,
                                                                        label='SGC_hdim{}_batch'.format(hdim))
    SGC_hdim16_pair_acc_dict = torch.load(
        '{}/Model_SGC_Norm_pair_Trick_None/SGC{}layers_syn2_summary.pt'.format(root,num_layers))
    SGC_hdim16_pair_TrainTestDataContainer = plot_data_preprocess(SGC_hdim16_pair_acc_dict, num_layers_lst,
                                                                        label='SGC_hdim{}_pair'.format(hdim))

    # other methods
    APPNP_hdim16_acc_dict = torch.load(
        '{}/Model_APPNP_Norm_None_Trick_None/APPNP{}layers_syn2_summary.pt'.format(root,num_layers))
    APPNP_hdim16_TrainTestDataContainer = plot_data_preprocess(APPNP_hdim16_acc_dict, num_layers_lst,
                                                                label='APPNP_hdim{}'.format(hdim))
    DAGNN_hdim16_acc_dict = torch.load(
        '{}/Model_DAGNN_Norm_None_Trick_None/DAGNN{}layers_syn2_summary.pt'.format(root,num_layers))
    DAGNN_hdim16_TrainTestDataContainer = plot_data_preprocess(DAGNN_hdim16_acc_dict, num_layers_lst,
                                                                label='DAGNN_hdim{}'.format(hdim))
    GCNII_hdim16_acc_dict = torch.load(
        '{}/Model_GCNII_Norm_None_Trick_None/GCNII{}layers_syn2_summary.pt'.format(root,num_layers))
    GCNII_hdim16_TrainTestDataContainer = plot_data_preprocess(GCNII_hdim16_acc_dict, num_layers_lst,
                                                                label='GCNII_hdim{}'.format(hdim))
    GPRGNN_hdim16_acc_dict = torch.load(
        '{}/Model_GPRGNN_Norm_None_Trick_None/GPRGNN{}layers_syn2_summary.pt'.format(root,num_layers))
    GPRGNN_hdim16_TrainTestDataContainer = plot_data_preprocess(GPRGNN_hdim16_acc_dict, num_layers_lst,
                                                                label='GPRGNN_hdim{}'.format(hdim))
    JKNet_hdim16_acc_dict = torch.load(
        '{}/Model_JKNet_Norm_None_Trick_None/JKNet{}layers_syn2_summary.pt'.format(root,num_layers))
    JKNet_hdim16_TrainTestDataContainer = plot_data_preprocess(JKNet_hdim16_acc_dict, num_layers_lst,
                                                                label='JKNet_hdim{}'.format(hdim))

    GPRGNN_float64_hdim16_acc_dict = torch.load(
        'logs/precision/hdim16/Model_GPRGNN_Norm_None_Trick_None/GPRGNN_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32layers_syn2_summary.pt'.format(root,num_layers))
    GPRGNN_float64_hdim16_TrainTestDataContainer = plot_data_preprocess(GPRGNN_float64_hdim16_acc_dict, num_layers_lst,
                                                                label='GPRGNN_float64_hdim{}'.format(hdim))


    list_container = [GCN_hdim16_TrainTestDataContainer, GAT_hdim16_TrainTestDataContainer,
                      SGC_hdim16_TrainTestDataContainer]
    # list_container = [GCN_hdim16_TrainTestDataContainer, GCN_hdim16_batch_TrainTestDataContainer,GCN_hdim16_pair_TrainTestDataContainer,GCN_hdim16_residual_TrainTestDataContainer]
    # list_container = [GAT_hdim16_TrainTestDataContainer, GAT_hdim16_batch_TrainTestDataContainer,GAT_hdim16_pair_TrainTestDataContainer]
    # list_container = [SGC_hdim16_TrainTestDataContainer, SGC_hdim16_batch_TrainTestDataContainer,SGC_hdim16_pair_TrainTestDataContainer]
    # list_container = [APPNP_hdim16_TrainTestDataContainer, DAGNN_hdim16_TrainTestDataContainer,GCNII_hdim16_TrainTestDataContainer,GPRGNN_hdim16_TrainTestDataContainer,JKNet_hdim16_TrainTestDataContainer]
    # list_container = [APPNP_hdim16_TrainTestDataContainer, DAGNN_hdim16_TrainTestDataContainer,GCNII_hdim16_TrainTestDataContainer,GPRGNN_hdim16_TrainTestDataContainer,JKNet_hdim16_TrainTestDataContainer,GCN_hdim16_TrainTestDataContainer, GAT_hdim16_TrainTestDataContainer,
    #                   SGC_hdim16_TrainTestDataContainer]
    # list_container = [GPRGNN_hdim16_TrainTestDataContainer]
    # list_container = [JKNet_hdim16_TrainTestDataContainer]
    list_container = [GPRGNN_float64_hdim16_acc_dict, GPRGNN_hdim16_TrainTestDataContainer]
    # acc_type_list = ["train", "test"]
    acc_type_list = ['train']
    # acc_type_list = ['test']
    title = "{} accuracy curve with 64 dim 1000 pairs".format(acc_type_list)
    plot_train_test_accuracy_curve_with_covariance(list_container, save_path="acc_curve1.png",
                                                   acc_type_list=acc_type_list, title=title)
