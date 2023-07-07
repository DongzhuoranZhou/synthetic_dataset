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
            plt.plot(x, mean, color=mean_color, label=container.label + '_acc')
            plt.fill_between(x, mean - std, mean + std, color=std_color, alpha=0.3,
                             label=container.label + '_acc' + '_std')
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
                plt.plot(x, mean, color=mean_color, label=container.label + '_acc')
                plt.fill_between(x, mean - std, mean + std, color=std_color, alpha=0.3,
                                 label=container.label + '_acc' + '_std')
    plt.xlabel('#Layers')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='lower right', fontsize='x-small')
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
        (mean_train_acc, std_train_acc, mean_test_acc, std_test_acc) = acc_list[num_layer-3]
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
    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/reproduce/reproduce/cluster/hdim16'
    num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'

    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # hdim = 64
    # root = 'logs/reproduce/reproduce/cluster/hdim64'
    # num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'
    #
    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    hdim = 16
    # root = 'logs/reproduce/reproduce/cluster/10000pairs/hdim16'
    # num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'
    #
    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # hdim = 16
    # root = 'logs/reproduce/reproduce/cluster/WithACM/hdim16'
    # num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'
    #
    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    hdim = 16
    root = 'logs/reproduce/reproduce/cluster/undirected/hdim16'
    num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'
    #
    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # hdim = 16
    # root = 'logs/reproduce/reproduce/cluster/precision/float64/hdim16'
    # num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'

    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # hdim = 16
    # root = 'logs/reproduce/reproduce/cluster/precision/float16/hdim16'
    # num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'

    # num_layers_lst = [2,3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # root = 'logs/reproduce/reproduce/cluster/fixedDepth/hdim16'
    # num_layers = '_2_3_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/reproduce/reproduce/cluster/fixedDepth/Depth20/hdim16'
    num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'

    # num_layers_lst = list(range(20,66,2))
    # root = 'logs/fixedDepthNew/Depth32/hdim16'
    # num_layers = '_'+'_'.join([str(i) for i in num_layers_lst])
    #
    # num_layers_lst = list(range(32,128,2))
    # root = 'logs/fixedDepthNew/Depth64/hdim16'
    # num_layers = '_32_126'
    #
    # num_layers_lst = list(range(100,180,2))
    # root = 'logs/fixedDepthNew/Depth128'
    # num_layers = '_100_178'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/MiddleHiddenLayer64/MiddleHiddenLayer64'
    num_layers = '_' + '_'.join([str(i) for i in num_layers_lst])

    num_layers_lst = list(range(20, 66, 2))
    root = 'logs/reproduce/reproduce/cluster/fixedDepth/Depth32/hdim16'
    num_layers = '_20_64'

    num_layers_lst = list(range(32, 128, 2))
    root = 'logs/reproduce/reproduce/cluster/fixedDepth/Depth64/dataset_syn2/width_1/hdim16'
    num_layers = '_32_126'

    num_layers_lst = list(range(100, 178, 2))
    root = 'logs/reproduce/reproduce/cluster/fixedDepth/Depth128/dataset_syn2/width_1/hdim16'
    num_layers = '_100_178'


    num_layers_lst = [3,4,6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/reproduce/reproduce/cluster/Changinglayers/dataset_syn4/width_1/hdim16'
    num_layers = '_3_32'
    dataset = 'syn4'

    num_layers_lst = [3,4,6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22]
    root = 'logs/reproduce/reproduce/cluster/Changinglayers/numPairs10/dataset_syn4/width_2/hdim16'
    num_layers = '_3_22'
    dataset = 'syn4'



    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/changingDepth/dataset_syn2/width_2/hdim16'
    num_layers = '_4_32'
    dataset = 'syn2'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/changingDepth/dataset_syn4/width_1/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/WONorm/fixedDepth/Depth16/dataset_syn4/width_1/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/changingDepth/dataset_syn2/width_2/hdim16'
    num_layers = '_4_32'
    dataset = 'syn2'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/supplementary/supplementary/fixedDepth/Depth8/dataset_syn2/width_2/hdim16'
    num_layers = '_4_32'
    dataset = 'syn2'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/supplementary/supplementary/fixedDepth/Depth10/dataset_syn4/width_2/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'


    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/reproduce/reproduce/cluster/hdim16'
    num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'
    dataset = 'syn2'

    num_layers_lst = [2,3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/reproduce/reproduce/cluster/fixedDepth/Depth5/hdim16'
    num_layers = '_2_3_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'
    dataset = 'syn2'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/reproduce/reproduce/cluster/fixedDepth/Depth16/hdim16'
    num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'
    dataset = 'syn2'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/changingDepth/dataset_syn2/width_2/hdim16'
    num_layers = '_4_32'
    dataset = 'syn2'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/fixedDepth/Depth8/dataset_syn2/width_2/hdim16'
    num_layers = '_4_32'
    dataset = 'syn2'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/fixedDepth/Depth16/dataset_syn2/width_2/hdim16'
    num_layers = '_4_32'
    dataset = 'syn2'

    num_layers_lst = list(range(20, 64))
    root = 'logs/light_dataset/light_dataset/fixedDepth/Depth32/dataset_syn2/width_2/hdim16'
    num_layers = '_20_63'
    dataset = 'syn2'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/changingDepth/dataset_syn4/width_1/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/fixedDepth/Depth4/dataset_syn4/width_2/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/fixedDepth/Depth12/dataset_syn4/width_2/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'


    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/WONorm/changingDepth/dataset_syn4/width_1/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/WONorm/changingDepth/dataset_syn4/width_1/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/WONorm/fixedDepth/Depth4/dataset_syn4/width_1/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/WONorm/fixedDepth/Depth16/dataset_syn4/width_1/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/WONorm/changingDepth/dataset_syn4/width_2/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'

    num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    root = 'logs/light_dataset/light_dataset/WONorm/fixedDepth/Depth4/dataset_syn4/width_2/hdim16'
    num_layers = '_4_32'
    dataset = 'syn4'

    num_layers_lst = [6,7, 9, 11, 13, 14, 15, 16, 18, 19, 21, 23, 25, 27, 29, 31, 33,35]
    root = 'logs/deeper/deeper'
    num_layers = '_3_32'
    dataset = 'syn2'

    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # root = 'logs/reproduce/reproduce/cluster/fixedDepth/Depth20/hdim16'
    # num_layers = '_4_6_8_10_11_12_13_14_15_16_18_20_22_24_26_28_30_32'
    # dataset = 'syn2'

    # num_layers_lst = list(range(20, 66, 2))
    # root = 'logs/reproduce/reproduce/cluster/fixedDepth/Depth32/hdim16'
    # num_layers = '_20_64'
    # dataset = 'syn2'
    #
    # num_layers_lst = list(range(32, 128, 2))
    # root = 'logs/reproduce/reproduce/cluster/fixedDepth/Depth64/dataset_syn2/width_1/hdim16'
    # num_layers = '_32_126'
    # dataset = 'syn2'

    # num_layers_lst = list(range(100, 177, 2))
    # root = 'logs/reproduce/reproduce/cluster/fixedDepth/Depth128/dataset_syn2/width_1/hdim16'
    # num_layers = '_100_178'
    # dataset = 'syn2'

    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # root = 'logs/light_dataset/light_dataset/fixedDepth/Depth16/dataset_syn2/width_2/hdim16'
    # num_layers = '_4_32'
    # dataset = 'syn2'
    #
    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # root = 'logs/light_dataset/light_dataset/WONorm/changingDepth/dataset_syn4/width_1/hdim16'
    # num_layers = '_4_32'
    # dataset = 'syn4'
    #
    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # root = 'logs/light_dataset/light_dataset/WONorm/changingDepth/dataset_syn4/width_2/hdim16'
    # num_layers = '_4_32'
    # dataset = 'syn4'
    #
    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # root = 'logs/light_dataset/light_dataset/WONorm/fixedDepth/Depth4/dataset_syn4/width_2/hdim16'
    # num_layers = '_4_32'
    # dataset = 'syn4'
    #
    # num_layers_lst = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # root = 'logs/light_dataset/light_dataset/WONorm/fixedDepth/Depth16/dataset_syn4/width_1/hdim16'
    # num_layers = '_4_32'
    # dataset = 'syn4'

    #### hdim 16
    # GCN
    # GCN_hdim16_acc_dict = torch.load(
    #     '{}/Model_GCN_Norm_None_Trick_None/GCN{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # GCN_hdim16_TrainTestDataContainer = plot_data_preprocess(GCN_hdim16_acc_dict, num_layers_lst,
    #                                                          label='GCN_hdim{}'.format(hdim))
    # GCN_hdim16_batch_acc_dict = torch.load(
    #     '{}/Model_GCN_Norm_batch_Trick_None/GCN{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # GCN_hdim16_batch_TrainTestDataContainer = plot_data_preprocess(GCN_hdim16_batch_acc_dict, num_layers_lst,
    #                                                                label='GCN_hdim{}_batch'.format(hdim))
    # GCN_hdim16_pair_acc_dict = torch.load(
    #     '{}/Model_GCN_Norm_pair_Trick_None/GCN{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # GCN_hdim16_pair_TrainTestDataContainer = plot_data_preprocess(GCN_hdim16_pair_acc_dict, num_layers_lst,
    #                                                               label='GCN_hdim{}_pair'.format(hdim))
    # # # TODO train with 5 grounds
    # GCN_hdim16_group_acc_dict = torch.load(
    #     '{}/Model_GCN_Norm_group_Trick_None/GCN{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # GCN_hdim16_group_TrainTestDataContainer = plot_data_preprocess(GCN_hdim16_group_acc_dict, num_layers_lst,
    #                                                               label='GCN_hdim{}_ground'.format(hdim))

    # GCN_hdim16_residual_acc_dict = torch.load(
    #     '{}/Model_GCN_Norm_None_Trick_Residual/GCN{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # GCN_hdim16_residual_TrainTestDataContainer = plot_data_preprocess(GCN_hdim16_residual_acc_dict, num_layers_lst,
    #                                                                   label='GCN_hdim{}_residual'.format(hdim))

    # GAT
    GAT_hdim16_acc_dict = torch.load(
        '{}/Model_GAT_Norm_None_Trick_None/GAT{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    GAT_hdim16_TrainTestDataContainer = plot_data_preprocess(GAT_hdim16_acc_dict, num_layers_lst,
                                                             label='GAT_hdim{}'.format(hdim))

    GAT_hdim16_batch_acc_dict = torch.load(
        '{}/Model_GAT_Norm_batch_Trick_None/GAT{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    GAT_hdim16_batch_TrainTestDataContainer = plot_data_preprocess(GAT_hdim16_batch_acc_dict, num_layers_lst,
                                                                   label='GAT_hdim{}_batch'.format(hdim))
    GAT_hdim16_pair_acc_dict = torch.load(
        '{}/Model_GAT_Norm_pair_Trick_None/GAT{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    GAT_hdim16_pair_TrainTestDataContainer = plot_data_preprocess(GAT_hdim16_pair_acc_dict, num_layers_lst,
                                                                  label='GAT_hdim{}_pair'.format(hdim))
    # GAT_hdim16_group_acc_dict = torch.load(
    #     '{}/Model_GAT_Norm_group_Trick_None/GAT{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # GAT_hdim16_group_TrainTestDataContainer = plot_data_preprocess(GAT_hdim16_pair_acc_dict, num_layers_lst,
    #                                                               label='GAT_hdim{}_group'.format(hdim))

    # simpleGCN
    # simpleGCN_hdim16_acc_dict = torch.load(
    #     '{}/Model_simpleGCN_Norm_None_Trick_None/simpleGCN{}layers_{}_summary.pt'.format(root,num_layers, dataset))
    # simpleGCN_hdim16_TrainTestDataContainer = plot_data_preprocess(simpleGCN_hdim16_acc_dict, num_layers_lst, label='simpleGCN_hdim{}'.format(hdim))
    # simpleGCN_hdim16_batch_acc_dict = torch.load(
    #     '{}/Model_simpleGCN_Norm_batch_Trick_None/simpleGCN{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # simpleGCN_hdim16_batch_TrainTestDataContainer = plot_data_preprocess(simpleGCN_hdim16_batch_acc_dict,
    #                                                                      num_layers_lst,
    #                                                                      label='simpleGCN_hdim{}_batch'.format(hdim))
    # simpleGCN_hdim16_pair_acc_dict = torch.load(
    #     '{}/Model_simpleGCN_Norm_pair_Trick_None/simpleGCN{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # simpleGCN_hdim16_pair_TrainTestDataContainer = plot_data_preprocess(simpleGCN_hdim16_pair_acc_dict, num_layers_lst,
    #                                                                     label='simpleGCN_hdim{}_pair'.format(hdim))
    # simpleGCN_hdim16_group_acc_dict = torch.load(
    #     '{}/Model_simpleGCN_Norm_group_Trick_None/simpleGCN{}layers_{}_summary.pt'.format(root,num_layers, dataset))
    # simpleGCN_hdim16_group_TrainTestDataContainer = plot_data_preprocess(simpleGCN_hdim16_group_acc_dict, num_layers_lst,
    #                                                                 label='simpleGCN_hdim{}_ground'.format(hdim))

    # SGC
    # SGC_hdim16_acc_dict = torch.load(
    #     '{}/Model_SGC_Norm_None_Trick_None/SGC{}layers_{}_summary.pt'.format(root,num_layers, dataset))
    # SGC_hdim16_TrainTestDataContainer = plot_data_preprocess(SGC_hdim16_acc_dict, num_layers_lst, label='SGC_hdim{}'.format(hdim))
    #
    # SGC_hdim16_batch_acc_dict = torch.load(
    #     '{}/Model_SGC_Norm_batch_Trick_None/SGC{}layers_{}_summary.pt'.format(root,num_layers, dataset))
    # SGC_hdim16_batch_TrainTestDataContainer = plot_data_preprocess(SGC_hdim16_batch_acc_dict, num_layers_lst,
    #                                                                     label='SGC_hdim{}_batch'.format(hdim))
    # SGC_hdim16_pair_acc_dict = torch.load(
    #     '{}/Model_SGC_Norm_pair_Trick_None/SGC{}layers_{}_summary.pt'.format(root,num_layers, dataset))
    # SGC_hdim16_pair_TrainTestDataContainer = plot_data_preprocess(SGC_hdim16_pair_acc_dict, num_layers_lst,
    #                                                                     label='SGC_hdim{}_pair'.format(hdim))
    # SGC_hdim16_group_acc_dict = torch.load(
    #     '{}/Model_SGC_Norm_group_Trick_None/SGC{}layers_{}_summary.pt'.format(root,num_layers, dataset))
    # SGC_hdim16_group_TrainTestDataContainer = plot_data_preprocess(SGC_hdim16_pair_acc_dict, num_layers_lst,
    #                                                                     label='SGC_hdim{}_pair'.format(hdim))

    # other methods
    # APPNP_hdim16_acc_dict = torch.load(
    #     '{}/Model_APPNP_Norm_None_Trick_None/APPNP{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # APPNP_hdim16_TrainTestDataContainer = plot_data_preprocess(APPNP_hdim16_acc_dict, num_layers_lst,
    #                                                            label='APPNP_hdim{}'.format(hdim))
    # DAGNN_hdim16_acc_dict = torch.load(
    #     '{}/Model_DAGNN_Norm_None_Trick_None/DAGNN{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # DAGNN_hdim16_TrainTestDataContainer = plot_data_preprocess(DAGNN_hdim16_acc_dict, num_layers_lst,
    #                                                            label='DAGNN_hdim{}'.format(hdim))
    # GCNII_hdim16_acc_dict = torch.load(
    #     '{}/Model_GCNII_Norm_None_Trick_None/GCNII{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # GCNII_hdim16_TrainTestDataContainer = plot_data_preprocess(GCNII_hdim16_acc_dict, num_layers_lst,
    #                                                            label='GCNII_hdim{}'.format(hdim))
    # GPRGNN_hdim16_acc_dict = torch.load(
    #     '{}/Model_GPRGNN_Norm_None_Trick_None/GPRGNN{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # GPRGNN_hdim16_TrainTestDataContainer = plot_data_preprocess(GPRGNN_hdim16_acc_dict, num_layers_lst,
    #                                                             label='GPRGNN_hdim{}'.format(hdim))
    # JKNet_hdim16_acc_dict = torch.load(
    #     '{}/Model_JKNet_Norm_None_Trick_None/JKNet{}layers_{}_summary.pt'.format(root, num_layers, dataset))
    # JKNet_hdim16_TrainTestDataContainer = plot_data_preprocess(JKNet_hdim16_acc_dict, num_layers_lst,
    #                                                            label='JKNet_hdim{}'.format(hdim))

    # list_container = [GCN_hdim16_TrainTestDataContainer, GAT_hdim16_TrainTestDataContainer,simpleGCN_hdim16_batch_TrainTestDataContainer,simpleGCN_hdim16_pair_TrainTestDataContainer,SGC_hdim16_TrainTestDataContainer,SGC_hdim16_batch_TrainTestDataContainer,SGC_hdim16_pair_TrainTestDataContainer]
    # list_container = [GCN_hdim16_TrainTestDataContainer, GAT_hdim16_TrainTestDataContainer,
    #                   SGC_hdim16_TrainTestDataContainer]
    # list_container = [GCN_hdim16_TrainTestDataContainer, GAT_hdim16_TrainTestDataContainer,
    #                   simpleGCN_hdim16_TrainTestDataContainer]
    # list_container = [GCN_hdim16_TrainTestDataContainer, GAT_hdim16_TrainTestDataContainer, ]
    # list_container = [SGC_hdim16_batch_TrainTestDataContainer,SGC_hdim16_pair_TrainTestDataContainer]
    # list_container = [ GCN_hdim16_batch_TrainTestDataContainer,GCN_hdim16_pair_TrainTestDataContainer]
    # list_container = [GCN_hdim16_TrainTestDataContainer,GCN_hdim16_batch_TrainTestDataContainer,GCN_hdim16_pair_TrainTestDataContainer,GCN_hdim16_group_TrainTestDataContainer]
    # list_container = [GCN_hdim16_TrainTestDataContainer,GCN_hdim16_batch_TrainTestDataContainer,GCN_hdim16_pair_TrainTestDataContainer]
    list_container = [GAT_hdim16_TrainTestDataContainer, GAT_hdim16_batch_TrainTestDataContainer,GAT_hdim16_pair_TrainTestDataContainer]
    # list_container = [SGC_hdim16_TrainTestDataContainer, SGC_hdim16_batch_TrainTestDataContainer,SGC_hdim16_pair_TrainTestDataContainer]
    # list_container = [DAGNN_hdim16_TrainTestDataContainer,GPRGNN_hdim16_TrainTestDataContainer,GCNII_hdim16_TrainTestDataContainer,APPNP_hdim16_TrainTestDataContainer,JKNet_hdim16_TrainTestDataContainer]
    # list_container = [DAGNN_hdim16_TrainTestDataContainer,GCNII_hdim16_TrainTestDataContainer,GPRGNN_hdim16_TrainTestDataContainer,JKNet_hdim16_TrainTestDataContainer,APPNP_hdim16_TrainTestDataContainer]
    #
    # list_container = [APPNP_hdim16_TrainTestDataContainer, DAGNN_hdim16_TrainTestDataContainer,GCNII_hdim16_TrainTestDataContainer,JKNet_hdim16_TrainTestDataContainer]
    # list_container = [GPRGNN_hdim16_TrainTestDataContainer]
    # list_container = [JKNet_hdim16_TrainTestDataContainer]
    # list_container = [GPRGNN_float64_hdim16_acc_dict, GPRGNN_hdim16_TrainTestDataContainer]
    # list_container = [DAGNN_hdim16_TrainTestDataContainer, GCNII_hdim16_TrainTestDataContainer, GPRGNN_hdim16_TrainTestDataContainer]
    # list_container = [DAGNN_hdim16_TrainTestDataContainer,
    #                   GPRGNN_hdim16_TrainTestDataContainer]
    # list_container = [GCN_hdim16_TrainTestDataContainer, GAT_hdim16_TrainTestDataContainer,simpleGCN_hdim16_batch_TrainTestDataContainer,simpleGCN_hdim16_pair_TrainTestDataContainer,SGC_hdim16_TrainTestDataContainer]

    # list_container = [
    #                   GPRGNN_hdim16_TrainTestDataContainer]
    # acc_type_list = ["train", "test"]
    acc_type_list = ['train']
    # acc_type_list = ['test']
    title = "{} accuracy, fixed depth=4, width=2, sum,  topology dataset".format(' '.join(acc_type_list))
    plot_train_test_accuracy_curve_with_covariance(list_container, save_path="acc_curve1.png",
                                                   acc_type_list=acc_type_list, title=title)
