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


def plot_accuracy_curve_with_covariance(list_container, save_path=None,acc_type='test'):
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

def plot_train_test_accuracy_curve_with_covariance(list_container, save_path=None,acc_type=None):
    # colors = [
    #     '#008080', '#420420', '#7fe5f0', '#065535',
    #     '#ffd700', '#ffc0cb', '#bada55',
    # ]
    colors = [('blue', 'lightblue'), ('green', 'lightgreen'), ('red', 'pink'), ('black', 'gray'),
              ('yellow', 'lightyellow'), ('purple', 'violet'), ('orange', 'yellow'), ('brown', 'pink'),
              ('gray', 'lightgray'), ('pink', 'lightpink'), ('violet', 'brown')]
    for index, container in enumerate(list_container):
        mean_color, std_color = colors[index]
        if type(acc_type) == str:
            acc_type = [acc_type]
        for index_acc, acc_type in enumerate(acc_type):
            mean_color, std_color = colors[index+index_acc]
            accuracy = getattr(container, "{}_accuracy".format(acc_type))
            covariance = getattr(container, "{}_covariance".format(acc_type))
            num_layer = container.num_layer
            x = np.array(num_layer)
            mean = np.array(accuracy)
            std = np.sqrt(np.array(covariance))
            plt.plot(x, mean, color=mean_color, label=container.label + '_' + acc_type + '_Accuracy')
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
    container = TrainTestDataContainer(num_layer, mean_train_acc_list, std_train_acc_list,mean_test_acc_list,std_test_acc_list, label=label)
    return container

if __name__ == '__main__':

    test_SGC_acc_list = {4: (0.6031666666666666, 0.4860453796829171, 0.6025, 0.48685213360937424), 6: (0.607, 0.48139859899163717, 0.6045, 0.4844388506302937), 8: (0.6645, 0.40982740269532975, 0.657, 0.4172601107223168), 10: (0.6753333333333332, 0.30084769125478317, 0.6740000000000002, 0.2786736442507616), 11: (0.6491666666666667, 0.20039336316355388, 0.6289999999999999, 0.2243746866293076), 12: (0.591, 0.12864831475339616, 0.5995, 0.1447618734335806), 13: (0.5579999999999999, 0.08957089060874877, 0.56, 0.09352807065261208), 14: (0.5369999999999999, 0.04725991959366836, 0.5315, 0.04468221122549777), 15: (0.531, 0.03036994127971493, 0.529, 0.0371012129181783), 16: (0.5096666666666667, 0.018978057505094313, 0.514, 0.026296387584609424), 32: (0.5003333333333334, 0.0045521667612492284, 0.5049999999999999, 0.01565247584249852)}
    test_SGC_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    test_SGC_TrainTestDataContainer = plot_data_preprocess(test_SGC_acc_list, test_SGC_num_layer, label='test_SGC')

    # single split GCN
    GCN_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 16, 32]
    GCN_accuracy = [1, 1, 1, 1, 0.924000000, 0.799, 0.599, 0.5, 0.5, 0.5]
    GCN_covariance = [0, 0, 0, 0, 0.152000000, 0.246178797, 0.200521819, 0, 0, 0]
    GCN_container = DataContainer(GCN_num_layer, GCN_accuracy, GCN_covariance, label="GCN")

    # single split GCN hdim 16
    GCN_acc_dict = {4: (1.0, 0.0, 1.0, 0.0), 6: (1.0, 0.0, 1.0, 0.0), 8: (1.0, 0.0, 1.0, 0.0), 10: (0.9970000000000001, 0.0012472191289246374, 0.9925, 0.0031622776601683646), 11: (0.9668333333333333, 0.0032231799343022975, 0.969, 0.004062019202317962), 12: (0.8880000000000001, 0.005691123692987942, 0.8870000000000001, 0.02575849374478251), 13: (0.756, 0.0035512126254437573, 0.75, 0.012449899597988716), 14: (0.6546666666666667, 0.010200762498732906, 0.651, 0.027820855486487092), 15: (0.5970000000000001, 0.004428443418528813, 0.5780000000000001, 0.012288205727444499), 16: (0.552, 0.011506037062728802, 0.5315, 0.01178982612255161), 32: (0.5003333333333334, 0.0045521667612492284, 0.5049999999999999, 0.01565247584249852)}
    GCN_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    GCN_TrainTestDataContainer = plot_data_preprocess(GCN_acc_dict, GCN_num_layer, label='GCN')

    # single split GCN hdim 16
    GAT_acc_dict = {4: (1.0, 0.0, 1.0, 0.0), 6: (0.9913333333333334, 0.017333333333333336, 0.9894999999999999, 0.020999999999999998), 8: (0.9918333333333333, 0.004484541349024567, 0.9834999999999999, 0.010198039027185565), 10: (0.8971666666666668, 0.19859828241396696, 0.8915, 0.19585453785909585), 11: (0.9966666666666667, 0.003836954811073786, 0.9905000000000002, 0.0029154759474226757), 12: (0.852, 0.1745475102467329, 0.8505, 0.16716309401300278), 13: (0.8946666666666667, 0.19701283094142763, 0.8925000000000001, 0.19648791311426766), 14: (0.797, 0.24250864818485224, 0.795, 0.24092011124022006), 15: (0.6980000000000001, 0.24354933335523174, 0.7064999999999999, 0.2357413837237747), 16: (0.5993333333333333, 0.19950661364698885, 0.5980000000000001, 0.1972523764115404), 32: (0.5015000000000001, 0.003887301263230194, 0.5005, 0.0009999999999999788)}
    GAT_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    GAT_TrainTestDataContainer = plot_data_preprocess(GAT_acc_dict, GAT_num_layer, label='GAT')


    list_container = [GCN_TrainTestDataContainer,GAT_TrainTestDataContainer]

    plot_train_test_accuracy_curve_with_covariance(list_container, save_path="acc_curve.png",acc_type=["train",'test'])

    # single split GCN hdim 64
    GCN_hdim64_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    GCN_hdim64_accuracy = [1, 1, 1, 1, 0.702500000, 0.600000000, 0.506500000, 0.504500000, 0.500500000, 0.5, 0.5]
    GCN_hdim64_covariance = [0, 0, 0, 0, 0.242950612, 0.200000000, 0.013000000, 0.009000000, 0.001000000, 0, 0]
    GCN_hdim64_container = DataContainer(GCN_hdim64_num_layer, GCN_hdim64_accuracy, GCN_hdim64_covariance,
                                         label="GCN_hdim64")

    # single split GCN hdim 128


    # single split PairnormGCN
    PairnormGCN_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    PairnormGCN_accuracy = [1, 1, 1, 0.993250000, 0.972000000, 0.777000000, 0.802500000, 0.839250000, 0.685500000,
                            0.495500000, 0.505000000]
    PairnormGCN_covariance = [0, 0, 0, 0.020250000, 0.061783898, 0.181874957, 0.216685256, 0.201286643, 0.158251382,
                              0.006000000, 0.007500000]
    PairnormGCN_container = DataContainer(PairnormGCN_num_layer, PairnormGCN_accuracy, PairnormGCN_covariance,
                                          label="PairnormGCN")

    # single split GAT
    GAT_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    GAT_accuracy = [1.000000000, 1.00000000, 0.983500000, 0.891500000, 0.990500000, 0.850500000, 0.892500000,
                    0.795000000, 0.706500000, 0.598000000, 0.500500000]
    GAT_covariance = [0.000000000, 0.000000000, 0.010198039, 0.195854538, 0.002915476, 0.167163094, 0.196487913,
                      0.240920111, 0.235741384, 0.197252376, 0.001000000]
    GAT_container = DataContainer(GAT_num_layer, GAT_accuracy, GAT_covariance, label="GAT")

    # single split GAT hdim 64
    GAT_hdim64_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    GAT_hdim64_accuracy = [1.000000000, 1, 0.751000000, 0.727500000, 0.661000000, 0.793000000, 0.664000000, 0.582500000,
                           0.574000000, 0.501500000, 0.504500000]
    GAT_hdim64_covariance = [0.000000000, 0.000000000, 0.211101871, 0.185889752, 0.197816834, 0.793000000, 0.181704430,
                             0.165000000, 0.148000000, 0.007348469, 0.010295630]
    GAT_hdim64_container = DataContainer(GAT_hdim64_num_layer, GAT_hdim64_accuracy, GAT_hdim64_covariance,
                                         label="GAT_hdim64")
    # single split PairnormGAT
    # PairnormGAT_num_layer = [4,6,8,10,11,12,13,14,15,16,32]
    # PairnormGAT_accuracy = [1,1,1,1,1]
    # PairnormGAT_covariance = [0,0,0,0,0]
    # PairnormGAT_container = DataContainer(PairnormGAT_num_layer, PairnormGAT_accuracy, PairnormGAT_covariance, label="PairnormGAT")

    # single split SGC
    SGC_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    SGC_accuracy = [1, 1, 1, 0.992500000, 0.969000000, 0.891000000, 0.752000000, 0.650000000, 0.579000000, 0.531000000,
                    0.486000000, 0.493000000, 0.497500000, 0.500000000, 0.496000000, 0.500500000, 0.500000000,
                    0.508500000]
    SGC_covariance = [0, 0, 0, 0.003162278, 0.004062019, 0.027685736, 0.013820275, 0.027838822, 0.015049917,
                      0.012708265, 0.028398944, 0.018934096, 0.024949950, 0.009354143, 0.022781571, 0.019196354,
                      0.021965883, 0.010440307]
    SGC_container = DataContainer(SGC_num_layer, SGC_accuracy, SGC_covariance, label="SGC")

    # single split SGC hdim 64
    SGC_hdim64_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    SGC_hdim64_accuracy = [1, 1, 1, 1, 1, 0.985500000, 0.893500000, 0.769000000, 0.645000000, 0.482000000, 0.472000000,
                           0.516000000, 0.511500000, 0.506500000, 0.494500000, 0.503500000, 0.496500000, 0.486000000]
    SGC_hdim64_covariance = [0, 0, 0, 0, 0, 0.003316625, 0.012103718, 0.019209373, 0.014747881, 0.024969982,
                             0.016537835, 0.023958297, 0.023484037, 0.019532025, 0.005338539, 0.027685736, 0.019786359,
                             0.011357817]
    SGC_hdim64_container = DataContainer(SGC_hdim64_num_layer, SGC_hdim64_accuracy, SGC_hdim64_covariance,
                                         label="SGC_hdim64")

    # single split SGC original
    SGC_original_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    SGC_original_accuracy = [1, 1, 1, 0.993000000,0.967500000,0.889000000,0.746500000,0.645500000,0.579000000,0.533000000,0.507500000]
    SGC_original_covariance = [0, 0, 0, 0.002915476,0.005477226,0.021011901,0.011247222,0.030306765,0.011022704,0.013546217,0.009617692]
    SGC_original_container = DataContainer(SGC_original_num_layer, SGC_original_accuracy, SGC_original_covariance,
                                            label="SGC_original")

    # single split batchnormSGC
    batchnormSGC_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    batchnormSGC_accuracy = [1, 1, 1, 0.992500000, 0.965500000, 0.894500000, 0.753500000, 0.651000000, 0.580000000,
                             0.531500000, 0.492500000]
    batchnormSGC_covariance = [0, 0, 0, 0.003162278, 0.003316625, 0.025268558, 0.016170962, 0.032503846, 0.010000000,
                               0.019912308, 0.015411035]
    batchnormSGC_container = DataContainer(batchnormSGC_num_layer, batchnormSGC_accuracy, batchnormSGC_covariance,
                                           label="batchnormSGC")

    # # single split PairnormSGC # TODO: check the code
    PairnormSGC_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    PairnormSGC_accuracy = [1, 1, 1, 0.992500000, 0.969000000, 0.887000000, 0.750000000, 0.651000000, 0.578000000,
                            0.531500000, 0.505000000]
    PairnormSGC_covariance = [0, 0, 0, 0.003162278, 0.004062019, 0.025758494, 0.012449900, 0.027820855, 0.012288206,
                              0.019912308, 0.015652476]
    PairnormSGC_container = DataContainer(PairnormSGC_num_layer, PairnormSGC_accuracy, PairnormSGC_covariance,
                                          label="PairnormSGC")

    # single split JKNet GCN # TODO: check the code
    JKNet_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    JKNet_accuracy = [1, 1, 1, 0.887500000, 0.493500000, 0.488000000, 0.490500000, 0.501500000, 0.513500000,
                      0.488000000, 0.497500000]
    JKNet_covariance = [0, 0, 0, 0.225000000, 0.013472194, 0.010885771, 0.012786712, 0.040236799, 0.016477257,
                        0.016309506, 0.005700877]
    JKNet_container = DataContainer(JKNet_num_layer, JKNet_accuracy, JKNet_covariance, label="JKNetGCN")

    JKNet_train_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    JKNet_train_accuracy = [1, 1, 1, 0.902166667, 0.503666667, 0.506166667, 0.503833333, 0.499833333, 0.500166667,
                            0.511166667, 0.503666667]
    JKNet_train_covariance = [0, 0, 0, 0.195666667, 0.009480975, 0.005126185, 0.007023769, 0.018313626, 0.005878397,
                              0.004169999, 0.003965126]
    JKNet_train_container = DataContainer(JKNet_num_layer, JKNet_accuracy, JKNet_covariance, label="JKNetGCN_Train")

    # single split EdgeDrop GCN

    # single split APPNP
    APPNP_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    APPNP_accuracy = [0.995000000, 0.767000000, 0.620000000, 0.548500000, 0.564500000, 0.505500000, 0.505500000,
                      0.494500000, 0.496500000, 0.486500000, 0.500500000]
    APPNP_covariance = [0.010000000, 0.180460522, 0.132693255, 0.036694686, 0.050950957, 0.012884099, 0.007810250,
                        0.005787918, 0.004898979, 0.014628739, 0.001000000]
    APPNP_container = DataContainer(APPNP_num_layer, APPNP_accuracy, APPNP_covariance, label="APPNP")

    # single split DAGNN
    DAGNN_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    DAGNN_accuracy = [1, 1, 1, 1, 1, 1, 1, 0.957000000, 0.689000000, 0.460000000, 0.488500000]
    DAGNN_covariance = [0, 0, 0, 0, 0, 0, 0, 0.086000000, 0.251030875, 0.026598872, 0.010074721]
    DAGNN_container = DataContainer(DAGNN_num_layer, DAGNN_accuracy, DAGNN_covariance, label="DAGNN")

    # single split # TODO: check the code
    GCNII_num_layer = [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 32]
    GCNII_accuracy = [1, 1, 1, 1, 1, 0.632500000, 0.482000000, 0.503500000, 0.502500000, 0.495500000, 0.483000000]
    GCNII_covariance = [0, 0, 0, 0, 0, 0.156468847, 0.019455076, 0.022781571, 0.007416198, 0.014177447, 0.015198684]
    GCNII_container = DataContainer(GCNII_num_layer, GCNII_accuracy, GCNII_covariance, label="GCNII")

    # list_container = [GCN_container,PairnormGCN_container,GAT_container,SGC_container,JKNet_container,GCNII_container,APPNP_container,DAGNN_container,batchnormSGC_container,PairnormSGC_container]
    # list_container = [GCN_container,JKNet_container,JKNet_train_container]
    # list_container = [GCN_container,GCN_hdim64_container]
    # list_container = [GCN_container,GCN_hdim64_container,GAT_container,GAT_hdim64_container]
    list_container = [SGC_container,SGC_hdim64_container]

    plot_accuracy_curve_with_covariance(list_container, save_path="acc_curve.png")
