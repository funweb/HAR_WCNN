import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings

from keras.models import Model
from keras import backend as K

from classifierModels import modelsLY
from tools import general
from tools.DataGenerator import MyDataGenerator

from tools.configure.constants import METHOD_PARAMETER_TEMPLATE, DATASETS_CONSTANT


mpl.style.use('seaborn-paper')
warnings.simplefilter('ignore', category=DeprecationWarning)


def load_model(dict_config):
    if dict_config['model_name'] == 'LSTM':
        model = modelsLY.get_LSTM(vocabulary_size=dict_config['vocabulary_size'],
                                  output_dim=dict_config['units'],
                                  data_lenght=dict_config['data_lenght'],
                                  no_activities=dict_config['no_activities'])

    elif dict_config['model_name'] == 'WCNN':
        model = modelsLY.WCNN(dict_config['no_activities'],
                              vocabulary_size=dict_config['vocabulary_size'],
                              output_dim=dict_config['units'],
                              data_lenght=dict_config['data_lenght'],
                              kernel_number_base=dict_config['kernel_number_base'],
                              kernel_wide_base=dict_config['kernel_wide_base'])
    elif dict_config["model_name"] == "WCNNR":
        model = modelsLY.WCNNR(dict_config['no_activities'],
                              vocabulary_size=dict_config['vocabulary_size'],
                              output_dim=dict_config['units'],
                              data_lenght=dict_config['data_lenght'],
                              kernel_number_base=dict_config['kernel_number_base'],
                              kernel_wide_base=dict_config['kernel_wide_base'],
                               net_deep_base=dict_config['net_deep_base'])
    elif dict_config["model_name"] == "inception_model":
        model = modelsLY.inception_model(no_activities=dict_config['no_activities'],
                                         data_lenght=dict_config['data_lenght'],
                                         kernel_number_base=dict_config['kernel_number_base'],
                                         kernel_wide_base=dict_config['kernel_wide_base']
                                         )

    else:
        print(
            'Your model name is:%s, but it does not exist. What are the names of the other models? Quickly specify the name of the model. Wow...' % (
                dict_config['model_name']))
        sys.exit(-1)

    model.summary()
    return model



# Load data in the following format:
# cutdatadir + '\\' + dataset_name + '-test-y-' + str(k) + '.npy', datas_y[test]
def load_data(dataset_name, cutdatadir, data_type='train', k=0, data_lenght=2000):
    data_x_path = os.path.join(cutdatadir, dataset_name + '-' + data_type + '-x-' + str(k) + '.npy')
    data_x = np.load(data_x_path, allow_pickle=True)

    data_y_path = os.path.join(cutdatadir, dataset_name + '-' + data_type + '-y-' + str(k) + '.npy')
    data_y = np.load(data_y_path, allow_pickle=True)

    data_labels_path = os.path.join(cutdatadir, dataset_name + '-labels.npy')
    dictActivities = np.load(data_labels_path, allow_pickle=True).item()

    # return data_x[:, -300:], data_y, dictActivities

    # x_train = x_train / (x_range + 1)  # 归一化处理  # TODO: 感觉没必要
    data_x = data_x[:, -data_lenght:]  # 控制数据长度

    return data_x, data_y, dictActivities


def build_function(model, layer_names=None, outputs=None):
    """
    Builds a Keras Function which retrieves the output of a Layer.
    构建一个 Keras 函数，用于检索层的输出。

    Args:
        model: Keras Model.
        layer_names: Name of the layer whose output is required.  # 需要输出的图层的名称。
        outputs: Output tensors.

    Returns:
        List of Keras Functions.
    """
    inp = model.input

    if layer_names is not None and (type(layer_names) != list and type(layer_names) != tuple):  # layer_names: 如果来自 filters, 那就是 list
        layer_names = [layer_names]  # 最后转为 list

    if outputs is None:
        if layer_names is None:
            outputs = [layer.output for layer in model.layers]  # all layer outputs
        else:
            outputs = [layer.output for layer in model.layers if layer.name in layer_names]
    else:
        outputs = outputs

    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions 评价函数
    return funcs


def get_outputs(model, inputs, eval_functions, verbose=False):
    """
    Gets the outputs of the Keras model.  获取Keras模型的输出。

    Args:
        model: Unused.  # 无使用
        inputs: Input numpy arrays.  # 输入数据
        eval_functions: Keras functions for evaluation.
        verbose: Whether to print evaluation metrics.  是否打印评估指标。

    Returns:
        List of outputs of the Keras Model.
    """
    if verbose: print('----- activations -----')
    outputs = []
    layer_outputs = [func([inputs, 1.])[0] for func in eval_functions]
    for layer_activations in layer_outputs:
        outputs.append(layer_activations)
    return outputs




def visualize_cam(dataset_name, k, cutdatadir, dict_config, class_id, conv_id, seed=0):
    """
    Used to visualize the Class Activation Maps of the Keras Model.  用于可视化Keras模型的类激活图。

    Args:
        model: A Keras Model.
        dataset_id: Integer id representing the dataset index containd in `utils/constants.py`.
                  : 表示 utils/constants.py 中包含的数据集索引的整数 id。
        dataset_prefix: Name of the dataset. Used for weight saving.
        class_id: Index of the class whose activation is to be visualized.
                : 要可视化其激活的类的索引。
        cutoff: Optional integer which slices of the first `cutoff` timesteps from the input signal.
              : 可选整数，它从输入信号中分割第一个“cutoff”时间步长。
        normalize_timeseries: Bool / Integer. Determines whether to normalize the timeseries.
                            ：布尔/整数。 确定是否标准化时间序列。
            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise z-normalization.
            If 2: Performs full dataset z-normalization.
        seed: Random seed number for Numpy.
    """
    """
     用于可视化 Keras 模型的类激活图。

     参数：
         模型：一个 Keras 模型。
         dataset_id：整数id，表示包含在中的数据集索引
             `utils/constants.py`。
         dataset_prefix：数据集的名称。 Used for weight saving.
         class_id：要可视化其激活的类的索引。
         cutoff: 可选整数，它从输入信号中分割第一个“cutoff”时间步长。
         normalize_timeseries：布尔/整数。 确定是否归一化
             时间序列。

             如果为 False，则不对时间序列进行标准化。
             如果 True / int 不等于 2，则执行标准样本 z 归一化。
             如果为 2：执行完整的数据集 z 归一化。
         seed：Numpy 的随机种子数。
     """

    np.random.seed(seed)

    # X_train, y_train, _, _, is_timeseries = load_dataset_at(dataset_id, normalize_timeseries=normalize_timeseries)
    # _, sequence_length = calculate_dataset_metrics(X_train)

    data_type = 'train'
    X_train, y_train, dictActivities = load_data(dataset_name, cutdatadir, data_type=data_type, k=k,
                                                 data_lenght=dict_config["data_lenght"])
    sequence_length = dict_config["data_lenght"]

    data_type = 'test'
    test_x, test_y, dictActivities = load_data(dataset_name, cutdatadir, data_type=data_type, k=k,
                                               data_lenght=dict_config["data_lenght"])

    dict_config['no_activities'] = len(dictActivities)
    dict_config['vocabulary_size'] = len(X_train)  # TODO:这个服务于 embedding 的字典大小, 默认的都是 188, 现在需要看一下一共有多少种状态, 可以 set 一下.

    dict_config['n_channels'] = 0  # 默认是 0
    dict_config['is_to_categorical'] = False
    dict_config['n_classes'] = dict_config['no_activities']
    dict_config['shuffle'] = True
    dict_config['is_embedding'] = True  # 决定了要不要扩充维度

    training_generator = MyDataGenerator(X_train, y_train, dict_config['batch_size'], dict_config['data_lenght'],
                                         dict_config['n_channels'], dict_config['is_to_categorical'],
                                         dict_config['n_classes'], dict_config['shuffle'], dict_config['is_embedding'])
    validation_generator = MyDataGenerator(test_x, test_y, dict_config['batch_size'], dict_config['data_lenght'],
                                           dict_config['n_channels'], dict_config['is_to_categorical'],
                                           dict_config['n_classes'], dict_config['shuffle'],
                                           dict_config['is_embedding'])

    model = load_model(dict_config)

    model.load_weights("./weights/%s-final.hdf5" % k)

    class_weights = model.layers[-1].get_weights()[0]  # 获得最后一层权重

    conv_layers = [layer for layer in model.layers if layer.__class__.__name__ == 'Conv1D']  # 全部卷积层
    final_conv = conv_layers[conv_id].name  # 最后一层卷积层  # 我其实是想得到下一层

    concat_layers = [layer for layer in model.layers if layer.__class__.__name__ == 'Concatenate']  # 连结层
    final_concat = concat_layers[-1].name  # 最后一层连接层, 其实也就是一层

    final_softmax = model.layers[-1].name  # softmax 分类层
    out_names = [final_conv, final_softmax]  # 要输出哪些层, 请放入list

    if class_id > 0:
        class_id = class_id - 1

    y_train = np.reshape(y_train, (-1,1))
    y_train_ids = np.where(y_train[:, 0] == class_id)  # 找出来其中的训练数据集
    sequence_input = X_train[y_train_ids[0], ...]  # 所有的训练数据集
    choice = np.random.choice(range(len(sequence_input)), 1)  # 随机选择一个

    for i in range(len(sequence_input)):
        if sequence_input[i][400] != 0:
            print(i)
            choice = np.array([i])
            break

    sequence_input = sequence_input[choice, :]

    eval_functions = build_function(model, out_names)
    conv_out, predictions = get_outputs(model, sequence_input, eval_functions)

    conv_out = conv_out[0, :, :]  # (T, C)

    conv_out = (conv_out - conv_out.min(axis=0, keepdims=True)) / \
               (conv_out.max(axis=0, keepdims=True) - conv_out.min(axis=0, keepdims=True))
    conv_out = (conv_out * 2.) - 1.  # 不知道这个操作是为了啥

    conv_out = conv_out.transpose((1, 0))  # (C, T)  # 维度转换  (长度, C) -> (C, T)
    conv_channels = conv_out.shape[0]

    conv_cam = class_weights[:conv_channels, [class_id]] * conv_out
    conv_cam = np.sum(conv_cam, axis=0)

    conv_cam /= conv_cam.max()

    sequence_input = sequence_input.reshape((-1, 1))
    conv_cam = conv_cam.reshape((-1, 1))

    sequence_df = pd.DataFrame(sequence_input,
                               index=range(sequence_input.shape[0]),
                               columns=range(sequence_input.shape[1]))

    sequence_df.to_csv("data.csv")

    conv_cam_df = pd.DataFrame(conv_cam,
                               index=range(conv_cam.shape[0]),
                               columns=[1])

    conv_cam_df.to_csv("{}.csv".format(conv_id))

    fig, axs = plt.subplots(2, 1, squeeze=False,
                            figsize=(6, 6))

    class_label = class_id + 1

    sequence_df.plot(title='Sequence (class = %d)' % (class_label),
                     subplots=False,
                     legend=None,
                     ax=axs[0][0])

    conv_cam_df.plot(title='Convolution Class Activation Map (class = %d)' % (class_label),
                     subplots=False,
                     legend=None,
                     ax=axs[1][0])

    plt.show()


if __name__ == '__main__':

    from tools.configure.constants import WCNNR_CONSTANT as MODEL_DEFAULT_CONF

    # os.chdir('../')
    print(os.getcwd())

    distance_int = 9999
    dataset_name = 'cairo'
    calculation_unit = "0"

    # 修订论文所需要的
    batch_size = 64
    data_lenght = 2000

    kernel_wide_base = 8
    kernel_number_base = 1
    net_deep_base = 1

    nb_epochs = 1000  # 公平起见, 默认都是 1000 吧.

    MODEL_DEFAULT_CONF["kernel_number_base"] = kernel_number_base
    MODEL_DEFAULT_CONF["kernel_wide_base"] = kernel_wide_base
    MODEL_DEFAULT_CONF["net_deep_base"] = net_deep_base

    dict_config_cus = {

        "datasets_dir": DATASETS_CONSTANT["base_dir"],  # 这是公共数据集常量
        "archive_name": DATASETS_CONSTANT["archive_name"],
        "ksplit": DATASETS_CONSTANT["ksplit"],

        'distance_int': distance_int,
        'dataset_name': dataset_name,
        'calculation_unit': calculation_unit,

        'data_lenght': data_lenght,
        'nb_epochs': nb_epochs,
        "batch_size": batch_size,

        # 'datasetsNames': ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11'],
    }

    dict_config_cus = general.Merge(dict_config_cus, MODEL_DEFAULT_CONF)

    dict_config = general.Merge(METHOD_PARAMETER_TEMPLATE, dict_config_cus)

    # logger = MyLogger()

    dataset_name = dict_config['dataset_name']
    print("current dataset: %s" % dataset_name)
    datadir = os.path.join(dict_config["datasets_dir"], 'ende', dataset_name, str(dict_config['distance_int']), 'npy')
    cutdatadir = os.path.join(datadir, str(dict_config['ksplit']))  # 是否应该放到 for 循环外面呢?
    k = 0

    visualize_cam(dataset_name, k, cutdatadir, dict_config_cus, class_id=1, conv_id=-1)
