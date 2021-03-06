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

    # x_train = x_train / (x_range + 1)  # ???????????????  # TODO: ???????????????
    data_x = data_x[:, -data_lenght:]  # ??????????????????

    return data_x, data_y, dictActivities


def build_function(model, layer_names=None, outputs=None):
    """
    Builds a Keras Function which retrieves the output of a Layer.
    ???????????? Keras ????????????????????????????????????

    Args:
        model: Keras Model.
        layer_names: Name of the layer whose output is required.  # ?????????????????????????????????
        outputs: Output tensors.

    Returns:
        List of Keras Functions.
    """
    inp = model.input

    if layer_names is not None and (type(layer_names) != list and type(layer_names) != tuple):  # layer_names: ???????????? filters, ????????? list
        layer_names = [layer_names]  # ???????????? list

    if outputs is None:
        if layer_names is None:
            outputs = [layer.output for layer in model.layers]  # all layer outputs
        else:
            outputs = [layer.output for layer in model.layers if layer.name in layer_names]
    else:
        outputs = outputs

    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions ????????????
    return funcs


def get_outputs(model, inputs, eval_functions, verbose=False):
    """
    Gets the outputs of the Keras model.  ??????Keras??????????????????

    Args:
        model: Unused.  # ?????????
        inputs: Input numpy arrays.  # ????????????
        eval_functions: Keras functions for evaluation.
        verbose: Whether to print evaluation metrics.  ???????????????????????????

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
    Used to visualize the Class Activation Maps of the Keras Model.  ???????????????Keras????????????????????????

    Args:
        model: A Keras Model.
        dataset_id: Integer id representing the dataset index containd in `utils/constants.py`.
                  : ?????? utils/constants.py ???????????????????????????????????? id???
        dataset_prefix: Name of the dataset. Used for weight saving.
        class_id: Index of the class whose activation is to be visualized.
                : ???????????????????????????????????????
        cutoff: Optional integer which slices of the first `cutoff` timesteps from the input signal.
              : ??????????????????????????????????????????????????????cutoff??????????????????
        normalize_timeseries: Bool / Integer. Determines whether to normalize the timeseries.
                            ?????????/????????? ????????????????????????????????????
            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise z-normalization.
            If 2: Performs full dataset z-normalization.
        seed: Random seed number for Numpy.
    """
    """
     ??????????????? Keras ????????????????????????

     ?????????
         ??????????????? Keras ?????????
         dataset_id?????????id???????????????????????????????????????
             `utils/constants.py`???
         dataset_prefix???????????????????????? Used for weight saving.
         class_id??????????????????????????????????????????
         cutoff: ??????????????????????????????????????????????????????cutoff??????????????????
         normalize_timeseries?????????/????????? ?????????????????????
             ???????????????

             ????????? False??????????????????????????????????????????
             ?????? True / int ????????? 2???????????????????????? z ????????????
             ????????? 2??????????????????????????? z ????????????
         seed???Numpy ?????????????????????
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
    dict_config['vocabulary_size'] = len(X_train)  # TODO:??????????????? embedding ???????????????, ??????????????? 188, ?????????????????????????????????????????????, ?????? set ??????.

    dict_config['n_channels'] = 0  # ????????? 0
    dict_config['is_to_categorical'] = False
    dict_config['n_classes'] = dict_config['no_activities']
    dict_config['shuffle'] = True
    dict_config['is_embedding'] = True  # ??????????????????????????????

    training_generator = MyDataGenerator(X_train, y_train, dict_config['batch_size'], dict_config['data_lenght'],
                                         dict_config['n_channels'], dict_config['is_to_categorical'],
                                         dict_config['n_classes'], dict_config['shuffle'], dict_config['is_embedding'])
    validation_generator = MyDataGenerator(test_x, test_y, dict_config['batch_size'], dict_config['data_lenght'],
                                           dict_config['n_channels'], dict_config['is_to_categorical'],
                                           dict_config['n_classes'], dict_config['shuffle'],
                                           dict_config['is_embedding'])

    model = load_model(dict_config)

    model.load_weights("./weights/%s-final.hdf5" % k)

    class_weights = model.layers[-1].get_weights()[0]  # ????????????????????????

    conv_layers = [layer for layer in model.layers if layer.__class__.__name__ == 'Conv1D']  # ???????????????
    final_conv = conv_layers[conv_id].name  # ?????????????????????  # ??????????????????????????????

    concat_layers = [layer for layer in model.layers if layer.__class__.__name__ == 'Concatenate']  # ?????????
    final_concat = concat_layers[-1].name  # ?????????????????????, ?????????????????????

    final_softmax = model.layers[-1].name  # softmax ?????????
    out_names = [final_conv, final_softmax]  # ??????????????????, ?????????list

    if class_id > 0:
        class_id = class_id - 1

    y_train = np.reshape(y_train, (-1,1))
    y_train_ids = np.where(y_train[:, 0] == class_id)  # ?????????????????????????????????
    sequence_input = X_train[y_train_ids[0], ...]  # ????????????????????????
    choice = np.random.choice(range(len(sequence_input)), 1)  # ??????????????????

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
    conv_out = (conv_out * 2.) - 1.  # ?????????????????????????????????

    conv_out = conv_out.transpose((1, 0))  # (C, T)  # ????????????  (??????, C) -> (C, T)
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

    # ????????????????????????
    batch_size = 64
    data_lenght = 2000

    kernel_wide_base = 8
    kernel_number_base = 1
    net_deep_base = 1

    nb_epochs = 1000  # ????????????, ???????????? 1000 ???.

    MODEL_DEFAULT_CONF["kernel_number_base"] = kernel_number_base
    MODEL_DEFAULT_CONF["kernel_wide_base"] = kernel_wide_base
    MODEL_DEFAULT_CONF["net_deep_base"] = net_deep_base

    dict_config_cus = {

        "datasets_dir": DATASETS_CONSTANT["base_dir"],  # ???????????????????????????
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
    cutdatadir = os.path.join(datadir, str(dict_config['ksplit']))  # ?????????????????? for ????????????????
    k = 0

    visualize_cam(dataset_name, k, cutdatadir, dict_config_cus, class_id=1, conv_id=-1)
