#!/usr/bin/env python3
from __future__ import print_function
import csv
import re
import shutil
from datetime import datetime
import json
from glob import glob

import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import os

import time
import pandas as pd
import sys
import tensorflow as tf

# import modelsLY as models
from classifierModels import modelsLY
from tools import general, path2name
from tools.DataGenerator import MyDataGenerator
from tools.MyLogger import MyLogger
from tools.configure.constants import METHOD_PARAMETER_TEMPLATE, DATASETS_CONSTANT
from tools.general import ModelCheckpoint_cus


def get_dict_length(dataset_name, distant_int, P=False):
    file_url = os.path.join(os.getcwd(), 'datasets', 'ende', dataset_name, str(distant_int), 'sensor2dict.npy')
    data_content = np.load(file_url, allow_pickle=True)
    if P:
        print('operater dataset and the length dict is %s: %d' % (
            dataset_name + '_' + str(distant_int), len(data_content.item())))
    # sys.exit(-1)
    return len(data_content.item())


def datacut(dataset_name, datadir, dict_config):
    ksplit = dict_config['ksplit']
    opdir = os.path.join(os.getcwd(), 'datasets', 'ende', dataset_name, str(dict_config['distance_int']), 'npy',
                         str(ksplit))

    if not os.path.exists(opdir):
        print('Create directory: %s' % opdir)
        time.sleep(3)
        os.makedirs(opdir)

    # load data
    data_path_x = os.path.join(datadir, dataset_name + '-x.npy')
    datas_x = np.load(data_path_x, allow_pickle=True)

    data_path_y = os.path.join(datadir, dataset_name + '-y.npy')
    datas_y = np.load(data_path_y, allow_pickle=True)
    datas_y = np.array(datas_y, dtype=np.int32)

    data_path_labels = os.path.join(datadir, dataset_name + '-labels.npy')
    datas_labels = np.load(data_path_labels, allow_pickle=True)

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    k = 0
    for train, test in kfold.split(datas_x, datas_y):
        np.save(os.path.join(opdir, dataset_name + '-train-x-' + str(k) + '.npy'), datas_x[train])
        np.save(os.path.join(opdir, dataset_name + '-train-y-' + str(k) + '.npy'), datas_y[train])

        np.save(os.path.join(opdir, dataset_name + '-test-x-' + str(k) + '.npy'), datas_x[test])
        np.save(os.path.join(opdir, dataset_name + '-test-y-' + str(k) + '.npy'), datas_y[test])

        if k == 0:
            np.save(os.path.join(opdir, dataset_name + '-labels.npy'), datas_labels)

        k += 1


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

    return model


def train(dataset_name, k, cutdatadir, dict_config):
    data_type = 'train'
    train_x, train_y, dictActivities = load_data(dataset_name, cutdatadir, data_type=data_type, k=k,
                                                 data_lenght=dict_config["data_lenght"])

    data_type = 'test'
    test_x, test_y, dictActivities = load_data(dataset_name, cutdatadir, data_type=data_type, k=k,
                                               data_lenght=dict_config["data_lenght"])

    dict_config['no_activities'] = len(dictActivities)
    dict_config['vocabulary_size'] = len(train_x)  # TODO:这个服务于 embedding 的字典大小, 默认的都是 188, 现在需要看一下一共有多少种状态, 可以 set 一下.

    dict_config['n_channels'] = 0  # 默认是 0
    dict_config['is_to_categorical'] = False
    dict_config['n_classes'] = dict_config['no_activities']
    dict_config['shuffle'] = True
    dict_config['is_embedding'] = True  # 决定了要不要扩充维度

    training_generator = MyDataGenerator(train_x, train_y, dict_config['batch_size'], dict_config['data_lenght'],
                                         dict_config['n_channels'], dict_config['is_to_categorical'],
                                         dict_config['n_classes'], dict_config['shuffle'], dict_config['is_embedding'])
    validation_generator = MyDataGenerator(test_x, test_y, dict_config['batch_size'], dict_config['data_lenght'],
                                           dict_config['n_channels'], dict_config['is_to_categorical'],
                                           dict_config['n_classes'], dict_config['shuffle'],
                                           dict_config['is_embedding'])

    model = load_model(dict_config)

    if k is 0:
        model.summary()

    model = modelsLY.compileModelcus(model, dict_config['optimizer'])
    starttime = datetime.now().strftime('%Y%m%d-%H%M%S')

    # train the model
    # checkpointer_dir = os.path.join(cutdatadir, 'weight')  # 保存结果文件夹
    #
    # os.path.join(dict_config["datasets_dir"], 'ende', dataset_name, str(dict_config['distance_int']), 'npy')
    checkpointer_dir = path2name.get_checkpointer_dir(dict_config)

    weights_dir = os.path.join(checkpointer_dir, "weights")
    general.create_folder(weights_dir)

    weight_name = os.path.join(weights_dir, "%s-{epoch:06d}-{loss:.6f}-{val_acc:.6f}.hdf5" % (str(k)))
    model_checkpoint = ModelCheckpoint_cus(filepath=weight_name,
                                           monitor='loss', verbose=1,
                                           save_best_only=False,
                                           save_best_only_period=True,
                                           mode='min', period=100,  # int(self.nb_epochs/5)
                                           )

    early_stop = EarlyStopping(monitor='acc', patience=dict_config['patience'], verbose=1, mode='auto')

    print('Begin training ...')
    history_LY = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=dict_config['nb_epochs'],
                                     verbose=int(dict_config['verbose']),
                                     shuffle=dict_config['shuffle'],
                                     callbacks=[model_checkpoint, early_stop]
                                     )

    #### 保存 final_model 和 best_model
    weight_name = os.path.join(weights_dir, str(k) + '-final.hdf5')
    print('ModelCheckpoint 已经保存到 %s' % weight_name)

    model.save(weight_name)
    ### -----------  认为 最小loss 是最好的
    min_loss = 128
    best_model_name = ""
    pattern = r'[Pbest_]?{}-(\d+)-(\d*\.\d*)-(\d*\.\d*).hdf5'.format(str(k))

    prog = re.compile(pattern)
    for hdf5_name in os.listdir(weights_dir):
        matchObj = prog.match(os.path.basename(hdf5_name))
        if matchObj is not None:
            c_epoch = matchObj.group(1)
            c_loss = matchObj.group(2)
            c_acc = matchObj.group(3)
            if float(c_loss) < min_loss:
                best_model_name = hdf5_name
    shutil.copy(os.path.join(weights_dir, os.path.basename(best_model_name)),
                os.path.join(weights_dir, str(k) + "-best.hdf5"))
    ### -----------  认为 最小loss 是最好的

    # weight_name_final_epochs = os.path.join(checkpointer_dir, base_identifier + '-final.hdf5')
    # model.save(weight_name_final_epochs)

    log_file_path = os.path.join(checkpointer_dir, 'log')  # 日志文件保存路径
    general.create_folder(log_file_path)

    csvFileName = os.path.join(log_file_path, path2name.get_identifier_name(dict_config) + '.csv')
    pd.read_json(json.dumps(history_LY.history), encoding="utf-8", orient='records').to_csv(csvFileName)
    print('save in: %s' % (csvFileName))

    endtime = datetime.now().strftime('%Y%m%d-%H%M%S')
    print('time: %s s' % (datetime.strptime(endtime, '%Y%m%d-%H%M%S') - datetime.strptime(starttime, '%Y%m%d-%H%M%S')))
    print('time: %s s' % (datetime.strptime(endtime, '%Y%m%d-%H%M%S') - datetime.strptime(starttime,
                                                                                          '%Y%m%d-%H%M%S')).total_seconds())
    print('average %f s' % ((datetime.strptime(endtime, '%Y%m%d-%H%M%S') - datetime.strptime(starttime,
                                                                                             '%Y%m%d-%H%M%S')).total_seconds() /
                            dict_config['nb_epochs']))
    print('%d epoch finished' % k)


# load from weight
def validation_from_weight(dataset_name, weight_path, cutdatadir, k, dict_config, flag='best'):
    data_type = 'train'
    train_x, train_y, dictActivities = load_data(dataset_name, cutdatadir, data_type=data_type, k=k,
                                                 data_lenght=dict_config["data_lenght"])

    data_type = 'test'
    test_x, test_y, dictActivities = load_data(dataset_name, cutdatadir, data_type=data_type, k=k,
                                               data_lenght=dict_config["data_lenght"])

    dict_config['no_activities'] = len(dictActivities)
    dict_config['vocabulary_size'] = len(train_x)
    model = load_model(dict_config)

    checkpointer_dir = path2name.get_checkpointer_dir(dict_config)

    weights_dir = os.path.join(checkpointer_dir, "weights")

    weight_name = os.path.join(weights_dir, str(k) + '-' + flag + '.hdf5')

    model.load_weights(weight_name)
    model = modelsLY.compileModelcus(model, dict_config['optimizer'])
    print("Created model and loaded weights from file")

    print('Begin testing ...')
    scores = model.evaluate(test_x, test_y, batch_size=dict_config['batch_size'], verbose=1)

    print('%d-%s:\t%s: %.2f%%' % (k, dataset_name, model.metrics_names[1], scores[1] * 100))

    print('Report:')
    target_names = sorted(dictActivities, key=dictActivities.get)

    predict = model.predict(test_x, batch_size=dict_config['batch_size'])
    classes = np.argmax(predict, axis=1)
    print('*' * 20)
    print('list(Y[test]: %s' % (list(test_y)))
    print('classes: %s' % (list(classes)))
    print('*' * 20)

    multiply_evaluation = classification_report(list(test_y), classes, target_names=target_names, digits=6)  # 多个评价指标
    print(multiply_evaluation)

    print('Confusion matrix:')
    labels = list(dictActivities.values())
    print('confusion matrix\'s labels is: %s' % labels)
    Confusion_matrix = confusion_matrix(list(test_y), classes, labels)
    print(Confusion_matrix)

    dict_evaluation = {}
    dict_evaluation.update({'multiply_evaluation': multiply_evaluation})
    dict_evaluation.update({'Confusion_matrix': str(Confusion_matrix)})
    dict_evaluation.update({model.metrics_names[1]: str(scores[1] * 100)})

    str_classes = ''
    for c in classes:
        str_classes += str(c)
        str_classes += '\t'
    dict_evaluation.update({'predict': str_classes + '\n'})
    str_test_y = ''
    for t in test_y:
        str_test_y += str(t)
        str_test_y += '\t'
    dict_evaluation.update({'test_y': str_test_y + '\n'})

    return dict_evaluation, scores


# Merge the two dictionaries, that is, the parameters required by the algorithm
# def Merge(dict_config, dict_config_cus):
#     return dict_config.update(dict_config_cus)


def train_val(dict_config_cus):
    dict_config = general.Merge(METHOD_PARAMETER_TEMPLATE, dict_config_cus)

    # logger = MyLogger()

    dataset_name = dict_config['dataset_name']
    print("current dataset: %s" % dataset_name)
    datadir = os.path.join(dict_config["datasets_dir"], 'ende', dataset_name, str(dict_config['distance_int']), 'npy')

    want_cut_data = False
    if want_cut_data == True:  # 基本上都是不重新切分的, 因此这句是没必要的.
        datacut(dataset_name, datadir, dict_config)

    cvaccuracy_best = []
    cvaccuracy_final = []
    cvscores_best = []
    cvscores_final = []

    # modelname = dict_config['model_name']

    total_dict_evaluation_best = {}
    total_dict_evaluation_final = {}

    old_result_dir = path2name.get_checkpointer_dir(dict_config)
    print(old_result_dir)
    if os.path.exists(old_result_dir):
        shutil.rmtree(old_result_dir)

    for k in range(dict_config['ksplit']):
        cutdatadir = os.path.join(datadir, str(dict_config['ksplit']))  # 是否应该放到 for 循环外面呢?

        train(dataset_name, k, cutdatadir, dict_config)

        weight_path = os.path.join(datadir, str(dict_config['ksplit']), 'weight')
        print('-' * 20, '  best  ', '-' * 20)
        dict_evaluation_best, scores_best = validation_from_weight(dataset_name, weight_path, cutdatadir, k,
                                                                   dict_config, flag='best')

        print('-' * 20, '  final  ', '-' * 20)
        dict_evaluation_final, scores_final = validation_from_weight(dataset_name, weight_path, cutdatadir, k,
                                                                     dict_config, flag='final')

        cvaccuracy_best.append(scores_best[1] * 100)
        cvscores_best.append(scores_best)
        total_dict_evaluation_best.update({str(k): dict_evaluation_best})

        cvaccuracy_final.append(scores_final[1] * 100)
        cvscores_final.append(scores_final)
        total_dict_evaluation_final.update({str(k): dict_evaluation_final})

    print('best: current database: {} \t {:.2f}% (+/- {:.2f}%)'.format(dataset_name, np.mean(cvaccuracy_best),
                                                                       np.std(cvaccuracy_best)))
    print('final: current database: {} \t {:.2f}% (+/- {:.2f}%)'.format(dataset_name, np.mean(cvaccuracy_final),
                                                                        np.std(cvaccuracy_final)))

    base_identifier = path2name.get_identifier_name(dict_config)
    checkpointer_dir = path2name.get_checkpointer_dir(dict_config)

    csvfile_best = os.path.join(checkpointer_dir, 'log', base_identifier + '_best.csv')
    with open(csvfile_best, "w", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in cvscores_best:
            writer.writerow([",".join(str(el) for el in val)])

    csvfile_final = os.path.join(checkpointer_dir, 'log', base_identifier + '_final.csv')
    with open(csvfile_final, "w", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in cvscores_final:
            writer.writerow([",".join(str(el) for el in val)])

    evaluation_file_best = os.path.join(checkpointer_dir, 'log', base_identifier + '_best.txt')
    with open(evaluation_file_best, 'w', encoding="utf-8") as fw:
        for i in total_dict_evaluation_best:
            for ii in total_dict_evaluation_best[i]:
                fw.writelines(total_dict_evaluation_best[i][ii])
            fw.writelines('\n\n')

    evaluation_file_final = os.path.join(checkpointer_dir, 'log', base_identifier + '_final.txt')
    with open(evaluation_file_final, 'w', encoding="utf-8") as fw:
        for i in total_dict_evaluation_final:
            for ii in total_dict_evaluation_final[i]:
                fw.writelines(total_dict_evaluation_final[i][ii])
            fw.writelines('\n\n')

    with open(os.path.join(checkpointer_dir, dict_config["complete_flag"]), "w", encoding="utf-8") as fw:
        json.dump(dict_config, fw)


if __name__ == '__main__':

    from tools.configure.constants import WCNNR_CONSTANT as MODEL_DEFAULT_CONF

    # os.chdir('../')
    print(os.getcwd())

    distance_int = 9999
    dataset_name = 'cairo'
    calculation_unit = "0"

    # 修订论文所需要的
    batch_size = 64
    data_lenght = 30

    kernel_number_base = 8
    kernel_wide_base = 1

    nb_epochs = 1000  # 公平起见, 默认都是 1000 吧.

    MODEL_DEFAULT_CONF["kernel_number_base"] = kernel_number_base
    MODEL_DEFAULT_CONF["kernel_wide_base"] = kernel_wide_base

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
    train_val(dict_config_cus)
