#!/usr/bin/env python3
from __future__ import print_function
import csv
from datetime import datetime
import json

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

import modelsLY as models
from tools.DataGenerator import MyDataGenerator
from tools.MyLogger import MyLogger

# import argparse
# from tools.GPU_available import GPU_operator
# from sklearn.utils import compute_class_weight
# import keras

# 需要注意的是：默认路径都是在工作路径中调用的。

# 获取到各个数据编码后字典的长度
# PS：因为温度是以开关二进制编码的，因此字典都较小
# 参数1：数据库名字
# 参数2：距离大小
def get_dict_length(data_name, distant_int, P=False):
    """
    :param data_name: 数据集名称
    :param distant_int: 距离长度
    :param P: 是否打印
    :return: 距离<int>

    eg:
    distant_ints = 7
    data_names = ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11']

    for data_name in data_names:
        print('- - -'*4, data_name, '- - - '*4)
        for distant_int in range(distant_ints):
            if distant_int==0:
                distant_int = 999
            if distant_int==6:
                distant_int = 9999
            get_dict_length(data_name, distant_int, P=True)
    """
    file_url = os.path.join(os.getcwd(), 'datasets', 'ende', data_name, str(distant_int), 'sensor2dict.npy')
    data_content = np.load(file_url, allow_pickle=True)
    if P:
        print('operater dataset and the length dict is %s: %d' % (
        data_name + '_' + str(distant_int), len(data_content.item())))
    # sys.exit(-1)
    return len(data_content.item())


# 从数据中切分模型
# 一次切分一个数据集,数据集的格式为：
#   kyoto7-x.npy
#   kyoto7-y.npy
#   kyoto7-labels.npy

def datacut(data_name, datadir, dict_config):
    ksplit = dict_config['ksplit']
    opdir = os.path.join(os.getcwd(), 'datasets', 'ende', data_name, str(dict_config['distance_int']), 'npy',
                         str(ksplit))
    print('即将切分数据为:%d 份,并且保存到 %s' % (ksplit, opdir))

    if not os.path.exists(opdir):
        print('不存在 %s, 即将创建' % opdir)
        time.sleep(3)
        os.makedirs(opdir)

    # 首先载入数据
    data_path_x = os.path.join(datadir, data_name + '-x.npy')
    datas_x = np.load(data_path_x, allow_pickle=True)

    data_path_y = os.path.join(datadir, data_name + '-y.npy')
    datas_y = np.load(data_path_y, allow_pickle=True)
    datas_y = np.array(datas_y, dtype=np.int32)  # 真是奇怪，竟然读出来的是 str，不得不这样做

    data_path_labels = os.path.join(datadir, data_name + '-labels.npy')
    datas_labels = np.load(data_path_labels, allow_pickle=True)

    # 数据切分
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    k = 0  # 用来计数
    for train, test in kfold.split(datas_x, datas_y):
        np.save(os.path.join(opdir, data_name + '-train-x-' + str(k) + '.npy'), datas_x[train])
        np.save(os.path.join(opdir, data_name + '-train-y-' + str(k) + '.npy'), datas_y[train])

        np.save(os.path.join(opdir, data_name + '-test-x-' + str(k) + '.npy'), datas_x[test])
        np.save(os.path.join(opdir, data_name + '-test-y-' + str(k) + '.npy'), datas_y[test])

        if k == 0:
            np.save(os.path.join(opdir, data_name + '-labels.npy'), datas_labels)

        k += 1
    print('数据正常切分和保存，并且执行完毕。。。')


# 载入数据，命名格式为：
# cutdatadir + '\\' + data_name + '-test-y-' + str(k) + '.npy', datas_y[test]
def load_data(data_name, cutdatadir, data_type='train', k=0):

    data_x_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-x-' + str(k) + '.npy')
    data_x = np.load(data_x_path, allow_pickle=True)

    data_y_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-y-' + str(k) + '.npy')
    data_y = np.load(data_y_path, allow_pickle=True)

    data_labels_path = os.path.join(cutdatadir, data_name + '-labels.npy')
    dictActivities = np.load(data_labels_path, allow_pickle=True).item()

    return data_x, data_y, dictActivities

def load_model(dict_config):
    if dict_config['model_name'] == 'LSTM':
        model = models.get_LSTM(dict_config['input_dim'], dict_config['units'], dict_config['data_max_lenght'], dict_config['no_activities'])  # 原论文：len(X[train]), 64, 2000
    elif dict_config['model_name'] == 'LSTM_LY':
        model = models.get_LSTM(get_dict_length(dict_config['data_name'], dict_config['distant_int'], P=False) + 1, dict_config['units'],
                                dict_config['data_max_lenght'], dict_config['no_activities'])  # 这个遵循 embedding 理论编码，缩小了长度

    elif dict_config['model_name'] == 'BiLSTM':
        model = models.get_biLSTM(dict_config['input_dim'], dict_config['units'], dict_config['data_max_lenght'],
                                  dict_config['no_activities'])
    elif dict_config['model_name'] == 'Ensemble2LSTM':
        model = models.get_Ensemble2LSTM(dict_config['input_dim'], dict_config['units'], dict_config['data_max_lenght'],
                                         dict_config['no_activities'])
    elif dict_config['model_name'] == 'no_embedding_LSTM':
        dict_config['is_embedding'] = False  # 因为第一个维度不是 embedding， 因此设为 False 表示不执行维度扩充
        model = models.no_embedding_LSTM(dict_config['input_dim'], dict_config['units'], dict_config['data_max_lenght'],
                                         dict_config['no_activities'])
    elif dict_config['model_name'] == 'WCONN':
        model = models.WCONN(dict_config['no_activities'])
    elif dict_config['model_name'] == 'zzz':
        model = models.zzz(dict_config['no_activities'])
    elif dict_config['model_name'] == 'deep_model':
        model = models.deep_model(dict_config['input_dim'], dict_config['units'], dict_config['data_max_lenght'],
                                  dict_config['no_activities'])
    else:
        print('你的模型名字为：%s，但是不存在。别的模型名字是什么呢？赶快指定模型的名字哇。。。' % (dict_config['model_name']))
        sys.exit(-1)

    return model


def train(data_name, k, cutdatadir, dict_config):
    # 训练和验证数据载入
    data_type = 'train'
    train_x, train_y, dictActivities = load_data(data_name, cutdatadir, data_type=data_type, k=k)

    data_type = 'test'
    test_x, test_y, dictActivities = load_data(data_name, cutdatadir, data_type=data_type, k=k)

    print('训练数据：%d个，验证数据：%d个，活动种类：%d类' % (len(train_y), len(test_y), len(dictActivities)))

    training_generator = MyDataGenerator(train_x, train_y, dict_config['batch_size'], dict_config['dim'],
                                         dict_config['n_channels'], dict_config['is_to_categorical'],
                                         dict_config['n_classes'], dict_config['shuffle'], dict_config['is_embedding'])
    validation_generator = MyDataGenerator(test_x, test_y, dict_config['batch_size'], dict_config['dim'],
                                         dict_config['n_channels'], dict_config['is_to_categorical'],
                                         dict_config['n_classes'], dict_config['shuffle'], dict_config['is_embedding'])

    dict_config['no_activities'] = len(dictActivities)  # 行为种类
    dict_config['input_dim'] = len(train_x)
    model = load_model(dict_config)

    if k is 0:
        model.summary()

    model = models.compileModelcus(model, dict_config['optimizer'])
    # modelname = model.name
    starttime = datetime.now().strftime('%Y%m%d-%H%M%S')
    print('开始训练，当前时间是：%s' % (starttime))

    # train the model
    checkpointer_dir = os.path.join(cutdatadir, 'weight')
    if not os.path.exists(checkpointer_dir):
        os.makedirs(checkpointer_dir)

    base_identifier = '%s_%s_%s_%s_%s_%s_%s' % (
        dict_config['identifier'], data_name, dict_config['model_name'], dict_config['optimizer'],
        dict_config['epochs'],
        str(dict_config['distance_int']), k)  # 作为基础的命名格式

    # weight_save_name =  checkpointer_dir + '/' + dict_config['model_name'] + '-' + k + '-' + data_name + '-{epoch:02d}-{loss:.2f}.hdh5'  # 动态的
    weight_name = os.path.join(checkpointer_dir, base_identifier + '-best.hdh5')  # 静态的
    print('ModelCheckpoint 已经保存到 %s' % weight_name)
    model_checkpoint = ModelCheckpoint(weight_name,
                                       monitor='loss',
                                       mode='min',
                                       period=1,
                                       save_best_only=True)
    early_stop = EarlyStopping(monitor='acc', patience=dict_config['patience'], verbose=1, mode='auto')

    print('Begin training ...')
    history_LY = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=dict_config['epochs'],
                                     verbose=dict_config['verbose'],
                                     shuffle=dict_config['shuffle'],
                                     # use_multiprocessing=True,
                                     # workers=1,
                                     callbacks=[model_checkpoint, early_stop]
                                     )

    log_file_path = os.path.join(cutdatadir, 'log')
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    csvFileName = os.path.join(log_file_path, base_identifier + '.csv')
    pd.read_json(json.dumps(history_LY.history), encoding="utf-8", orient='records').to_csv(csvFileName)
    print('保存训练数据到：%s' % (csvFileName))

    endtime = datetime.now().strftime('%Y%m%d-%H%M%S')
    print('结束了第 %d 轮训练，当前时间是：%s' % (k, endtime))
    print('一共经过了 %s s' % (datetime.strptime(endtime, '%Y%m%d-%H%M%S') - datetime.strptime(starttime, '%Y%m%d-%H%M%S')))
    print('一共经过了 %s s' % (datetime.strptime(endtime, '%Y%m%d-%H%M%S') - datetime.strptime(starttime,
                                                                                          '%Y%m%d-%H%M%S')).total_seconds())
    print('平均是 %f s' % ((datetime.strptime(endtime, '%Y%m%d-%H%M%S') - datetime.strptime(starttime,
                                                                                         '%Y%m%d-%H%M%S')).total_seconds() /
                        dict_config['epochs']))
    print('并没有返回值，第 %d epoch 训练完毕。。。' % k)


# 从权重中载入
def validation_from_weight(data_name, weight_path, cutdatadir, k, dict_config):
    # 数据准备
    data_type = 'train'
    train_x, train_y, dictActivities = load_data(data_name, cutdatadir, data_type=data_type, k=k)

    data_type = 'test'
    test_x, test_y, dictActivities = load_data(data_name, cutdatadir, data_type=data_type, k=k)

    # 模型载入，根据 model_name
    dict_config['no_activities'] = len(dictActivities)  # 行为种类
    dict_config['input_dim'] = len(train_x)
    model = load_model(dict_config)

    base_identifier = '%s_%s_%s_%s_%s_%s_%s' % (
        dict_config['identifier'], data_name, dict_config['model_name'], dict_config['optimizer'],
        dict_config['epochs'],
        str(dict_config['distance_int']), k)  # 作为基础的命名格式

    weight_name = os.path.join(weight_path, base_identifier + '-best.hdh5')  # 静态的

    model.load_weights(weight_name)
    model = models.compileModelcus(model, dict_config['optimizer'])
    print("Created model and loaded weights from file")

    # evaluate the model
    print('Begin testing ...')
    scores = model.evaluate(test_x, test_y, batch_size=dict_config['batch_size'], verbose=1)

    # test_generator = MyDataGenerator(test_x, test_y, **params)
    # scores = model.evaluate_generator(test_generator)

    print('%d-%s:\t%s: %.2f%%' % (k, data_name, model.metrics_names[1], scores[1] * 100))

    print('Report:')
    target_names = sorted(dictActivities, key=dictActivities.get)

    # classes = model.predict_classes(X_test_input, batch_size=batch_size)  # 这个需要交向选择
    # classes = model.predict(X_test_input, batch_size=batch_size)  # 也就是预测的类别
    predict = model.predict(test_x, batch_size=dict_config['batch_size'])
    classes = np.argmax(predict, axis=1)  # 这一句有点费接哦，我加上的
    # classes = model.predict_classes([X_test_input, X_test_input], batch_size=batch_size)
    print('*' * 20)
    print('list(Y[test]: %s' % (list(test_y)))
    print('classes: %s' % (list(classes)))
    print('*' * 20)

    # print(classification_report(list(Y[test]), classes, target_names=target_names))
    multiply_evaluation = classification_report(list(test_y), classes, target_names=target_names, digits=6)  # 多个评价指标
    print(multiply_evaluation)

    print('Confusion matrix:')
    labels = list(dictActivities.values())
    print('混淆矩阵：confusion matrix\'s labels is: %s' % labels)
    Confusion_matrix = confusion_matrix(list(test_y), classes, labels)
    print(Confusion_matrix)

    dict_evaluation = {}
    dict_evaluation.update({'multiply_evaluation': multiply_evaluation})
    dict_evaluation.update({'Confusion_matrix': str(Confusion_matrix)})
    dict_evaluation.update({model.metrics_names[1]: str(scores[1] * 100)})

    # 不得已这样先搞过去吧
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


# 合并两个字典，也就是算法所需要的参数
def Merge(dict_config, dict_config_cus):
    return dict_config.update(dict_config_cus)


def train_val(dict_config_cus):

    dict_config = {
        # 以下参数需要设置
        'model_name': '',
        'optimizer': '',
        'distance_int': '',
        'data_name': '',
        'calculation_unit': '0',  # 默认为 CPU

        'seed': 7,  # 随机种子
        'want_cut_data': 'False',  # 是否重新切分保存数据，一般第一次为 True，其余为 False
        'ksplit': 3,  # 切分的份数

        # 数据生成器参数设置，也就是 param 参数
        'dim': (2000),  # 暂时不更改，应该是数据长度 2000，但是要带上括号
        'is_embedding': True,  # 是否数据扩维
        'batch_size': 64,  # 批次大小
        'n_classes': 6,  # 可以删除么？这个参数不记得了
        'n_channels': 0,  # 为什么是 0，忘记了
        'is_to_categorical': False,  # 关系到用什么样的交叉熵
        'shuffle': True,  # 每一个 gen 的顺序是否打乱，通常为 True

        # 网络参数
        'input_dim': '',  # len(X[train])  # 按理说这个应该是设置为数据类型的大小的。PS：程序中设置
        'units': 64,  # deep 论文中的，这个小的可以尝试一下。。
        'data_max_lenght': 2000,  # 数据编码长度
        'no_activities': '',  # 类别。PS：程序中设置

        # 训练设置
        'epochs': 2,
        'patience': 200,  # 停止监测的轮数
        'fit_shuffle': False,  # fit 是否打乱顺序，一般这个为 False，但是生成器的一般为 True
        'verbose': 1,

        # 标志
        'identifier': '20201116',
        'purpose': '2020年11月16日跑基于LSTM的模型，整合了一下',  # 更改模型训练的目的
        'datasetsNames': ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11'],
    }

    Merge(dict_config, dict_config_cus)

    '''
    if dict_config['calculation_unit'] is '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    else:
        if dict_config['calculation_unit'] is '1':
            # 使用第0块GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 显示信息的等级  3：只显示error
    '''

    logger = MyLogger()  # 定义我的日志记录操作
    logger.info('首先说明是否限制了数据？用的什么优化器，训练了多少epoch，用的什么方法。目的是什么')
    logger.info('数据限制：%s， \t 优化器：%s, \t epochs: %d, \t model_name: %s, \t purpose: %s' % (
        dict_config['distance_int'], dict_config['optimizer'], dict_config['epochs'], dict_config['model_name'],
        dict_config['purpose']))

    data_name = dict_config['data_name']
    print("current dataset: %s" % data_name)
    datadir = os.path.join(os.getcwd(), 'datasets', 'ende', data_name, str(dict_config['distance_int']), 'npy')

    if dict_config['want_cut_data'] == True:
        datacut(data_name, datadir, dict_config)

    cvaccuracy = []  # 记录准确率
    cvscores = []  # 记录分数
    modelname = dict_config['model_name']

    total_dict_evaluation = {}  # 记录评价指标的一些值

    for k in range(dict_config['ksplit']):
        cutdatadir = os.path.join(datadir, str(dict_config['ksplit']))  # k折切分完之后数据的目录

        train(data_name, k, cutdatadir, dict_config)

        # 从保存的模型中进行测试
        weight_path = os.path.join(datadir, str(dict_config['ksplit']), 'weight')
        dict_evaluation, scores = validation_from_weight(data_name, weight_path, cutdatadir, k,
                                                         dict_config)  # 返回 scores
        cvaccuracy.append(scores[1] * 100)
        cvscores.append(scores)
        total_dict_evaluation.update({str(k): dict_evaluation})

    print('current database: {} \t {:.2f}% (+/- {:.2f}%)'.format(data_name, np.mean(cvaccuracy), np.std(cvaccuracy)))

    base_identifier = '%s_%s_%s_%s_%s_%s' % (
        dict_config['identifier'], data_name, dict_config['model_name'], dict_config['optimizer'],
        dict_config['epochs'],
        str(dict_config['distance_int']))  # 作为基础的命名格式

    csvfile = base_identifier + '.csv'
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in cvscores:
            writer.writerow([",".join(str(el) for el in val)])

    evaluation_file = base_identifier + '.txt'
    with open(evaluation_file, 'w') as fw:
        for i in total_dict_evaluation:
            for ii in total_dict_evaluation[i]:
                fw.writelines(total_dict_evaluation[i][ii])
            fw.writelines('\n\n')


if __name__ == '__main__':
    os.chdir('../')
    print(os.getcwd())  # 切换到上级目录
    dict_config_cus = {
        # 以下参数需要设置
        'model_name': 'LSTM',
        'optimizer': 'rms',
        'distance_int': '3',
        'data_name': 'kyoto7',
        'calculation_unit': '0',  # 默认为 CPU

        'epochs': 20,

        # 标志
        'identifier': '20201116',
        'purpose': '2020年11月16日跑基于LSTM的模型，整合了一下',  # 更改模型训练的目的
        'datasetsNames': ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11'],
    }
    train_val(dict_config_cus)
