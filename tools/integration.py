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

# import modelsLY as models
from classifierModels import modelsLY
from tools import general
from tools.DataGenerator import MyDataGenerator
from tools.MyLogger import MyLogger


def get_dict_length(data_name, distant_int, P=False):
    file_url = os.path.join(os.getcwd(), 'datasets', 'ende', data_name, str(distant_int), 'sensor2dict.npy')
    data_content = np.load(file_url, allow_pickle=True)
    if P:
        print('operater dataset and the length dict is %s: %d' % (
            data_name + '_' + str(distant_int), len(data_content.item())))
    # sys.exit(-1)
    return len(data_content.item())


def datacut(data_name, datadir, dict_config):
    ksplit = dict_config['ksplit']
    opdir = os.path.join(os.getcwd(), 'datasets', 'ende', data_name, str(dict_config['distance_int']), 'npy',
                         str(ksplit))

    if not os.path.exists(opdir):
        print('Create directory: %s' % opdir)
        time.sleep(3)
        os.makedirs(opdir)

    # load data
    data_path_x = os.path.join(datadir, data_name + '-x.npy')
    datas_x = np.load(data_path_x, allow_pickle=True)

    data_path_y = os.path.join(datadir, data_name + '-y.npy')
    datas_y = np.load(data_path_y, allow_pickle=True)
    datas_y = np.array(datas_y, dtype=np.int32)

    data_path_labels = os.path.join(datadir, data_name + '-labels.npy')
    datas_labels = np.load(data_path_labels, allow_pickle=True)

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    k = 0
    for train, test in kfold.split(datas_x, datas_y):
        np.save(os.path.join(opdir, data_name + '-train-x-' + str(k) + '.npy'), datas_x[train])
        np.save(os.path.join(opdir, data_name + '-train-y-' + str(k) + '.npy'), datas_y[train])

        np.save(os.path.join(opdir, data_name + '-test-x-' + str(k) + '.npy'), datas_x[test])
        np.save(os.path.join(opdir, data_name + '-test-y-' + str(k) + '.npy'), datas_y[test])

        if k == 0:
            np.save(os.path.join(opdir, data_name + '-labels.npy'), datas_labels)

        k += 1


# Load data in the following format:
# cutdatadir + '\\' + data_name + '-test-y-' + str(k) + '.npy', datas_y[test]
def load_data(data_name, cutdatadir, data_type='train', k=0):
    data_x_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-x-' + str(k) + '.npy')
    data_x = np.load(data_x_path, allow_pickle=True)

    data_y_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-y-' + str(k) + '.npy')
    data_y = np.load(data_y_path, allow_pickle=True)

    data_labels_path = os.path.join(cutdatadir, data_name + '-labels.npy')
    dictActivities = np.load(data_labels_path, allow_pickle=True).item()

    # return data_x[:, -300:], data_y, dictActivities
    return data_x, data_y, dictActivities


def load_model(dict_config):
    if dict_config['model_name'] == 'LSTM':
        model = modelsLY.get_LSTM(dict_config['input_dim'], dict_config['units'], dict_config['data_max_lenght'],
                                dict_config['no_activities'])
    elif dict_config['model_name'] == 'WCNN':
        model = modelsLY.WCNN(dict_config['no_activities'])
    else:
        print(
            'Your model name is:%s, but it does not exist. What are the names of the other models? Quickly specify the name of the model. Wow...' % (
            dict_config['model_name']))
        sys.exit(-1)

    return model


def train(data_name, k, cutdatadir, dict_config):
    data_type = 'train'
    train_x, train_y, dictActivities = load_data(data_name, cutdatadir, data_type=data_type, k=k)

    data_type = 'test'
    test_x, test_y, dictActivities = load_data(data_name, cutdatadir, data_type=data_type, k=k)

    training_generator = MyDataGenerator(train_x, train_y, dict_config['batch_size'], dict_config['dim'],
                                         dict_config['n_channels'], dict_config['is_to_categorical'],
                                         dict_config['n_classes'], dict_config['shuffle'], dict_config['is_embedding'])
    validation_generator = MyDataGenerator(test_x, test_y, dict_config['batch_size'], dict_config['dim'],
                                           dict_config['n_channels'], dict_config['is_to_categorical'],
                                           dict_config['n_classes'], dict_config['shuffle'],
                                           dict_config['is_embedding'])

    dict_config['no_activities'] = len(dictActivities)
    dict_config['input_dim'] = len(train_x)
    model = load_model(dict_config)

    if k is 0:
        model.summary()

    model = modelsLY.compileModelcus(model, dict_config['optimizer'])
    starttime = datetime.now().strftime('%Y%m%d-%H%M%S')

    # train the model
    checkpointer_dir = os.path.join(cutdatadir, 'weight')

    general.create_folder(checkpointer_dir)

    base_identifier = '%s_%s_%s_%s_%s_%s_%s' % (
        dict_config['identifier'], data_name, dict_config['model_name'], dict_config['optimizer'],
        dict_config['epochs'],
        str(dict_config['distance_int']), k)

    weight_name = os.path.join(checkpointer_dir, base_identifier + '-best.hdh5')
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
                                     callbacks=[model_checkpoint, early_stop]
                                     )
    weight_name_final_epochs = os.path.join(checkpointer_dir, base_identifier + '-final.hdh5')
    model.save(weight_name_final_epochs)
    log_file_path = os.path.join(cutdatadir, 'log')

    general.create_folder(log_file_path)

    csvFileName = os.path.join(log_file_path, base_identifier + '.csv')
    pd.read_json(json.dumps(history_LY.history), encoding="utf-8", orient='records').to_csv(csvFileName)
    print('save in: %s' % (csvFileName))

    endtime = datetime.now().strftime('%Y%m%d-%H%M%S')
    print('time: %s s' % (datetime.strptime(endtime, '%Y%m%d-%H%M%S') - datetime.strptime(starttime, '%Y%m%d-%H%M%S')))
    print('time: %s s' % (datetime.strptime(endtime, '%Y%m%d-%H%M%S') - datetime.strptime(starttime,
                                                                                          '%Y%m%d-%H%M%S')).total_seconds())
    print('average %f s' % ((datetime.strptime(endtime, '%Y%m%d-%H%M%S') - datetime.strptime(starttime,
                                                                                             '%Y%m%d-%H%M%S')).total_seconds() /
                            dict_config['epochs']))
    print('%d epoch finished' % k)


# load from weight
def validation_from_weight(data_name, weight_path, cutdatadir, k, dict_config, flag='best'):
    data_type = 'train'
    train_x, train_y, dictActivities = load_data(data_name, cutdatadir, data_type=data_type, k=k)

    data_type = 'test'
    test_x, test_y, dictActivities = load_data(data_name, cutdatadir, data_type=data_type, k=k)

    dict_config['no_activities'] = len(dictActivities)
    dict_config['input_dim'] = len(train_x)
    model = load_model(dict_config)

    base_identifier = '%s_%s_%s_%s_%s_%s_%s' % (
        dict_config['identifier'], data_name, dict_config['model_name'], dict_config['optimizer'],
        dict_config['epochs'],
        str(dict_config['distance_int']), k)

    weight_name = os.path.join(weight_path, base_identifier + '-' + flag + '.hdh5')

    model.load_weights(weight_name)
    model = modelsLY.compileModelcus(model, dict_config['optimizer'])
    print("Created model and loaded weights from file")

    print('Begin testing ...')
    scores = model.evaluate(test_x, test_y, batch_size=dict_config['batch_size'], verbose=1)

    print('%d-%s:\t%s: %.2f%%' % (k, data_name, model.metrics_names[1], scores[1] * 100))

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
def Merge(dict_config, dict_config_cus):
    return dict_config.update(dict_config_cus)


def train_val(dict_config_cus, opts):
    dict_config = {
        'model_name': '',
        'optimizer': '',
        'distance_int': '',
        'data_name': '',
        'calculation_unit': '0',

        'seed': 7,
        'want_cut_data': 'False',
        'ksplit': 3,

        'dim': (2000),
        'is_embedding': True,
        'batch_size': 64,
        'n_classes': 6,
        'n_channels': 0,
        'is_to_categorical': False,
        'shuffle': True,

        'input_dim': '',
        'units': 64,
        'data_max_lenght': 2000,
        'no_activities': '',

        'epochs': 2,
        'patience': 200,
        'fit_shuffle': False,
        'verbose': 1,

        # flag
        'identifier': '2021',
        'purpose': 'None',
        'datasetsNames': ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11'],
    }

    Merge(dict_config, dict_config_cus)

    logger = MyLogger()

    data_name = dict_config['data_name']
    print("current dataset: %s" % data_name)
    datadir = os.path.join(opts["datasets"]["base_dir"], 'ende', data_name, str(dict_config['distance_int']), 'npy')

    if dict_config['want_cut_data'] == True:
        datacut(data_name, datadir, dict_config)

    cvaccuracy_best = []
    cvaccuracy_final = []
    cvscores_best = []
    cvscores_final = []

    modelname = dict_config['model_name']

    total_dict_evaluation_best = {}
    total_dict_evaluation_final = {}

    for k in range(dict_config['ksplit']):
        cutdatadir = os.path.join(datadir, str(dict_config['ksplit']))

        train(data_name, k, cutdatadir, dict_config)

        weight_path = os.path.join(datadir, str(dict_config['ksplit']), 'weight')
        print('-' * 20, '  best  ', '-' * 20)
        dict_evaluation_best, scores_best = validation_from_weight(data_name, weight_path, cutdatadir, k,
                                                                   dict_config, flag='best')

        print('-' * 20, '  final  ', '-' * 20)
        dict_evaluation_final, scores_final = validation_from_weight(data_name, weight_path, cutdatadir, k,
                                                                     dict_config, flag='final')

        cvaccuracy_best.append(scores_best[1] * 100)
        cvscores_best.append(scores_best)
        total_dict_evaluation_best.update({str(k): dict_evaluation_best})

        cvaccuracy_final.append(scores_final[1] * 100)
        cvscores_final.append(scores_final)
        total_dict_evaluation_final.update({str(k): dict_evaluation_final})

    print('best: current database: {} \t {:.2f}% (+/- {:.2f}%)'.format(data_name, np.mean(cvaccuracy_best),
                                                                       np.std(cvaccuracy_best)))
    print('final: current database: {} \t {:.2f}% (+/- {:.2f}%)'.format(data_name, np.mean(cvaccuracy_final),
                                                                        np.std(cvaccuracy_final)))

    base_identifier = '%s_%s_%s_%s_%s_%s' % (
        dict_config['identifier'], data_name, dict_config['model_name'], dict_config['optimizer'],
        dict_config['epochs'],
        str(dict_config['distance_int']))

    csvfile_best = os.path.join(cutdatadir, 'weight', base_identifier + '_best.csv')
    with open(csvfile_best, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in cvscores_best:
            writer.writerow([",".join(str(el) for el in val)])

    csvfile_final = os.path.join(cutdatadir, 'weight', base_identifier + '_final.csv')
    with open(csvfile_final, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in cvscores_final:
            writer.writerow([",".join(str(el) for el in val)])

    evaluation_file_best = os.path.join(cutdatadir, 'weight', base_identifier + '_best.txt')
    with open(evaluation_file_best, 'w') as fw:
        for i in total_dict_evaluation_best:
            for ii in total_dict_evaluation_best[i]:
                fw.writelines(total_dict_evaluation_best[i][ii])
            fw.writelines('\n\n')

    evaluation_file_final = os.path.join(cutdatadir, 'weight', base_identifier + '_final.txt')
    with open(evaluation_file_final, 'w') as fw:
        for i in total_dict_evaluation_final:
            for ii in total_dict_evaluation_final[i]:
                fw.writelines(total_dict_evaluation_final[i][ii])
            fw.writelines('\n\n')


if __name__ == '__main__':
    os.chdir('../')
    print(os.getcwd())
    opts = general.load_config()

    dict_config_cus = {

        'model_name': 'LSTM',
        'optimizer': 'rms',
        'distance_int': '3',
        'data_name': 'kyoto7',
        'calculation_unit': '0',

        'epochs': 20,

        'identifier': '2021',
        'purpose': 'None',
        # 'datasetsNames': ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11'],
    }
    train_val(dict_config_cus, opts)
