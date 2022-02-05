import numpy as np
from queue import Queue, LifoQueue, PriorityQueue
import datetime
import re
import os
from tqdm import tqdm
import time
from collections import Counter
from operator import itemgetter
import json

from tools import general


def check_path(full_paht):
    print("data_name location: %s" % (full_paht))
    assert os.path.exists(full_paht), ('The following path does not exist, please check...\n %s' % (full_paht))


def tf(data_dir, data_name, save_file=False):
    print('\n\n\n', '*' * 20, data_name, '*' * 20)
    full_path = os.path.join(data_dir, data_name)
    check_path(full_path)

    list_activitydata = os.listdir(full_path)
    savedir = os.path.join(data_dir, '../..', 'tf', data_name)

    for str_activitydata in list_activitydata:
        print('activity：%s' % (str_activitydata))
        with open(os.path.join(full_path, str_activitydata), 'r') as fr:
            dict_data = json.load(fr)
            int_total_num_activities = 0
            int_total_num_sensors = 0
            A_type = 0
            P_type = 0
            T_type = 0
            M_type = 0
            D_type = 0
            L_type = 0
            I_type = 0
            E_type = 0
            dict_total_sensors = {}
            for activity_index in dict_data:
                int_total_num_activities += 1
                for k, v in dict_data[activity_index].items():
                    if k[0] == 'A':
                        A_type += v
                    elif k[0] == 'P':
                        P_type += v
                    elif k[0] == 'T':
                        T_type += v
                    elif k[0] == 'M':
                        M_type += v
                    elif k[0] == 'D':
                        D_type += v
                    elif k[0] == 'L':
                        L_type += v
                    elif k[0] == 'I':
                        I_type += v
                    elif k[0] == 'E':
                        E_type += v
            dict_total_sensors.update({
                'str_activitydata': str_activitydata,
                'int_total_num_activities': int_total_num_activities,
                'A_type': A_type,
                'P_type': P_type,
                'T_type': T_type,
                'M_type': M_type,
                'D_type': D_type,
                'L_type': L_type,
                'I_type': I_type,
                'E_type': E_type,
            })
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            with open(os.path.join(savedir, str_activitydata), 'w', encoding="utf-8") as fw:
                json_total_sensors = json.dumps(dict_total_sensors)
                fw.writelines(json_total_sensors)
                print('TF calculation: %s' % os.path.join(savedir, str_activitydata))
        # break
    pass


def df(data_dir, data_name, save_file=False):
    print('\n\n\n', '*' * 20, data_name, '*' * 20)
    full_path = os.path.join(data_dir, data_name)
    check_path(full_path)

    begin_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    list_activitydata = os.listdir(full_path)

    savedir = os.path.join(data_dir, '..', 'df', data_name)

    for str_activitydata in list_activitydata:
        print('activity：%s' % (str_activitydata))

        with open(os.path.join(full_path, str_activitydata), 'r', encoding="utf-8") as fr:
            lines = fr.readlines()
            int_count_activities = 0
            A_type = 0
            P_type = 0
            T_type = 0
            M_type = 0
            D_type = 0
            L_type = 0
            I_type = 0
            E_type = 0
            A_flag = False
            P_flag = False
            T_flag = False
            M_flag = False
            D_flag = False
            L_flag = False
            I_flag = False
            E_flag = False
            dict_df = {}

            for i, line in enumerate(lines):
                f_info = line.split()
                if len(f_info) < 5:
                    A_flag = False
                    P_flag = False
                    T_flag = False
                    M_flag = False
                    D_flag = False
                    L_flag = False
                    I_flag = False
                    E_flag = False
                    int_count_activities += 1
                elif f_info[3][0] == 'A' and A_flag is False:
                    A_type += 1
                    A_flag = True
                elif f_info[3][0] == 'P' and P_flag is False:
                    P_type += 1
                    P_flag = True
                elif f_info[3][0] == 'T' and T_flag is False:
                    T_type += 1
                    T_flag = True
                elif f_info[3][0] == 'M' and M_flag is False:
                    M_type += 1
                    M_flag = True
                elif f_info[3][0] == 'D' and D_flag is False:
                    D_type += 1
                    D_flag = True
                elif f_info[3][0] == 'L' and L_flag is False:
                    L_type += 1
                    L_flag = True
                elif f_info[3][0] == 'I' and I_flag is False:
                    I_type += 1
                    I_flag = True
                elif f_info[3][0] == 'E' and E_flag is False:
                    E_type += 1
                    E_flag = True

            dict_df.update({
                str(int_count_activities): {
                    'A_type': A_type,
                    'P_type': P_type,
                    'T_type': T_type,
                    'M_type': M_type,
                    'D_type': D_type,
                    'L_type': L_type,
                    'I_type': I_type,
                    'E_type': E_type,
                }
            })
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            with open(os.path.join(savedir, str_activitydata + '.json'), 'w') as fw:
                json_total_sensors = json.dumps(dict_df)
                fw.writelines(json_total_sensors)
                print('DF calculation, save in: %s' % os.path.join(savedir, str_activitydata + '.json'))

    pass


def tf_df(data_dir, data_name, save_file=False, multiply=3):
    print('\n\n\n', '*' * 20, data_name, '*' * 20)

    tf_dir = os.path.join(data_dir, '..', 'tf')
    df_dir = os.path.join(data_dir, '..', 'df')

    tf_full_path = os.path.join(tf_dir, data_name)
    check_path(tf_full_path)

    df_full_path = os.path.join(df_dir, data_name)
    check_path(df_full_path)

    list_activitydata = os.listdir(tf_full_path)

    dict_total_tf = {}
    dict_total_df = {}
    for activitydata in list_activitydata:

        # TF
        tf_d = open(os.path.join(tf_full_path, activitydata), 'r')
        dict_tf = json.load(tf_d)
        # The frequency of occurrence has been calculated in the document, so the total frequency is counted here
        count_sensors = dict_tf['A_type'] + dict_tf['P_type'] + dict_tf['T_type'] + dict_tf['M_type'] + dict_tf[
            'D_type'] + dict_tf['L_type'] + dict_tf['I_type'] + dict_tf['E_type']
        dict_total_tf.update({dict_tf['str_activitydata'].split('.')[0]:
            {
                'int_total_num_activities': dict_tf['int_total_num_activities'],
                'A_type_frequency': dict_tf['A_type'] / count_sensors,
                'P_type_frequency': dict_tf['P_type'] / count_sensors,
                'T_type_frequency': dict_tf['T_type'] / count_sensors,
                'M_type_frequency': dict_tf['M_type'] / count_sensors,
                'D_type_frequency': dict_tf['D_type'] / count_sensors,
                'L_type_frequency': dict_tf['L_type'] / count_sensors,
                'I_type_frequency': dict_tf['I_type'] / count_sensors,
                'E_type_frequency': dict_tf['E_type'] / count_sensors,
            }
        })

        # DF
        df_d = open(os.path.join(df_full_path, activitydata), 'r')
        dict_df = json.load(df_d)

        if len(dict_df) > 1:
            print('There is a problem. There was only one, and there were still a few behaviors')
            exit(-1)

        for k in dict_df:
            dict_total_df.update({dict_tf['str_activitydata'].split('.')[0]: {
                'int_total_num_activities': dict_tf['int_total_num_activities'],
                'A_type_recall': dict_df[k]['A_type'] / dict_tf['int_total_num_activities'],
                'P_type_recall': dict_df[k]['P_type'] / dict_tf['int_total_num_activities'],
                'T_type_recall': dict_df[k]['T_type'] / dict_tf['int_total_num_activities'],
                'M_type_recall': dict_df[k]['M_type'] / dict_tf['int_total_num_activities'],
                'D_type_recall': dict_df[k]['D_type'] / dict_tf['int_total_num_activities'],
                'L_type_recall': dict_df[k]['L_type'] / dict_tf['int_total_num_activities'],
                'I_type_recall': dict_df[k]['I_type'] / dict_tf['int_total_num_activities'],
                'E_type_recall': dict_df[k]['E_type'] / dict_tf['int_total_num_activities']
            }
            })

        df_d.close()
        tf_d.close()

    dict_total_tf_df = {}
    dict_total_tf_df_norm = {}  # normalization  # 归一化
    for activity in dict_total_tf:

        dict_total_tf_df.update({activity: {
            'A_tf_df': (1 if dict_total_tf[activity]['A_type_frequency'] == 0 or dict_total_tf[activity][
                'A_type_frequency'] == 1 else -np.log(dict_total_tf[activity]['A_type_frequency']) *
                                              dict_total_tf[activity]['A_type_frequency']) * pow(
                dict_total_df[activity]['A_type_recall'], multiply),
            'P_tf_df': (1 if dict_total_tf[activity]['P_type_frequency'] == 0 or dict_total_tf[activity][
                'P_type_frequency'] == 1 else -np.log(dict_total_tf[activity]['P_type_frequency']) *
                                              dict_total_tf[activity]['P_type_frequency']) * pow(
                dict_total_df[activity]['P_type_recall'], multiply),
            'T_tf_df': (1 if dict_total_tf[activity]['T_type_frequency'] == 0 or dict_total_tf[activity][
                'T_type_frequency'] == 1 else -np.log(dict_total_tf[activity]['T_type_frequency']) *
                                              dict_total_tf[activity]['T_type_frequency']) * pow(
                dict_total_df[activity]['T_type_recall'], multiply),
            'M_tf_df': (1 if dict_total_tf[activity]['M_type_frequency'] == 0 or dict_total_tf[activity][
                'M_type_frequency'] == 1 else -np.log(dict_total_tf[activity]['M_type_frequency']) *
                                              dict_total_tf[activity]['M_type_frequency']) * pow(
                dict_total_df[activity]['M_type_recall'], multiply),
            'D_tf_df': (1 if dict_total_tf[activity]['D_type_frequency'] == 0 or dict_total_tf[activity][
                'D_type_frequency'] == 1 else -np.log(dict_total_tf[activity]['D_type_frequency']) *
                                              dict_total_tf[activity]['D_type_frequency']) * pow(
                dict_total_df[activity]['D_type_recall'], multiply),
            'L_tf_df': (1 if dict_total_tf[activity]['L_type_frequency'] == 0 or dict_total_tf[activity][
                'L_type_frequency'] == 1 else -np.log(dict_total_tf[activity]['L_type_frequency']) *
                                              dict_total_tf[activity]['L_type_frequency']) * pow(
                dict_total_df[activity]['L_type_recall'], multiply),
            'I_tf_df': (1 if dict_total_tf[activity]['I_type_frequency'] == 0 or dict_total_tf[activity][
                'I_type_frequency'] == 1 else -np.log(dict_total_tf[activity]['I_type_frequency']) *
                                              dict_total_tf[activity]['I_type_frequency']) * pow(
                dict_total_df[activity]['I_type_recall'], multiply),
            'E_tf_df': (1 if dict_total_tf[activity]['E_type_frequency'] == 0 or dict_total_tf[activity][
                'E_type_frequency'] == 1 else -np.log(dict_total_tf[activity]['E_type_frequency']) *
                                              dict_total_tf[activity]['E_type_frequency']) * pow(
                dict_total_df[activity]['E_type_recall'], multiply),
        }
        })

        list_data = []
        norm_data = []
        for values in dict_total_tf_df[activity].values():
            list_data.append(values)
        for x in np.array(list_data):
            norm_data.append(
                float((x - np.min(np.array(list_data))) / (np.max(np.array(list_data) - np.min(np.array(list_data))))))

        dict_total_tf_df_norm[activity] = {
            'A_tf_df': norm_data[0],
            'P_tf_df': norm_data[1],
            'T_tf_df': norm_data[2],
            'M_tf_df': norm_data[3],
            'D_tf_df': norm_data[4],
            'L_tf_df': norm_data[5],
            'I_tf_df': norm_data[6],
            'E_tf_df': norm_data[7],
        }

    json_dict_total_tf_df = json.dumps(dict_total_tf_df)
    json_dict_total_tf_df_norm = json.dumps(dict_total_tf_df_norm)

    save_dir = os.path.join(data_dir, '..', 'tfidf')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, data_name), 'w') as fw:
        fw.writelines(json_dict_total_tf_df)
        print('save in: %s' % (os.path.join(save_dir, data_name)))

    with open(os.path.join(save_dir, data_name + '-norm'), 'w') as fw:
        fw.writelines(json_dict_total_tf_df_norm)
        print('save in %s: ' % (os.path.join(save_dir, data_name + '-norm')))

    pass


if __name__ == '__main__':
    opts = general.load_config()

    data_dir = os.path.join(opts["datasets"]["base_dir"], 'tfidf', 'log')
    data_names = ['cairo', 'kyoto7', 'kyoto8', 'kyoto11', 'milan']
    data_names = opts["datasets"]["names"]

    for data_name in data_names:
        tf(data_dir, data_name, save_file=False)  # TF calculation
        pass

    data_dir = os.path.join(opts["datasets"]["base_dir"], 'cutdata')
    for data_name in data_names:
        df(data_dir, data_name, save_file=False)  # DF calculation
        pass

    for data_name in data_names:
        multiply = opts["tfidf"]["power"]
        tf_df(data_dir, data_name, save_file=False, multiply=multiply)
        pass

    print('Finish all!')
