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


def validate_date(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


def statistical_date(data_dir, data_name, save_file=False):
    print('\n\n\n', '*' * 20, data_name, '*' * 20)
    full_path = os.path.join(data_dir, data_name)
    check_path(full_path)

    list_activitydata = os.listdir(full_path)

    for str_activitydata in list_activitydata:
        print('activity：%s' % (str_activitydata))

        with open(os.path.join(full_path, str_activitydata), 'r') as fr:
            lines = fr.readlines()
            list_activities_lines = []
            list_sensor = []
            list_sensors = []
            counter_sensors = Counter()
            dict_sensors = {}
            j = 0
            str_begin_time = ''
            str_end_time = ''
            for i, line in enumerate(lines):
                f_info = line.split()
                if len(f_info) < 5:
                    counter_sensors = Counter(list_sensor)
                    j += 1
                    order_sorted_counter = sorted(counter_sensors.items(), key=itemgetter(1),
                                                  reverse=True)
                    end_data = lines[i - 1].split()
                    str_end_time = end_data[1] + ' ' + end_data[2]

                    interval_seconds = (datetime.datetime.strptime(str_end_time, '%Y-%m-%d %H:%M:%S.%f') -
                                        datetime.datetime.strptime(str_begin_time,
                                                                   '%Y-%m-%d %H:%M:%S.%f')).total_seconds()

                    begin_hour = datetime.datetime.strptime(str_begin_time, '%Y-%m-%d %H:%M:%S.%f').strftime("%H")
                    end_hour = datetime.datetime.strptime(str_end_time, '%Y-%m-%d %H:%M:%S.%f').strftime("%H")

                    dict_key = str(j) + ':' + str_begin_time + '--' + str_end_time + '__' \
                               + begin_hour + ':' + end_hour + '--' + str(round(interval_seconds))
                    dict_sensors.update({dict_key: dict(order_sorted_counter)})
                    list_sensor = []
                    str_begin_time = ''
                else:
                    if str_begin_time == '':
                        str_begin_time = f_info[1] + ' ' + f_info[2]

                    if f_info[3][0] == 'A' or f_info[3][0] == 'P':
                        list_sensor.append(f_info[3])
                    elif f_info[3][0] == 'T':
                        list_sensor.append(f_info[3])
                    elif f_info[3][0] == 'M' or f_info[3][0] == 'D' or f_info[3][0] == 'L' or f_info[3][0] == 'I' or \
                            f_info[3][0] == 'E':
                        list_sensor.append(f_info[3])

            log_path = os.path.join(data_dir, "..", 'tfidf', 'log', data_name)
            general.create_folder(log_path)

            with open(os.path.join(log_path, str_activitydata + '.json'), 'w', encoding="utf-8") as fw:
                json_sensors = json.dumps(dict_sensors)

                fw.writelines(json_sensors)

    pass


if __name__ == '__main__':
    opts = general.load_config()
    data_dir = os.path.join(opts["datasets"]["base_dir"], 'cutdata')

    data_names = ['cairo', 'kyoto7', 'kyoto8', 'kyoto11', 'milan']
    data_names = opts["datasets"]["names"]

    for data_name in data_names:
        statistical_date(data_dir, data_name, save_file=True)

    print('Statistical analysis of data... \n   Finish!')
