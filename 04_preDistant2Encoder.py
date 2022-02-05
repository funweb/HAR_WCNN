import numpy as np
import sys
import re
import time
import os
from datetime import datetime
import json
from tools import general


def statistics_sensors(data_dir, data_names, debuge=True):
    sensor_set = set()
    print('data: %s,  \t %s' % (data_dir, data_names))

    for data_name in data_names:
        list_activitydata = os.listdir(os.path.join(data_dir, data_name))

        for str_activitydata in list_activitydata:
            print('activity: %s type:%s' % (data_name, str_activitydata))
            with open(os.path.join(data_dir, data_name, str_activitydata), 'r') as fr:
                lines = fr.readlines()

                for i, line in enumerate(lines):
                    f_info = line.split()
                    if len(f_info) >= 5:
                        sensor = f_info[3]
                        sensor_set.add(sensor)

    print('The returned is the sensor list, which can be coded after numbering and adding the status code')
    list_sensor = list(sensor_set)
    list_sensor.sort()
    return list_sensor


def conver_sensor2dict(list_sensors):
    dict = {}
    for i, sensor in enumerate(list_sensors):
        dict.update({sensor: i})
    return dict


if __name__ == '__main__':
    opts = general.load_config()

    data_dir = os.path.join(opts["datasets"]["base_dir"], 'cutdata')
    data_namess = [['cairo'], ['milan'], ['kyoto7', 'kyoto8', 'kyoto11']]

    for data_names in data_namess:
        list_sensors = statistics_sensors(data_dir, data_names, debuge=True)
        sensor2dict = conver_sensor2dict(list_sensors)

        savedir = os.path.join(opts["datasets"]["base_dir"], 'distant', 'distancepredata')
        general.create_folder(savedir)

        with open(os.path.join(savedir, str(data_names) + '.json'), 'w', encoding="utf-8") as fw:
            json.dump(sensor2dict, fw)
