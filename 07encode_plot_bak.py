import numpy as np
import re
from datetime import datetime
from tqdm import tqdm
import os
import time

from tools import general


def statistics_sensors(data_dir, data_name, debuge=True):
    sensor_set = set()
    print('dataset: %s' % os.path.join(data_dir, data_name))

    list_activitydata = os.listdir(os.path.join(data_dir, data_name))

    for str_activitydata in list_activitydata:
        print('activity: %s : %s' % (data_name, str_activitydata))
        with open(os.path.join(data_dir, data_name, str_activitydata), 'r') as fr:
            lines = fr.readlines()

            for i, line in enumerate(lines):
                f_info = line.split()
                if len(f_info) >= 5:
                    sensor = f_info[3]
                    sensor_set.add(sensor)

    print('The returned is the sensor list, which can be coded after numbering and adding the status code')
    return sensor_set


# Input: sensor list
# Return: sensor code dictionary
def conver_sensor2dict(set_sensors, data_name, distant_int, opts):
    serial_number = 1
    dict_sensor2serial_number = {}
    for sensor in sorted(set_sensors):
        if sensor[0] == 'A' or sensor[0] == 'P' or sensor[0] == 'T':
            dict_sensor2serial_number.update({sensor: str(serial_number)})
            serial_number += 1
        elif sensor[0] == 'M':  # ON/OFF
            dict_sensor2serial_number.update({sensor + 'ON': str(serial_number)})
            serial_number += 1
            dict_sensor2serial_number.update({sensor + 'OFF': str(serial_number)})
            serial_number += 1
        elif sensor[0] == 'L':  # ON/OFF
            dict_sensor2serial_number.update({sensor + 'ON': str(serial_number)})
            serial_number += 1
            dict_sensor2serial_number.update({sensor + 'OFF': str(serial_number)})
            serial_number += 1
        elif sensor[0] == 'D':  # OPEN/CLOSE
            dict_sensor2serial_number.update({sensor + 'OPEN': str(serial_number)})
            serial_number += 1
            dict_sensor2serial_number.update({sensor + 'CLOSE': str(serial_number)})
            serial_number += 1
        elif sensor[0] == 'I':  # PRESENT/ABSENT
            dict_sensor2serial_number.update({sensor + 'PRESENT': str(serial_number)})
            serial_number += 1
            dict_sensor2serial_number.update({sensor + 'ABSENT': str(serial_number)})
            serial_number += 1

    opdir = os.path.join(opts["datasets"]["base_dir"], 'ende', data_name, str(distant_int))
    general.create_folder(opdir)
    opdoc = os.path.join(opdir, 'sensor2dict.npy')

    np.save(opdoc, dict_sensor2serial_number)
    print('Save in: %s \n' % (opdoc))
    return dict_sensor2serial_number


# Input: dictionary, data directory, dataset name
# Return: save the encoded file to
def encoder_sensors(read_sensor2dict, data_dir, data_name, distant_int, opts, debuge=True):
    print('\n' + '*' * 40)
    print("data_name location: %s" % (os.path.join(data_dir, data_name)))

    dict_ids = {}

    list_activitydata = os.listdir(os.path.join(data_dir, data_name))
    for str_activitydata in list_activitydata:
        list_ids = []
        with open(os.path.join(data_dir, data_name, str_activitydata), 'r') as fr:
            lines = fr.readlines()
            list_id = []
            for i, line in enumerate(lines):
                f_info = line.split()
                if len(f_info) >= 5:
                    sensor = f_info[3]
                    if sensor[0] == 'M' or sensor[0] == 'D' or sensor[0] == 'I' or sensor[
                        0] == 'L':
                        if f_info[4] in ['ON', 'OFF', 'ABSENT', 'PRESENT', 'OPEN', 'CLOSE']:
                            list_id.append(read_sensor2dict[f_info[3] + f_info[4]])
                        else:
                            print('This data is abnormal status data, which may be marked incorrectly. The data is: %s' % line)
                    elif sensor[0] == 'A' or sensor[0] == 'P' or sensor[0] == 'T':
                        list_id.append(read_sensor2dict[f_info[3]])
                    elif sensor[0] == 'E':
                        pass
                    else:
                        print('dataset：%s\tactivity:%s\trow:%d data:%s\n' % (data_name, str_activitydata, i, line))
                else:
                    list_ids.append(list_id)
                    list_id = []

        dict_ids.update({str_activitydata: list_ids})

    opdir = os.path.join(opts["datasets"]["base_dir"], 'ende', data_name, str(distant_int))
    opdoc = os.path.join(opdir, data_name + '-dict_ids.npy')
    np.save(opdoc, dict_ids)
    print('save in: %s ' % opdir)


# Input: dictionary, data directory, dataset name
# Return: the decoded file is saved to
def decoder_sensors(read_sensor2dict, data_dir, data_name, opts, debuge=True):
    data_dir = os.path.join(opts["datasets"]["base_dir"], 'ende', data_name, str(distant_int))
    print('location: %s' % data_dir)
    dict_sensors = {}
    dict_ids = np.load(os.path.join(data_dir, data_name + '-dict_ids.npy'), allow_pickle=True).item()
    for str_activities in dict_ids:
        list_sensors = []
        for list_activities in dict_ids[str_activities]:
            list_sensor = []
            for str_id in list_activities:
                try:
                    str_sensor = list(read_sensor2dict.keys())[
                        list(read_sensor2dict.values()).index(str_id)]
                except IndexError:
                    print('%s:Illegal code value exists, please check the data...' % (str_id))
                list_sensor.append(str_sensor)
            list_sensors.append(list_sensor)
        dict_sensors.update({str_activities: list_sensors})

    opdir = os.path.join(opts["datasets"]["base_dir"], 'ende', data_name, str(distant_int))
    opname = os.path.join(opdir, data_name + '-dict_sensors.npy')
    np.save(opname, dict_sensors)
    print('save in: %s ' % opdir)


# Input: code dictionary, list_ id
# Return: List sensors
def list_id2sensor(read_sensor2dict, list_ids):
    list_sensor = []
    for str_id in list_ids:
        str_sensor = list(read_sensor2dict.keys())[list(read_sensor2dict.values()).index(str_id)]
        # print(str_sensor)
        list_sensor.append(str_sensor)
    return list_sensor


if __name__ == '__main__':
    opts = general.load_config()
    for i in range(6):
        distant_int = i
        if distant_int == 0:
            data_dir = os.path.join(opts["datasets"]["base_dir"], 'cutdata')
        elif str(distant_int) in os.listdir(os.path.join(opts["datasets"]["base_dir"], 'constraintdata')):
            data_dir = os.path.join(opts["datasets"]["base_dir"], 'constraintdata', str(distant_int))
        else:
            print('The current distance limit is incorrect, please re-enter')
            exit(-1000)

        data_names = ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11']

        for data_name in data_names:

            # Setp 1: Count the sensor data and save the sensor dictionary
            set_sensors = statistics_sensors(data_dir, data_name)
            sensor2dict = conver_sensor2dict(set_sensors, data_name, distant_int, opts)

            encoder_dict_path = os.path.join(opts["datasets"]["base_dir"], 'ende', data_name, str(distant_int), 'sensor2dict.npy')
            read_sensor2dict = np.load(encoder_dict_path, allow_pickle=True).item()

            # Setp 2: Encode data
            encoder_sensors(read_sensor2dict, data_dir, data_name, distant_int, opts, debuge=True)

            # Setp 3: Decode data(Optional)
            decoder_sensors(read_sensor2dict, data_dir, data_name, opts, debuge=True)
            # Decoded data test view
            # ids_dir = r'D:\Anaconda3\workspace\py36TF\dpcnnfdal\processdata\09-encodersensors'
            # ids_name = r'kyoto7-dict_sensors.npy'
            # dict_ids = np.load(ids_dir + '\\' + ids_name, allow_pickle=True).item()
            # for activity_name in dict_ids:
            #     for list_sensors in dict_ids[activity_name]:
            #         print(list_sensors)

            # Test and decode a single piece of data
            ids_dir = os.path.join(opts["datasets"]["base_dir"], 'ende', data_name, str(distant_int))
            ids_name = data_name + r'-dict_ids.npy'
            dict_ids = np.load(os.path.join(ids_dir, ids_name), allow_pickle=True).item()
            for str_activity in dict_ids:
                for list_ids in dict_ids[str_activity]:
                    # list_sensor = list_id2sensor(read_sensor2dict, list_ids)
                    # print(list_sensor)
                    pass

    print('success, all finished!')
