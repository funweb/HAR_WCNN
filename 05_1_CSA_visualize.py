import numpy as np
from queue import Queue, LifoQueue, PriorityQueue
import datetime
import re
import os
from tqdm import tqdm
import time
from collections import Counter
from tools import general


def check_path(full_paht):
    print("data_name location: %s" % (full_paht))
    assert os.path.exists(full_paht), ('The following path does not exist, please check...\n %s' % (full_paht))


def validate_date(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


def sort_date(list_data):
    # list_data = ['2010-11-23', '1989-3-7', '2010-1-5', '978-12-1', '2010-2-4']
    patt = '(\d+)-(\d+)-(\d+)'
    # 交换排序
    for i in range(len(list_data) - 1):
        for x in range(i + 1, len(list_data)):
            j = 1
            while j < 4:
                lower = re.match(patt, list_data[i]).group(j)
                upper = re.match(patt, list_data[x]).group(j)
                # print lower,upper
                if int(lower) < int(upper):
                    j = 4
                elif int(lower) == int(upper):
                    j += 1
                else:
                    list_data[i], list_data[x] = list_data[x], list_data[i]
                    j = 4
    return list_data


def calculate_date_interval(counter_date, interval=1):
    dict_date = dict(counter_date)
    list_sort_data = list(dict_date)
    for i in range(len(list_sort_data) - 2):
        prior_date = list_sort_data[i]
        later_date = list_sort_data[i + 1]
        if interval < (datetime.datetime.strptime(later_date, '%Y-%m-%d') - datetime.datetime.strptime(prior_date,
                                                                                                       '%Y-%m-%d')).days:
            print('The time is not continuous within the specified interval:%d, please check: %s -- %s' % (
                interval, prior_date, later_date))

    list_sort_data = sorted(list_sort_data)
    list_time_is = []
    begin_date = list_sort_data[0]
    end_date = list_sort_data[-1]
    current_through_date = begin_date
    while (datetime.datetime.strptime(end_date, '%Y-%m-%d') - datetime.datetime.strptime(current_through_date,
                                                                                         '%Y-%m-%d')).days >= 0:
        if current_through_date in list_sort_data:
            list_time_is.append(dict_date[current_through_date])
        else:
            list_time_is.append(0)
        current_through_date = datetime.datetime.strptime(current_through_date, '%Y-%m-%d') + datetime.timedelta(days=1)
        current_through_date = current_through_date.strftime('%Y-%m-%d')

    int_weekly_begin_date = int(datetime.datetime.strptime(begin_date, '%Y-%m-%d').strftime("%w"))
    print('\n\nbegin: %s\nend: %s\n\n' % (begin_date, end_date))
    print('\tSun.\t\tMon.\t\tTues.\t\tWed.\t\tThur.\t\tFri.\t\tSat.')
    print('\t\t' * int_weekly_begin_date, end='')
    for i, v in enumerate(list_time_is):
        print('%6d\t' % (v), end='')
        if (i + int_weekly_begin_date) % 7 == 6:
            print('\n', end='')


def statistical_date(data_dir, data_name, save_file=False):
    print('\n\n\n', '*' * 20, data_name, '*' * 20)
    full_path = os.path.join(data_dir, data_name)
    check_path(full_path)

    begin_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    with open(full_path, 'r') as fr:
        datas = fr.readlines()
        set_date = set()
        list_date = []

        for i, line in enumerate(datas):
            data = line.split()
            assert not len(data) < 4, ('The data in this row is incomplete, please check: %d' % (i))
            validate_date(data[0])
            list_date.append(data[0])
            set_date.add(data[0])

    print('There are %d days of data in total.' % (len(set_date)))
    counter_date = Counter(list_date)
    print('Number of Statistics: \n%s' % (counter_date))
    counter_date = sorted(counter_date.items())
    print('Sort date information: \n%s' % (counter_date))

    calculate_date_interval(counter_date, interval=1)

    print('\ndone')
    pass


if __name__ == '__main__':

    opts = general.load_config()

    data_dir = os.path.join(opts["datasets"]["base_dir"], 'repairdata')

    data_names = ['repair-kyoto7', 'repair-kyoto8', 'repair-kyoto11']
    data_names = ['repair-milan', 'repair-cairo', 'repair-kyoto7', 'repair-kyoto8', 'repair-kyoto11']

    for data_name in data_names:
        statistical_date(data_dir, data_name, save_file=False)

    print("done...")
