import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import time
import json
from tools import general


@general.clock
def cutdataclass(data_dir, data_name, save_file=False):
    print('\n' + '*' * 20)
    op_data = os.path.join(data_dir, 'repair-' + data_name)
    print("data_name location: %s" % (op_data))

    with open(op_data, 'r', encoding="utf-8") as fr:
        datas = fr.readlines()

        sensor_data = ''
        list_data = []
        class_data = {}

        # log
        begin_index = ''
        end_index = ''
        pair_index = {}
        dic_index = {}

        # load data
        data_dir = os.path.join(data_dir, '..', 'static')
        data_static = os.path.join(data_dir, 'activities.json')

        with open(data_static, 'r', encoding="utf-8") as fr2:
            dictclass = json.load(fr2)

        for classtype in dictclass[data_name].keys():
            flag = False

            list_data = []

            # log
            begin_index = ''
            end_index = ''
            pair_index = {}
            logindex = 0

            for i, line in enumerate(datas):
                try:
                    if flag:
                        list_data.append(line)
                    if classtype in line.split():
                        if 'begin' in line:
                            list_data.append(line)

                            if flag == True:
                                print('This indicates that the previous activity did not end normally...')

                            flag = True

                            # log
                            begin_index = i

                        if 'end' in line:
                            if flag == False:
                                print('Borrow previous start and end(%d, %d) ' % (begin_index, end_index))
                                print('Data from：%s, row: %d \n\n' % (classtype, i))
                            flag = False

                            # log
                            end_index = i
                            pair_index.update({str(logindex): str(begin_index) + ',' + str(end_index)})
                            if (end_index - begin_index) < 4:
                                print('What iss the problem? Why is it so short. The length is: %d' % (end_index - begin_index))
                                print('Data from：%s, row: %d \n\n' % (classtype, i))
                            logindex = logindex + 1

                except IndexError:
                    print(i, line)

            class_data.update({classtype: list_data})

            # log
            dic_index.update({classtype: pair_index})

            if flag == True:  # I just want to do a verification to avoid some data having a beginning and no end
                print('You donot have a normal end, brother...')

    # 是否保存文件
    if save_file == True:
        file_dir = os.path.join(data_dir, '..', 'temp', 'cutedata')
        subdir = os.path.join(file_dir, data_name)
        general.create_folder(subdir)

        for activity_name in class_data:
            with open(os.path.join(subdir, activity_name), 'w', encoding="utf-8") as fw:
                fw.writelines(class_data[activity_name])

        with open(data_static, 'w', encoding="utf-8") as fw:
            for activity_name in dic_index:
                dictclass[data_name].update({activity_name: dic_index[activity_name]})
            json.dump(dictclass, fw)

        dataindex_dir = os.path.join(data_dir, '..', 'dataindex')
        # log_dir = file_dir + 'log\\'
        general.create_folder(dataindex_dir)

        log_name = data_name + '-log'
        with open(os.path.join(dataindex_dir, log_name), 'w', encoding="utf-8") as fw:
            # log
            for activity_name in dic_index:
                fw.writelines('\n%s:\n' % (activity_name))
                fw.writelines(str(dic_index[activity_name]))
        print('The operation is completed. Now the file has been saved to: %s' % (subdir))


@general.clock
def verifyindex(data_dir, data_name, save_file=False):

    static_data_dir = os.path.join(data_dir, '..', 'static')
    data_static = os.path.join(static_data_dir, 'activities.json')
    with open(data_static, 'r', encoding="utf-8") as fr:
        dictclass = json.load(fr)

    with open(os.path.join(data_dir, 'repair-' + data_name), 'r', encoding="utf-8") as f:
        datasets = f.readlines()
        data_index = dictclass[data_name]
        for activity_type in data_index:
            if activity_type == 'other':
                continue
            print('dataset: %s _ %s ...' % (data_name, activity_type))
            for str_index in data_index[activity_type]:
                index_array = np.array(data_index[activity_type][str_index].split(','), dtype=int)

                try:
                    begin_activity_type = datasets[index_array[0]].split()
                    if activity_type != begin_activity_type[4] or 'begin' != begin_activity_type[5] or len(
                            begin_activity_type) != 6:
                        print('\nWarning at the end of the sequence：%s: %s \n' % (str_index, index_array))

                    end_activity_type = datasets[index_array[1]].split()
                    if activity_type != end_activity_type[4] or 'end' != end_activity_type[5] or len(
                            end_activity_type) != 6:
                        print('\nWarning at the begin of the sequence：%s: %s\n' % (str_index, index_array))
                except IndexError:
                    print('There is a data problem. Please check the start and end index array values...')
                    print(index_array)
    pass


def saveother_index(data_dir, data_name, save_file=False):

    static_data_dir = os.path.join(data_dir, '..', 'static')
    data_static = os.path.join(static_data_dir, 'activities.json')
    with open(data_static, 'r', encoding="utf-8") as fr:
        dictclass = json.load(fr)

    with open(os.path.join(data_dir, 'repair-' + data_name), 'r', encoding="utf-8") as fr:
        datasets = fr.readlines()
        int_index_set = set()
        dict_other_index = {}
        other_activity_index = 0

        data_index = dictclass[data_name]
        for activity_type in data_index:
            if activity_type == 'other':
                continue
                pass
            print('dataset: %s _ %s ...' % (data_name, activity_type))
            for str_index in data_index[activity_type]:
                index_array = np.array(data_index[activity_type][str_index].split(','), dtype=int)
                for i in range(index_array[0], index_array[1] + 1):
                    int_index_set.add(i)

        for i in range(0, len(datasets)):
            if i == 0 and i not in int_index_set:
                other_index_begin = i
            elif i - 1 in int_index_set and i not in int_index_set:
                other_index_begin = i
            elif i not in int_index_set and i + 1 in int_index_set:
                other_index_end = i
                dict_other_index.update(
                    {str(other_activity_index): str(other_index_begin) + ',' + str(other_index_end)})

                other_activity_index += 1
                other_index_begin = float('Inf')
            elif i == len(datasets) - 1 and other_index_begin != float('Inf'):
                dict_other_index.update(
                    {str(other_activity_index): str(other_index_begin) + ',' + str(len(datasets) - 1)})

        with open(data_static, 'w', encoding="utf-8") as fw:
            dictclass[data_name].update({'other': dict_other_index})
            json.dump(dictclass, fw)


@general.clock
def savecute_data(data_dir, data_name, save_file=False):
    static_data_dir = os.path.join(data_dir, '..', 'static')
    data_static = os.path.join(static_data_dir, 'activities.json')
    with open(data_static, 'r', encoding="utf-8") as fr:
        dictclass = json.load(fr)

    with open(os.path.join(data_dir, 'repair-' + data_name), 'r', encoding="utf-8") as f:
        datasets = f.readlines()

        data_index = dictclass[data_name]
        for activity_type in data_index:
            lines = []
            print('datasets: %s _ %s...' % (data_name, activity_type))
            for str_index in data_index[activity_type]:
                index_array = np.array(data_index[activity_type][str_index].split(','), dtype=int)

                for i in range(index_array[0], index_array[1] + 1):
                    lines.append(str(i).zfill(7) + '\t' + datasets[i])
                lines.append('\n')

            activity_dir = os.path.join(data_dir, "..", 'cutdata', data_name)
            general.create_folder(activity_dir)
            with open(os.path.join(activity_dir, activity_type), 'w', encoding="utf-8") as fw:
                fw.writelines(lines)


if __name__ == '__main__':
    opts = general.load_config()

    data_dir = os.path.join(opts["datasets"]["base_dir"], 'repairdata')

    # data_names = ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11']
    data_names = opts["datasets"]["names"]

    # Step 1
    for data_name in data_names:
        cutdataclass(data_dir, data_name, save_file=True)
        pass
    print('Now the first step is to: \n\tthe index value of the source data is obtained. For the next step, it needs to be stored in the corresponding file of config. PS: there is no other type of data yet')

    # Step 2
    for data_name in data_names:
        verifyindex(data_dir, data_name, save_file=True)
        pass
    print('\n\n第二步：\n\tIf there is no problem in the program, it means there is no problem, and it is guaranteed that each line has only the beginning or end of an action')
    print('This second step is only for verification and has no substantive effect.')

    # Step 3
    for data_name in data_names:
        saveother_index(data_dir, data_name, save_file=True)
        pass
    print('\n\nStep 3：\n\tOther index data is generated to the folder and manually copied to config / index + data_ In the name file, add the index of other')


    # The first step is to log, get the start and end of all activity data, and cut the data into corresponding folders
    # The second step is just verification. If there is no problem data, it indicates that the index value segmentation is correct
    # The third step is to find out the active index value of other according to the active index value, and then save the corresponding config folder (including the log in the first step and the otherindex in the third step)
    # Now in step 4, correctly segment the corresponding data into the folder according to the config index value
    for data_name in data_names:
        savecute_data(data_dir, data_name, save_file=True)
        pass

    print('Now that the data has been segmented, we will move to the next step to segment the data according to the distance!')
