import numpy as np
import re
from datetime import datetime
from tqdm import tqdm
import os
import time

from tools import general


@general.clock
def constraint_data(data_dir, data_name, distance_matrix_path, My_constrain_distant, opts, debuge=True):
    list_non_distant_constraint = []
    list_distant_constraint = []

    if data_name == 'cairo':
        list_non_distant_constraint = ['T']
        list_distant_constraint = ['M']
    elif data_name == 'kyoto7':
        list_distant_constraint = ['M', 'D', 'A']
    elif data_name == 'kyoto8':
        list_non_distant_constraint = ['P', 'T']
        list_distant_constraint = ['M', 'D']
    elif data_name == 'kyoto11':
        list_non_distant_constraint = ['P']
        list_distant_constraint = ['M', 'D']
    elif data_name == 'milan':
        list_non_distant_constraint = ['T']
        list_distant_constraint = ['M', 'D']
    else:
        print('The constraint you are preparing does not exist in the scheduled database, please check...')
        exit(-1000)

    list_OFF_state = ['OFF', 'CLOSE', 'ABSENT']
    list_ON_state = ['ON', 'OPEN', 'PRESENT']

    distance_int_normal = My_constrain_distant['normal']
    distance_int_other = My_constrain_distant['other']
    distance_int = 0

    short_len = opts["datasets"]["short_len"]

    print('dataset：%s,  \t %s' % (data_dir, data_name))

    list_activitydata = os.listdir(os.path.join(data_dir, data_name))
    dictlist_activities = {}
    dictlist_log = {}
    int_original = 0
    int_change = 0

    # Load distance constraint matrix
    distance_matrix = np.loadtxt(distance_matrix_path)

    for str_activitydata in list_activitydata:
        print('atcivity: %s' % (str_activitydata))
        if str_activitydata == 'other':
            distance_int = distance_int_other
        else:
            distance_int = distance_int_normal
        with open(os.path.join(data_dir, data_name, str_activitydata), 'r') as fr:
            lines = fr.readlines()
            list_activities_lines = []

            list_len_sensors = []
            new_len = 0
            static_short = 0

            anterior_index = 0

            anterior_sensor_on = ''
            for i, line in enumerate(lines):
                f_info = line.split()
                if len(f_info) < 5:
                    int_change += new_len

                    anterior_sensor_on = ''
                    list_activities_lines.append(line)

                    list_len_sensors.append(
                        str(i - anterior_index) + ':' + str(new_len) + '\t')
                    if new_len < short_len:
                        static_short += 1
                    new_len = 0
                    anterior_index = i
                else:
                    int_original += 1

                    if f_info[3][0] in list_non_distant_constraint:
                        list_activities_lines.append(line)
                        new_len += 1

                    use_sensor = ["AD1-A", "AD1-B", "AD1-C", "D001", "D002", "D003", "D004", "D005", "D006", "D007",
                                  "D008", "D009", "D010", "D011", "D012", "D013", "D014", "D015", "I003",
                                  "I006", "I010", "I011", "I012", "M001", "M002", "M003", "M004", "M005", "M006",
                                  "M007", "M008", "M009", "M010", "M011", "M012", "M013", "M014", "M015", "M016",
                                  "M017", "M018", "M019", "M020", "M021", "M022", "M023", "M024", "M025", "M026",
                                  "M027", "M028", "M029", "M030", "M031", "M032", "M033", "M034", "M035", "M036",
                                  "M037", "M038", "M039", "M040", "M041", "M042", "M043", "M044", "M045", "M046",
                                  "M047", "M048", "M049", "M050", "M051"]

                    if f_info[3][0] in list_distant_constraint:
                        if f_info[3][0] == 'I':
                            print('here')

                        if anterior_sensor_on != '' and (f_info[4] in list_OFF_state):

                            current_sensor_on = f_info[3]
                            try:
                                anterior_sensor_on_id = int(distantmatrix.dict_sensor2id[anterior_sensor_on])
                                current_sensor_on_id = int(distantmatrix.dict_sensor2id[current_sensor_on])
                            except IndexError:
                                print('Current sensor not include in dict: %s / %s '
                                      % (anterior_sensor_on_id, current_sensor_on_id))

                            distance_sensor = distance_matrix[anterior_sensor_on_id, current_sensor_on_id]

                            if distance_sensor < distance_int:
                                list_activities_lines.append(line)
                                new_len += 1

                        if f_info[4] in list_ON_state or f_info[3] in ['AD1-A', 'AD1-B', 'AD1-C']:
                            if anterior_sensor_on == '':
                                anterior_sensor_on = f_info[3]
                                list_activities_lines.append(line)
                                new_len += 1
                                continue

                            current_sensor_on = f_info[3]
                            try:
                                anterior_sensor_on_id = int(distantmatrix.dict_sensor2id[anterior_sensor_on])
                                current_sensor_on_id = int(distantmatrix.dict_sensor2id[current_sensor_on])
                            except IndexError:
                                print('Current sensor not include in dict:%s / %s '
                                      % (anterior_sensor_on_id, current_sensor_on_id))

                            distance_sensor = distance_matrix[anterior_sensor_on_id, current_sensor_on_id]
                            if distance_sensor < distance_int:
                                anterior_sensor_on = current_sensor_on
                                list_activities_lines.append(line)
                                new_len += 1

        dictlist_activities.update({str_activitydata: list_activities_lines})
        dictlist_log.update({str_activitydata: list_len_sensors})
        print('There are %d data lengths shorter than %d \n' % (static_short, short_len))

    if debuge == False:
        file_dir = os.path.join(data_dir, '..', 'constraintdata', str(distance_int_normal))
        subdir = os.path.join(file_dir, data_name)
        print('The path where the current operation is stored is: %s' % (subdir))
        general.create_folder(subdir)

        for activity_name in dictlist_activities:
            with open(os.path.join(subdir, activity_name), 'w') as fw:
                fw.writelines(dictlist_activities[activity_name])

        # -----------------------------   log   ----------------------------- #
        log_dir = os.path.join(file_dir, 'log')
        general.create_folder(log_dir)

        log_name = data_name + '-log'
        with open(os.path.join(log_dir, log_name), 'w') as fw:
            for activity_name in dictlist_activities:
                fw.writelines('\n\n%s:  %d 个\n' % (activity_name, len(dictlist_log[activity_name])))
                fw.writelines(dictlist_log[activity_name])

            fw.writelines('\n\nold amount: %d \nnew amount: %d \nchange: %f \n' % (
            int_original, int_change, (1 - int_change / int_original)))

    print('success')

    pass


if __name__ == '__main__':
    opts = general.load_config()

    for i in range(6):
        My_constrain_distant = {}
        My_constrain_distant['normal'] = i
        My_constrain_distant['other'] = 999
        if i==0:
            My_constrain_distant['normal'] = 999

        data_dir = os.path.join(opts["datasets"]["base_dir"], 'cutdata')
        data_names = ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11']
        data_names = opts["datasets"]["names"]

        for data_name in data_names:
            if data_name == 'cairo':
                import distantmatrix_cairo as distantmatrix
                distance_matrix_path = os.path.join(opts["datasets"]["base_dir"], 'distant', 'cairo-distance_matrix.txt')
            elif data_name == 'milan':
                import distantmatrix_milan as distantmatrix
                distance_matrix_path = os.path.join(opts["datasets"]["base_dir"], 'distant', 'milan-distance_matrix.txt')
            else:
                import distantmatrix_kyoto as distantmatrix
                distance_matrix_path = os.path.join(opts["datasets"]["base_dir"], 'distant', 'kyoto-distance_matrix.txt')

            constraint_data(data_dir, data_name, distance_matrix_path, My_constrain_distant, opts, debuge=False)

    print('finish all')
