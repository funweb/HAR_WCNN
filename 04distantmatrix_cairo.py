import numpy as np
import sys
import re
import time
import os


dict_sensor2id = {"M001": 0, "M002": 1, "M003": 2, "M004": 3, "M005": 4, "M006": 5, "M007": 6, "M008": 7, "M009": 8, "M010": 9, "M011": 10, "M012": 11, "M013": 12, "M014": 13, "M015": 14, "M016": 15, "M017": 16, "M018": 17, "M019": 18, "M020": 19, "M021": 20, "M022": 21, "M023": 22, "M024": 23, "M025": 24, "M026": 25, "M027": 26, "T001": 27, "T002": 28, "T003": 29, "T004": 30, "T005": 31}

cairo_dict_first_stairs_distance = {"M001": 2, "M002": 2, "M003": 1, "M004": 3, "M005": 4, "M006": 3, "M007": 4, "M008": 4, "M009": 4, "M010": 0}

cairo_dict_second_stairs_distance = {"M011": 0, "M012": 1, "M013": 1, "M014": 0, "M015": 0, "M016": 1, "M017": 2, "M018": 3, "M019": 3, "M020": 4, "M021": 2, "M022": 2, "M023": 1, "M024": 2}

cairo_dict_third_stairs_distance = {"M025": 0, "M026": 1, "M027": 1}


def dict_key2value(dict, key):
    try:
        return dict[key]
    except:
        print('Key input error, not found!')


def dict_value2key(dict, value):
    try:
        return list(dict.keys())[list(dict.values()).index(value)]
    except:
        print('Value input error, not found!')


def init_matrix(shape=(94, 94)):
    distance_matrix = np.zeros(shape)
    return distance_matrix


def save_matrix(matrix_name, matrix_value, path='./', delimiter='\t', encoding=None):
    np.savetxt(matrix_name, matrix_value, fmt='%.0f', delimiter='\t')
    return 'OK'


if __name__ == '__main__':
    dataset_name = 'cairo'
    matrix_value = ''
    opdir = os.path.join(os.getcwd(), 'datasets', 'distant')
    matrix_name = os.path.join(opdir, dataset_name + '-distance_matrix.txt')

    assert (os.path.exists(matrix_name),
            'Please confirm that the file distance matrix file exists first! Otherwise, please create it first and initialize an empty one.')

    operate_select = ''  # 0: initialize 1: view 2: repair 3: save

    while operate_select != 10:  # Setting 10 is because it requires pressing two keys at a long distance to avoid wrong pressing
        print(operate_select)
        operate_select = int(input('\n''\nPlease enter the action you want to perform: \n%s\n%s\n%s\n%s\n%s\n' %
                                   ('0\tinitialization',
                                    '1\tView data',
                                    '2\tRepair and improve data',
                                    '3\tmodel save',
                                    '10\texit')
                                   ))
        if operate_select == 0:
            print('Initialization operation, all zeroing processing...')
            matrix_value = init_matrix(shape=(len(dict_sensor2id), len(dict_sensor2id)))
            print('shape: %s' % str(matrix_value.shape))
            print(save_matrix(matrix_name, matrix_value))  # The default is to save as an integer
        if operate_select == 1:
            matrix_value = np.loadtxt(matrix_name)
            print('shape: %s' % str((matrix_value.shape)))
            print(dict_sensor2id.keys())

            sensor_one = input('Please enter the name of the sensor to view 1：')
            sensor_two = input('Please enter the name of the sensor to view 2：')
            sensor_one_value = int(dict_key2value(dict_sensor2id, sensor_one))
            sensor_two_value = int(dict_key2value(dict_sensor2id, sensor_two))
            print('(%s,%s)=>(%d,%d) distance: %d' %
                  (sensor_one, sensor_two,
                   sensor_one_value, sensor_two_value,
                   matrix_value[sensor_one_value, sensor_two_value]))
            print('location: \n%s' % (matrix_name))
        if operate_select == 2:
            operate_sensors = ['AD1-A', 'AD1-B', 'AD1-C', 'M', 'D']
            matrix_value = np.loadtxt(matrix_name)
            copy_matrix_value = matrix_value.copy()
            s = matrix_value.shape[0]

            row_number = -1
            for i in range(1, int(s)):
                if float(copy_matrix_value[i][0]) == 0:
                    row_number = i
                    break
            if row_number == -1:
                print('You have completed all the data')
            else:
                print('Suggestion: last operation to% d lines' % (row_number))

            print('Generally, it is the lower triangular matrix of operation...\n')
            row_number = int(input('Please enter the line you want to modify from? Cannot be greater than %s' % (s)))
            column_number = int(input('Please enter the column you want to modify from?'))
            if row_number > int(s) - 1 or column_number >= row_number:
                print('There is a problem with your input.')
                sys.exit(999)
            keypress = ''
            while keypress != 'Q':
                try:
                    if int(s) > int(row_number) > int(column_number):
                        row_sensor = dict_value2key(dict_sensor2id, int(row_number))
                        column_sensor = dict_value2key(dict_sensor2id, int(column_number))

                        print('The row sensor to operate is: %s' % row_sensor)
                        print('The column sensor to operate is: %s' % column_sensor)
                        if re.sub('[\d]{3}', '', row_sensor) not in operate_sensors or \
                                re.sub('[\d]{3}', '', column_sensor) not in operate_sensors:
                            print(
                                'The current distance is not within the calculation range，(%s, %s) is initialized as inf' % (
                                row_sensor, column_sensor))
                            input_value = np.PINF  # np.PINF   np.NINF
                            time.sleep(0.2)

                        elif row_sensor in cairo_dict_first_stairs_distance.keys() and column_sensor in cairo_dict_second_stairs_distance.keys():
                            input_value = cairo_dict_first_stairs_distance.get(
                                row_sensor) + cairo_dict_second_stairs_distance.get(column_sensor)
                            print('Not on the same floor, (%s, %s) distance is automatically calculated as: %d' % (
                            row_sensor, column_sensor, input_value))
                            time.sleep(0.2)

                        elif row_sensor in cairo_dict_first_stairs_distance.keys() and column_sensor in cairo_dict_third_stairs_distance.keys():
                            input_value = cairo_dict_first_stairs_distance.get(
                                row_sensor) + cairo_dict_third_stairs_distance.get(column_sensor) + 2
                            print('Not on the same floor, (%s, %s) distance is automatically calculated as: %d' % (
                            row_sensor, column_sensor, input_value))
                            time.sleep(0.2)

                        elif row_sensor in cairo_dict_second_stairs_distance.keys() and column_sensor in cairo_dict_first_stairs_distance.keys():
                            input_value = cairo_dict_second_stairs_distance.get(
                                row_sensor) + cairo_dict_first_stairs_distance.get(column_sensor)
                            print('Not on the same floor, (%s, %s) distance is automatically calculated as: %d' % (
                            row_sensor, column_sensor, input_value))
                            time.sleep(0.2)

                        elif row_sensor in cairo_dict_second_stairs_distance.keys() and column_sensor in cairo_dict_third_stairs_distance.keys():
                            input_value = cairo_dict_second_stairs_distance.get(
                                row_sensor) + cairo_dict_third_stairs_distance.get(column_sensor)
                            print('Not on the same floor, (%s, %s) distance is automatically calculated as: %d' % (
                            row_sensor, column_sensor, input_value))
                            time.sleep(0.2)

                        elif row_sensor in cairo_dict_third_stairs_distance.keys() and column_sensor in cairo_dict_first_stairs_distance.keys():
                            input_value = cairo_dict_third_stairs_distance.get(
                                row_sensor) + cairo_dict_first_stairs_distance.get(column_sensor) + 2
                            print('Not on the same floor, (%s, %s) distance is automatically calculated as: %d' % (
                            row_sensor, column_sensor, input_value))
                            time.sleep(0.2)

                        elif row_sensor in cairo_dict_third_stairs_distance.keys() and column_sensor in cairo_dict_second_stairs_distance.keys():
                            input_value = cairo_dict_third_stairs_distance.get(
                                row_sensor) + cairo_dict_second_stairs_distance.get(column_sensor)
                            print('Not on the same floor, (%s, %s) distance is automatically calculated as: %d' % (
                            row_sensor, column_sensor, input_value))
                            time.sleep(0.2)

                        else:
                            input_value = input('current value：%d\t row:%d column:%d\t\t[(%s, %s)]\t：'
                                                % (copy_matrix_value[row_number][column_number],
                                                   row_number, column_number,
                                                   row_sensor,
                                                   column_sensor))

                    copy_matrix_value[row_number][column_number] = input_value
                    column_number += 1
                    if column_number == row_number:
                        column_number = 0
                        row_number += 1
                        print('-' * 30)
                        print('\nThis line is over and will go to the next new line, sensor is：%s ' %
                              (dict_value2key(dict_sensor2id, int(row_number))))
                    if row_number > s - 1:
                        break
                except ValueError:
                    keypress = input_value
                    print('exit \'Q\'')
                    print('What you entered is %s' % (keypress))
                finally:
                    pass
            confirm_edit = input('Are you sure you want to save the new matrix, Y/ ')
            if confirm_edit:
                print('Saving...')
                from tools.uppertriangular2symmetry import low2sym

                matrix_value = low2sym(copy_matrix_value)
                print(save_matrix(matrix_name, matrix_value))
            else:
                print('You didnot save, that is, you didnot do anything...')

        elif operate_select == 3:
            save_matrix(matrix_name, matrix_value, path='./')
        pass

    print('exit')

