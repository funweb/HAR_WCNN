import numpy as np
import sys
import re
import time
import os


dict_sensor2id = {'AD1-A': '0', 'AD1-B': '1', 'AD1-C': '2', 'D001': '3', 'D002': '4', 'D003': '5', 'D004': '6',
                  'D005': '7', 'D006': '8', 'D007': '9', 'D008': '10', 'D009': '11', 'D010': '12', 'D011': '13',
                  'D012': '14', 'D013': '15', 'D014': '16', 'D015': '17', 'E002': '18', 'I003': '19', 'I006': '20',
                  'I010': '21', 'I011': '22', 'I012': '23', 'L001': '24', 'L002': '25', 'L003': '26', 'L004': '27',
                  'L005': '28', 'L006': '29', 'L007': '30', 'L008': '31', 'L009': '32', 'L010': '33', 'L011': '34',
                  'L012': '35', 'L013': '36', 'M001': '37', 'M002': '38', 'M003': '39', 'M004': '40', 'M005': '41',
                  'M006': '42', 'M007': '43', 'M008': '44', 'M009': '45', 'M010': '46', 'M011': '47', 'M012': '48',
                  'M013': '49', 'M014': '50', 'M015': '51', 'M016': '52', 'M017': '53', 'M018': '54', 'M019': '55',
                  'M020': '56', 'M021': '57', 'M022': '58', 'M023': '59', 'M024': '60', 'M025': '61', 'M026': '62',
                  'M027': '63', 'M028': '64', 'M029': '65', 'M030': '66', 'M031': '67', 'M032': '68', 'M033': '69',
                  'M034': '70', 'M035': '71', 'M036': '72', 'M037': '73', 'M038': '74', 'M039': '75', 'M040': '76',
                  'M041': '77', 'M042': '78', 'M043': '79', 'M044': '80', 'M045': '81', 'M046': '82', 'M047': '83',
                  'M048': '84', 'M049': '85', 'M050': '86', 'M051': '87', 'P001': '88', 'T001': '89', 'T002': '90',
                  'T003': '91', 'T004': '92', 'T005': '93'}

kyoto_dict_first_stairs_distance = {'AD1-A': 5, 'AD1-B': 5, 'AD1-C': 5, 'D001': 5, 'D002': 7, 'D007': 7, 'D008': 6, 'D009': 6,
                              'D010': 7, 'D011': 7, 'D012': 2, 'D013': 3, 'D014': 6, 'D015': 6, 'I003': 6, 'I006': 6,
                              'I010': 6, 'I011': 3, 'I012': 3, 'M001': 2, 'M002': 3, 'M003': 4, 'M004': 5, 'M005': 5,
                              'M006': 4, 'M007': 3, 'M008': 3, 'M009': 4, 'M010': 5, 'M011': 6, 'M012': 7, 'M013': 6,
                              'M014': 6, 'M015': 4, 'M016': 5, 'M017': 6, 'M018': 5, 'M019': 4, 'M020': 5, 'M021': 3,
                              'M022': 2, 'M023': 1, 'M024': 4, 'M025': 5, 'M026': 0, 'M051': 6}

kyoto_dict_second_stairs_distance = {'D003': 3, 'D004': 3, 'D005': 4, 'D006': 7, 'E002': 4, 'M027': 0, 'M028': 1, 'M029': 2,
                               'M030': 3, 'M031': 4, 'M032': 5, 'M033': 6, 'M034': 6, 'M035': 5, 'M036': 4, 'M037': 3,
                               'M038': 4, 'M039': 5, 'M040': 6, 'M041': 7, 'M042': 3, 'M043': 2, 'M044': 3, 'M045': 5,
                               'M046': 5, 'M047': 6, 'M048': 6, 'M049': 5, 'M050': 4}





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
    matrix_value = ''
    opdir = os.path.join(os.getcwd(), 'datasets', 'distant')
    if not os.path.exists(opdir):
        print('Create directory' % (opdir))
        time.sleep(3)
        os.makedirs(opdir)

    matrix_name = os.path.join(opdir, 'kyoto-distance_matrix.txt')
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
            operate_sensors = ['AD1-A', 'AD1-B', 'AD1-C', 'M', 'D', 'I']
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
                        row_sensor = dict_value2key(dict_sensor2id, str(row_number))
                        column_sensor = dict_value2key(dict_sensor2id, str(column_number))

                        print('The row sensor to operate is: %s' % row_sensor)
                        print('The column sensor to operate is: %s' % column_sensor)

                        if re.sub('[\d]{3}', '', row_sensor) not in operate_sensors or \
                                re.sub('[\d]{3}', '', column_sensor) not in operate_sensors:
                            print(
                                'The current distance is not within the calculation range，(%s, %s) is initialized as inf' % (
                                row_sensor, column_sensor))
                            input_value = np.PINF  # np.PINF   np.NINF
                            time.sleep(0.2)

                        elif row_sensor in kyoto_dict_first_stairs_distance.keys() and column_sensor in kyoto_dict_second_stairs_distance.keys():
                            input_value = kyoto_dict_first_stairs_distance.get(
                                row_sensor) + kyoto_dict_second_stairs_distance.get(column_sensor)
                            print('Not on the same floor, (%s, %s) distance is automatically calculated as: %d' % (
                            row_sensor, column_sensor, input_value))
                            time.sleep(0.2)
                        elif column_sensor in kyoto_dict_first_stairs_distance.keys() and row_sensor in kyoto_dict_second_stairs_distance.keys():
                            input_value = kyoto_dict_first_stairs_distance.get(
                                column_sensor) + kyoto_dict_second_stairs_distance.get(row_sensor)
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
