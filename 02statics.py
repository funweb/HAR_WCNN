from tqdm import tqdm
from collections import Counter
import os
import json
from tools import general


@general.clock
def staticclass(data_dir, data_name, save_file=False):
    op_data = os.path.join(data_dir, 'repair-' + data_name)
    print('\n' + '*' * 20)
    print("data_name location: %s" % (op_data))

    with open(op_data, 'r') as fr:
        datas = fr.readlines()

        class_dict = []
        class_log = []
        begin_class = []
        end_class = []
        begin_index_log = []
        end_index_log = []

        for i, line in tqdm(enumerate(datas)):
            data = line.split()
            try:

                if len(data) != 4:

                    if 'begin' in data:
                        begin_class.append(' '.join(data[4:-1]))
                        begin_index_log.append(i)
                    if 'end' in data:
                        end_class.append(str(data[4:-1]).replace('\'', '').replace(' ', '').replace(',', '_').replace('[', '').replace(']', ''))
                        end_index_log.append(i)

            except IndexError:
                print(i, line)

    if save_file == True:
        file_dir = os.path.join(data_dir, '..', 'static')

        log_name = data_name + '-log.txt'

        static_data = os.path.join(file_dir, log_name)

        with open(static_data, 'w', encoding="utf-8") as fw:
            fw.writelines('Type of activity to start：%s\n' % (set(begin_class)))
            fw.writelines('Start activity type statistics：\nTotal：%d \n%s\n' % (len(begin_class), Counter(begin_class)))

            fw.writelines('End activity type：%s\n' % (set(end_class)))
            fw.writelines('End activity type statistics：\nTotal：%d \n%s\n' % (len(end_class), Counter(end_class)))

            if set(begin_class) != set(end_class):
                fw.writelines('The start and end activity types are inconsistent and need to be checked or removed...')
            if len(begin_class) != len(end_class):
                fw.writelines('Inconsistent number of start and end activities...')

            fw.writelines('Start index：%s\n\n' % (begin_index_log))
            fw.writelines('End index：%s\n\n' % (end_index_log))
            print('Save data to: %s' % static_data)


        with open(os.path.join(file_dir, 'activities.json'), 'r', encoding='utf-8') as fr:
            dict_class = json.load(fr)

        list_class = list(set(begin_class))
        list_class.sort()
        list_class.append('other')
        for activity in list_class:
            dict_class[data_name].update({activity: {}})

        with open(os.path.join(file_dir, 'activities.json'), 'w', encoding='utf-8') as fw:
            json.dump(dict_class, fw)

        print('log saved：%s' % os.path.join(file_dir, 'activities.json'))


if __name__ == '__main__':
    opts = general.load_config()

    data_dir = os.path.join(opts["datasets"]["base_dir"], 'repairdata')

    # data_names = ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11']
    data_names = opts["datasets"]["names"]

    file_dir = os.path.join(opts["datasets"]["base_dir"], 'static')

    general.create_folder(file_dir)

    with open(os.path.join(file_dir, 'activities.json'), 'w', encoding="utf-8") as fw:
        json_dict = {}
        for data_name in data_names:
            json_dict.update({data_name: {}})
        json.dump(json_dict, fw)

    for data_name in data_names:
        staticclass(data_dir, data_name, save_file=True)

    print("success all...")
