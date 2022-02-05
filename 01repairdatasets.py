import numpy as np
import re
import os
from tqdm import tqdm
import time

from tools import general


@general.clock
def repair_time(data_dir, data_name, save_file=False):
    print('*' * 20)
    print("data_name location: %s\\%s" % (data_dir, data_name))

    opdata = os.path.join(data_dir, data_name)
    with open(opdata, 'r') as fr:
        datas = fr.readlines()

        new_data = []  # Store all the data to the new file, that is, the repaired data
        repair_log = []  # The repaired records are reserved for viewing to avoid errors

        for i, line in tqdm(enumerate(datas)):
            data = line.split()
            try:
                if not ('.' in str(np.array(data[0])) + str(np.array(data[1]))):
                    new_time = data[1] + '.000000'  # Repair time format
                    old_line = line
                    new_line = line.replace(data[1], new_time)
                    line = new_line

                    # log
                    repair_log.append('The number of rows repaired is: %d\n' % (i))
                    repair_log.append('old data: %s' % (old_line))
                    repair_log.append('new data: %s\n' % (new_line))

                pattern = re.compile('(^\w)(\d+)')
                s = pattern.search(data[2])
                if s and len(s[2]) < 3:
                    new_line = line.replace(data[2], s[1] + s[2].zfill(3))
                    line = new_line

                # Activity classification is unified as_
                if len(data) > 4:
                    activities = ' '.join(data[4:])
                    pattern = re.compile(r'begin|end')
                    s = pattern.findall(activities)
                    if len(s) > 1:
                        print('Many kinds of activities...')

                    if 'begin' in line or 'end' in line:
                        xiahuaxian_activity_type = str(data[4:-1]).replace('\'', '').replace(' ', '').replace(',',
                                                                                                              '_').replace(
                            '[', '').replace(']', '')
                        kongge_activity_type = str(data[4:-1]).replace('\'', '').replace(' ', '').replace(',',
                                                                                                          ' ').replace(
                            '[', '').replace(']', '')
                        line = line.replace(kongge_activity_type, xiahuaxian_activity_type)
                new_data.append(line)

            except IndexError:
                print(i, line)

    # Save file or not
    if save_file == True:
        file_dir = os.path.join(data_dir, '..', 'repairdata')
        general.create_folder(file_dir)

        data_name = 'repair-' + data_name

        opdata = os.path.join(file_dir, data_name)
        with open(opdata, 'w', encoding="utf-8") as fw:
            fw.writelines(new_data)

        # log
        logdir = os.path.join(file_dir, 'logs')
        general.create_folder(logdir)

        logdata = os.path.join(logdir, 'repairlog-' + data_name)

        with open(logdata, 'w', encoding="utf-8") as f:
            f.writelines(repair_log)

    print('The data is saved in {} '.format(opdata))
    print('The log is saved in {} '.format(logdata))


if __name__ == '__main__':
    opts = general.load_config()

    data_dir = os.path.join(opts["datasets"]["base_dir"], 'origindata')

    assert (os.path.exists(data_dir)), ("{} not exits, please check.".format(data_dir))

    # data_names = ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11']
    data_names = opts["datasets"]["names"]

    for data_name in data_names:
        repair_time(data_dir, data_name, save_file=True)

    print("Success All.")
