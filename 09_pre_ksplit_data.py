import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold


# 从数据中切分模型
# 一次切分一个数据集,数据集的格式为：
#   kyoto7-x.npy
#   kyoto7-y.npy
#   kyoto7-labels.npy
from tools import general


def datacut(data_name, datadir, opts, ksplit=3, save_data=False):
    opdir = os.path.join(datadir, str(ksplit))
    print('即将切分数据为:%d 份,并且保存到 %s' % (ksplit, opdir))

    general.create_folder(opdir)

    # 首先载入数据
    data_path_x = os.path.join(datadir, data_name + '-x.npy')
    datas_x = np.load(data_path_x, allow_pickle=True)

    data_path_y = os.path.join(datadir, data_name + '-y.npy')
    datas_y = np.load(data_path_y, allow_pickle=True)
    datas_y = np.array(datas_y, dtype=np.int32)  # 真是奇怪，竟然读出来的是 str，不得不这样做

    data_path_labels = os.path.join(datadir, data_name + '-labels.npy')
    datas_labels = np.load(data_path_labels, allow_pickle=True)

    # 数据切分
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=opts["public"]["kfoldSeed"])
    k = 0  # 用来计数
    for train, test in kfold.split(datas_x, datas_y):
        np.save(os.path.join(opdir, data_name + '-train-x-' + str(k) + '.npy'), datas_x[train])
        np.save(os.path.join(opdir, data_name + '-train-y-' + str(k) + '.npy'), datas_y[train])

        np.save(os.path.join(opdir, data_name + '-test-x-' + str(k) + '.npy'), datas_x[test])
        np.save(os.path.join(opdir, data_name + '-test-y-' + str(k) + '.npy'), datas_y[test])

        if k == 0:
            np.save(os.path.join(opdir, data_name + '-labels.npy'), datas_labels)

        k += 1
    print('数据正常切分和保存，并且执行完毕。。。')


if __name__ == '__main__':
    opts = general.load_config()

    datasetsNames = ['kyoto11', 'kyoto8', 'kyoto7', 'milan', 'cairo']
    datasetsNames = opts["datasets"]["names"]

    for data_name in datasetsNames:
        print("current dataset: %s" % (data_name))
        for i in os.listdir(os.path.join(opts["datasets"]["base_dir"], 'ende', data_name)):
            datadir = os.path.join(opts["datasets"]["base_dir"], 'ende', data_name, i, 'npy')
            # print(datadir)
            datacut(data_name, datadir, opts, opts["datasets"]['ksplit'], save_data=True)

    print('all Success!')