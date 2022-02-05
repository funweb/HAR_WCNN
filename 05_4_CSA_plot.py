import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from tools import general


def draw_heatmap(array_data, x_activity, y_sensors, data_name, img_dir):
    print(array_data.shape)

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(8, 6))
    plt.imshow(array_data)

    plt.colorbar()

    plt.xlabel('activities')
    plt.ylabel('sensor')
    plt.xticks(np.arange(len(x_activity)), x_activity, rotation=90)
    plt.yticks(np.arange(len(y_sensors)), y_sensors)

    plt.tick_params(labelsize=8)

    plt.title('significance analysis')
    general.create_folder(img_dir)

    plt.savefig(os.path.join(img_dir, data_name + '-tfdf3.png'), dpi=300)
    # plt.show()
    plt.close()


def draw_test():
    import matplotlib.pylab as plt
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 100))
    ax[0].plot([0, 1], [0, 1])
    ax[1].plot([0, 1], [0, 1])
    fig.savefig('test3.png')
    plt.close()


if __name__ == '__main__':
    '''
    @param: multiply, Represents the power
    '''

    opts = general.load_config()

    data_dir = os.path.join(opts["datasets"]["base_dir"], 'tfidf')
    data_names = ['cairo', 'kyoto7', 'kyoto8', 'kyoto11', 'milan']
    data_names = opts["datasets"]["names"]

    y_sensors = ["A_tf_df", "P_tf_df", "T_tf_df", "M_tf_df", "D_tf_df", "L_tf_df", "I_tf_df", "E_tf_df", ]

    x_activity = []
    list_values = []

    for data_name in data_names:
        list_values = []
        x_activity = []
        with open(os.path.join(data_dir, data_name + '-norm'), "r", encoding="utf-8") as fr:
            dict_data = json.load(fr)

        for activity in dict_data:
            x_activity.append(data_name + '_' + activity)
            for v in dict_data[activity].values():
                list_values.append(v)
        array_data = np.array(list_values, dtype='float64').reshape([-1, len(y_sensors)]).T
        draw_heatmap(array_data, x_activity, y_sensors, data_name, img_dir=os.path.join(data_dir, 'pic', str(opts["tfidf"]["power"])))
    # draw_test()

    print('Finish all!')
