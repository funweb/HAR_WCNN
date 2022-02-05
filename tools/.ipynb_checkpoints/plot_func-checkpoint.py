# 导入相关模块
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pandas as pd
import sys


def plot_log(model_name, data_name, ksplite, data_day, epochs, distant_ints):
    for distant_int in distant_ints:
        log_dir = os.path.join(os.getcwd(), 'datasets', 'ende', data_name, str(distant_int), 'npy', str(ksplite), 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        for k in range(ksplite):
            csvFileName = os.path.join(log_dir, '%s_%s_%s-%s-%s.csv' % (model_name, data_day, k, data_name, epochs))
            df = pd.read_csv(csvFileName)

            df.__len__()

            print('读取数据：%s' % (csvFileName))
            # X_scale = np.arange(dict_config['epochs'])
            X_scale = np.arange(len(df.loc[:, 'val_loss']))

            val_loss_scale = np.array(df.loc[:, 'val_loss'], dtype='float32')
            val_acc_scale = np.array(df.loc[:, 'val_acc'], dtype='float32')
            loss_scale = np.array(df.loc[:, 'loss'], dtype='float32')
            acc_scale = np.array(df.loc[:, 'acc'], dtype='float32')

            fig, axs = plt.subplots(1, 2, figsize= [20, 5], constrained_layout=True)  # 设备图的尺寸大小
            fig.suptitle('%s_%s_%s_%d' % (model_name, distant_int, data_name, k), fontsize=26)

            # plt.subplot(121)
            axs[0].plot(X_scale, loss_scale, 'r-', label='train')
            axs[0].plot(X_scale, val_loss_scale, 'b-', label='valid')
            axs[0].set_xlabel('epoch', fontsize=24)
            axs[0].set_ylabel('loss', fontsize=24)
            axs[0].set_title('loss: %d' % k)
            axs[0].legend(fontsize=24)
            axs[0].grid()  # 生成网格
            # plt.show()

            # plt.subplot(122)
            axs[1].plot(X_scale, acc_scale, 'r-', label='train')
            axs[1].plot(X_scale, val_acc_scale, 'b-', label='valid')
            axs[1].set_xlabel('epoch', fontsize=24)
            axs[1].set_ylabel('acc %', fontsize=24)
            axs[1].set_title('acc: %d' % k)
            axs[1].legend(fontsize=24)
            axs[1].set_yticks(np.linspace(0,1,num=11))
            axs[1].set_ylim(0,1)
            axs[1].grid()  # 生成网格

            min_indx_loss = np.argmin(val_loss_scale)  # min value index
            max_indx_val = np.argmax(val_acc_scale)  # max value index

            roots = np.linspace(0, len(X_scale)-1, num=6, dtype=int)

            for i in roots:
                axs[0].text(i, val_loss_scale[-1]*2, '%s\n%s' % (i, val_loss_scale[i]))
                axs[1].text(i, val_acc_scale[i]*0.7,  '%s\n%s' % (i, val_acc_scale[i]))

            axs[0].plot(min_indx_loss, val_loss_scale[min_indx_loss], 'gs')
            axs[1].plot(max_indx_val,val_acc_scale[max_indx_val],'gs')

            axs[0].annotate(
                'MIN:%s' % val_loss_scale[min_indx_loss],
                xy=(min_indx_loss,val_loss_scale[min_indx_loss]), 
                arrowprops=dict(arrowstyle='->'), 
                xytext=(min_indx_loss,val_loss_scale[min_indx_loss]*1.5),
                # textcoords='offset points',
                fontsize=16
            )
            axs[1].annotate(
                'MAX:%s' % val_acc_scale[max_indx_val],
                xy=(max_indx_val,val_acc_scale[max_indx_val]), 
                arrowprops=dict(arrowstyle='->'), 
                xytext=(max_indx_val,val_loss_scale[max_indx_val]),
                fontsize=16
            )

            plt.show()
            if k == 2:
                print('\n\n', '-'*20, '  ', '-'*20)
            # sys.exit(0)

            save_dir = os.path.join(log_dir, 'img')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(os.path.join(save_dir, '%s_%s_%s_%d' % (model_name, distant_int, data_name, k)), quality=95)

            plt.close(fig)  # 关闭，防止内存溢出