import os
import tensorflow as tf

from tools import general
from tools.configure.constants import EXTRA_CONSTANT, WCNN_CONSTANT as MODEL_DEFAULT_CONF
from tools.integration import train_val

if __name__ == '__main__':
    # opts = general.load_config()

    distance_int = 9999
    data_name = 'cairo'
    calculation_unit = "0"

    # 修订论文所需要的
    data_max_lenght = 300
    kernel_number_base = 8
    kernel_wide_base = 1

    epochs = 1000  # 公平起见, 默认都是 1000 吧.

    dict_config_cus = {

        'model_name': MODEL_DEFAULT_CONF["model_name"],
        'optimizer': MODEL_DEFAULT_CONF["optimizer"],
        'distance_int': distance_int,
        'data_name': data_name,
        'calculation_unit': calculation_unit,

        'data_max_lenght': data_max_lenght,
        'kernel_number_base': kernel_number_base,
        'kernel_wide_base': kernel_wide_base,
        'epochs': epochs,

        'identifier': EXTRA_CONSTANT["identifier"],
        'purpose': EXTRA_CONSTANT["purpose"],
        # 'datasetsNames': ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11'],
    }
    train_val(dict_config_cus)
