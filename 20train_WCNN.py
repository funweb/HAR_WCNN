import os
import tensorflow as tf

from tools import general
from tools.integration import train_val

if __name__ == '__main__':
    opts = general.load_config()

    method = "WCNN"
    distance_int = 9999
    data_name = 'cairo'
    calculation_unit = "0"

    dict_config_cus = {

        'model_name': opts[method]["model_name"],
        'optimizer': opts[method]["optimizer"],
        'distance_int': distance_int,
        'data_name': data_name,
        'calculation_unit': calculation_unit,

        'epochs': opts[method]["epochs"],

        'identifier': opts["public"]["identifier"],
        'purpose': opts["public"]["purpose"],
        # 'datasetsNames': ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11'],
    }
    train_val(dict_config_cus, opts)
