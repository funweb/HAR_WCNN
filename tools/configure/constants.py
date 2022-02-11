DATASETS_CONSTANT = {
    "archive_name": "casas",
    "base_dir": "../datasets/casas",  # 相对路径会更好
    "names": ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11'],
    "short_len": 5,
    "ksplit": 3,
    "shuffle:": False,
    "data_max_lenght": 2000,
}

# EXTRA_CONSTANT = {
#     "reTrain": True,
#     "kfoldSeed": 7,
#     "seed": 7,
#     "verbose": 2,
#     "identifier": "20220205",
#     "complete_flag": "success",
#     "purpose": "2020年7月25日跑基于kyoto11LY的模型"  # 更改模型训练的目的
# }

TFIDF_CONSTANT = {
    "power": 1,
}

DISTANCE_CONSTANT = {
    "distances": [9999, 999, 1, 2, 3, 4, 5]
}

# 需要寻找的超参
EXPLORE_CONSTANT = {
    "epochs": [200, 300, 500, 1000, 1500],
    "data_max_lenghts": [300, 2000],
    "batch_sizes": [32, 64, 128, 256],
    "input_dims": [64, 128, 256]
}

# jupyter
JUPYTER_TOKEN = {
    "token_lists": ["root",  # win
                    "8786ba7fd6db486eb13a6e4e79d5951a",  # 7920
                    "ef6c4f2832454a65a1d9c2c7551af431"]  # 5810
}

# methods
# _CONSTANT = {
WCNN_CONSTANT = {
    "model_name": "WCNN",
    # "patience": 200,  # 提前停止
    "input_dim": 128,  # 按理说这个应该是设置为数据类型的大小的
    "units": 64,  # deep 论文中的，这个小的可以尝试一下。。
    # "data_max_lenght": 2000,
    # "epochs": 2000,
    # "batchSize": 64,
    "optimizer": "adagrad",
    "kernel_number_base": "",
    "kernel_wide_base": ""
}

LSTM_CONSTANT = {
    "model_name": "LSTM",
    # "patience": 200,  # 提前停止
    "input_dim": 128,  # 按理说这个应该是设置为数据类型的大小的
    "units": 64,  # deep 论文中的，这个小的可以尝试一下。。
    # "data_max_lenght": 2000,
    # "epochs": 2000,
    # "batchSize": 64,
    "optimizer": "adam"
}

INCEPTION_CONSTANT = {
    "model_name": "inception",
    "nb_iter_": 5,  # inception 个数
    # "batch_size": 64,
    "nb_filters": 32,
    "use_residual": True,
    "use_bottleneck": True,
    "depth": 6,
    "kernel_size": 41,
    "optimizer": "adam",
    # "nb_epochs": 15,
}

METHOD_PARAMETER_TEMPLATE = {
    "kfoldSeed": "7",
    "seed": "7",

    "datasets_dir": "",
    "ksplit": "",
    "archive_name": "",
    "dataset_name": "",
    "data_lenght": "",
    "distance_int": "",

    "model_name": "",
    "nb_epochs": "",
    "batch_size": "",
    "optimizer": "",
    "patience": 200,  # 提前停止

    "reTrain": False,
    "calculation_unit": "0",
    "result_dir": "results",
    "complete_flag": "configParameter.json",
    "verbose": "1",
    "identifier": "20220201",
    "purpose": "validation",  # 更改模型训练的目的
}
