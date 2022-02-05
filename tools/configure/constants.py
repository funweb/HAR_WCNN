DATASETS_CONSTANT = {
    "base_dir": "F:/PaperEdit/HARedit/datasets/casas",  # 相对路径会更好
    "names": ['cairo', 'milan', 'kyoto7', 'kyoto8', 'kyoto11'],
    "short_len": 5,
    "ksplit": 3,
    "shuffle:": False,
    "data_max_lenght": 2000,
}

EXTRA_CONSTANT = {
    "kfoldSeed": 7,
    "seed": 7,
    "verbose": 2,
    "identifier": "20220205",
    "purpose": "2020年7月25日跑基于kyoto11LY的模型"  # 更改模型训练的目的
}

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

# methods
# _CONSTANT = {
WCNN_CONSTANT = {
    "model_name": "WCNN",
    "patience": 200,  # 提前停止
    "input_dim": 128,  # 按理说这个应该是设置为数据类型的大小的
    "units": 64,  # deep 论文中的，这个小的可以尝试一下。。
    "data_max_lenght": 2000,
    "epochs": 2000,
    "batchSize": 64,
    "optimizer": "rms"
}

LSTM_CONSTANT = {
    "model_name": "LSTM",
    "patience": 200,  # 提前停止
    "input_dim": 128,  # 按理说这个应该是设置为数据类型的大小的
    "units": 64,  # deep 论文中的，这个小的可以尝试一下。。
    "data_max_lenght": 2000,
    "epochs": 2000,
    "batchSize": 64,
    "optimizer": "adam"
}
