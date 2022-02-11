import os


def get_identifier_name(dict_config, other_parameter=""):
    # [tag_数据长度_距离_bs_核宽度_核数量_k]
    base_identifier = '%s_%s_%s' % (
        dict_config['identifier'],
        dict_config['data_lenght'],
        dict_config['batch_size'],
        )

    if "kernel_wide_base" in dict_config:
        base_identifier = "%s_%s" % (base_identifier, str(dict_config['kernel_wide_base']))

    if "kernel_number_base" in dict_config:
        base_identifier = "%s_%s" % (base_identifier, str(dict_config['kernel_number_base']))
    if other_parameter != "":
        base_identifier = "%s_%s" % (base_identifier, other_parameter)

    return base_identifier

def get_checkpointer_dir(dict_config):
    checkpointer_dir = os.path.join(dict_config["datasets_dir"],
                                    dict_config["result_dir"],
                                    dict_config["model_name"],
                                    dict_config["dataset_name"],
                                    str(dict_config['distance_int']))
    return checkpointer_dir


if __name__ == '__main__':
    d = {"a": 2}
    pass


