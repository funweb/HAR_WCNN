import os
import shutil
import time

op_dir = os.path.join("..")

source_file_name = "531_kyoto11_WCNNR_2000_64_1000_1_1_1_1.ipynb"
target_file_name = "{tag}_{dataname}_{method}_{data_length}_{batch_size}_{epochs}_{kernel_wide_base}_{kernel_number_base}_{net_deep_base}_1.ipynb"

tag = 531
dataname = "kyoto11"
method = "WCNNR"
data_length = 2000
batch_size = 64
epochs = 1000
kernel_wide_base = 1
kernel_number_base = 1
net_deep_base = 1

copy_num = 14

source_file = os.path.join(op_dir, source_file_name)

for i in range(copy_num):
    tag += 1
    kernel_wide_base += 1

    target_file = os.path.join(op_dir, target_file_name.format(tag=tag, dataname=dataname, method=method,
                                                               data_length=data_length, batch_size=batch_size,
                                                               epochs=epochs, kernel_wide_base=kernel_wide_base,
                                                               kernel_number_base=kernel_number_base,
                                                               net_deep_base=net_deep_base))
    assert not os.path.exists(target_file), "文件已存在, 请查验"

    print("{}->\n{}\n\n".format(source_file, target_file))
    time.sleep(3)
    shutil.copy(source_file, target_file)
