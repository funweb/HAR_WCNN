import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.backend.tensorflow_backend import set_session
import os
import pynvml


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


class GPU_operator(object):
    def test_GPU(self):
        log_dict = {
            'is_gpu_available': False,
            'is_built_with_cuda': False,
        }

        if tf.test.is_gpu_available() is True:
            log_dict['is_gpu_available'] = True

            # List available GPUs
            from tensorflow.python.client import device_lib

        if tf.test.is_built_with_cuda() is True:
            log_dict['is_built_with_cuda'] = True

        print('%s' % log_dict)

    def set_GPU(self):
        if tf.test.is_gpu_available():
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True  
            # sess = tf.Session(config=config)
            # KTF.set_session(sess)  # 设置session

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
            config.log_device_placement = True  # to log device placement (on which device the operation ran)
            sess = tf.Session(config=config)
            return sess  # set this TensorFlow session as the default session for Keras

        else:
            assert 'GPU is not available'

    def set_CPU(self):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # List the equipment available for training
    def list_devices(self):
        from tensorflow.python.client import device_lib
        return device_lib.list_local_devices()

    # List available GPUs
    def list_GPUs(self):
        from keras import backend as K
        list_gpu = K.tensorflow_backend._get_available_gpus()
        return list_gpu

    # Detects the number of available computing devices
    def automatic_allocation(self, list_gpu):
        if list_gpu == []:
            return 0

        KB = 1024
        MB = KB * 1024
        GB = MB * 1024
        list_gpu_nums = []
        for gpu in list_gpu:
            list_gpu_nums.append(int(gpu.split(':')[-1]))
        pynvml.nvmlInit()  # initialization
        L = pynvml.nvmlDeviceGetCount()  # This can count the number of GPUs, but it may not be used for deep learning calculation
        for list_gpu_num in list_gpu_nums:
            handle = pynvml.nvmlDeviceGetHandleByIndex(list_gpu_num)
            meninfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if meninfo.free / GB > 1:  # That is, when the available memory of the GPU is greater than 1GB, try to use the GPU for so long
                return list_gpu_num + 1
        
        return 0  # There are no available computing units, so you can only use CPU


    # Switch training equipment
    def change_device(self, device_number='0'):
        # '-1': CPU
        # '0': Represents the first GPU
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use, usually either "0" or "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = device_number


if __name__ == '__main__':
    # test GPU
    GPU_operator().test_GPU()

    # list devices for calculate
    dict_device = GPU_operator().list_devices()

    # list GPUs for calculate
    list_gpu = GPU_operator().list_GPUs()

    # Adapted computing unit
    Adapted_num = GPU_operator().automatic_allocation(list_gpu)

    # set CPU
    GPU_operator().set_CPU()

    # set GPU
    GPU_operator().set_GPU()


