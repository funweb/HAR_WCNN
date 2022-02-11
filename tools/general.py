import copy

import keras
import numpy as np
import yaml
import os
import time
import shutil
import emoji  # https://carpedm20.github.io/emoji/all.html
from keras.callbacks import Callback


def clock(func):
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    interval_time = (
            datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f") -
            datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")).total_seconds()

    """

    def clocked(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("{}: {}".format(func.__name__, end - start))
        return result

    return clocked


@clock
def load_config(config_path="tools/configure/config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fr:
            return yaml.load(fr, Loader=yaml.FullLoader)
    else:
        assert False, "config file does not exits: {}".format(os.path.join(os.getcwd(), config_path))


def create_folder(path='./new', remake=False):
    # Create folder
    if not os.path.exists(path):
        print('Create subdir directory: %s...' % (path))
        time.sleep(3)
        os.makedirs(path)
    elif remake:
        shutil.rmtree(path)  # delete output folder
        os.makedirs(path)


# Merge the two dictionaries, that is, the parameters required by the algorithm
def Merge(dict_config, dict_config_cus):
    dict_config.update(dict_config_cus)
    for k in dict_config:  # 保证值不为空, 也就是保证参数的有效性
        assert dict_config[k] != "", "Please set value for: {}".format(k)
    return dict_config


def reTrain(p):
    if os.path.exists(p):
        shutil.rmtree(p)


import urllib, json, os, ipykernel, ntpath
from notebook import notebookapp as app


def lab_or_notebook():
    length = len(list(app.list_running_servers()))
    if length:
        return "notebook"
    else:
        return "lab"


def ipy_nb_name(token_lists):
    """ Returns the short name of the notebook w/o .ipynb
        or get a FileNotFoundError exception if it cannot be determined
        NOTE: works only when the security is token-based or there is also no password
    """

    if lab_or_notebook() == "lab":
        from jupyter_server import serverapp as app
    else:
        from notebook import notebookapp as app
    #         from jupyter_server import serverapp as app

    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    #     from notebook import notebookapp as app
    for srv in app.list_running_servers():
        for token in token_lists:
            srv['token'] = token

            try:
                # print(token)
                if srv['token'] == '' and not srv['password']:  # No token and no password, ahem...
                    req = urllib.request.urlopen(srv['url'] + 'api/sessions')
                    print('no token or password')
                else:
                    req = urllib.request.urlopen(srv['url'] + 'api/sessions?token=' + srv['token'])
            except:
                pass
                # print("Token is error")

        sessions = json.load(req)

        for sess in sessions:
            if sess['kernel']['id'] == kernel_id:
                nb_path = sess['notebook']['path']
                return ntpath.basename(nb_path).replace('.ipynb', '')  # handles any OS

    raise FileNotFoundError("Can't identify the notebook name, Please check [token]")


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


class ModelCheckpoint_cus(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_best_only_period=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint_cus, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            print('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        self.init_value = self.best  # 设置默认值
        self.save_best_only_period = save_best_only_period
        self.best_period = self.init_value
        self.best_model_period = ""
        self.filepath_period = ""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        current = logs.get(self.monitor)
        if current is None:
            print('Can save best model only with %s available, skipping.' % (self.monitor), RuntimeWarning)
            return

        if (self.period - (epoch % self.period) < self.period*0.4) and (epoch % int(self.period*0.4 / 10) == 0):  # 为了节省算力, 仅仅抽查.  取最后的20%中 每间隔 X==>(self.period*0.4 / 10) 就抽 1 个进行比较
            if self.monitor_op(current, self.best_period):

                if self.best_period != self.init_value:
                    os.remove(self.filepath_period)

                self.filepath_period = os.path.join(os.path.dirname(self.filepath), "Pbest-"+os.path.basename(self.filepath)).format(epoch=epoch + 1, **logs)
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f, saving model to %s'
                          % (epoch + 1, self.monitor, self.best_period, current, self.filepath_period))
                self.best_period = current
                # self.best_model_period = copy.deepcopy(self.model)  # 仅仅保存在内存中, 而不是写到文件/// 解释, 尽管在内存中, 但是 copy.deepcopy 这个方法特别慢...
                self.model.save(self.filepath_period, overwrite=True)

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.best_period = self.init_value  # 清空 [期间] 列表
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch + 1, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve from %0.5f' %
                              (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


if __name__ == '__main__':
    # print(load_config("configure/config.yaml"))
    print(colorstr("bold", "bright_red", 222))
    print(colorstr("bright_red", 222))
