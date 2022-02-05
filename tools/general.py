import yaml
import os
import time
import shutil


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


if __name__ == '__main__':
    print(load_config("configure/config.yaml"))
