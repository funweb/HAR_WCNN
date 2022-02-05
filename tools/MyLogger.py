import os
import json
import logging.config


# Load profile from profile
class MyLogger:
    def __init__(self,
                 setting_path=os.path.join('./', 'tools', 'configure', 'logging_setting.json'),
                 default_level=logging.INFO,
                 env_key='LOG_CFG',  # environment variable
                 logger_name='Customize_info'
                 ):
        self.setting_path = setting_path
        self.default_level = default_level
        self.env_key = env_key
        self.logger_name = logger_name

        self.logger = self.load_setting()


    # Load settings from configuration file
    def load_setting(self):
        """
        Setup logging configuration
        """
        path = self.setting_path
        value = os.getenv(self.env_key, None)  # Determine whether there are environment variables and the path of the unified planning log file
        if value:
            # logging.info('load setting from: %s and %d' % (self.setting_path, 1111))
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)

            for my_handler in config['handlers']:
                if 'filename' in config['handlers'][my_handler]:
                    file_name_dir = os.path.dirname(config['handlers'][my_handler]['filename'])
                    if file_name_dir == '':
                        file_name_dir = './'

                    if not os.path.exists(file_name_dir):
                        os.makedirs(file_name_dir)
                        print('Create directory: %s' % file_name_dir.replace(os.sep, '/'))

            logging.config.dictConfig(config)
        else:
            # assert
            assert os.path.exists(path), 'file: %s Does not exist. If you need a default configuration, please log out of this line' % path
            logging.basicConfig(level=self.default_level)

        return logging.getLogger(self.logger_name)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


import time
def main():
    logger = MyLogger()
    for i in range(100):
        logger.info(i)
        logger.error(i)
        logger.debug(i)
        logger.critical(i)
        time.sleep(0.1)


if __name__ == '__main__':
    main()
