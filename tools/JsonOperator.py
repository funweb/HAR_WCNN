import json
import logging
import os
import time


class JsonOperator(object):
    '''
    1. Serialize the text in the incoming dictionary format
    2. Deserialize JSON format in text
    '''


    '''
    General expression: file and dict are not specified. Specify if specified
    '''
    def __init__(self, file_name=None, dict_data={}, json_data={}):
        self.file_name = file_name
        self.dict_data = dict_data
        self.json_data = json_data

    # Serialization
    # If you do not specify the contents of a dictionary, you call it from the initialization class object.
    # Return JSON
    def Dumps(self, dict_data={}):
        try:
            if dict_data != {}:
                self.dict_data = dict_data
            return json.dumps(self.dict_data)

        except BaseException as baseerr:
            logging.error(baseerr.message)
            print(baseerr.message)


    # Deserialization
    # Return to Dict
    def Loads(self, json_data={}):
        try:
            if json_data != {}:
                self.json_data = json_data

            return json.loads(self.json_data)

        except BaseException as baseerr:
            logging.error(baseerr.message)
            print(baseerr.message)


    def Write(self, file_name=None, dict_data={}, mode='w', indent=4):
        '''
        @param mode:default: w
        @param indent: Number of spaces / table 
        @param dict: dump
        @return: None.JSON successfully wrote the file
        '''
        # #Ensure that the parameters are passed correctly. The child parameter is greater than the parent parameter
        if file_name is not None:
            self.file_name = file_name
        if dict_data != {}:
            self.dict_data = dict_data

        # Ensure that the directory path exists

        if self.file_name is None or self.file_name == '':
            logging.error(self.file_name + "not exits")
            raise FileNotFoundError(self.file_name + "not exits")

        file_name_dir = os.path.dirname(self.file_name)
        if file_name_dir == '':
            file_name_dir = './'

        if not os.path.exists(file_name_dir):
            os.makedirs(file_name_dir)
            print('Create directory: %s' % file_name_dir.replace(os.sep, '/'))

        if self.file_name is not None:
            try:
                if os.path.exists(self.file_name) and os.path.getsize(self.file_name) != 0:
                    print('Null: %s bytes' % os.path.getsize(self.file_name))
                    if mode == 'w':
                        print('The current operation overwrites the source file: %s' % os.path.abspath(self.file_name))
                    elif mode == 'a':
                        print('The current operation is append: %s' % os.path.abspath(self.file_name))

                with open(self.file_name, mode) as fw:
                    json.dump(self.dict_data, fw, indent=indent)
            except IOError:
                print('Failed to write file!%s' % IOError)
            finally:
                print('Dictionary written to file: %s' % self.file_name.replace(os.sep, '/'))
        return True


    # Deserialize the contents of the file
    def Read(self, file_name=None,mode='r'):
        '''
        @param mode: default: read
        @return: dict
        '''
        if file_name is not None:
            self.file_name = file_name
        if not os.path.exists(file_name):
            print('File does not exist: %s' % self.file_name.replace(os.sep, '/'))
            logging.error(self.file_name + "not exits")
            raise FileNotFoundError(self.file_name + "not exits")

        try:
            with open(self.file_name, mode) as fr:
                return json.load(fr)
        except OSError as oserr:
            print(oserr)
            logging.error(oserr.message + "Json Read Error")


if __name__ == '__main__':
    test = {
        'a': 1,
        'b': 2,
    }

    # First: very specific instantiated objects
    n = JsonOperator(dict_data=test)
    print(n.Dumps())
    print('-' * 20)

    # The second: abstract objects, more personalized
    m = JsonOperator()
    print(m.Dumps(dict_data=test))  

    logging.error('aaaa')

    if 1==1:
        file_name = 'test.json'
        if m.Write(file_name=file_name, dict_data=test, mode='w'):
            print('file: %s will delete after %ss!' % (file_name, 10))
            time.sleep(10)
            # print(os.remove(log_name))
            print('file: %s has been deleted!' % (file_name))
