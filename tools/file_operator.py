import os
import numpy as np
import sys
import matplotlib.pyplot as plt


class file_operator:
    def __init__(self, path='./'):
        self.path = path

    def isDir(self, path=None):
        if path is not None:
            return os.path.isdir(path)
        return os.path.isdir(self.path)

    def isFile(self, path=None):
        if path is not None:
            return os.path.isfile(path)
        return os.path.isfile(self.path)

    def listDir(self, path=None, pathTyle=0):
        if path is not None:
            self.path = path

        if not os.path.exists(self.path):
            return None

        listDirName = []
        for dirName in os.listdir(self.path):
            if self.isDir(os.path.join(self.path, dirName)):
                if pathTyle == 1:
                    listDirName.append(os.path.join(self.path, dirName).replace('\\', '/'))
                elif pathTyle == 2:
                    listDirName.append(os.path.abspath(os.path.join(self.path, dirName)).replace('\\', '/'))
                else:
                    listDirName.append(dirName)
        return listDirName

    # Pathtype: 0: file name, 1: relative path, 2: absolute path
    def listFile(self, path=None, PathType=0):
        if path is not None:
            self.path = path

        if not os.path.exists(self.path):
            return []

        listFileName = []
        for FileName in os.listdir(self.path):
            if self.isFile(os.path.join(self.path, FileName)):
                if PathType == 1:
                    listFileName.append(os.path.join(self.path, FileName).replace('\\', '/'))
                if PathType == 2:
                    listFileName.append(os.path.abspath(os.path.join(self.path, FileName)).replace('\\', '/'))
                else:
                    listFileName.append(FileName)
        return listFileName
    
    
if __name__ == '__main__':
    checkLog()

    f = file_operator()
    # print(f.listDir('./', 2))
    # print(f.listFile('../../', 1))