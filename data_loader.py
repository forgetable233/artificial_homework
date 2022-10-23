import numpy as np


class DataLoader:
    def __init__(self):
        self._file_path = None
        self._data = None

    def __init__(self, file_path):
        self._file_path = file_path
        self._data = None
        self.load_file()

    def load_file(self):
        f = open(self._file_path, encoding='UTF-8')
        line = f.readline()
        cols = len(line.split(' '))
        info = []
        while line:
            array = line.split(' ')
            temp = []
            for j in range(0, cols):
                temp.append(int(array[j]))
            info.append(temp)
            line = f.readline()
        self._data = np.array(info)
        f.close()
