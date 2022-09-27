import numpy as np


class DataLoader:
    def __init__(self):
        self._file_path = None
        self._data = None

    def __init__(self, file_path):
        self._file_path = file_path
        self._data = np.zeros((38405, 3), dtype=int)
        self.load_file()

    def load_file(self):
        f = open(self._file_path, encoding='UTF-8')
        line = f.readline()
        i = 0
        while line:
            array = line.split('\t')
            self._data[i, 0] = int(array[0])
            self._data[i, 1] = int(array[1])
            self._data[i, 2] = int(array[2])
            i = i + 1
            line = f.readline()
        f.close()
