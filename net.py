import numpy as np
import matplotlib as plot


class Net:
    def __init__(self):
        self._net = None
        self._edge_number = 0
        self._point_number = 0

    def __init__(self, input_data):
        self._net = input_data._data
        self._edge_number = len(self._net)
        self._point_number = 0
        self._degree = np.zeros((2000, 2), dtype=int)
        for i in range(0, self._edge_number):
            if self._degree[self._net[i, 0], 1] == 0:
                self._degree[self._net[i, 0], 0] = self._net[i, 0]
                self._point_number = self._point_number + 1
            self._net[self._net[i, 0]] += 1
            if self._net[self._net[i, 1], 1] == 0:
                self._degree[self._net[i, 1], 0] = self._net[i, 1]
                self._point_number = self._point_number + 1
            self._degree[self._net[i, 1]] += 1
        print('The size of the points is :')
        print(self._point_number)
        print('The size of the edge is :')
        print(self._edge_number)

    def ComputeAverageDegree(self):
        print('The average degree is :')
        print(self._edge_number / (self._point_number / 2))
