import numpy
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
        self._degree_dict = {}
        degree = np.zeros(2000, dtype=int)

        for i in range(0, self._edge_number):
            if degree[self._net[i, 0]] == 0:
                self._degree_dict.update({self._net[i, 0]: 0})
                self._point_number += 1
                degree[self._net[i, 0]] = 1
            self._degree_dict[self._net[i, 0]] += 1
            if degree[self._net[i, 1]] == 0:
                self._degree_dict.update({self._net[i, 1]: 0})
                self._point_number += 1
                degree[self._net[i, 1]] = 1
            self._degree_dict[self._net[i, 1]] += 1

        print('=======================================================================')
        print('Q1:  The size of the points is :', end='\n')
        print(len(self._degree_dict))
        print('     The size of the edge is :', end='\n')
        print(self._edge_number)

    def ComputeAverageDegree(self):
        print('=======================================================================')
        print('Q2:  The average degree is :')
        print(self._edge_number / (self._point_number / 2))
        print('Q2   The points are :')
        temp_list = sorted(self._degree_dict.items(), key=lambda temp: (temp[1], temp[0]), reverse=True)
        for item in temp_list[0:20]:
            print(item[0])
