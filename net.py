import numpy
import numpy as np
import matplotlib.pyplot as plt


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

        self._network = np.zeros((self._point_number, self._point_number), dtype=int)
        self._min_dis = np.full((self._point_number, self._point_number), -1, dtype=int)
        self._min_route = [None] * self._point_number

        # 构建对应的路线矩阵
        for i in range(0, self._point_number):
            self._min_route[i] = [None] * self._point_number
            for j in range(0, self._point_number):
                self._min_route[i][j] = [i]
                self._min_route[i][j].append(j)

        # 构建对应的距离矩阵
        for item in self._net:
            self._network[item[0], item[1]] = item[2]
            self._network[item[1], item[0]] = item[2]

        print('=======================================================================')
        print('Q1:  The size of the points is :', end='\n')
        print(len(self._degree_dict))
        print('     The size of the edge is :', end='\n')
        print(self._edge_number)

    def ComputeAverageDegree(self):
        print('=======================================================================')
        print('Q2:  The average degree is :')
        print(self._edge_number / (self._point_number / 2))
        print('Q2:   The points are :')
        temp_list = sorted(self._degree_dict.items(), key=lambda temp: (temp[1], temp[0]), reverse=True)
        for item in temp_list[0:20]:
            print(item[0])

    def ComputeEdgeType(self):
        list = np.zeros(1000, dtype=int)
        number = 0
        for item in self._net:
            if list[item[2]] == 0:
                number += 1
            list[item[2]] += 1
        print('=======================================================================')
        print('Q3:  The number of the types is')
        print(number)
        print('Q3: The number of the edge that is 47 type is ')
        print(list[47])

    def DrawDegreeDis(self):
        print('=======================================================================')
        list = np.zeros(2000)
        for item in self._degree_dict.items():
            list[item[1]] += 1
        index = np.nonzero(list)
        list = list[list != 0] / (self._point_number * 2)
        plt.scatter(index, list)
        plt.show()

    def ResetRoute(self, i, j, k):
        self._min_dis[i][j] = self._min_dis[i][k] + self._min_dis[k][j]
        self._min_route[i][j] = self._min_route[i][k] + self._min_route[k][j]
        self._min_route[i][j].remove(k)

    def FindMinRoute(self):
        print('Begin to find the min route')
        # 读入对应的测试数据
        f = open('test_data.txt', encoding='UTF-8')
        test_data = np.zeros((4, 2), dtype=int)
        line = f.readline()
        i = 0
        while line:
            data = line.split(' ')
            test_data[i, 0] = data[0]
            test_data[i, 1] = data[1]
            i += 1
            line = f.readline()

        print('test')
        # 下面使用弗洛伊德算法计算最短路径
        for i in range(0, self._point_number):
            print(i)
            for j in range(0, self._point_number):
                for k in range(0, self._point_number):
                    if self._min_dis[i][k] > 0 & self._min_dis[k][j] > 0:
                        if self._min_dis[i][j] > 0:
                            if self._min_dis[i][k] + self._min_dis[k][j] < self._min_dis[i][j]:
                                self.ResetRoute(i, j, k)
                        else:
                            self.ResetRoute(i, j, k)

        print('Finish floyd')
        for item in test_data:
            print(self._min_route[item[0]][item[j]])
        f.close()
