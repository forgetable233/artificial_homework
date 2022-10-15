import numpy as np
import matplotlib.pyplot as plt
import math


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

        self._min_dis = np.full((self._point_number, self._point_number), 9999999, dtype=int)
        self._edge_type = np.full((self._point_number, self._point_number), -1, dtype=int)
        self._min_route = [None] * self._point_number
        self._connect_net = [None] * self._point_number

        # 构建对应的路线矩阵
        for i in range(0, self._point_number):
            self._min_route[i] = [None] * self._point_number
            self._connect_net[i] = [None]
            for j in range(0, self._point_number):
                self._min_route[i][j] = [[i]]
                self._min_route[i][j][0].append(j)

        # 构建对应的连接矩阵
        for item in self._net:
            self._connect_net[item[0]].append(item[1])
        # 构建对应的距离矩阵
        for item in self._net:
            self._min_dis[item[0], item[1]] = 1
            self._min_dis[item[1], item[0]] = 1
            self._edge_type[item[0], item[1]] = item[2]
            self._edge_type[item[1], item[0]] = item[2]

        print('=======================================================================')
        print('Q1:  The size of the points is :', end='\n')
        print(self._point_number)
        print('Q1:  The size of the edge is :', end='\n')
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
        print('Q3:  The number of the edge that is 47 type is ')
        print(list[47])

    def DrawDegreeDis(self):
        print('=======================================================================')
        degree_graph = np.zeros(2000)
        for item in self._degree_dict.items():
            degree_graph[item[1]] += 1
        index = np.nonzero(degree_graph)
        degree_graph = degree_graph[degree_graph != 0] / (self._point_number * 2)
        plt.scatter(index, degree_graph)
        plt.show()

    def ResetRoute(self, i, j, k):
        self._min_dis[i, j] = self._min_dis[i, k] + self._min_dis[k, j]
        self._min_route[i][j] = self._min_route[i][k] + self._min_route[k][j]
        self._min_route[i][j].remove(k)

    def FindMinRoute(self):
        # print('Begin to find the min route')
        # 读入对应的测试数据
        f = open('test_data.txt', encoding='UTF-8')
        test_data = np.zeros((7, 2), dtype=int)
        line = f.readline()
        i = 0
        while line:
            data = line.split(' ')
            test_data[i, 0] = data[0]
            test_data[i, 1] = data[1]
            i += 1
            line = f.readline()

        print('=======================================================================')
        print('Q5:  The min routes are as follows:')
        # 下面使用迪杰斯特拉算法计算最短路径
        for item in test_data:
            joined = [item[0]]
            non_joined = [x for x in range(0, self._point_number) if x != item[0]]
            while len(non_joined):
                idx = non_joined[0]
                for i in non_joined:
                    if self._min_dis[item[0], i] < self._min_dis[item[0], idx]:
                        idx = i
                joined.append(idx)
                non_joined.remove(idx)
                for i in non_joined:
                    if self._min_dis[item[0], i] >= self._min_dis[item[0], idx] + self._min_dis[idx, i]:
                        if self._min_dis[item[0], i] > self._min_dis[item[0], idx] + self._min_dis[idx, i]:
                            self._min_route[item[0]][i] = []
                        self._min_dis[item[0], i] = self._min_dis[item[0], idx] + self._min_dis[idx, i]
                        for list1 in self._min_route[item[0]][idx]:
                            for list2 in self._min_route[idx][i]:
                                if list1 + list2[1:] not in self._min_route[item[0]][i]:
                                    self._min_route[item[0]][i].append(list1 + list2[1:])
            print(self._min_route[item[0]][item[1]])
        f.close()

    def DFS(self, begin_point, joined_list, joined_number, not_joined_list):
        for item in self._connect_net[begin_point][1:]:
            if joined_list[item] == 0:
                not_joined_list.remove(item)
                joined_list[item] = 1
                # joined_number += 1
                joined_list, joined_number, not_joined_list = self.DFS(item, joined_list, joined_number + 1,
                                                                       not_joined_list)
        return joined_list, joined_number, not_joined_list

    def ComputeGather(self):
        gather_dirt = {}
        temp_gather = 0
        zero_number = 0
        for i in range(0, self._point_number):
            k = len(self._connect_net[i][1:])
            e = 0
            temp_list = self._connect_net[i][1:]
            if k == 1:
                gather_dirt.update({i: 0})
                zero_number += 1
                continue
            for item1 in temp_list:
                for item2 in temp_list:
                    if item2 in self._connect_net[item1]:
                        e += 1
            if e == 0:
                gather_dirt.update({i: 0})
                zero_number += 1
                continue
            temp_gather += 2 * e / (k * (k - 1))
            gather_dirt.update({i: format(2 * e / (k * (k - 1)), '.3f')})
        temp_gather /= self._point_number
        # TODO 平均数据不知道为啥为0.75倍
        print('=======================================================================')
        # print(zero_number / self._point_number)
        print('Q6: The average gather degree is :')
        print(format(temp_gather, '.3f'))
        print('The points gather degrees are :')
        print(gather_dirt)

    def ComputeConnect(self):
        joined_list = np.zeros(self._point_number, dtype=int)
        joined_list[0] = 1
        joined_number = 1
        not_joined_list = [x for x in range(1, self._point_number)]
        number = 1
        self.DFS(0, joined_list, joined_number, not_joined_list)
        while len(not_joined_list) > 0:
            number += 1
            temp = not_joined_list[0]
            not_joined_list.remove(temp)
            joined_list[temp] = 1
            joined_list, joined_number, not_joined_list = self.DFS(temp, joined_list, joined_number + 1,
                                                                   not_joined_list)
            # print(joined_number)
        print('=======================================================================')
        print('Q7: The number of different connected nets is:')
        print(number)

    def ComputeSubGraphNumber(self):
        child_net_number = 0
        list = [x for x in self._net if x[2] == 73]
        print(list)
        for item in list:
            for i in range(0, self._point_number):
                if self._edge_type[i][item[0]] == 47 and self._edge_type[i][item[1]] == 47:
                    child_net_number += 1
        print('=======================================================================')
        print('Q8: The number of the child net is :')
        print(child_net_number)

