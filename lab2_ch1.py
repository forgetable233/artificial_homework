import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris


class graph:
    def __init__(self, input_data):
        self._net = input_data._data
        self._reco_mat = None
        self._num = None

    def my_pca(self, data, mean):
        input_data = np.array(data)
        cov_mat = np.cov(data, rowvar=False)
        e_vals, e_vector = np.linalg.eig(cov_mat)
        sorted_index = np.argsort(e_vals)
        reco_mat = None
        # 得到目标下标
        tar_index = sorted_index[:10]
        tar_vector = e_vector[:, tar_index]
        reco_mat = np.matmul(input_data, tar_vector)
        print(reco_mat[0])
        return reco_mat

    def get_mean_net(self):
        mean_val = np.mean(self._net, axis=0)
        new_data = self._net - mean_val
        return new_data, mean_val

    def my_k_means(self, mat, num):
        n = int(math.sqrt(num / 2))
        k_list = []
        k_center = np.zeros((n, len(mat[0])))
        begin_center = (np.random.random(n) * num).astype(np.int32)
        unjoined_list = [x for x in range(0, num) if x not in begin_center]
        for i in range(0, n):
            k_list.append([begin_center[i]])
            k_center[i, :] = k_center[i, :] + np.array(mat[i, :])
        while len(unjoined_list) != 0:
            tar = unjoined_list[0]
            unjoined_list.remove(tar)
            min_dis = float('inf')
            min_dis_index = -1
            for i in range(0, n):
                temp_dis = np.linalg.norm(mat[tar, :] - k_center[i, :])
                if temp_dis < min_dis:
                    min_dis_index = i
                    min_dis = temp_dis
            length = len(k_list[min_dis_index])
            # 重新计算中心
            k_center[min_dis_index, :] = \
                k_center[min_dis_index, :] * (length / (length + 1)) + \
                mat[min_dis_index, :] * (1 / (length + 1))
            k_list[min_dis_index].append(tar)
        print(n)
        # 返回列表以及k个中心
        return k_list, k_center

    def get_heat_graph(self):
        self._num = len(self._net)
        pca = PCA(n_components='mle')
        self._reco_mat = pca.fit_transform(self._net)
        k_list, k_center = self.my_k_means(self._reco_mat, self._num)
        sim_mat = np.zeros((len(k_center), len(k_center)))
        for i in range(0, len(k_center)):
            for j in range(0, len(k_center)):
                sim_mat[i, j] = np.linalg.norm(k_center[i, :] - k_center[j, :])
        sns.heatmap(sim_mat)
        sns.color_palette('deep')
        # sns.set(style='whitegrid')
        plt.show()
        print('finish heat map')
