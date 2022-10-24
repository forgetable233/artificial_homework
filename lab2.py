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

    def get_heat_graph(self):
        num = len(self._net)
        pca = PCA(n_components='mle')
        new_data = pca.fit_transform(self._net)
        sim_mat = np.zeros([num, num])
        for i in range(0, num):
            for j in range(0, num):
                sim_mat[i, j] = np.linalg.norm(new_data[i, :] - new_data[j, :])
        sns.set(style='whitegrid', color_codes=True)
        test = sim_mat[0:10, 0:10]
        sns.heatmap(sim_mat)
        plt.show()
        print('finish heat map')
