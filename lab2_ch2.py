import numpy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class ch2:
    def __init__(self, ppi, pdi):
        self._reco_mat = None
        self._ppi = np.array(ppi._data)
        self._pdi = np.array(pdi._data)

    def add_pca(self):
        pca = PCA(n_components='mle')
        temp = numpy.array(pca.fit_transform(self._ppi))
        self._reco_mat = temp
    def compute_dis_inclass(self, proteins):
        dis_euler = 0
        dis_cos = 0
        for item in proteins:
            temp1 = np.array(item)
            for item2 in proteins:
                temp2 = np.array(item2)
                dis_euler += np.linalg.norm(temp1 - temp2)
                if not (temp1 == temp2).all():
                    dis_cos += np.dot(temp1, temp2) / (np.linalg.norm(temp1) * np.linalg.norm(temp2))
        if len(proteins) != 1:
            number = (len(proteins) * (len(proteins) - 1)) / 2
            dis_euler /= number
            dis_cos /= number
        else:
            dis_euler = 0
            dis_cos = 0
        return dis_euler, dis_cos

    def out_excel(self):
        data = []
        in_euler = []
        in_cos = []
        self.add_pca()
        for i in range(0, len(self._pdi[0])):
            temp = self._pdi[:, i]
            rel_pro_index = np.nonzero(temp)
            proteins = []
            for index in rel_pro_index[0]:
                # temp = self._ppi[index, :]
                temp = self._reco_mat[index]
                proteins.append(list(temp))
            dis_euler, dis_cos = self.compute_dis_inclass(proteins)
            in_euler.append(dis_euler)
            in_cos.append(dis_cos)

        for i in range(0, len(self._pdi[0])):
            dis_euler = 0
            dis_cos = 0
            list1 = np.nonzero(self._pdi[:, i])
            num = len(list1)
            for j in [x for x in range(0, len(self._pdi[0])) if x != i]:
                local_dis_euler = 0
                local_dis_cos = 0
                list2 = np.nonzero(self._pdi[:, j])
                num *= len(list2)
                for item1 in list1[0]:
                    # temp1 = self._ppi[item1, :]
                    temp1 = self._reco_mat[item1]
                    for item2 in list2[0]:
                        # temp2 = self._ppi[item2, :]
                        temp2 = self._reco_mat[item2]
                        local_dis_euler += np.linalg.norm(temp1 - temp2)
                        local_dis_cos += np.dot(temp1, temp2) / (np.linalg.norm(temp1) * np.linalg.norm(temp2))
                local_dis_euler /= (len(list2[0]) * len(list1[0]))
                local_dis_cos /= (len(list2[0]) * len(list1[0]))
                dis_euler += local_dis_euler
                dis_cos += local_dis_cos
            dis_euler /= (len(self._pdi[0]) - 1)
            dis_cos /= (len(self._pdi[0]) - 1)
            data.append(['D' + str(i + 1), in_euler[i], in_cos[i], dis_euler, dis_cos])
        print(len(self._pdi[0]))
        out_data = pd.DataFrame(data)
        out_data.columns = ['疾病', '类内距离（欧式）', '类内距离（余弦）', '类间距离（欧式）', '类间距离（余弦）']
        out_data.to_excel('pca.xlsx')
        print(out_data)
        print(data)
