import numpy as np
import pandas as pd


class ch2:
    def __init__(self, ppi, pdi):
        self._ppi = np.array(ppi._data)
        self._pdi = np.array(pdi._data)

    def compute_dis_inclass(self, proteins):
        dis_euler = 0
        dis_cos = 0
        center = np.zeros(383)
        for item in proteins:
            temp1 = np.array(item)
            for item2 in proteins:
                temp2 = np.array(item2)
                dis_euler += np.linalg.norm(temp1 - temp2)
                if not (temp1 == temp2).all():
                    dis_cos += np.dot(temp1, temp2) / (np.linalg.norm(temp1) * np.linalg.norm(temp2))
            center += temp1
        if len(proteins) != 1:
            number = len(proteins) * (len(proteins) - 1) / 2
            dis_euler /= number
            dis_cos /= number
        else:
            dis_euler = 0
            dis_cos = 0
        center /= len(proteins)
        return dis_euler, dis_cos, center

    def out_excel(self):
        data = []
        centers = []
        in_euler = []
        in_cos = []
        for i in range(0, len(self._pdi[0])):
            temp = self._pdi[:, i]
            rel_pro_index = np.nonzero(temp)
            proteins = []
            for index in rel_pro_index[0]:
                temp = self._ppi[index, :]
                proteins.append(list(temp))
            dis_euler, dis_cos, center = self.compute_dis_inclass(proteins)
            centers.append(list(center))
            in_euler.append(dis_euler)
            in_cos.append(dis_cos)

        for i in range(0, len(self._pdi[0])):
            dis_euler = 0
            dis_cos = 0
            temp1 = np.array(centers[i])
            for j in [x for x in range(0, len(self._pdi[0])) if x != i]:
                temp2 = np.array(centers[j])
                dis_euler += np.linalg.norm(temp1 - temp2)
                dis_cos += np.dot(temp1, temp2) / (np.linalg.norm(temp1) * np.linalg.norm(temp2))
            dis_euler /= (len(self._pdi[0]) - 1)
            dis_cos /= (len(self._pdi[0]) - 1)
            data.append(['D' + str(i + 1), in_euler[i], in_cos[i], dis_euler, dis_cos])
        out_data = pd.DataFrame(data)
        out_data.columns = ['疾病', '类内距离（欧式）', '类内距离（余弦）', '类间距离（欧式）', '类间距离（余弦）']
        out_data.to_excel('dis.xlsx')
        print(out_data)
        print(data)

