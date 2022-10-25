import numpy as np
import pandas as pd
import matplotlib as plot
import seaborn as sb
from data_loader import DataLoader
from lab1 import Net
from lab2_ch1 import graph
from lab2_ch2 import *

FILE_PATH = 'ddi_with_type_latest.txt'
PDI_PATH = 'pdi.txt'
PPI_PATH = 'ppi.txt'
if __name__ == '__main__':
    print('Begin to do homework')

    pdi = DataLoader(PDI_PATH)
    ppi = DataLoader(PPI_PATH)

    # ch1 = graph(ppi)
    # ch1.get_heat_graph()
    ch2 = ch2(ppi, pdi)
    ch2.out_excel()
    # net = Net(data)
    # net.ComputeAverageDegree()
    # net.ComputeEdgeType()
    # net.DrawDegreeDis()
    # net.FindMinRoute()
    # net.ComputeGather()
    # net.ComputeConnect()
    # net.ComputeSubGraphNumber()

    # test = np.array([[3, 4, 5], [3, 4, 5], [3, 4, 5]])
    # test2 = []
    # test2.append(list(test[1, :]))
    # print(test2)


