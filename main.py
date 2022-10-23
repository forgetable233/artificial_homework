import numpy as np
import pandas as pd
import matplotlib as plot
import seaborn as sb
from data_loader import DataLoader
from lab1 import Net
from lab2 import graph

FILE_PATH = 'ddi_with_type_latest.txt'
PDI_PATH = 'pdi.txt'
PPI_PATH = 'ppi.txt'
if __name__ == '__main__':
    print('Begin to do homework')

    pdi = DataLoader(PDI_PATH)
    ppi = DataLoader(PPI_PATH)

    lab2 = graph(ppi)
    lab2.get_heat_graph()
    # net = Net(data)
    # net.ComputeAverageDegree()
    # net.ComputeEdgeType()
    # net.DrawDegreeDis()
    # net.FindMinRoute()
    # net.ComputeGather()
    # net.ComputeConnect()
    # net.ComputeSubGraphNumber()
