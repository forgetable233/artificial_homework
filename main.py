import numpy as np
import matplotlib as plot
from data_loader import DataLoader
from net import Net

FILE_PATH = 'ddi_with_type_latest.txt'
if __name__ == '__main__':
    print('Begin to do homework')

    data = DataLoader(FILE_PATH)
    net = Net(data)
    net.ComputeAverageDegree()
    net.ComputeEdgeType()
    net.DrawDegreeDis()
    net.FindMinRoute()
    net.ComputeGather()
    net.ComputeConnect()
    net.ComputeSubGraphNumber()
