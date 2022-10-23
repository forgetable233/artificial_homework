import numpy as np
import matplotlib as plt
import pandas as pd
import seaborn as sns


class graph:
    def __init__(self, data):
        self._net = data._data

    def get_heat_graph(self):
        covariance = self._net * np.transpose(self._net)
        print(covariance)
