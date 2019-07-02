import numpy as np
from lmfit import minimize, Parameters, Model
from lmfit.models import LinearModel, ExponentialModel
import matplotlib.pyplot as plt


from .Measurement import Measurements

class PCH(Measurements):

    def __init__(self, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment="", logger=None):
        super().__init__(exp_param, num_channel, start_tick, end_tick, "PCH", name, comment, logger)

    def create_histogram(self, chronogram, timestamps=None,  bin_size=1):
        if chronogram is not None :
            self.time_axis = np.arange(0, chronogram.data.max() + 1)
            #FIXME binsize different than 1 ?
            self.bin_size = bin_size
            self.data = np.bincount(chronogram.data)

    def create_canonic_graph(self):
        pass
