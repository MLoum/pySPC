import numpy as np
from lmfit import minimize, Parameters, Model
from lmfit.models import LinearModel, ExponentialModel
import matplotlib.pyplot as plt


from .Measurement import Measurements

class Chronogram(Measurements):

    def __init__(self, exps, exp, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exps, exp, exp_param, num_channel, start_tick, end_tick, "chronogram", name, comment)

    def create_chronogram(self, timestamps,  bin_in_tick=1E5):
        """
        The x axis is in ->microsecond<-
        """
        self.timestamps = timestamps
        self.start_tick = timestamps.min()
        self.end_tick = timestamps.max()
        self.bin_in_tick = bin_in_tick

        # There is a +1 because the first bin is [0-1]
        self.nb_of_bin = int((self.end_tick - self.start_tick) / bin_in_tick)

        timestamps_ = np.copy(timestamps)
        timestamps_ -= self.start_tick

        # FIXME moins de divisions, ici on prend tout le fichier

        num_bin = (timestamps_ / bin_in_tick).astype(np.int64)
        self.data = np.bincount(num_bin)

        self.time_axis = np.arange(0, self.nb_of_bin + 1, dtype=np.float64)
        self.time_axis *= bin_in_tick
        self.time_axis += self.start_tick
        self.time_axis *= self.exp_param.mAcrotime_clickEquivalentIn_second * 1E6

    def get_num_photon_of_edge(self, is_absolute_number=True):
        edges_time = np.arange(0, self.nb_of_bin + 1)*self.bin_in_tick + self.start_tick
        if is_absolute_number:
            absolute_num_photon_of_edge = np.searchsorted(self.timestamps, edges_time)
            # The first num is zero consquently we discard it
            return absolute_num_photon_of_edge[1:]
        else:
            absolute_num_photon_of_edge = np.searchsorted(self.timestamps, edges_time)
            return absolute_num_photon_of_edge[1:] - absolute_num_photon_of_edge[0]

    def get_first_and_last_num_photon(self):
        return np.searchsorted(self.timestamps, (self.start_tick, self.end_tick))


    def create_canonic_graph(self):
        pass

