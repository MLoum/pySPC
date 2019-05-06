import numpy as np
from lmfit import minimize, Parameters, Model
import matplotlib.pyplot as plt


from .Measurement import Measurements


class PTOFS(Measurements):

    def __init__(self, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exp_param, num_channel, start_tick, end_tick, "lifetime", name, comment)

    def from_microtime_to_spectrum(self, microtimes, fiber_length, wl_calib, microtime_calib, wl_min, wl_max):
        self.fiber_length = fiber_length
        self.microtimes = microtimes
        self.wl_calib = wl_calib
        self.microtime_calib = microtime_calib
        self.wl_min = wl_min
        self.wl_max = wl_max

        self.data = np.zeros(microtimes.size)
        self.data[microtime_calib] = 0



        def calculate_index_and_derivative(wl):
            index = np.sqrt(1 + (0.6961663 * wl * wl) / (wl * wl - 0.0684043 * 0.0684043)
                            + (0.4079426 * wl * wl) / (wl * wl - 0.1162414 * 0.1162414)
                            + (0.8974794 * wl * wl) / (wl * wl - 9.896161 * 9.896161)
                            )

            index_derivative =\
                (
                        - (1.79496 * wl * wl * wl) / (pow(-97.934 + wl * wl, 2))
                        + (1.79496 * wl) / (-97.934 + wl * wl)

                        - (0.815885 * wl * wl * wl) / (pow(-0.0135121 + wl * wl, 2))
                        + (0.815885 * wl) / (-0.0135121 + wl * wl)

                        - (1.39233 * wl * wl * wl) / (pow(-0.00467915 + wl * wl, 2))
                        + (1.39233 * wl) / (-0.00467915 + wl * wl)
                )\
                /\
                (2 * np.sqrt(
                    1
                    + (0.897479 * wl * wl) / (-97.934 + wl * wl)
                    + (0.407943 * wl * wl) / (-0.0135121 + wl * wl)
                    + (0.696166 * wl * wl) / (-0.00467915 + wl * wl)
                )
                 )

    for microtime in microtimes:
            if microtime == microtime_calib:
                pass
