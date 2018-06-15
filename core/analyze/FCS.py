import multiprocessing as mp

import numpy as np
from lmfit import minimize, Parameters, Model
from lmfit.models import LinearModel, ExponentialModel

# from .correlate import correlate

from threading import Thread

from core.analyze.pycorrelate import pcorrelate

import numba




def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars

from core.analyze.Measurement import Measurements

class OneSpeDiffusion(Model):
    """A exponential decay with a shift in time, with four Parameters ``t0``, ``amp``, ``tau`` and ``cst``.

    Defined as:

    .. math::

        f(t; , G0, tdiff, cst) = cst + G0/(1+t/tdiff)

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def oneSpeDiffusion(t, G0, tdiff, cst):
            return cst + G0/(1+t/tdiff)

        super(OneSpeDiffusion, self).__init__(oneSpeDiffusion, **kwargs)

    def guess(self, data, x=None, **kwargs):
        G0, tdiff, cst = 0., 0., 0., 0.
        #if x is not None:
        G0 = data[0] #beawre afterpulsing...
        cst = np.mean(data[-10:])
        #Searching for position where G0 is divided by 2
        subarray = data[t0:]
        #tau = np.where(subarray < amp/np.exp(1))[0]
        tdiff = np.argmax(data < (float) (G0) / 2)

        pars = self.make_params(G0=G0, tdiff=tdiff, cst=cst)
        return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"


class CorrelationMeasurement(Measurements):
    def __init__(self, data_=None, time_axis_=None):
        super().__init__(data_, time_axis_)

    def correlateMonoProc(self, timestamps1, timestamps2, maxCorrelationTimeInTick, startCorrelationTimeInTick=1, nbOfPointPerCascade_aka_B=10, tick_duration_micros=1):
        self.tick_duration_micros = tick_duration_micros
        self.maxTimeInTick = timestamps1[-1]
        self.numLastPhoton = np.searchsorted(timestamps1, self.maxTimeInTick - maxCorrelationTimeInTick)
        self.endTimeCorrelation_tick = timestamps1[self.numLastPhoton]

        self.create_list_time_correlation(startCorrelationTimeInTick, maxCorrelationTimeInTick, pointPerDecade=nbOfPointPerCascade_aka_B)
        self.data = np.zeros(self.nbOfCorrelationPoint, dtype=np.int)

        # correlate(timestamps, self.data, self.timeAxis, self.numLastPhoton)
        self.data = pcorrelate(t=timestamps1, u=timestamps1, bins=self.timeAxis, normalize=True)
        # self.pcorrelate_me(timestamps1, self.timeAxis, self.data)

        # self.normalize_correlation()
        self.scale_time_axis()
        self.timeAxis = self.timeAxis[:-1]

    @numba.jit(nopython=True)
    def pcorrelate_me(self, timestamps, taus, G):

        numLastPhoton = np.size(timestamps)
        nbOfTau = np.size(taus)
        for n in range(numLastPhoton):
            # if n%100000==0:
            #     print(n)
            idx_tau = 0
            j = n + 1
            while(idx_tau < nbOfTau):
                # First tau is not necesseraly 1
                while timestamps[j] - timestamps[n]  < taus[0] - 1:
                    j += 1

                while timestamps[j] - timestamps[n]  < taus[idx_tau]:
                    G[idx_tau] += 1
                    j += 1
                    # if j == nbOfPhoton:
                    #     break
                idx_tau += 1

    def correlateFCS_multi(self, timestamps_1, timestamps_2, max_correlation_time_in_tick, start_correlation_time_in_tick=1, nb_of_point_per_cascade_aka_B=10, tick_duration_micros=1):
        self.tick_duration_micros = tick_duration_micros

        # TODO nb_of_chunk based on max correlation time and the max Time of the file.

        # nb_of_chunk = 10
        # nb_of_workers = 4
        #
        # # Split the timeStamps in 10 (?) array
        # self.maxTimeInTick_1 = timestamps_1[-1]
        # self.maxTimeInTick_2 = timestamps_2[-1]
        #
        # chunks_of_timestamps_1 = np.split(timestamps_1, nb_of_chunk)
        # chunks_of_timestamps_2 = np.split(timestamps_2, nb_of_chunk)
        #
        #
        #
        #
        # self.numLastPhoton = np.searchsorted(timestamps_1, self.maxTimeInTick - max_correlation_time_in_tick)
        # self.endTimeCorrelation_tick = timestamps_1[self.numLastPhoton]
        #
        # self.create_list_time_correlation(start_correlation_time_in_tick, max_correlation_time_in_tick,
        #                                   pointPerDecade=nb_of_point_per_cascade_aka_B)
        # self.data = np.zeros(self.nbOfCorrelationPoint, dtype=np.int)
        #
        # processes = [mp.Process(target=pcorrelate, args=(chunks_of_timestamps_1[i], chunks_of_timestamps_2[i], bins,
        #                                                  self.timeAxis, normalize=True)) for x in range(nb_of_workers)]
        # # Run processes
        # for p in processes:
        #     p.start()
        #
        # # Exit the completed processes
        # for p in processes:
        #     p.join()



        # correlate(timestamps, self.data, self.timeAxis, self.numLastPhoton)
        # self.data = pcorrelate(t=timestamps_1, u=timestamps_1, bins=self.timeAxis, normalize=True)
        # self.pcorrelate_me(timestamps1, self.timeAxis, self.data)

        # self.normalize_correlation()
        # self.scale_time_axis()
        # self.timeAxis = self.timeAxis[:-1]

    def normalize_correlation(self):
        self.data = self.data.astype(np.float64)
        B = self.pointPerDecade
        for n in range(1, self.nb_of_cascade):
            self.data[n * B:(n + 1) * B] /= np.power(2, n)

        self.data *= (self.maxTimeInTick - self.timeAxis) / (self.numLastPhoton ** 2)

        # maxCorrelationTimeInTick = 100000
        # G = np.zeros(maxCorrelationTimeInTick, dtype=np.int)
        # temp = np.zeros(maxCorrelationTimeInTick, dtype=np.int)
        # nbOfTick = np.size(self.data)
        # #correlate(timestamps1, G, temp, maxCorrelationTimeInTick, nbOfTick)
        # self.data = correlate(timestamps1, G, maxCorrelationTimeInTick)
        # #self.data = G
        # self.timeAxis = np.arange(maxCorrelationTimeInTick)

    def scale_time_axis(self):
        self.timeAxis = self.tick_duration_micros * self.timeAxis.astype(np.float64)

    def create_list_time_correlation(self, startCorrelationTimeInTick, maxCorrelationTime_tick, pointPerDecade):

        B = self.pointPerDecade = pointPerDecade
        # How many "cascade" do we need ?
        # maxCorrelationTime_tick =  2^(n_casc - 1/B)
        # then ln (maxCorrelationTime_tick) = (n_casc - 1/B) ln 2
        self.nb_of_cascade = int(np.log(maxCorrelationTime_tick) / np.log(2) + 1 / B)

        """
                 |
                 | 1                                si j = 1
         tau_j = |
                 |                  j - 1
                 | tau_(j-1) + 2^( -------)         si j > 1      ATTENTION division entre integer
                                      B                           i.e on prend la partie entiere !!

        """
        # TODO Total vectorisation ?
        self.nbOfCorrelationPoint = int(self.nb_of_cascade * B)  # +1 ?
        taus = np.zeros(self.nbOfCorrelationPoint, dtype=np.int)
        taus[:B] = np.arange(B) + 1
        for n in range(1, self.nb_of_cascade):
            taus[n * B:(n + 1) * B] = taus[:B] * np.power(2, n) + taus[n * B - 1]
        taus += startCorrelationTimeInTick
        self.timeAxis = taus

    def set_params(self, params):
        if self.modelName == "1 Diff":
            self.params['G0'].set(value=params[0], vary=True, min=0, max=None)
            self.params['tdiff'].set(value=params[1], vary=True, min=0, max=None)
            self.params['cst'].set(value=params[2], vary=True, min=0, max=None)

    def set_model(self, modelName):
        # il existe une  possibilité pour autoriser le passage d’une infinité de paramètres ! Cela se fait avec *
        if modelName == "1 Diff":
            self.modelName = modelName
            self.model = OneSpeDiffusion()
            self.params = self.model.make_params(G0=1.5, tdiff=500, cst=1)

class FCSMeasurements(CorrelationMeasurement):

    def __init__(self, correlationCurve=None, time_axis_= None):
        super().__init__(correlationCurve, time_axis_)


    def set_params(self, params):
        if self.modelName == "1 Diff":
            self.params['G0'].set(value=params[0],  vary=True, min=0, max=None)
            self.params['tdiff'].set(value=params[1], vary=True, min=0, max=None)
            self.params['cst'].set(value=params[2], vary=True, min=0, max=None)



    def set_model(self, modelName):
        #il existe une  possibilité pour autoriser le passage d’une infinité de paramètres ! Cela se fait avec *
        if modelName == "1 Diff":
            self.modelName = modelName
            self.model = OneSpeDiffusion()
            self.params = self.model.make_params(G0=1.5, tdiff=500, cst=1)




