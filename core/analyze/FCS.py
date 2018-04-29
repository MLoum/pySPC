from multiprocessing import Pool

import numpy as np
from lmfit import minimize, Parameters, Model
from lmfit.models import LinearModel, ExponentialModel

from .correlate import correlate

from threading import Thread







def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars

from .Measurement import Measurements

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
    def __init__(self, data_=None, timeAxis_=None):
        super().__init__(data_, timeAxis_)

    def correlateMonoProc(self, timestamps1, timestamps2, maxCorrelationTimeInTick, startCorrelationTimeInTick=1, nbOfPointPerCascade_aka_B=10):
        self.maxTimeInTick = timestamps1[-1]
        self.numLastPhoton = np.searchsorted(timestamps1, self.maxTimeInTick - maxCorrelationTimeInTick)
        self.endTimeCorrelation_tick = timestamps1[self.numLastPhoton]

        self.createListTimeCorrelation(startCorrelationTimeInTick, maxCorrelationTimeInTick, pointPerDecade=nbOfPointPerCascade_aka_B)
        self.data = np.zeros(self.nbOfCorrelationPoint, dtype=np.int)
        correlate(timestamps1, self.data, self.timeAxis, self.numLastPhoton)
        self.normalizeCorrelation()

    def correlateFCS_multi(self, timestamps1, timestamps2, maxCorrelationTimeInTick):
        self.maxTimeInTick = timestamps1[-1]
        self.numLastPhoton = np.searchsorted(timestamps1, self.maxTimeInTick - maxCorrelationTimeInTick)
        self.endTimeCorrelation_tick = timestamps1[self.numLastPhoton]

        nbOfChunk = 10
        timeStamp1_chunck = np.array_split(timestamps1, nbOfChunk)
        timeStamp2_chunck = np.array_split(timestamps2, nbOfChunk)
        self.createListTimeCorrelation(maxCorrelationTimeInTick, pointPerDecade=10)
        correlation_chunck = np.zeros(self.nbOfCorrelationPoint, nbOfChunk)
        #
        # pool = Pool(processes=3)
        # print(pool.map(target = correlate, numbers))

    def normalizeCorrelation(self):
        self.data = self.data.astype(np.float64)
        B = self.pointPerDecade
        for n in range(1, self.nbOfcascade):
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

    def createListTimeCorrelation(self, startCorrelationTimeInTick, maxCorrelationTime_tick, pointPerDecade):

        B = self.pointPerDecade = pointPerDecade
        # How many "cascade" do we need ?
        # maxCorrelationTime_tick =  2^(n_casc - 1/B)
        # then ln (maxCorrelationTime_tick) = (n_casc - 1/B) ln 2
        self.nbOfcascade = int(np.log(maxCorrelationTime_tick) / np.log(2) + 1 / B)

        """
                 |
                 | 1                                si j = 1
         tau_j = |
                 |                  j - 1
                 | tau_(j-1) + 2^( -------)         si j > 1      ATTENTION division entre integer
                                      B                           i.e on prend la partie entiere !!

        """
        # TODO Total vectorisation ?
        self.nbOfCorrelationPoint = int(self.nbOfcascade * B)  # +1 ?
        taus = np.zeros(self.nbOfCorrelationPoint, dtype=np.int)
        taus[:B] = np.arange(B) + 1
        for n in range(1, self.nbOfcascade):
            taus[n * B:(n + 1) * B] = taus[:B] * np.power(2, n) + taus[n * B - 1]
        self.timeAxis = taus

    def setParams(self, params):
        if self.modelName == "1 Diff":
            self.params['G0'].set(value=params[0], vary=True, min=0, max=None)
            self.params['tdiff'].set(value=params[1], vary=True, min=0, max=None)
            self.params['cst'].set(value=params[2], vary=True, min=0, max=None)

    def setModel(self, modelName):
        # il existe une  possibilité pour autoriser le passage d’une infinité de paramètres ! Cela se fait avec *
        if modelName == "1 Diff":
            self.modelName = modelName
            self.model = OneSpeDiffusion()
            self.params = self.model.make_params(G0=1.5, tdiff=500, cst=1)

class FCSMeasurements(CorrelationMeasurement):

    def __init__(self, correlationCurve=None, timeAxis_= None):
        super().__init__(correlationCurve, timeAxis_)


    def setParams(self, params):
        if self.modelName == "1 Diff":
            self.params['G0'].set(value=params[0],  vary=True, min=0, max=None)
            self.params['tdiff'].set(value=params[1], vary=True, min=0, max=None)
            self.params['cst'].set(value=params[2], vary=True, min=0, max=None)



    def setModel(self, modelName):
        #il existe une  possibilité pour autoriser le passage d’une infinité de paramètres ! Cela se fait avec *
        if modelName == "1 Diff":
            self.modelName = modelName
            self.model = OneSpeDiffusion()
            self.params = self.model.make_params(G0=1.5, tdiff=500, cst=1)




