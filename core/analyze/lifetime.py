import numpy as np
from lmfit import minimize, Parameters, Model
from lmfit.models import LinearModel, ExponentialModel

from .Measurement import Measurements
from .histogram import histogram

def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars

class OneExpDecay(Model):
    """A exponential decay with a shift in time, with four Parameters ``t0``, ``amp``, ``tau`` and ``cst``.

    Defined as:

    .. math::

        f(t; , t0, amp, tau, cst) = cst + amp * exp(-(t + t0) / tau)

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='propagate',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def oneExpDecay(t, t0, amp, tau, cst):
            return cst + amp * np.exp(-(t - t0) / tau)

        super(OneExpDecay, self).__init__(oneExpDecay, **kwargs)

    def guess(self, data, x=None, **kwargs):
        t0, amp, tau, cst = 0., 0., 0., 0.
        #if x is not None:
        idx_t0 = np.argmax(data)
        t0 = x[idx_t0]
        amp = np.max(data)
        #Searching for position where amp is divided by e=2.71
        subarray = data[idx_t0:]
        #tau = np.where(subarray < amp/np.exp(1))[0]
        idx_tau = np.argmax(subarray < (float) (amp) / np.exp(1))
        tau = x[idx_tau] - t0
        #TODO check if it is not the case
        cst = np.min(data[np.nonzero(data)]) #Attention aux canaux à zeros

        pars = self.make_params(t0=t0, amp=amp, tau=tau, cst=cst)
        return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"

#TODO creer une classe mère pour les analyses.
class lifeTimeMeasurements(Measurements):

    def __init__(self, lifetimeHistogram_=None, time_axis_= None):
        super().__init__(lifetimeHistogram_, time_axis_)

        self.IR = None
        self.data = lifetimeHistogram_
        self.timeAxis = time_axis_

        self.eval_x_axis = None
        self.eval_y_axis = None

        self.residuals = None

        self.fitResults = None


    def createHistogramm(self, nanotimes, nbOfMicrotimeChannel, mIcrotime_clickEquivalentIn_second):
        #self.data, self.timeAxis = np.histogram(nanotimes, nbOfMicrotimeChannel)
        self.data = np.zeros(nbOfMicrotimeChannel, dtype=np.uint)
        #self.timeAxis = self.timeAxis[:-1]
        self.timeAxis = np.arange(nbOfMicrotimeChannel)*mIcrotime_clickEquivalentIn_second*1E9
        #Doesnot work ??? TODO more pythonic
        #self.data[nanotimes] += 1
        histogram(nanotimes, self.data)
        self.trimLifeTimeCurve()

    def trimLifeTimeCurve(self):
        nonzero = np.nonzero(self.data)
        idxStartNonZero = nonzero[0][0]
        idxEndNonZero = nonzero[0][-1]
        self.data = self.data[idxStartNonZero:idxEndNonZero]
        self.timeAxis = self.timeAxis[idxStartNonZero:idxEndNonZero]


    def shiftHistogramm(self, shift):
        self.data = np.roll(self.data, shift)

    def setIR(self, IR):
        self.IR = IR

    def shiftIR(self, shift):
        """
        Shift in nb of microtime channel
        """
        self.IR = np.roll(self.IR, shift)

    def generateArtificialIR(self, mainWidth, secondaryWidth, secondaryAmplitude, timeOffset):
        self.IR = (1-secondaryWidth) * np.exp( - (self.eval_x_axis - timeOffset)**2/mainWidth) + secondaryAmplitude * np.exp( - (self.eval_x_axis - timeOffset)**2/secondaryWidth)
        #scale
        self.IR *= np.max(self.data)

    def convolveWithIR(self):
        return np.convolve(self.data , self.IR)

    # def fit(self, idxStart=0, idxEnd=-1):
    #     self.fitResults = self.model.fit(self.lifetimeHistogram[idxStart:idxEnd], self.params)
    #     self.evalParams(idxStart, idxEnd)
    #
    # def evalParams(self, idxStart=0, idxEnd=-1):
    #     x = self.timeAxis[idxStart:idxEnd]
    #     self.eval_y_axis = self.model.eval(self.params, t=x)
    #     self.residuals = self.eval_y_axis - self.lifetimeHistogram[idxStart:idxEnd]
    #     self.eval_x_axis = self.timeAxis[idxStart:idxEnd]
    #
    # def guess(self, idxStart=0, idxEnd=-1):
    #     self.params = self.model.guess(self.lifetimeHistogram)
    #     self.evalParams(idxStart, idxEnd)

    def set_params(self, params):
        if self.modelName == "One Decay":
            self.params['t0'].set(value=params[0],  vary=True, min=0, max=None)
            self.params['amp'].set(value=params[1], vary=True, min=0, max=None)
            self.params['tau'].set(value=params[2], vary=True, min=0, max=None)
            self.params['cst'].set(value=params[3], vary=True, min=0, max=None)


    def set_model(self, modelName):
        #il existe une  possibilité pour autoriser le passage d’une infinité de paramètres ! Cela se fait avec *
        if modelName == "One Decay":
            self.modelName = modelName
            self.model = OneExpDecay()
            self.params = self.model.make_params(t0=0, amp=1, tau=1, cst=0)




