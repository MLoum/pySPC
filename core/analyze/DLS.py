import numpy as np
from lmfit import minimize, Parameters, Model
from lmfit.models import LinearModel, ExponentialModel

from .FCS import CorrelationMeasurement


# Pour les fits :
def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars

from .correlate import correlate

from threading import Thread

class Cumulant(Model):
    """ with four Parameters ``t0``, ``amp``, ``tau`` and ``cst``.

    Defined as:

    .. math::

        f(t; , t0, amp, tau, cst) = B + beta*np.exp(-2*Gamma*t)*((1 + mu2/2*t**2 - mu3/6*t**3 +  mu4/24*t**4)**2)

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='propagate',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def cumulant(t, B, beta, Gamma, mu2, mu3, mu4):
            return B + beta*np.exp(-2*Gamma*t)*((1 + mu2/2*t**2 - mu3/6*t**3 +  mu4/24*t**4)**2)

        super(Cumulant, self).__init__(cumulant, **kwargs)

    def guess(self, data, x=None, **kwargs):
        B, beta, Gamma, mu2, mu3, mu4 = 0., 0., 0., 0., 0., 0.

        if x is not None:
            B = data[-1] #TODO moyenne des dix (?) derniers points
            beta = data[0] - 1
            #TODO Gamma l'inverse du temps pour laquelle l'amplitude est divisée par e^2
            #Gamma = 1
            #
            # t0 = np.argmax(data)
            # amp = np.max(data)
            # #Searching for position where amp is divided by e=2.71
            # subarray = data[t0:]
            # #tau = np.where(subarray < amp/np.exp(1))[0]
            # tau = np.argmax(subarray < amp / np.exp(1))
            #TODO check if it is not the case
            # cst = np.min(data[np.nonzero(data)]) #Attention aux canaux à zeros

        pars = self.make_params(B=B, beta=beta, Gamma=Gamma, mu2=mu2, mu3=mu3, mu4=mu4)
        return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"


class DLS_Measurements(CorrelationMeasurement):

    def __init__(self, data_=None, timeAxis_= None):
        super().__init__(data_, timeAxis_)



    # def fit(self, idxStart=0, idxEnd=-1):
    #     self.fitResults = self.model.fit(data, self.params)
    #
    # def evalParams(self, idxStart=0, idxEnd=-1):
    #     y = self.model.eval(self.params, t=self.timeAxis)
    #     residuals = y - self.lifetimeHistogram
    #     return y, residuals
    #
    # def guess(self, idxStart=0, idxEnd=-1):
    #     pass
    #     #self.params = self.model.guess(data)



    def setParams(self, params):
        if self.modelName == "Cumulant":
            self.params['B'].set(value=params[0],  vary=True, min=0, max=None)
            self.params['beta'].set(value=params[1], vary=True, min=0, max=None)
            self.params['Gamma'].set(value=params[2], vary=True, min=0, max=None)
            self.params['mu2'].set(value=params[4], vary=True, min=0, max=None)
            self.params['mu3'].set(value=params[5], vary=True, min=None, max=None)
            self.params['mu4'].set(value=params[6], vary=True, min=0, max=None)


    def setModel(self, modelName):
        #il existe une  possibilité pour autoriser le passage d’une infinité de paramètres ! Cela se fait avec *
        if modelName == "Cumulant":
            self.modelName = modelName
            self.model = Cumulant()
            self.params = self.model.make_params(t0=0, amp=1, tau=1, cst=0)



