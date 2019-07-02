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

        f(t; , t0, amp, tau, cst) = B + beta*np.exp(-2*Gamma*t)*((1 + mu2/2*t**2/Gamma**2 - mu3/6*t**3/Gamma**3 +  mu4/24*t**4/Gamma**4)**2)

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='propagate',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def cumulant(t, B, beta, tau, mu2, mu3, mu4):
            return B + beta*np.exp(-t/tau)*((1 + mu2/2*t**2/tau**2 - mu3/6*t**3/tau**3 +  mu4/24*t**4/tau**4)**2)

        super(Cumulant, self).__init__(cumulant, **kwargs)

    def guess(self, data, x=None, **kwargs):
        B, beta, tau, mu2, mu3, mu4 = 0., 0., 0., 0., 0., 0.

        if x is not None:
            B = np.mean(data[-10:-1]) #TODO moyenne des dix (?) derniers points
            beta = np.mean(data[0:10]) - 1
            #TODO Gamma l'inverse du temps pour laquelle l'amplitude est divisée par e^2
            # t0 = np.argmax(data)
            # amp = np.max(data)
            # #Searching for position where amp is divided by e=2.71

            idx_tau = np.where(data < B + beta/np.exp(1))[0][0]
            tau = x[idx_tau]
            #TODO check if it is not the case
            # cst = np.min(data[np.nonzero(data)]) #Attention aux canaux à zeros

        pars = self.make_params(B=B, beta=beta, tau=tau, mu2=mu2, mu3=mu3, mu4=mu4)
        return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"


class DLS_Measurements(CorrelationMeasurement):

    def __init__(self, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment="", logger=None):
        super().__init__(exp_param, num_channel, start_tick, end_tick, "DLS", name, comment, logger)


    def set_params(self, params):
        if self.modelName == "Cumulant":
            self.params['B'].set(value=params[0],  vary=True, min=0, max=None)
            self.params['beta'].set(value=params[1], vary=True, min=0, max=None)
            self.params['tau'].set(value=params[2], vary=True, min=0, max=None)
            self.params['mu2'].set(value=params[3], vary=True, min=0, max=None)
            self.params['mu3'].set(value=params[4], vary=True, min=None, max=None)
            self.params['mu4'].set(value=params[5], vary=True, min=0, max=None)


    def set_model(self, modelName):
        if modelName == "Cumulant":
            self.modelName = modelName
            self.model = Cumulant()
            self.params = self.model.make_params(B=1, beta=1, tau=100, mu2=0, mu3=0, mu4=0)

    def export(self, file_path = None):
        data = np.column_stack((self.data, self.timeAxis))
        np.savetxt(file_path, data)




