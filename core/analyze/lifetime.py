import numpy as np
from lmfit import minimize, Parameters, Model, fit_report, Minimizer
from lmfit.models import LinearModel, ExponentialModel
import matplotlib.pyplot as plt


from scipy.special import erfc
from scipy.ndimage.interpolation import shift as shift_scipy
from scipy.linalg import lstsq

from .Measurement import Measurements
from scipy.integrate import cumtrapz
import copy
from .histogram import histogram

def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars



class lifetimeModelClass():
    def __init__(self, IRF=None):
        self.IRF = IRF
        # self.x_range = None
        self.data = None
        self.non_convoluted_decay = None
        self.observed_count = 0

        # self.t0 = None
        self.data_bckgnd = None

        self.ini_params = {}

    def guess(self, t, params):
        pass

    def eval(self, t, params):
        pass

class OneExpDecay(lifetimeModelClass):
    """A normalized exponential decay with a shift in time, with two Parameters ``tau``, ''shift, and a fixed ``bckgnd``.

    Defined as:

    .. math::

        f(t; , tau, shift) = IRF(shift) x exp(-t / tau) + bckgnd

    The area under the decay curves obtained from the observed counts Cexp and from the predicted counts Ĉtheo must be
    conserved during optimization of the fitting parameters. Hence, the exponential does'nt have a amplitude parameter
    """
    def __init__(self, IRF=None):
        super().__init__(IRF)

    def guess(self, t, params):

        # if x is not None:
        # cst is the background and is taken, as a first guess, as the minimum of the lifetime curve
        # We trim the last point because the TAC tipycally has some problem here.
        # cst = np.min(self.data[np.nonzero(data[:-10])])
        amp = np.max(self.data)

        # if we don't take into account the IR, t0 is simply the max of the lifetime curve
        idx_t0 = idx_max = np.argmax(self.data)

        t0 = t[idx_t0]

        # Searching for position where amp is divided by e=2.71
        subarray = self.data[idx_max:]
        x_subarray = t[idx_max:]
        # tau = np.where(subarray < amp/np.exp(1))[0]
        idx_tau = np.argmax(subarray < (float)(amp) / np.exp(1))
        params['tau'].value = x_subarray[idx_tau] - t0
        self.ini_params['tau'] = params['tau'].value

        # Mean
        dt = t[1] - t[0]
        tau = np.sum(subarray)/amp*dt
        params['tau'].value = tau
        self.ini_params['tau'] = tau

    def eval(self, t,  params):
        tau = params['tau'].value
        shift = params['IRF_shift'].value
        self.data_bckgnd = params['bckgnd'].value
        bckgnd_corrected_data = self.data - self.data_bckgnd
        bckgnd_corrected_data[bckgnd_corrected_data < 0] = 0
        self.observed_count = (bckgnd_corrected_data).sum()
        # self.observed_count = (self.data - self.data_bckgnd).sum()
        self.non_convoluted_decay = np.exp(-(t) / tau)
        # t_0 is in the shift

        if self.IRF is not None:
            IR = shift_scipy(self.IRF.processed_data, shift, mode='wrap')
            # IR = np.roll(self.IR, shift)
            conv = np.convolve(self.non_convoluted_decay, IR)[0:np.size(self.non_convoluted_decay)]
            # Test if sum is different from zero ?
            conv /= conv.sum()
            # return conv[self.x_range[0]:self.x_range[1]] + self.data_bckgnd
            return self.observed_count*conv + self.data_bckgnd
        else:
            self.non_convoluted_decay /= self.non_convoluted_decay.sum()
            return self.observed_count*self.non_convoluted_decay + self.data_bckgnd

    def make_params(self):
        params = Parameters()
        params.add(name="tau", value=1, min=0.01, max=np.inf, brute_step=0.1)
        params.add(name="IRF_shift", value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        params.add(name="bckgnd", vary=False, value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        return params


class OneExpDecay_tail(lifetimeModelClass):
    """A normalized exponential decay with a shift in time, with two Parameters ``tau``, ''shift, and a fixed ``bckgnd``.

    Defined as:

    .. math::

        f(t; , tau, t0, bckgnd) =  exp(-(t-t0) / tau) + bckgnd

    The area under the decay curves obtained from the observed counts Cexp and from the predicted counts Ĉtheo must be
    conserved during optimization of the fitting parameters. Hence, the exponential does'nt have a amplitude parameter
    """

    def __init__(self, IRF=None):
        super().__init__(IRF)

    def guess(self, t, params):
        # Trouver t0 de manière algorithmique

        # Integrer la courbe pour obtenir Amp*tau

        # Le décalage t0 max connaissant le temps de l'irf permet d'en deduire la vraie amplitude.
        amp = np.max(self.data)

        # if we don't take into account the IR, t0 is simply the max of the lifetime curve
        idx_t0 = idx_max = np.argmax(self.data)

        t0 = t[idx_t0]


        # Searching for position where amp is divided by e=2.71
        subarray = self.data[idx_max:]
        x_subarray = t[idx_max:]
        # tau = np.where(subarray < amp/np.exp(1))[0]
        idx_tau = np.argmax(subarray < (float)(amp) / np.exp(1))
        params['tau'].value = x_subarray[idx_tau] - t0
        self.ini_params['tau'] = params['tau'].value

        dt = t[1] - t[0]
        tau = np.sum(subarray) / amp * dt
        params['tau'].value = tau
        self.ini_params['tau'] = tau

    def eval(self, t,  params):
        tau = params['tau'].value
        t0 = params['t0'].value
        self.data_bckgnd = params['bckgnd'].value
        bckgnd_corrected_data = self.data - self.data_bckgnd
        bckgnd_corrected_data[bckgnd_corrected_data < 0] = 0
        self.observed_count = (bckgnd_corrected_data).sum()
        # self.observed_count = (self.data - self.data_bckgnd).sum()
        self.non_convoluted_decay = np.exp(-(t-t0) / tau)
        self.non_convoluted_decay[t < t0] = 0
        # t_0 is in the shift

        self.non_convoluted_decay /= self.non_convoluted_decay.sum()
        return self.observed_count*self.non_convoluted_decay + self.data_bckgnd

    def make_params(self):
        params = Parameters()
        params.add(name="tau", value=1, min=0.01, max=np.inf, brute_step=0.1)
        params.add(name="t0", value=0, min=0, max=np.inf, brute_step=0.1)
        params.add(name="bckgnd", vary=False, value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        return params

class TwoExpDecay(lifetimeModelClass):
    """A normalized exponential decay with a shift in time, with two Parameters ``tau``, ''shift, and a fixed ``bckgnd``.

    Defined as:

    .. math::

        f(t; , tau1, a1, tau2, shift) = IRF(shift) x (a1 . exp(-t / tau1) + (1-a1) . exp(-t / tau2) ) + bckgnd

    The area under the decay curves obtained from the observed counts Cexp and from the predicted counts Ĉtheo must be
    conserved during optimization of the fitting parameters. Hence, the exponential does'nt have a amplitude parameter
    """
    def __init__(self, IRF=None):
        super().__init__(IRF)

    def guess(self):
        pass

    def eval(self, t,  params):
        tau1 = params['tau1'].value
        a1 = params['a1'].value
        tau2 = params['tau2'].value
        shift = params['shift'].value
        self.data_bckgnd = params['bckgnd'].value
        bckgnd_corrected_data = self.data - self.data_bckgnd
        bckgnd_corrected_data[bckgnd_corrected_data < 0] = 0
        self.observed_count = (bckgnd_corrected_data).sum()

        self.non_convoluted_decay = a1*np.exp(-t/tau1) + (1-a1)*np.exp(-t/tau2)
        # t_0 is in the shift

        if self.IRF is not None:
            IR = shift_scipy(self.IRF.processed_data, shift, mode='wrap')
            # IR = np.roll(self.IR, shift)
            IR /= IR.sum()
            conv = np.convolve(self.non_convoluted_decay, IR)[0:np.size(self.non_convoluted_decay)]
            # transform count to probability
            conv /= conv.sum()
            # return conv[self.x_range[0]:self.x_range[1]] + self.data_bckgnd
            return self.observed_count*conv + self.data_bckgnd
        else:
            self.non_convoluted_decay /= self.non_convoluted_decay.sum()
            return self.non_convoluted_decay + self.data_bckgnd

    def make_params(self):
        params = Parameters()
        params.add(name="tau1", value=1, min=0, max=np.inf, brute_step=0.1)
        params.add(name="a1", value=1, min=0, max=np.inf, brute_step=0.1)
        params.add(name="tau2", value=1, min=0, max=np.inf, brute_step=0.1)
        params.add(name="shift", value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        params.add(name="bckgnd", vary=False, value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        return params

class TwoExpDecay_a1a2(lifetimeModelClass):
    """A normalized exponential decay with a shift in time, with two Parameters ``tau``, ''shift, and a fixed ``bckgnd``.

    Defined as:

    .. math::

        f(t; , tau1, a1, tau2, shift) = IRF(shift) x (a1 . exp(-t / tau1) + (1-a1) . exp(-t / tau2) ) + bckgnd

    The area under the decay curves obtained from the observed counts Cexp and from the predicted counts Ĉtheo must be
    conserved during optimization of the fitting parameters. Hence, the exponential does'nt have a amplitude parameter
    """
    def __init__(self, IRF=None):
        super().__init__(IRF)

    def guess(self, t, params):
        pass

    def eval(self, t,  params):
        tau1 = params['tau1'].value
        a1 = params['a1'].value
        a2 = params['a2'].value
        tau2 = params['tau2'].value
        shift = params['shift'].value
        self.data_bckgnd = params['bckgnd'].value

        self.non_convoluted_decay = a1*np.exp(-t/tau1) + a2*np.exp(-t/tau2)
        # t_0 is in the shift

        if self.IRF is not None:
            IR = shift_scipy(self.IRF.processed_data, shift, mode='wrap')
            IR /= IR.sum()
            conv = np.convolve(self.non_convoluted_decay, IR)[0:np.size(self.non_convoluted_decay)]
            return conv + self.data_bckgnd
        else:
            # self.non_convoluted_decay /= self.non_convoluted_decay.sum()
            return self.non_convoluted_decay + self.data_bckgnd

    def make_params(self):
        params = Parameters()
        params.add(name="tau1", value=1, min=0, max=np.inf, brute_step=0.1)
        params.add(name="a1", value=1, min=0, max=np.inf, brute_step=0.1)
        params.add(name="tau2", value=1, min=0, max=np.inf, brute_step=0.1)
        params.add(name="a2", value=1, min=0, max=np.inf, brute_step=0.1)
        params.add(name="shift", value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        params.add(name="bckgnd", vary=False, value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        return params

class TwoExpDecay_tail(lifetimeModelClass):
    """A normalized exponential two decays with a shift in time, with two Parameters ``tau``, ''shift, and a fixed ``bckgnd``.

    Defined as:

    .. math::

        f(t; , tau, t0, bckgnd) =  exp(-(t-t0) / tau) + bckgnd

    The area under the decay curves obtained from the observed counts Cexp and from the predicted counts Ĉtheo must be
    conserved during optimization of the fitting parameters. Hence, the exponential does'nt have a amplitude parameter
    """

    def __init__(self, IRF=None):
        super().__init__(IRF)

    def guess(self, t, params):
        pass



    def eval(self, t,  params):
        tau1 = params['tau1'].value
        tau2 = params['tau2'].value
        a1 = params['a1'].value
        t0 = params['t0'].value
        self.data_bckgnd = params['bckgnd'].value
        bckgnd_corrected_data = self.data - self.data_bckgnd
        bckgnd_corrected_data[bckgnd_corrected_data < 0] = 0
        self.observed_count = (bckgnd_corrected_data).sum()
        # self.observed_count = (self.data - self.data_bckgnd).sum()
        self.non_convoluted_decay = a1*np.exp(-t/tau1) + (1-a1)*np.exp(-t/tau2)
        self.non_convoluted_decay[t < t0] = 0
        # t_0 is in the shift

        self.non_convoluted_decay /= self.non_convoluted_decay.sum()
        return self.observed_count*self.non_convoluted_decay + self.data_bckgnd

    def make_params(self):
        params = Parameters()
        params.add(name="t0", value=0, min=0, max=np.inf, brute_step=0.1)
        params.add(name="tau1", value=1, min=0.01, max=np.inf, brute_step=0.1)
        params.add(name="a1", value=0.5, min=0, max=1, brute_step=0.1)
        params.add(name="tau2", value=1, min=0.01, max=np.inf, brute_step=0.1)
        params.add(name="bckgnd", vary=False, value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        return params

class ThreeExpDecay_tail_a1a2a3(lifetimeModelClass):
    """A normalized exponential two decays with a shift in time, with two Parameters ``tau``, ''shift, and a fixed ``bckgnd``.

    Defined as:

    .. math::

        f(t; , tau, t0, bckgnd) =  a1*exp(-(t-t0) / tau1) + a2*exp(-(t-t0) / tau2) + bckgnd

    The area under the decay curves obtained from the observed counts Cexp and from the predicted counts Ĉtheo must be
    conserved during optimization of the fitting parameters. Hence, the exponential does'nt have a amplitude parameter
    """

    def __init__(self, IRF=None):
        super().__init__(IRF)

    def guess(self, t, params):
        x = t
        y = self.data
        #TODO better (?) integration (simpson, smoothed gaussian quadrature https://austingwalters.com/gaussian-quadrature/)
        # Use Lagrange polynomial to approximate noisy data with a smooth function.
        # Integrate these smooth function with a Gauss-Legendre quadrature, which guarantee to obtain exact integral of smooth approximation.

        iy1 = np.empty_like(y)
        iy1[0] = 0
        iy1[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))

        iy2 = np.empty_like(y)
        iy2[0] = 0
        iy2[1:] = np.cumsum(0.5 * (iy1[1:] + iy1[:-1]) * np.diff(x))

        Y = np.array([iy1, iy2, x**2, x, np.ones(x.size)])
        Y = np.transpose(Y)

        A = np.dot(np.linalg.pinv(Y), y)

        A_tranfer = np.array([[A[0], A[1]], [1, 0]])

        lambdas = np.linalg.eig(A_tranfer)[0]

        X = [np.ones(x.size), np.exp(lambdas[0] * x), np.exp(lambdas[1] * x)]
        X = np.transpose(X)
        P = np.dot(np.linalg.pinv(X), y)

        params['tau1'].value = lambdas[0]
        params['tau2'].value = lambdas[1]
        params['bckgnd'].value = P[0]
        params['a1'].value = P[1]
        params['a2'].value = P[2]

    def eval(self, t,  params):
        tau1 = params['tau1'].value
        a1 = params['a1'].value
        tau2 = params['tau2'].value
        a2 = params['a2'].value
        tau3 = params['tau3a'].value
        a3 = params['a3'].value
        t0 = params['t0'].value
        self.data_bckgnd = params['bckgnd'].value
        self.non_convoluted_decay = a1*np.exp(-(t-t0) / tau1) + a2*np.exp(-(t-t0) / tau2) + a3*np.exp(-(t-t0) / tau3)
        self.non_convoluted_decay[t < t0] = 0

        return self.non_convoluted_decay + self.data_bckgnd

    def make_params(self):
        params = Parameters()
        params.add(name="tau1", value=1, min=0.01, max=np.inf, brute_step=0.1)
        params.add(name="a1", value=0.5, min=0, max=np.inf, brute_step=0.1)
        params.add(name="tau2", value=3, min=0.01, max=np.inf, brute_step=0.1)
        params.add(name="a2", value=0.5, min=0, max=np.inf, brute_step=0.1)
        params.add(name="tau3", value=5, min=0.01, max=np.inf, brute_step=0.1)
        params.add(name="a3", value=0.5, min=0, max=np.inf, brute_step=0.1)
        params.add(name="t0", value=0, min=0, max=np.inf, brute_step=0.1)
        params.add(name="bckgnd", vary=False, value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        return params


class TwoExpDecay_tail_a1a2(lifetimeModelClass):
    """A normalized exponential two decays with a shift in time, with two Parameters ``tau``, ''shift, and a fixed ``bckgnd``.

    Defined as:

    .. math::

        f(t; , tau, t0, bckgnd) =  a1*exp(-(t-t0) / tau1) + a2*exp(-(t-t0) / tau2) + bckgnd

    The area under the decay curves obtained from the observed counts Cexp and from the predicted counts Ĉtheo must be
    conserved during optimization of the fitting parameters. Hence, the exponential does'nt have a amplitude parameter
    """

    def __init__(self, IRF=None):
        super().__init__(IRF)

    def guess(self, t, params):
        #TODO better (?) integration (simpson, smoothed gaussian quadrature https://austingwalters.com/gaussian-quadrature/)
        # Use Lagrange polynomial to approximate noisy data with a smooth function.
        # Integrate these smooth function with a Gauss-Legendre quadrature, which guarantee to obtain exact integral of smooth approximation.

        """
        iy1 = np.empty_like(y)
        iy1[0] = 0
        iy1[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))

        iy2 = np.empty_like(y)
        iy2[0] = 0
        iy2[1:] = np.cumsum(0.5 * (iy1[1:] + iy1[:-1]) * np.diff(x))

        Y = np.array([iy1, iy2, x**2, x, np.ones(x.size)])
        Y = np.transpose(Y)

        A = np.dot(np.linalg.pinv(Y), y)

        A_tranfer = np.array([[A[0], A[1]], [1, 0]])

        lambdas = np.linalg.eig(A_tranfer)[0]

        X = [np.ones(x.size), np.exp(lambdas[0] * x), np.exp(lambdas[1] * x)]
        X = np.transpose(X)
        P = np.dot(np.linalg.pinv(X), y)
        """





        x = np.array(t)


        y = np.array(self.data, dtype=np.float64)
        max_y = np.max(y)
        y /= y.sum()

        # find t0 (crudely) Starting from the maximum, we go back microtime per microtime until the previous microtime is smaller than the precedent
        pos_max = np.argmax(y)
        pos_t0 = pos_max
        while y[pos_t0] - y[pos_t0-1] > 0:
            pos_t0 -= 1
        if pos_t0 < 0:
            pos_t0 = 0

        x -= x[pos_t0]
        x = x[pos_t0:]
        y = y[pos_t0:]

        # plt.plot(x, y)
        # plt.show()

        S = np.empty_like(y)
        S[0] = 0
        S[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))

        # plt.plot(x, S)
        # plt.show()

        SS = np.empty_like(y)
        SS[0] = 0
        SS[1:] = np.cumsum(0.5 * (S[1:] + S[:-1]) * np.diff(x))

        # plt.plot(x, SS)
        # plt.show()

        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2

        M = [[sum(SS * SS), sum(SS * S), sum(SS * x2), sum(SS * x), sum(SS)],
             [sum(SS * S), sum(S * S), sum(S * x2), sum(S * x), sum(S)],
             [sum(SS * x2), sum(S * x2), sum(x4), sum(x3), sum(x2)],
             [sum(SS * x), sum(S * x), sum(x3), sum(x2), sum(x)],
             [sum(SS), sum(S), sum(x2), sum(x), len(x)]]

        Ycol = np.array([sum(SS * y), sum(S * y), sum(x2 * y), sum(x * y), sum(y)])

        (A, B, C, D, E), residues, rank, singulars = list(lstsq(M, Ycol))

        '''
        Minv = np.linalg.inv(M)    
        A,B,C,D,E = list( np.matmul(Minv,Ycol) )
        '''
        if B*B + 4*A > 0:
            p = (1 / 2.) * (B + np.sqrt(B * B + 4 * A))
            q = (1 / 2.) * (B - np.sqrt(B * B + 4 * A))
        else:
            return None

        beta = np.exp(p * x)
        eta = np.exp(q * x)

        betaeta = beta * eta

        L = [[len(x), sum(beta), sum(eta)],
             [sum(beta), sum(beta * beta), sum(betaeta)],
             [sum(eta), sum(betaeta), sum(eta * eta)]]

        Ycol = np.array([sum(y), sum(beta * y), sum(eta * y)])

        (a, b, c), residues, rank, singulars = list(lstsq(L, Ycol))

        '''
        Linv = np.linalg.inv(L)
        a,b,c = list( np.matmul( Linv, Ycol ) )
        '''

        # sort in ascending order (fastest negative rate first)
        (b, p), (c, q) = sorted([[b, p], [c, q]], key=lambda x: x[1])

        #return a, b, c, p, q

        ratio = b/c
        b = max_y / (1/ratio + 1)
        c = max_y / (ratio + 1)


        params['tau1']["value"]= -1.0/p
        params['tau2']["value"]= -1.0/q
        params['bckgnd']["value"] = a
        params['a1']["value"] = b
        params['a2']["value"] = c

        return params

    def eval(self, t,  params):
        tau1 = params['tau1'].value
        a1 = params['a1'].value
        tau2 = params['tau2'].value
        a2 = params['a2'].value
        t0 = params['t0'].value
        self.data_bckgnd = params['bckgnd'].value
        self.non_convoluted_decay = a1*np.exp(-(t-t0) / tau1) + a2*np.exp(-(t-t0) / tau2)
        self.non_convoluted_decay[t < t0] = 0

        return self.non_convoluted_decay + self.data_bckgnd

    def make_params(self):
        params = Parameters()
        params.add(name="tau1", value=1, min=0.01, max=np.inf, brute_step=0.1)
        params.add(name="a1", value=0.5, min=0, max=np.inf, brute_step=0.1)
        params.add(name="tau2", value=3, min=0.01, max=np.inf, brute_step=0.1)
        params.add(name="a2", value=0.5, min=0, max=np.inf, brute_step=0.1)
        params.add(name="t0", value=0, min=0, max=np.inf, brute_step=0.1)
        params.add(name="bckgnd", vary=False, value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        return params

class IRF_Becker(lifetimeModelClass):
    """This function is used the IRF.

    Defined as:

    .. math::

        f(x;\mu,\sigma,\lambda) = \frac{\lambda}{2} e^{\frac{\lambda}{2} (2 \mu + \lambda \sigma^2 - 2 x)} \operatorname{erfc} (\frac{\mu + \lambda \sigma^2 - x}{ \sqrt{2} \sigma})

    """
    def __init__(self, IRF=None):
        super().__init__(IRF)

    def guess(self):
        pass

    def eval(self, t,  params):
        t_irf = params['tau_irf'].value
        t0 = params['t0'].value
        amp = params['amp'].value
        bckgnd = params['bckgnd'].value
        irf = amp*(t-t0)/t_irf*np.exp(-(t-t0)/t_irf) + bckgnd
        irf[t < t0] = 0
        return irf

    def make_params(self):
        params = Parameters()
        params.add(name="tau_irf", value=0.5, min=0.001, max=np.inf, brute_step=0.1)
        params.add(name="t0", value=0, min=-100, max=100, brute_step=0.1)
        params.add(name="amp", value=3000, min=0, max=np.inf, brute_step=10)
        params.add(name="bckgnd", value=5, min=0, max=np.inf, brute_step=1)
        return params

class MonoExp_for_IRF(lifetimeModelClass):
    """This function is used the IRF.

    Defined as:

    .. math::

        f(x;\mu,\sigma,\lambda) = \frac{\lambda}{2} e^{\frac{\lambda}{2} (2 \mu + \lambda \sigma^2 - 2 x)} \operatorname{erfc} (\frac{\mu + \lambda \sigma^2 - x}{ \sqrt{2} \sigma})

    """
    def __init__(self, IRF=None):
        super().__init__(IRF)

    def guess(self):
        pass

    def eval(self, t,  params):
        tau = params['tau'].value
        amp = params['amp'].value
        t0 = params['t0'].value
        t_irf = params['tau_irf'].value
        bckgnd = params['bckgnd'].value
        decay = amp * np.exp(-(t-t0)/tau)
        irf = (t-t0)/t_irf*np.exp(-(t-t0)/t_irf)
        irf[t < t0] = 0
        irf /= irf.sum()
        conv = np.convolve(decay, irf)[0:np.size(decay)]
        return conv + bckgnd

    def make_params(self):
        params = Parameters()
        params.add(name="tau", value=3, min=0.001, max=np.inf, brute_step=0.1)
        params.add(name="amp", value=3000, min=0, max=np.inf, brute_step=10)
        params.add(name="t0", value=0, min=-100, max=100, brute_step=0.1)
        params.add(name="tau_irf", value=0.5, min=0.001, max=np.inf, brute_step=0.1)
        params.add(name="bckgnd", value=5, min=0, max=np.inf, brute_step=1)
        return params


class IRF_GaAs(Model):
    """This function is used the IRF.

    Defined as:

    .. math::

        f(x;\mu,\sigma,\lambda) = \frac{\lambda}{2} e^{\frac{\lambda}{2} (2 \mu + \lambda \sigma^2 - 2 x)} \operatorname{erfc} (\frac{\mu + \lambda \sigma^2 - x}{ \sqrt{2} \sigma})

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='propagate',
                 **kwargs):

        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def irf(t, t_irf, t0):
            return (t-t0)/t_irf*np.exp(-(t-t0)/t_irf)

        super(IRF_GaAs, self).__init__(irf, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pass
        # return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"

class ExpConvGauss(Model):
    """representing the convolution of a gaussian and an exponential. This function is used to model the IRF.

    Defined as:

    .. math::

        f(x;\mu,\sigma,\lambda) = \frac{\lambda}{2} e^{\frac{\lambda}{2} (2 \mu + \lambda \sigma^2 - 2 x)} \operatorname{erfc} (\frac{\mu + \lambda \sigma^2 - x}{ \sqrt{2} \sigma})

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='propagate',
                 **kwargs):

        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def exgauss(t, mu, sig, tau):
            lam = 1. / tau
            return 0.5 * lam * np.exp(0.5 * lam * (2 * mu + lam * (sig ** 2) - 2 * t)) * \
                   erfc((mu + lam * (sig ** 2) - t) / (np.sqrt(2) * sig))

        super(ExpConvGauss, self).__init__(exgauss, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pass
        # return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"



#TODO creer une classe mère pour les analyses.
class lifeTimeMeasurements(Measurements):

    def __init__(self, exps=None, exp=None, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exps, exp, exp_param, num_channel, start_tick, end_tick, "lifetime", name, comment)
        self.IRF = None
        self.is_plot_log = True
        self.use_IR = False

    def set_additional_param_for_calculation(self, params):
        pass

    def calculate(self):
        timeStamps = self.exp.data.channels[self.num_channel].photons['timestamps']
        nanotimes = self.exp.data.channels[self.num_channel].photons['nanotimes']
        if self.start_tick == 0:
            start_tick = self.exp.data.channels[self.num_channel].start_tick
        else:
            start_tick = self.start_tick

        if self.end_tick == -1:
            end_tick = self.exp.data.channels[self.num_channel].end_tick
        else:
            end_tick = self.end_tick

        idxStart, idxEnd = np.searchsorted(timeStamps, (start_tick, end_tick))
        nanotimes = nanotimes[idxStart:idxEnd]

        self.create_histogramm(nanotimes)
        # TODO check validity
        self.error_bar = np.sqrt(self.data + 1)
        return self

    def create_histogramm(self, nanotimes):
        # self.data = np.zeros(self.exp_param.nb_of_microtime_channel, dtype=np.uint)
        # self.time_axis = np.arange(self.exp_param.nb_of_microtime_channel) * self.exp_param.mIcrotime_clickEquivalentIn_second*1E9
        self.data = np.bincount(nanotimes)
        self.time_axis = np.arange(0, self.data.size) * self.exp_param.mIcrotime_clickEquivalentIn_second*1E9
        self.trim_life_time_curve()
        if self.model is not None:
            self.model.total_count = self.data.sum()

    def trim_life_time_curve(self):
        nonzero = np.nonzero(self.data)
        idxStartNonZero = nonzero[0][0]
        idxEndNonZero = nonzero[0][-1]
        self.data = self.data[idxStartNonZero:idxEndNonZero]
        self.time_axis = self.time_axis[idxStartNonZero:idxEndNonZero]

    def shift_histogramm(self, shift):
        self.data = np.roll(self.data, shift)

    def set_model_IRF(self):
        self.model.IRF = self.IRF

    def set_use_IR(self, use_IR=False):
        self.use_IR = use_IR
        if self.model is not None:
            self.model.use_IR = use_IR

    def fit(self, params=None):
        # ATTENTION au -1 -> cela créé une case de moins dans le tableau.
        if self.model is not None:
            self.model.data = self.data

        if params is not None:
            self.set_params(params)

        y_eval_range = self.data[self.idx_start:self.idx_end]
        x_eval_range = self.time_axis[self.idx_start:self.idx_end]
        self.model.x_range = (self.idx_start, self.idx_end)

        self.model.observed_count = (self.data - self.params['bckgnd'].value).sum()

        threshold_MLE = 5

        if self.qty_to_min == "auto":
            if max(y_eval_range) < threshold_MLE:
                qty_to_min = "max. likelyhood (MLE)"
            else:
                qty_to_min = "chi2"
        else:
            qty_to_min = self.qty_to_min

        #FIXME
        # if self.use_IR:
        if True:
            # self.set_params(params)

            def residuals(params, x, y):
                """
                Returns the array of residuals for the current parameters.
                """
                ymodel = self.model.eval(x, params)
                return (y[self.idx_start:self.idx_end] - ymodel[self.idx_start:self.idx_end])

            def weighted_residuals(params, x, y, weights):
                """
                Returns the array of weighted residuals for the current parameters.
                The weight is sqrt(N).
                """
                ymodel = self.model.eval(x, params)
                return (y[self.idx_start:self.idx_end] - ymodel[self.idx_start:self.idx_end])/weights[self.idx_start:self.idx_end]


            def chi2(params, x, y, weights):
                """
                Returns the array of residuals for the current parameters.
                """
                ymodel = self.model.eval(x, params)
                return (y[self.idx_start:self.idx_end] - ymodel[self.idx_start:self.idx_end])**2 / weights[self.idx_start:self.idx_end]**2


            def loglike(params, x, ydata):
                """
                return loglike of model vs data
                """
                # tau, baseline, offset, ampl = params
                ymodel = self.model.eval(x, params)
                return (ymodel - ydata * np.log(ymodel)).sum()

            def maximum_likelihood_method(params, x, ydata):
                """
                Return a scalar.
                :param params:
                :param x:
                :param ydata:
                :return:
                """

                # NB : we all the data on the time_axis in order to calculate the Convolution by the IRF.
                # But the likelihood has to be calculated only on the selection
                ymodel = self.model.eval(x, params)
                ymodel = ymodel[self.idx_start:self.idx_end]
                ydata_reduc = ydata[self.idx_start:self.idx_end]
                likelyhood = 2 * (ydata_reduc*np.log(ydata_reduc/ymodel)).sum()
                # likelyhood = -2*(ydata_reduc * np.log(ymodel)).sum()
                return likelyhood

            def callback_iteration(params, iter, resid, *args, **kws):
                """
                Function to be called at each fit iteration. This function should have the signature
                iter_cb(params, iter, resid, *args, **kws), where params will have the current parameter values,
                iter the iteration number, resid the current residual array, and *args and **kws as passed
                to the objective function.
                :param params:
                :param iter:
                :param resid:
                :param args:
                :param kws:
                :return:
                """
                # TODO draw and report progress
                pass

            minimization_quantity_fct = maximum_likelihood_method
            callback_iteration_fct = None
            nb_step_per_param = 10
            nb_of_best_candidate_to_keep = 25

            if self.fitting_method1 == "brute":
                # Set brute parameters steps
                params['x'].set(min=-4, max=4, brute_step=0.25)
                params['y'].set(min=-4, max=4, brute_step=0.25)

                # workers = -1 -> use all core to parse the brute grid.
                fitter = Minimizer(maximum_likelihood_method, params)
                self.fit_results = minimize(minimization_quantity_fct, self.params, args=(self.time_axis, self.data), method='brute', iter_cb=callback_iteration_fct, nan_policy='propagate', keep=nb_of_best_candidate_to_keep, workers=-1)
                # grid_x, grid_y = [np.unique(par.ravel()) for par in self.fit_results.brute_grid]
                # result.brute_x0 – A 1-D array containing the coordinates of a point at which the objective function had its minimum value.
                # print(self.fit_results.brute_x0)
                # result.brute_fval – Function value at the point x0.
                # print(result.brute_fval)

                best_result = copy.deepcopy(self.fit_results)

                # TODO workers and multiprocessor -> CUDA ?
                for i, candidate in enumerate(self.fit_results.candidates):
                    self.log(self.fit_results.show_candidates(i))

                    self.log("Applying " + self.fitting_method1 + " to optimize around candidate\n")
                    trial = fitter.minimize(method='leastsq', params=candidate.params)
                    if trial.chisqr < best_result.chisqr:
                        best_result = trial
                        self.log("Minimization result chi square : " + str(trial.chisqr) + "this is the new best candidate\n")

                    else:
                        self.log("Minimization result chi square : " + str(trial.chisqr) + "bigger than the others candidates -> Discarding the candidate\n")

                self.fit_results = best_result
            else:
                pass

            #FIXME
            callback_iteration = None
            # if qty_to_min == "chi2":
            #     fct_min = chi2
            #     args = (self.time_axis, self.data, self.error_bar)
            # elif qty_to_min == "max. likelyhood (MLE)":
            #     fct_min = maximum_likelihood_method
            #     args = (self.time_axis, self.data)
            #
            # self.fit_results = minimize(fct_min, self.params, args=args, method=self.fitting_method1, iter_cb=callback_iteration, nan_policy='propagate')
            #
            # if self.fitting_method2 != "None":
            #     self.fit_results = minimize(fct_min, self.fit_results.params, args=args,
            #                                 method=self.fitting_method2, iter_cb=callback_iteration, nan_policy='propagate')
            #

            # if self.fitting_method2 != "None":
            #     self.fit_results = minimize(fct_min, self.fit_results.params, args=args,
            #                                 method=self.fitting_method2, iter_cb=callback_iteration, nan_policy='propagate')
            #


            if qty_to_min == "chi2":
                self.fit_results = minimize(weighted_residuals, self.params, args=(self.time_axis, self.data, self.error_bar), method=self.fitting_method1,
                                            iter_cb=callback_iteration, nan_policy='propagate')
            elif qty_to_min == "max. likelyhood (MLE)":
                self.fit_results = minimize(maximum_likelihood_method, self.params, args=(self.time_axis, self.data), method=self.fitting_method1,
                                            iter_cb=callback_iteration, nan_policy='propagate')



            if self.fitting_method2 != "None":
                if qty_to_min == "chi2":
                    self.fit_results = minimize(fcn=weighted_residuals, params=self.params, args=(self.time_axis, self.data, self.error_bar),
                                                method=self.fitting_method2,
                                                iter_cb=callback_iteration, nan_policy='propagate')
                elif qty_to_min == "max. likelyhood (MLE)":
                    self.fit_results = minimize(fcn=maximum_likelihood_method, params=self.params,
                                                args=(self.time_axis, self.data), method=self.fitting_method2,
                                                iter_cb=callback_iteration, nan_policy='propagate')


            #FIXME do eval_y_axis and eval_x_axis still in use ?
            # self.eval_y_axis = self.fit_results.best_fit
            self.eval_y_axis = self.model.eval(self.time_axis, self.fit_results.params)
            self.eval_x_axis = x_eval_range

            self.residuals = self.data[self.idx_start:self.idx_end] - self.eval_y_axis[self.idx_start:self.idx_end]
            self.fit_x = self.time_axis
            self.residual_x = self.time_axis[self.idx_start:self.idx_end]

            return fit_report(self.fit_results)
        else:
            self.error_bar = np.sqrt(self.data + 1)
            return super().fit(params)

    def guess(self, params_=None):

        self.set_params(params_)

        if self.model is not None:
            self.model.data = self.data

        # y = self.data[self.idx_start:self.idx_end]
        # x_eval_range = self.time_axis[self.idx_start:self.idx_end]

        result= self.model.guess(params=params_, t=self.time_axis)

        if result is not None:
            self.set_params(result)
            self.eval(params_)




    def eval(self, params_=None):

        # Because of the convolution with the IRF, the eval function needs to access all the data and not only
        # the x-selected area

        self.set_params(params_)
        # y = self.data[self.idx_start:self.idx_end]
        # x_eval_range = self.time_axis[self.idx_start:self.idx_end]
        # self.model.x_range = (self.idx_start, self.idx_end)
        # self.model.observed_count = (self.data - self.params['bckgnd'].value).sum()
        self.model.data = self.data

        # x = self.time_axis[self.idx_start:self.idx_end]
        # y = self.data[self.idx_start:self.idx_end]

        # self.model.data = self.data
        self.eval_y_axis = self.model.eval(params=self.params, t=self.time_axis)

        self.eval_y_axis = self.eval_y_axis[self.idx_start:self.idx_end]
        x = self.time_axis[self.idx_start:self.idx_end]
        y = self.data[self.idx_start:self.idx_end]

        self.residuals = self.eval_y_axis - y
        self.residual_x = x
        self.eval_x_axis = self.fit_x = x


    def estimate_nb_of_exp(self, params_):
        # get data
        if params_ is not None:
            self.set_params(params_)

        y = self.data[self.idx_start:self.idx_end]
        x = self.time_axis[self.idx_start:self.idx_end]

        # Max nb of exp
        n = 4

        # TODO calculate integral trapeze
        iy1 = np.empty_like(y)
        iy1[0] = 0
        iy1[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))

        iy2 = np.empty_like(y)
        iy2[0] = 0
        iy2[1:] = np.cumsum(0.5 * (iy1[1:] + iy1[:-1]) * np.diff(x))

        iy3 = np.empty_like(y)
        iy3[0] = 0
        iy3[1:] = np.cumsum(0.5 * (iy2[1:] + iy2[:-1]) * np.diff(x))

        iy4 = np.empty_like(y)
        iy4[0] = 0
        iy4[1:] = np.cumsum(0.5 * (iy3[1:] + iy3[:-1]) * np.diff(x))

        # get exponentials lambdas
        Y = np.array([iy1, iy2, iy3, iy4, np.ones(x.size), x, x ** 2, x ** 3])
        Y = np.transpose(Y)

        for i in range(len(Y)):
            for j in range(len(Y[i])):
                print('{:.2}'.format(Y[i][j]), end=' ')
            print()

        u, s, vh = np.linalg.svd(Y)
        ysv_scaled = 100 * s / np.sum(s)
        print("s :", s)

    def explore_chi2_surf(self, params_):
        self.set_model(params_["model_name"])
        self.params = self.model.make_params()

        self.set_params(params_)
        self.model.data = self.data

        def fcn2min(params_, x, data, weights):
            ymodel = self.model.eval(params=params_, t=x)
            return (data[self.idx_start:self.idx_end] - ymodel[self.idx_start:self.idx_end])**2 / (weights[self.idx_start:self.idx_end]**2)

        self.model.observed_count = (self.data - self.params['bckgnd'].value).sum()


        weights = np.sqrt(self.data + 1)


        y = self.data[self.idx_start:self.idx_end]
        x = self.time_axis[self.idx_start:self.idx_end]

        fitter = Minimizer(fcn2min, self.params, fcn_args=(self.time_axis, self.data, weights), nan_policy='propagate')
        # result_brute = fitter.minimize(method='brute', Ns=25, keep=25, workers=-1)
        result_brute = fitter.minimize(method='brute', Ns=25, keep=25)
        return result_brute

    def create_canonic_graph(self, is_plot_error_bar=False, is_plot_text=False):
        self.canonic_fig, self.canonic_fig_ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(hspace=0)
        ax = self.canonic_fig_ax
        ax[0].semilogy(self.time_axis, self.data, "ro", alpha=0.5)
        for a in ax:
            a.grid(True)
            a.grid(True, which='minor', lw=0.3)


        if self.fit_results is not None:
            ax[0].semilogy(self.time_axis, self.fit_results.best_fit, "b-", linewidth=3)
        if self.residuals is not None:
            ax[1].plot(self.time_axis, self.fit_results.residual, 'k')
            ym = np.abs(self.fit_results.residual).max()
            ax[1].set_ylim(-ym, ym)

        if is_plot_text:
            pass
            # TODO Changer le texte selon les modeles
            if self.modelName == "One Decay":
                pass
            elif self.modelName == "Two Decays":
                msg = ((r'$\tau_1$ = {tau1:.2f} ns' + '\n' + r'$\tau_2$ = {tau2:.0f} ns' + '\n' + r'$A_1$ = {A1:.0f}' + '\n' + r'$A_2$ = {A2:.0f}')
                       .format(tau1=self.fit_results.values['tau1'], tau2=self.fit_results.values['tau2'], A1=self.fit_results.values['amp1'],
                               A2=self.fit_results.values['amp2']))
            # ax[0].text(.75, .9, msg,
            #            va='top', ha='left', transform=ax[0].transAxes, fontsize=18)

        ax[0].set_ylabel('Occurence', fontsize=40)
        ax[1].set_ylabel('residuals', fontsize=20)
        ax[1].set_xlabel('time (ns)', fontsize=40)

        ax[0].tick_params(axis='both', which='major', labelsize=20)
        ax[0].tick_params(axis='both', which='minor', labelsize=8)

        ax[1].tick_params(axis='x', which='major', labelsize=20)
        ax[1].tick_params(axis='x', which='minor', labelsize=8)
        ax[1].tick_params(axis='x', which='major', labelsize=20)
        ax[1].tick_params(axis='x', which='minor', labelsize=8)

        return self.canonic_fig

    def create_chronogram_overlay(self, chronogram, micro_1, micro_2, photon_threshold=5):
        if micro_1 > micro_2:
            micro_1, micro_2 = micro_2, micro_1
        micro_1 = int(micro_1 / (self.exp_param.mIcrotime_clickEquivalentIn_second*1E9))
        micro_2 = int(micro_2 / (self.exp_param.mIcrotime_clickEquivalentIn_second * 1E9))
        nanotimes = chronogram.get_raw_data("nanotimes")

        # Test if photon are in the interval
        nanotimes_mask = (nanotimes >= micro_1) & (nanotimes <= micro_2)

        nanotimes_mask = nanotimes_mask.astype(int)
        # Count for each chronogram bin the ratio photon inside the nanotimes boundaries vs total number of photon
        num_photon_edge = chronogram.get_num_photon_of_edge(is_absolute_number=False)

        multi_dim_mask = np.hsplit(nanotimes_mask, num_photon_edge)
        # multi_dim_selected_photon_by_bin = np.cumsum(multi_dim_mask, axis=1)
        selected_photon_by_bin = np.zeros(len(multi_dim_mask))
        for i, dim in enumerate(multi_dim_mask):
            selected_photon_by_bin[i] = np.sum(dim)

        # selected_photon_by_bin = np.ravel(multi_dim_selected_photon_by_bin)

        # selected_photon_by_bin = np.ravel(np.cumsum(np.hsplit(nanotimes_mask, num_photon_edge)))
        chrono = np.copy(chronogram.data)
        # can't divide by zero...
        chrono[chrono == 0] = 1
        overlay_data = selected_photon_by_bin / chrono
        overlay_data[chrono < photon_threshold] = 0
        return overlay_data

    def set_model(self, model_name):
        if model_name == "One Decay IRF":
            self.modelName = model_name
            self.model = OneExpDecay(self.IRF)
            self.params = self.model.make_params()
            self.model.fit_formula = r"f(t; \tau, shift) = IRF(shift) \times (e^{-t/\tau}) + bckgnd"

        elif model_name == "One Decay Tail":
            self.modelName = model_name
            self.model = OneExpDecay_tail()
            self.params = self.model.make_params()
            self.model.fit_formula = r"e^{-(t-t_0)/\tau} + bckgnd"

        elif model_name == "Two Decays IRF":
            self.modelName = model_name
            self.model = TwoExpDecay(self.IRF)
            self.params = self.model.make_params()
            self.model.fit_formula = r"IRF(shift) \times  (a1 . e^{-t/\tau_1} + (1-a_1).e^{-t/\tau_2}) + bckgnd"

        elif model_name == "Two Decays IRF A1 A2":
            self.modelName = model_name
            self.model = TwoExpDecay_a1a2(self.IRF)
            self.params = self.model.make_params()
            self.model.fit_formula = r"e^{-(t-t_0)/\tau} + bckgnd"

        elif model_name == "Two Decays Tail":
            self.modelName = model_name
            self.model = TwoExpDecay_tail()
            self.params = self.model.make_params()
            self.model.fit_formula = r"a1 . e^{-(t-t_0)/\tau_1} + (1-a_1).e^{-(t-t_0)/\tau_2}) + bckgnd"

        elif model_name == "Two Decays Tail A1 A2":
            self.modelName = model_name
            self.model = TwoExpDecay_tail_a1a2()
            self.params = self.model.make_params()
            self.model.fit_formula = r" a1.e^{-(t-t_0)/\tau_1} + a2.e^{-(t-t_0)/\tau_2} + bckgnd"

        elif model_name == "IRF Becker":
            self.modelName = model_name
            self.model = IRF_Becker()
            self.params = self.model.make_params()
            self.model.fit_formula = r"amp.(t-t0)/tau_irf e^{-(t-t0)/tau_irf}"

        elif model_name == "MonoExp for IRF":
            self.modelName = model_name
            self.model = MonoExp_for_IRF()
            self.params = self.model.make_params()
            self.model.fit_formula = r"((t-t0)/tau_irf e^{-(t-t0)/tau_irf}) \times amp . (e^{-(t-t0)/\tau}) + bckgnd"


        elif model_name == "Three Decays Tail A1 A2 A3":
            self.modelName = model_name
            self.model = ThreeExpDecay_tail_a1a2a3()
            self.params = self.model.make_params()
            self.model.fit_formula = r"a1.e^{-(t-t_0)/\tau_1} + a2.e^{-(t-t_0)/\tau_2} + a3.e^{-(t-t_0)/\tau_3} + bckgnd"



        self.model.use_IR = self.use_IR
        self.model.IR = self.IRF


class IRF:
    def __init__(self, exps=None, exp=None, file_path=None):
        self.exps = exps
        self.exp = exp
        if exp is not None:
            self.exp_param = exp.exp_param
        else:
            self.exp_param = None
        self.raw_data, self.processed_data, self.name = None, None, None
        self.start, self.end = None, None
        self.bckgnd = 0
        self.shift = None
        self.time_axis, self.time_axis_processed = None, None
        if file_path is not None:
            self.get_data(file_path)

    def generate(self, params_dict, algo="Becker"):
        if algo == "Becker":
            tau = params_dict["tau"]
            t0 = params_dict["t0"]
            irf_shift = params_dict["irf_shift"]

            if "time_step_ns" in params_dict:
                time_step_ns = params_dict["time_step_ns"]
            else:
                time_step_ns = self.exp_param.mIcrotime_clickEquivalentIn_second * 1E9
            if "nb_of_microtime_channel" in params_dict:
                time_nbins = params_dict["nb_of_microtime_channel"]  # number of time bins
            else:
                time_nbins = self.exp_param.nb_of_microtime_channel

            time_idx = np.arange(time_nbins)  # time axis in index units
            self.time_axis = time_idx * time_step_ns  # time axis in nano-seconds

            non_zero_cst = 0.001
            non_zero_cst = 0
            self.raw_data = (self.time_axis - t0) / tau * np.exp(-(self.time_axis - t0) / tau) + non_zero_cst
            self.raw_data[self.raw_data < 0] = 0
            if irf_shift is not None:
                self.raw_data = shift_scipy(self.raw_data, irf_shift, mode='wrap')

            self.raw_data /= self.raw_data.sum()
            #FIXME
            nb_of_photon_irf = 1e6

            self.raw_data *= nb_of_photon_irf
            self.raw_data[self.raw_data < 0.1] = 0.1
            self.name = "generated Becker tau=" + str(tau)
            self.process()
            return self

    def get_data(self, file_path):
        self.name, self.raw_data, self.time_axis = self.exps.get_IRF_from_file(file_path)

    def fit(self, ini_params):
        # TODO test
        self.model_IR = ExpConvGauss()
        self.params_fit_IR = self.model_IR.make_params(mu=ini_params[0], sig=ini_params[1], tau=ini_params[2])
        self.fit_results = self.model_IR.fit(self.processed_data, self.params_fit_IR, t=self.time_axis_processed)

        self.fit_y_axis = self.fit_results.best_fit
        self.fit_x_axis = self.time_axis_processed

        self.residuals = self.fit_results.residual
        return self.fit_results

    def process(self):
        """
        Shift in nb of microtime channel
        """

        # shift = int(self.shift/100.0 * self.exp_param.nb_of_microtime_channel)
        # self.IR_processed = np.roll(self.IR_raw[idx_begin:idx_end], int(self.IR_shift/100.0*self.exp_param.nb_of_microtime_channel))
        if self.raw_data is not None:
            if self.start is not None and self.end is not None :
                idx_begin = int(self.start / 100.0 * self.exp_param.nb_of_microtime_channel)
                idx_end = int(self.end / 100.0 * self.exp_param.nb_of_microtime_channel)
                self.processed_data = self.raw_data[idx_begin:idx_end].astype(np.float64)
                self.time_axis_processed = self.time_axis[idx_begin:idx_end]
            else:
                self.processed_data = self.raw_data.astype(np.float64)


            # background removal
            self.processed_data -= self.bckgnd
            self.processed_data[self.processed_data < 0] = 0

            # We divide by the sum of the IR so that the convolution doesn't change the amplitude of the signal.
            self.processed_data /= float(self.processed_data.sum())

            return "OK"
        else:
            return "No IR was loaded"



