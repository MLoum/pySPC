import numpy as np
from lmfit import minimize, Parameters, Model
from lmfit.models import LinearModel, ExponentialModel


def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars


#TODO creer une classe mère pour les analyses.
class Measurements():

    def __init__(self, data_=None, timeAxis_= None):
        self.params = Parameters()
        self.modelName = ""
        self.model = None

        self.timeAxis = timeAxis_

        self.data = data_

        self.eval_x_axis = None
        self.eval_y_axis = None

        self.idxStart, self.idxEnd  = 0, -1

        self.residuals = None

        self.fitResults = None

    def findIdxOfFitLim(self, idxStart, idxEnd):
        if idxStart != 0:
            self.idxStart = np.searchsorted(self.timeAxis, idxStart)
        if idxEnd != -1:
            self.idxEnd = np.searchsorted(self.timeAxis, idxEnd)

    def fit(self, idxStart=0, idxEnd=-1):
        self.findIdxOfFitLim(idxStart, idxEnd)
        y = self.data[self.idxStart:self.idxEnd]
        x = self.timeAxis[self.idxStart:self.idxEnd]
        self.fitResults = self.model.fit(y, self.params, t=x)
        #self.setParams(self.fitResults.params)
        self.eval_y_axis = self.fitResults.best_fit
        self.eval_x_axis = x
        self.residuals = self.fitResults.residual

        #self.evalParams(idxStart, idxEnd)
        return self.fitResults



    def evalParams(self, idxStart=0, idxEnd=-1):
        self.findIdxOfFitLim(idxStart, idxEnd)

        x = self.timeAxis[self.idxStart:self.idxEnd]
        y = self.data[self.idxStart:self.idxEnd]

        self.eval_y_axis = self.model.eval(self.params, t=x)
        self.residuals = self.eval_y_axis - y
        self.eval_x_axis = x

    def guess(self, idxStart=0, idxEnd=-1):
        self.findIdxOfFitLim(idxStart, idxEnd)

        y = self.data[self.idxStart:self.idxEnd]
        x = self.timeAxis[self.idxStart:self.idxEnd]

        self.params = self.model.guess(y, x)
        self.evalParams(idxStart, idxEnd)

    def setParams(self, params):
        pass

    def setModel(self, modelName):
        pass




