from . import Data
from . import ExpParam
from . import Results
import os

class Experiment(object):
    def __init__(self, expParam=None, results=None, data=None):
        if expParam is None:
            self.expParam = ExpParam.ExperimentParam()
        else :
            self.expParam = expParam

        if results is None:
            self.results = Results.Results()
        else :
            self.results = results

        if data is None:
            self.data = Data.Data(self.results, self.expParam)
        else :
            self.data = data

        self.fileName = None
        self.defaultBinSize_s = 0.01 #default : 10ms

    #TODO put convert function where it belongs
    def convertTicksInSeconds(self, ticks):
        return self.expParam.mAcrotime_clickEquivalentIn_second * ticks

    def convertSecondsInTicks(self, seconds):
        return seconds / self.expParam.mAcrotime_clickEquivalentIn_second

    def newExp(self, mode, params):
        if mode == "file":
            filePath = params[0]
            self.data.loadFromFile(filePath)
            head, self.fileName = os.path.split(filePath)

        elif mode== "generate":
            type = params[0]
            time_s = params[1]
            count_per_second_s = params[2]
            self.data.newGeneratedExp(type, [time_s, count_per_second_s])
            self.fileName = "Generated Poisson Noise - count per second : %f" % (count_per_second_s)
        elif mode == "simulation":
            pass

        #Display chronogram as proof od new exp.
        #chronogram(self, numChannel, startTick, endTick, binInTick):
        binInTick = self.convertSecondsInTicks(self.defaultBinSize_s)
        self.results.navigationChronogram  = self.data.chronogram(0, 0, self.data.channels[0].endTick, binInTick)
        # self.results.mainChronogram = self.data.chronogram(0, 0, self.data.channels[0].endTick, binInTick)
        #self.data.PCH(self.results.mainChronogram)

    def saveState(self, shelf):
        shelf['expParam'] = self.expParam
        shelf['results'] = self.results
        shelf['data'] = self.data
        shelf['fileName'] = self.fileName

    def loadState(self, shelf):
        self.expParam = shelf['expParam']
        self.results = shelf['results']
        self.data = shelf['data']
        self.fileName = shelf['fileName']

    def update(self):
        pass

