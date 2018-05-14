from . import Data
from . import ExpParam
from . import Results
import os


class Experiment(object):

    def __init__(self, exp_param=None, results=None, data=None):
        if exp_param is None:
            self.expParam = ExpParam.ExperimentParam()
        else:
            self.expParam = exp_param

        if results is None:
            self.results = Results.Results()
        else:
            self.results = results

        if data is None:
            self.data = Data.Data(self.results, self.expParam)
        else:
            self.data = data

        self.fileName = None
        self.defaultBinSize_s = 0.01  # default : 10ms

    # TODO put convert function where it belongs
    def convert_ticks_in_seconds(self, ticks):
        return self.expParam.mAcrotime_clickEquivalentIn_second * ticks

    def convert_seconds_in_ticks(self, seconds):
        return seconds / self.expParam.mAcrotime_clickEquivalentIn_second

    def new_exp(self, mode, params):
        if mode == "file":
            filePath = params[0]
            self.data.loadFromFile(filePath)
            head, self.fileName = os.path.split(filePath)

        elif mode == "generate":
            type = params[0]
            time_s = params[1]
            count_per_second_s = params[2]
            self.data.newGeneratedExp(type, [time_s, count_per_second_s])
            self.fileName = "Generated Poisson Noise - count per second : %f" % count_per_second_s
        elif mode == "simulation":
            pass

        # Display chronogram as proof od new exp.
        # chronogram(self, numChannel, startTick, endTick, binInTick):
        binInTick = self.convert_seconds_in_ticks(self.defaultBinSize_s)
        self.results.navigationChronogram  = self.data.chronogram(0, 0, self.data.channels[0].endTick, binInTick)
        # self.results.mainChronogram = self.data.chronogram(0, 0, self.data.channels[0].endTick, binInTick)
        # self.data.PCH(self.results.mainChronogram)

    def save_state(self, shelf):
        shelf['expParam'] = self.expParam
        shelf['results'] = self.results
        shelf['data'] = self.data
        shelf['fileName'] = self.fileName

    def load_state(self, shelf):
        self.expParam = shelf['expParam']
        self.results = shelf['results']
        self.data = shelf['data']
        self.fileName = shelf['fileName']

    def update(self):
        pass

