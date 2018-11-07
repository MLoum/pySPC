from core import Data
from core import ExpParam
from core import Results
import os
import numpy as np

from core.analyze import lifetime, FCS, DLS

"""
Test Experiment
"""
class Experiment(object):
    """
    Root class of the model (named here "core") of the MVC pattern.

    main members :
    - exp_param
    - results
    - data

    main methods :
    - new_exp
    - save_state/load_state
    """

    def __init__(self, filepath=None):
        self.exp_param = ExpParam.Experiment_param()
        self.results = Results.Results()
        self.data = Data.Data(self.results, self.exp_param)

        self.file_name = None
        self.defaultBinSize_s = 0.01  # default : 10ms

        if filepath is not None :
            self.new_exp("file", [filepath])


    # TODO put convert function where it belongs
    def convert_ticks_in_seconds(self, ticks):
        return self.exp_param.mAcrotime_clickEquivalentIn_second * ticks

    def convert_seconds_in_ticks(self, seconds):
        return seconds / self.exp_param.mAcrotime_clickEquivalentIn_second

    def new_exp(self, mode, params):
        """
        Create a new experiment from a file or from simulated data.

        :param mode: a string telling how to create the new experiment. Can be :
            - "file" : a recorded file spc or pt3 file
            - "generate" : Poissonian Noise
            - "simulation" : Not Implemented Yet.
        :param params: params passed to the function based on the mode :
            - for "file mode :
                - params[0] = filePath
            - for "generate" :
                - params[0] = filePath
                - params[1] = time_s
                - params[2] = count_per_second_s
        :return:
        """
        if mode == "file":
            filePath = params[0]
            result = self.data.load_from_file(filePath)
            if result is not "OK":
                return

            head, self.file_name = os.path.split(filePath)

        elif mode == "generate":
            type = params[0]
            time_s = params[1]
            count_per_second_s = params[2]
            self.data.new_generated_exp(type, [time_s, count_per_second_s])
            self.file_name = "Generated Poisson Noise - count per second : %f" % count_per_second_s
        elif mode == "simulation":
            pass

        # Display chronogram as proof od new exp.
        # chronogram(self, numChannel, startTick, endTick, binInTick):
        binInTick = self.convert_seconds_in_ticks(self.defaultBinSize_s)
        self.results.navigationChronogram = self.chronogram(0, 0, self.data.channels[0].end_tick, binInTick)
        # self.results.mainChronogram = self.data.chronogram(0, 0, self.data.channels[0].endTick, binInTick)
        # self.data.PCH(self.results.mainChronogram)

    def save_state(self, shelf):
        """

        :param shelf:
        :return:
        """
        shelf['exp_param'] = self.exp_param
        shelf['results'] = self.results
        shelf['data'] = self.data
        shelf['fileName'] = self.file_name

    def load_state(self, shelf):
        """

        :param shelf:
        :return:
        """
        self.exp_param = shelf['exp_param']
        self.results = shelf['results']
        self.data = shelf['data']
        self.file_name = shelf['fileName']

    def update(self):
        pass

    # TODO put into a file in analyze and call it Bin.
    def chronogram(self, num_channel=0, start_tick=0, end_tick=-1, bin_in_tick=1E5):
        """
        The x axis is in ->microsecond<-
        """
        timeStamps = self.data.channels[num_channel].photons['timestamps']

        if start_tick == 0:
            start_tick = self.data.channels[num_channel].start_tick
        if end_tick == -1:
            end_tick = self.data.channels[num_channel].end_tick


        start_tick = np.uint64(start_tick)
        end_tick = np.uint64(end_tick)

        # TODO Expliquer le +1, je pense que c'est du à des pb de valeurs arrondies... Au pire la derniere case est vide.
        nbOfBin = int((end_tick - start_tick) / bin_in_tick) + 1
        # Find indices where elements should be inserted to maintain order
        idxStart, idxEnd = np.searchsorted(timeStamps, (start_tick, end_tick))

        timesStamps = np.copy(timeStamps[idxStart:idxEnd])
        timesStamps -= start_tick

        # numStartBin = int(start_tick / bin_in_tick)
        # # numEndBin = int( end_tick / binInTick)
        # numEndBin = numStartBin + nbOfBin

        # FIXME moins de divisions, ici on prend tout le fichier
        # numOfBinForEachPhoton =  timesStamps / binInTick
        # numOfBinForEachPhoton = numOfBinForEachPhoton.astype(int)

        # #Default value for "range" seems fine  -> The lower and upper range of the bins. If not provided, range is simply (a.min(), a.max()). Values outside the range are ignored.
        # self.results.mainChronogram = Results.Chronogram()
        # self.results.mainChronogram.tickStart, self.results.mainChronogram.tickEnd, self.results.mainChronogram.nbOfBin = start_tick, end_tick, nbOfBin
        # self.results.mainChronogram.data, self.results.mainChronogram.xAxis  =  np.histogram(numOfBinForEachPhoton, nbOfBin)
        # #time axis in millisecond
        # self.results.mainChronogram.xAxis *= binInTick * self.expParam.mAcrotime_clickEquivalentIn_second*1E6 #ms
        # #We have to cut by one element the x axis because it is on element longer than the data (NB : there is no copy, just "a view")
        # self.results.mainChronogram.xAxis = self.results.mainChronogram.xAxis[:-1]

        chronogram = Results.Chronogram()
        chronogram.tickStart, chronogramtickEnd, chronogram.nbOfBin = start_tick, end_tick, int(nbOfBin)
        # chronogram.data, chronogram.xAxis = np.histogram(numOfBinForEachPhoton, int(nbOfBin))

        # chronogram.data = np.zeros(chronogram.nbOfBin + 1, dtype=np.int)
        # bin.bin(timesStamps, chronogram.data, bin_in_tick)

        num_bin = (timesStamps / bin_in_tick).astype(np.int64)
        chronogram.data = np.bincount(num_bin)

        # chronogram.data
        # TODO UNDERSTAND We have to cut by one element the x axis because it is on element longer than the data (NB : there is no copy, just "a view")
        # chronogram.data = chronogram.data[:-1]

        # chronogram.xAxis = np.arange(numStartBin, numEndBin, dtype=np.float64)
        chronogram.xAxis = np.arange(nbOfBin, dtype=np.float64)
        chronogram.xAxis *= bin_in_tick
        chronogram.xAxis += chronogram.tickStart
        chronogram.xAxis *= self.exp_param.mAcrotime_clickEquivalentIn_second * 1E6

        # time axis in millisecond
        # chronogram.xAxis += chronogram.tickStart
        # chronogram.xAxis *= binInTick * self.expParam.mAcrotime_clickEquivalentIn_second * 1E6  # microsecond
        # We have to cut by one element the x axis because it is on element longer than the data (NB : there is no copy, just "a view")
        # chronogram.xAxis = chronogram.xAxis[:-1]

        return chronogram

    def PCH(self, chronogram):
        """
        Photon Counting Histogramm
        :param chronogram:
        :return:
        """
        pass
        # self.results.mainPCH = Results.PCH()
        # self.results.mainPCH.data, self.results.mainPCH.xAxis = np.histogram(chronogram.data, chronogram.data.max())
        #
        # self.results.mainPCH.nbOfBin = len(self.results.mainPCH.data)
        #
        # # We have to cut by one element the x axis because it is on element longer than the data (NB : there is no copy, just "a view")
        # self.results.mainPCH.xAxis = self.results.mainPCH.xAxis[:-1]

    def timeDifference(self, numChannel=0):
        """
        Compute the tume difference between consecutive photons
        :param numChannel:
        :return:
        """
        self.results.timeDifference[numChannel] = np.diff(self.data.timestamps[numChannel])

    def micro_time_life_time(self, num_channel=0, start_tick=0, end_tick=-1):
        """
        Calculate the histogramm of the microtime and fill the corresponding "results" member
        with a lifeTimeMeasurement object

        :param num_channel: Default is 0.
        :param start_tick: Default value is 0
        :param end_tick: Default value is -1
        :return: Fill the "results" member with a lifeTimeMeasurement object
        """
        timeStamps = self.data.channels[num_channel].photons['timestamps']
        nanotimes = self.data.channels[num_channel].photons['nanotimes']
        if start_tick == 0:
            start_tick = self.data.channels[num_channel].start_tick

        if end_tick == -1:
            end_tick = self.data.channels[num_channel].end_tick

        idxStart, idxEnd = np.searchsorted(timeStamps, (start_tick, end_tick))

        nanotimes = nanotimes[idxStart:idxEnd]

        if self.results.lifetimes[num_channel] is None:
            self.results.lifetimes[num_channel] = lifetime.lifeTimeMeasurements()

        self.results.lifetimes[num_channel].create_histogramm(nanotimes, self.exp_param.nb_of_microtime_channel,
                                                              self.exp_param.mIcrotime_clickEquivalentIn_second)
        return self.results.lifetimes[num_channel]

    def FCS(self, num_c1=0, num_c2=0, start_tick=0, end_tick=-1, start_cor_time_micros = 0.5, max_cor_time_ms=1000):
        """
        Fluctuation Correlation Spectroscopy

        Create a FCS_Measurements object and launch calculation

        :param num_c1:
        :param num_c2:
        :param start_tick:
        :param end_tick:
        :param max_cor_time_ms:
        :return:
        """
        # TODO cross correlation
        num_channel = num_c1

        if start_tick == 0:
            start_tick = self.data.channels[num_c1].start_tick

        if end_tick == -1:
            end_tick = self.data.channels[num_c1].end_tick



        timeStamps = self.data.channels[num_channel].photons['timestamps']
        nanotimes = self.data.channels[num_channel].photons['nanotimes']
        idxStart, idxEnd = np.searchsorted(timeStamps, (start_tick, end_tick))
        timeStamps_reduc = timeStamps[idxStart:idxEnd]

        if self.results.FCS_Measurements[num_channel] == None:
            self.results.FCS_Measurements[num_channel] = FCS.FCSMeasurements()

        max_correlation_time_in_tick = int(
            max_cor_time_ms / 1E3 / self.exp_param.mAcrotime_clickEquivalentIn_second)

        start_correlation_time_in_tick = int(
            start_cor_time_micros / 1E6 / self.exp_param.mAcrotime_clickEquivalentIn_second)

        # self.results.FCS_Measurements[num_channel].correlateMonoProc(timeStamps_reduc, timeStamps_reduc,
        #                                                             max_correlation_time_in_tick, start_correlation_time_in_tick)
        self.results.FCS_Measurements[num_channel].correlateFCS_multicore(timeStamps_reduc, timeStamps_reduc,
                                                                    max_correlation_time_in_tick, start_correlation_time_in_tick)
        return self.results.FCS_Measurements[num_channel]

    def DLS(self, num_channel_1, num_channel_2, start_tick, end_tick, max_correlation_time_ms=1000, start_time_mu_s=1,
            precision=10):
        """
        Dynamic Light Scattering

        Create a DLS_Measurements object and launch calculation

        :param num_channel_1:
        :param num_channel_2:
        :param start_tick:
        :param end_tick:
        :param max_correlation_time_ms:
        :param start_time_mu_s:
        :param precision:
        :return:
        """
        numChannel = num_channel_1

        timeStamps = self.data.channels[numChannel].photons['timestamps']
        nanotimes = self.data.channels[numChannel].photons['nanotimes']
        idxStart, idxEnd = np.searchsorted(timeStamps, (start_tick, end_tick))
        timeStamps_reduc = timeStamps[idxStart:idxEnd]

        if self.results.DLS_Measurements[numChannel] == None:
            self.results.DLS_Measurements[numChannel] = DLS.DLS_Measurements()

        max_correlation_time_tick = int(
            max_correlation_time_ms / 1000.0 / self.exp_param.mAcrotime_clickEquivalentIn_second)

        start_time_tick = int(start_time_mu_s / 1E6 / self.exp_param.mAcrotime_clickEquivalentIn_second)

        self.results.DLS_Measurements[numChannel].correlateMonoProc(timeStamps_reduc,
                                                                    timeStamps_reduc, max_correlation_time_tick,
                                                                    startCorrelationTimeInTick=start_time_tick,
                                                                    nbOfPointPerCascade_aka_B=int(precision),
                                                                    tick_duration_micros=
                                                                    self.exp_param.mAcrotime_clickEquivalentIn_second * 1E6)

