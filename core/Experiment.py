from core import Data
from core import ExpParam
from core import Results
import os
import numpy as np

from core.analyze import lifetime, FCS, DLS, chronogram, PCH, burstDetection, phosphorescence

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

    def __init__(self, mode, params, exps=None):
        self.exps = exps
        self.exp_param = ExpParam.Experiment_param()
        # self.results = Results.Results()
        self.measurements = {}
        self.data = Data.Data(self.exp_param)



        self.navigation_chronograms = None
        self.time_zoom_chronograms = None
        self.mini_PCHs = None


        self.file_name = None
        self.comment = ""

        self.defaultBinSize_s = 0.01  # default : 10ms

        self.new_exp(mode, params)


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

        # self.results.mainChronogram = self.data.chronogram(0, 0, self.data.channels[0].endTick, binInTick)
        # self.data.PCH(self.results.mainChronogram)

    def calculate_chronogram(self, chronogram_, bin_in_tick):
        """
        The x axis is in ->microsecond<-
        """
        num_channel = chronogram_.num_channel
        start_tick = chronogram_.start_tick
        end_tick = chronogram_.end_tick
        time_stamps = self.data.channels[num_channel].photons['timestamps']

        if start_tick == 0:
            start_tick = self.data.channels[num_channel].start_tick
        if end_tick == -1:
            end_tick = self.data.channels[num_channel].end_tick

        start_tick = np.uint64(start_tick)
        end_tick = np.uint64(end_tick)

        idx_start, idx_end = np.searchsorted(time_stamps, (start_tick, end_tick))
        chronogram_.create_chronogram(time_stamps[idx_start:idx_end], bin_in_tick)

        # self.store_measurement(chrono)
        return chronogram_

    def get_measurement(self, name):
        if name in self.measurements:
            return self.measurements[name]
        else:
            return None

    # def create_measurement(self, num_channel, start_tick, end_tick, type, name, comment, is_store=True):
    def create_measurement(self, num_channel, start_tick, end_tick, type_, name, comment, is_store=True):
        # TODO compact code
        if type_ == "FCS":
            fcs = FCS.FCSMeasurements(self.exps, self, self.exp_param, num_channel, start_tick, end_tick, name, comment)
            if is_store:
                self.store_measurement(fcs)
            return fcs
        elif type_ == "chronogram":
            chrono = chronogram.Chronogram(self.exps, self, self.exp_param, num_channel, start_tick, end_tick, name, comment)
            if is_store:
                self.store_measurement(chrono)
            return chrono
        elif type_ == "lifetime":
            lifetime_ = lifetime.lifeTimeMeasurements(self.exps, self, self.exp_param, num_channel, start_tick, end_tick, name, comment)
            if is_store:
                self.store_measurement(lifetime_)
            return lifetime_
        elif type_ == "DLS":
            dls = DLS.DLS_Measurements(self.exps, self, self.exp_param, num_channel, start_tick, end_tick, name, comment)
            if is_store:
                self.store_measurement(dls)
            return dls
        elif type_ == "PCH":
            pch = PCH.PCH(self.exps, self, self.exp_param, num_channel, start_tick, end_tick, name, comment)
            if is_store:
                self.store_measurement(pch)
            return pch
        elif type_ == "burst":
            burst = burstDetection.DetectBurst(self.exps, self, self.data, self.exp_param, num_channel, start_tick, end_tick, name, comment)
            if is_store:
                self.store_measurement(burst)
            return burst
        elif type_ == "phosphorescence":
            phospho = phosphorescence.PhosphoMeasurements(self.exps, self, self.exp_param, num_channel, start_tick, end_tick, name, comment)
            if is_store:
                self.store_measurement(phospho)
            return phospho


    def del_measurement(self, name):
        self.measurements.pop(name, None)

    def add_measurement(self, measurement):
        #TODO Test etc
        self.measurements[measurement.name] = measurement

    def duplicate_measurement(self, name):
        new_key = self.measurements[name] + "_b"
        self.measurements[new_key] = self.measurements[name]

    def set_measurement_channel(self, measurement, channel):
        measurement.num_channel = channel


    def calculate_PCH(self, pch, chronogram, bin_size=1):
        """
        Photon Counting Histogramm
        :param chronogram:
        :return:
        """
        # pch = PCH.PCH(self.exp_param, num_channel, start_tick, end_tick, name=name, comment=comment)
        pch.create_histogram(chronogram, timestamps=None,  bin_size=1)

        return pch

    #TODO put it in the controller or view module ?
    def create_navigation_chronograms(self, t1_tick, t2_tick, bin_in_tick, logger=None):
        binInTick = self.convert_seconds_in_ticks(self.defaultBinSize_s)
        if self.navigation_chronograms is None:
            self.navigation_chronograms = []
        for i in range(self.exp_param.nbOfChannel):
            self.navigation_chronograms.append(self.create_measurement(i, t1_tick, t2_tick, type_="chronogram", name="dont_store", comment="", is_store=False))
            self.navigation_chronograms[i] = self.calculate_chronogram(self.navigation_chronograms[i], bin_in_tick)

    def create_time_zoom_chronograms(self, t1_tick, t2_tick, bin_in_tick, logger=None):
        """
        Create a chronogram with a time zoom for each channel
        :param t1_tick:
        :param t2_tick:
        :param bin_in_tick:
        :param logger:
        :return:
        """

        #FIXME check if the channel actually has data in the time bin.
        if self.time_zoom_chronograms is None:
            self.time_zoom_chronograms = []
        for i in range(self.exp_param.nbOfChannel):
            self.time_zoom_chronograms.append(self.create_measurement(i, t1_tick, t2_tick, type_="chronogram", name="dont_store", comment="", is_store=False))
            self.time_zoom_chronograms[i].num_channel = i
            self.time_zoom_chronograms[i].start_tick = t1_tick
            self.time_zoom_chronograms[i].end_tick = t2_tick
            self.time_zoom_chronograms[i] = self.calculate_chronogram(self.time_zoom_chronograms[i], bin_in_tick)
        return self.time_zoom_chronograms

    def create_mini_PCH(self, logger=None):
        """
        Create mini PCH for all channels
        :param logger:
        :return:
        """
        if self.time_zoom_chronograms is not None:
            if self.mini_PCHs is None:
                self.mini_PCHs = []
                for i in range(self.exp_param.nbOfChannel):
                    start_tick, end_tick = self.time_zoom_chronograms[i].start_tick, self.time_zoom_chronograms[i].end_tick
                    self.mini_PCHs.append(self.create_measurement(i, start_tick, end_tick, "PCH",
                                                         name="dont_store", comment="", is_store=False))
                    self.mini_PCHs[i] = self.calculate_PCH(self.mini_PCHs[i], self.time_zoom_chronograms[i], bin_size=1)


    def get_available_name_for_measurement(self, type):
        nb_of_existing_type = 0
        for m in self.measurements.values():
            if m.type ==type:
                nb_of_existing_type += 1
        return type + "_" + str(nb_of_existing_type)

    def store_measurement(self, measurement):

        def auto_generate_name(measurement):
            nb_of_existing_type = 0
            for m in self.measurements.values():
                if m.type == measurement.type:
                    nb_of_existing_type += 1
            return measurement.type + "_" + str(nb_of_existing_type)

        if measurement.name == "":
            name = auto_generate_name(measurement)

        if measurement.name != "dont_store":
            self.measurements[measurement.name] = measurement


    def calculate_FCS(self, measurement, num_c1=0, num_c2=0, start_cor_time_micros = 0.5, max_cor_time_ms=100, is_multi_proc=False, algo="Whal"):
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
        start_tick = measurement.start_tick
        end_tick = measurement.end_tick

        if start_tick == 0:
            start_tick = self.data.channels[num_c1].start_tick

        if end_tick == -1:
            end_tick = self.data.channels[num_c1].end_tick

        timeStamps = self.data.channels[num_channel].photons['timestamps']
        idxStart, idxEnd = np.searchsorted(timeStamps, (start_tick, end_tick))
        timeStamps_reduc = timeStamps[idxStart:idxEnd]

        max_correlation_time_in_tick = int(
            max_cor_time_ms / 1E3 / self.exp_param.mAcrotime_clickEquivalentIn_second)

        start_correlation_time_in_tick = int(
            start_cor_time_micros / 1E6 / self.exp_param.mAcrotime_clickEquivalentIn_second)

        # self.results.FCS_Measurements[num_channel].correlateMonoProc(timeStamps_reduc, timeStamps_reduc,
        #                                                             max_correlation_time_in_tick, start_correlation_time_in_tick)
        tick_duration_micros = self.exp_param.mAcrotime_clickEquivalentIn_second*1E6
        B = 10
        # TODO store in measurement all the aqvuisiiton parameters

        #FIXME from afterpulsing filter
        coeff_1 = np.ones(timeStamps_reduc.size, dtype=np.uint32)
        coeff_2 = np.ones(timeStamps_reduc.size, dtype=np.uint32)

        if is_multi_proc:
            measurement.correlate_multicore(timeStamps_reduc, timeStamps_reduc, coeff_1, coeff_2,
                                            max_correlation_time_in_tick, start_correlation_time_in_tick, B, tick_duration_micros, algo=algo)
        else:
            measurement.correlate_mono_proc(timeStamps_reduc, timeStamps_reduc, coeff_1, coeff_2,
                                            max_correlation_time_in_tick, start_correlation_time_in_tick, B,
                                            tick_duration_micros, algo=algo)

        return measurement

    def calculate_DLS(self, measurement):
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

        # TODO cross correlation

        self.num_c1 = None
        self.num_c2 = None
        self.start_cor_time_micros = None
        self.max_cor_time_ms = None
        self.precision = None

        num_channel = measurement.num_c1
        start_tick = measurement.start_tick
        end_tick = measurement.end_tick

        if start_tick == 0:
            start_tick = self.data.channels[num_c1].start_tick

        if end_tick == -1:
            end_tick = self.data.channels[num_c1].end_tick

        timeStamps = self.data.channels[num_channel].photons['timestamps']
        idxStart, idxEnd = np.searchsorted(timeStamps, (start_tick, end_tick))
        timeStamps_reduc = timeStamps[idxStart:idxEnd]

        max_correlation_time_in_tick = int(
            measurement.max_cor_time_ms / 1E3 / self.exp_param.mAcrotime_clickEquivalentIn_second)

        start_correlation_time_in_tick = int(
            measurement.start_cor_time_micros / 1E6 / self.exp_param.mAcrotime_clickEquivalentIn_second)

        # self.results.FCS_Measurements[num_channel].correlateMonoProc(timeStamps_reduc, timeStamps_reduc,
        #                                                             max_correlation_time_in_tick, start_correlation_time_in_tick)
        tick_duration_micros = self.exp_param.mAcrotime_clickEquivalentIn_second*1E6

        B = measurement.precision
        # TODO store in measurement all the aqvuisiiton parameters

        # measurement.correlateFCS_multicore(timeStamps_reduc, timeStamps_reduc,
        #                                                             max_correlation_time_in_tick, start_correlation_time_in_tick, B, tick_duration_micros)

        measurement.correlate_mono_proc(timeStamps_reduc, timeStamps_reduc,
                                        max_correlation_time_in_tick, start_correlation_time_in_tick, B,
                                        tick_duration_micros)

        return measurement

    def get_info(self):
        print(self.file_name)
        print("%d channel(s)" % self.data.channels.size)
        i = 1
        for c in self.data.channels:
            print("Channel %d" % i)
            print("Total time %0.1f" % c.end_tick / self.exp_param.mAcrotime_clickEquivalentIn_second)
            print("Number of Photon %d" % c.nb_of_tick)
            print("CPS %d" % c.CPS)
            i += 1