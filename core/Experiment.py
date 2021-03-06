from core import Data
from core import ExpParam
from core import Results
import os
import numpy as np

from core.analyze import lifetime, FCS, DLS, chronogram, PCH, burstDetection, phosphorescence, PTOFS

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

    def __init__(self, mode, params_dict, exps=None):
        self.exps = exps
        self.exp_param = ExpParam.Experiment_param()
        # self.results = Results.Results()
        self.measurements = {}
        self.data = Data.Data(self)

        self.navigation_chronograms = None
        self.time_zoom_chronograms = None
        self.mini_PCHs = None


        self.file_name = None
        self.comment = ""

        self.defaultBinSize_s = 0.01  # default : 10ms

        self.new_exp(mode, params_dict)


    # TODO put convert function where it belongs
    def convert_ticks_in_seconds(self, ticks):
        return self.exp_param.mAcrotime_clickEquivalentIn_second * ticks

    def convert_seconds_in_ticks(self, seconds):
        return seconds / self.exp_param.mAcrotime_clickEquivalentIn_second

    def new_exp(self, mode, params_dict):
        """
        Create a new experiment from a file or from simulated data.

        :param mode: a string telling how to create the new experiment. Can be :
            - "file" : a recorded file spc or pt3 file
            - "generate" : Poissonian Noise
            - "simulation" : Not Implemented Yet.
        :param params_dict: params dictionnary passed to the function based on the mode :

        :return:
        """
        if mode == "file":
            filePath = params_dict["file_path"]
            result = self.data.load_from_file(filePath)
            if result is not "OK":
                return

            head, self.file_name = os.path.split(filePath)

        elif mode == "generate":
            type = params_dict["type"]
            time_s = params_dict["time"]
            count_per_second_s = params_dict["cps"]
            #FIXME could change
            self.exp_param.nbOfChannel = 1
            self.data.new_generated_exp(type, params_dict)
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
            lifetime_ = lifetime.lifeTimeMeasurements(self.exps, self, self.exp_param, num_channel, start_tick, end_tick, "lifetime", name, comment)
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
        elif type_ == "PTOFS":
            ptofs = PTOFS.PTOFS(self.exps, self, self.exp_param, num_channel, start_tick, end_tick, type_="PTOFS",  name=name, comment=comment)
            if is_store:
                self.store_measurement(ptofs)
            return ptofs


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
        # if self.time_zoom_chronograms is None:
        #     self.time_zoom_chronograms = []

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

    def get_raw_data(self, num_channel=0, start_tick=0, end_tick=-1, type="timestamp", mode="data"):

        timestamps = self.data.channels[num_channel].photons['timestamps']
        nanotimes = self.data.channels[num_channel].photons['nanotimes']

        idxStart, idxEnd = np.searchsorted(timestamps, (start_tick, end_tick))

        if type == "nanotimes":
            if mode == "data":
                return nanotimes[idxStart:idxEnd]
            elif mode == "full":
                return nanotimes

        if type == "timestamp":
            if mode == "data":
                return timestamps[idxStart:idxEnd]
            elif mode == "full":
                return timestamps

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