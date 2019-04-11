import numpy as np
from .Measurement import Measurements
from .lifetime import lifeTimeMeasurements
from .chronogram import Chronogram
from .PCH import PCH


class Burst():
    """
    Should maybe be a numpy custom dtype ?
    """
    def __init__(self):
        self.nb_bin = 0
        self.num_bin_start = 0
        self.num_bin_end = 0
        self.nb_photon = 0
        self.num_photon_start = 0
        self.num_photon_stop = 0
        self.tick_start = 0
        self.tick_end = 0
        self.duration_tick = 0
        self.CPS = 0
        self.measurement = None

import numba

@numba.jit(nopython=True)
def threshold_loop_numba(idx, data, burst_threshold, flank_threshold):
        while idx < data.size and data[idx] < burst_threshold:
            idx += 1
        if idx >= data.size:
            return -1, -1
        # We are inside a burst, we will know go in the left and right direction until we goes under the flank threshold
        idx_center_burst = idx

        # going left
        while idx >= 0 and data[idx] > flank_threshold:
            idx -= 1
        num_bin_start = idx + 1

        # going right
        idx = idx_center_burst
        while idx < data.size and data[idx] > flank_threshold:
            idx += 1
        # End of the burst is the start of a bin that is under the flank_threshold
        # burst.num_bin_end = idx - 1
        num_bin_end = idx

        # TODO dark count.

        return num_bin_start, num_bin_end


class DetectBurst(Measurements):
    def __init__(self, data, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exp_param, num_channel, start_tick, end_tick, "burst", name, comment)
        self.data = data
        self.bursts = []
        self.parameters = None
        self.binned_timestamps = None
        self.threshold = None
        self.bin_in_tick = 0

    def bin(self, bin_in_tick):
        self.bin_in_tick = bin_in_tick
        self.timestamps = self.data.channels[self.num_channel].photons['timestamps']
        self.chronogram = Chronogram(self.exp_param, self.num_channel, self.start_tick, self.end_tick, self.name + "_chrono", self.comment)
        self.chronogram.create_chronogram(self.timestamps, bin_in_tick)

        self.PCH = PCH(self.exp_param, self.num_channel, self.start_tick, self.end_tick, self.name + "_PCH", self.comment)
        self.PCH.create_histogram(self.chronogram)
        return self.PCH

    def do_threshold(self, burst_threshold, flank_threshold, min_succesive_bin, max_succesive_noise_bin, min_nb_photon):

        self.burst_threshold = burst_threshold
        self.flank_threshold = flank_threshold
        self.bursts = []

        data = self.chronogram.data

        self.nb_too_short_burst = 0
        num_photon = 0
        idx = 0
        while idx < data.size:
            burst = Burst()
            # Delegated to exterior numba pre-compiled function for efficiency purpose
            burst.num_bin_start, burst.num_bin_end  = threshold_loop_numba(idx, data, burst_threshold, flank_threshold)
            if burst.num_bin_start == -1:
                break
            idx = burst.num_bin_end
            # while idx < data.size and data[idx] < burst_threshold:
            #     idx += 1
            # if idx >= data.size:
            #     break
            #
            # # We are inside a burst, we will know go in the left and right direction until we goes under the flank threshold
            # idx_center_burst = idx
            #
            # #going left
            # while idx >=0 and data[idx] > flank_threshold:
            #     idx -=1
            # burst.num_bin_start = idx + 1
            #
            # #going right
            # idx = idx_center_burst
            # while idx < data.size and data[idx] > flank_threshold:
            #     idx += 1
            # # End of the burst is the start of a bin that is under the flank_threshold
            # # burst.num_bin_end = idx - 1
            # burst.num_bin_end = idx

            # Look at burst characteristics and see, considering the filter parameters, if we have to put it in the burst list
            burst.nb_bin = burst.num_bin_end - burst.num_bin_start
            if burst.nb_bin > min_succesive_bin:
                # if idx_end_of_burst + 1 < idx_candidate_bin.size:
                burst.num_photon_start = np.searchsorted(self.timestamps,
                                                                     burst.num_bin_start * self.chronogram.bin_in_tick)
                # num_photon = burst.num_photon_start
                burst.tick_start = self.timestamps[burst.num_photon_start]
                burst.num_photon_stop = np.searchsorted(self.timestamps, burst.num_bin_end*self.chronogram.bin_in_tick)
                # else:
                #     burst.num_photon_stop = num_photon = np.searchsorted(self.timestamps[num_photon:],
                #                                                          idx_candidate_bin[
                #                                                              idx_end_of_burst] * self.chronogram.bin_in_tick)
                # num_photon = burst.num_photon_stop
                burst.tick_end = self.timestamps[burst.num_photon_stop]
                burst.duration_tick = burst.tick_end - burst.tick_start
                burst.nb_photon = burst.num_photon_stop - burst.num_photon_start
                burst.CPS = burst.nb_photon / (burst.duration_tick*self.exp_param.mAcrotime_clickEquivalentIn_second)
                if burst.nb_photon > min_nb_photon:
                    self.bursts.append(burst)
            else:
                self.nb_too_short_burst += 1







        # idx = 0
        # while idx < idx_candidate_bin.size - 1:
        #     # Every bin indexed by idx_candidate_bin are potentially bursts since they are above the threshold
        #     # Start of the burst
        #     burst = Burst()
        #     burst.num_bin_start = idx_candidate_bin[idx]
        #     burst.num_photon_start = num_photon = np.searchsorted(self.timestamps[num_photon:],
        #                                                           idx_candidate_bin[idx] * self.chronogram.bin_in_tick)
        #     burst.tick_start = self.timestamps[burst.num_photon_start]
        #
        #     # Let's see if the next bin is also in the burst. This is the case if the difference of index is
        #     # smaller than max_succesive_noise_bin
        #     # NB : the index idx contain the difference between the index idx+1 and idx of the idx_candidate_bin array
        #
        #     inside_burst = True
        #     # idx += 1
        #
        #     while inside_burst is True:
        #         if idx >= idx_candidate_bin.size - 1:
        #             break
        #         if diff_idx_candidate_bin[idx] <= max_succesive_noise_bin:
        #             # Still in the burst
        #             idx += 1
        #         else:
        #             inside_burst = False
        #     idx_end_of_burst = idx
        #
        #     # Test if burst can be append to the list
        #
        #     # cf nb de piquet et de haie.
        #     burst.num_bin_end = idx_candidate_bin[idx_end_of_burst] + 1
        #     burst.nb_bin = burst.num_bin_end - burst.num_bin_start
        #     if burst.nb_bin > min_succesive_bin:
        #         if idx_end_of_burst + 1 < idx_candidate_bin.size:
        #             burst.num_photon_stop = num_photon = np.searchsorted(self.timestamps[num_photon:], idx_candidate_bin[idx_end_of_burst+1]*self.chronogram.bin_in_tick)
        #         else:
        #             burst.num_photon_stop = num_photon = np.searchsorted(self.timestamps[num_photon:],
        #                                                                  idx_candidate_bin[
        #                                                                      idx_end_of_burst] * self.chronogram.bin_in_tick)
        #         burst.tick_end = self.timestamps[burst.num_photon_stop]
        #         burst.duration_tick = burst.tick_end - burst.tick_start
        #         burst.nb_photon = burst.num_photon_stop - burst.num_photon_start
        #         burst.CPS = burst.nb_photon / burst.duration_tick
        #         if burst.nb_photon > min_nb_photon:
        #             self.bursts.append(burst)
        #     else:
        #         self.nb_too_short_burst += 1
        #     idx += 1


        # From python list to numpy array for statistical analysis
        self.nb_of_bursts = len(self.bursts)
        self.bursts_length = np.zeros(len(self.bursts))
        self.bursts_intensity = np.zeros(len(self.bursts))
        self.bursts_CPS = np.zeros(len(self.bursts))
        for i, burst in enumerate(self.bursts):
            self.bursts_length[i] = burst.duration_tick
            self.bursts_intensity[i] = burst.nb_photon
            self.bursts_CPS[i] = burst.CPS

        self.bursts_length_histogram, self.bin_edges_bursts_length = np.histogram(self.bursts_length*self.exp_param.mAcrotime_clickEquivalentIn_second*1E6, bins="auto")
        self.bursts_intensity_histogram, self.bin_edges_bursts_intensity = np.histogram(self.bursts_intensity, bins="auto")
        self.bursts_CPS_histogram, self.bin_edges_bursts_CPS = np.histogram(self.bursts_CPS/(self.exp_param.mAcrotime_clickEquivalentIn_second*1E6),
                                                                                        bins="auto")
        self.perform_measurements()

    def perform_measurements(self, type=None):
        num = 0
        for burst in self.bursts:
            burst.measurement = lifeTimeMeasurements(self.exp_param, self.num_channel, burst.tick_start, burst.tick_end, "burst_" + str(num) + "_lifetime")
            nanotimes = self.data.channels[self.num_channel].photons['nanotimes'][burst.num_photon_start:burst.num_photon_stop]
            burst.measurement.create_histogramm(nanotimes)
            num += 1

    def cusum_sprt(self, macrotimes, I0, IB):
        """

        :param macrotimes:
        :param I0: Estimation of the Intensity when the particle is at the center of the focal volume in CPS
        :param IB: Estimation of the noise in CPS.
        :return:
        """
        pass

        delta_i = np.diff(macrotimes)
        alpha = 1 / macrotimes.size
        alpha = 1 / 100.0
        # 5% of missed occurence
        beta = 0.05

        I1 = I0*np.exp(-2) + IB

        """
        // NB la formule est a priori fausse Dans le cadre d'un signal Poissonien
        // Param->lambda_1 = -1 + B/S + Log(S/B)
        Param->lambda_1 = Param->I_B_EnSecondeMoinsUn / (double) Param->I_1_EnSecondeMoinsUn ;
        Param->lambda_1 = -1 + Param->I_B_EnSecondeMoinsUn / (double) Param->I_1_EnSecondeMoinsUn + log(Param->I_B_EnSecondeMoinsUn / (double) Param->I_1_EnSecondeMoinsUn);     
        """
        lambda_1 = -1 + IB/I1 + np.log(IB/I1)

        # Formule de Lorden
        h = - np.log(alpha * np.log(1/alpha)) / (3*(lambda_1 - 1)*(lambda_1 - 1))

        A = (1-beta)/alpha # >>1
        B = beta /(1-alpha) # ~beta

    def get_burst(self, num):
        return self.bursts[num]

    def get_nb_of_burst(self):
        return len(self.bursts)

