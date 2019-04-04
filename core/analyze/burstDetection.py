import numpy as np
from .Measurement import Measurements
from .chronogram import Chronogram
from .PCH import PCH

class Burst():
    """
    Should maybe be a numpy custom dtype ?
    """
    def __init__(self):
        self.nb_photon = 0
        self.num_photon_start = 0
        self.num_photon_stop = 0
        self.duration_tick = 0
        self.measurement = None

class DetectBurst(Measurements):
    def __init__(self, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exp_param, num_channel, start_tick, end_tick, "burst", name, comment)
        self.bursts = []
        self.parameters = None
        self.binned_timestamps = None


    def bin(self, timestamps, bin_in_tick):
        self.timestamps = timestamps
        self.chronogram = Chronogram(self.exp_param, self.num_channel, self.start_tick, self.end_tick, self.name + "_chrono", self.comment)
        self.chronogram.create_chronogram(timestamps, bin_in_tick)

        self.PCH = PCH(self.exp_param, self.num_channel, self.start_tick, self.end_tick, self.name + "_PCH", self.comment)
        self.PCH.create_histogram(self.chronogram)
        return self.PCH

    def threshold(self, threshold, min_succesive_bin, max_succesive_noise_bin, min_nb_photon):

        self.bursts = []

        binned_ts_thresholded = np.copy(self.chronogram.data)
        # binned_ts_thresholded[binned_ts_thresholded < 0] = 0
        # Then, the burst are the consecutive non-zero values
        # mark the bin that are under the threshold, they are put to Nan.
        binned_ts_thresholded[binned_ts_thresholded < threshold] = 0

        #Find consecutive burst bins
        idx_candidate_bin = np.where(binned_ts_thresholded > 0)[0]
        diff_idx_candidate_bin = np.diff(idx_candidate_bin) - 1
        num_photon = 0

        consecutive_noise_flag = False
        idx = 0
        while idx < idx_candidate_bin.size - 1:
            if diff_idx_candidate_bin[idx] == 0:
                # Start of the burst
                burst = Burst()
                burst.num_photon_start = num_photon = np.searchsorted(self.timestamps[num_photon:], idx_candidate_bin[idx]*self.chronogram.bin_in_tick)
                consecutive_noise_bin = 0
                inside_burst = True
                idx += 1

                while inside_burst is True:
                    if idx >= idx_candidate_bin.size - 1:
                        break
                    if diff_idx_candidate_bin[idx] == 0:
                        # Still in the burst
                        idx += 1
                        consecutive_noise_bin = 0
                    else:
                        if consecutive_noise_bin == max_succesive_noise_bin:
                            # end of the burst
                            inside_burst = False
                        else:
                            idx += 1
                        consecutive_noise_bin += 1

                # end of the burst

                # Test if burst can be append to the list
                burst.num_photon_stop = num_photon = np.searchsorted(self.timestamps[num_photon:], idx_candidate_bin[idx]*self.chronogram.bin_in_tick)
                burst.nb_photon = burst.num_photon_stop - burst.num_photon_start
                burst.duration_tick = self.timestamps[burst.num_photon_stop] - self.timestamps[burst.num_photon_start]
                if burst.num_photon_stop - burst.num_photon_start > min_succesive_bin:
                    if burst.nb_photon > min_nb_photon:
                        self.bursts.append(burst.nb_photon)
            else:
                idx += 1

        # From python list to numpy array for statistical analysis
        self.nb_of_bursts = len(self.bursts)
        self.bursts_length = np.zeros(len(self.bursts))
        self.bursts_intensity = np.zeros(len(self.bursts))
        for i, burst in enumerate(self.bursts):
            self.bursts_length[i] = burst.duration_tick
            self.bursts_intensity[i] = burst.nb_photon

        self.bursts_length_histogram = np.histogram(self.bursts_length, bins="auto")
        self.bursts_intensity_histogram = np.histogram(self.bursts_intensity, bins="auto")

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



