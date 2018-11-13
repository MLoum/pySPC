import numpy as np
import os
from core import Results
# from .importFormat import bh_SPC_SET
# from .importFormat import pq_PT3_PT2
from core.importFormat import pqreader
from core.importFormat import bhreader
from core.importFormat import nist_fpga
# from .analyze import bin




class Channel():
    def __init__(self, name=""):
        self.name = name
        self.photons = []
        self.nb_of_tick = 0
        self.start_tick = 0
        self.end_tick = 0
        self.CPS = 0

    def update(self, mAcrotime_clickEquivalentIn_second):
        """
        Update  nb_of_tick,  start_tick, end_tick and CPS
        :return:
        """
        self.start_tick = self.photons['timestamps'][0]
        self.end_tick = self.photons['timestamps'][-1]
        self.nb_of_tick = self.photons['timestamps'].size
        self.CPS = float(self.nb_of_tick) / (self.end_tick - self.start_tick) / mAcrotime_clickEquivalentIn_second


class Data():
    """
    Contains the SPC raw data, and method to process them (file opening, noise generation, filtering).
    """

    def __init__(self, expParam):

        # Ici cela serait du Data-Oriented Design
        # http://gamesfromwithin.com/data-oriented-design
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html
        # self.photonDataType = np.dtype([('timestamps', np.uint64), ('nanotimes', np.uint32) , ('detectors', np.uint16), ('isFiltered', np.bool_)])
        self.photonDataType = np.dtype([('timestamps', np.uint64), ('nanotimes', np.uint32), ('detectors', np.uint16)])

        # Je ne fait pas une approche 100% objet avec une classe detecteur et une classe photon, car cela complique bcp l'utilisation de numpy
        self.timestamps = []
        self.nanotimes = []
        self.isFiltered = []

        self.channels = []

        self.expParam = expParam

    def _del_data(self):
        del self.timestamps[:]
        del self.nanotimes[:]
        del self.isFiltered[:]

    def load_from_file(self, file_path):
        """

        :param file_path:
        :return:
        """
        if (os.path.isfile(file_path)) is False:
            print("File not existing")
            return "File not existing"


        filename, file_extension = os.path.splitext(file_path)
        if file_extension == ".spc":
            timestamps, detectors, nanotimes, timestamps_unit, meta = self.loadSPC_Bh_File(file_path)
            # Done inside the loadSPC_Bh_File function
            self.expParam.fill_with_SPC_meta_data(meta, timestamps_unit)

        elif file_extension == ".pt3":
            timestamps, detectors, nanotimes, meta = pqreader.load_pt3(file_path)
            self.expParam.fill_with_pt3_meta_data(meta)

        elif file_extension == ".ttt":
            timestamps, detectors, nanotimes, timestamps_unit, meta = nist_fpga.load_ttt(file_path)
            self.expParam.fill_with_ttt_meta_data(meta)

        # Les photons ne sont pas triès par detecteur, il le sont par ordre d'arrivée
        unique, return_index, unique_inverse, unique_counts = np.unique(detectors, return_index=True,
                                                                        return_inverse=True, return_counts=True)

        numChannel = 0
        nbOfChannel = 0
        for value in unique:
            if unique_counts[numChannel] > 50 and value >= 0:
                nbOfChannel += 1
            numChannel += 1

        self.expParam.nbOfChannel = nbOfChannel

        self._del_data()
        del self.channels[:]

        numChannel = 0
        soft_channel_value = 0
        for value in unique:

            # 50 is arbiratry, we do this to filter false count on some detector.
            if unique_counts[numChannel] < 50:
                numChannel += 1
                soft_channel_value += 1
                continue

            # On créé un masque et on fait une copie des elements non masqués vers le channel.
            m_ = np.ma.masked_where(detectors != value, timestamps)
            timestampsMasked = np.ma.compressed(m_)

            m_ = np.ma.masked_where(detectors != value, nanotimes)
            nanotimesMasked = np.ma.compressed(m_)

            photons = np.empty(unique_counts[soft_channel_value], self.photonDataType)

            photons['timestamps'] = timestampsMasked
            photons['nanotimes'] = nanotimesMasked

            # self.isFiltered.append(np.ones(unique_counts[soft_channel_value], dtype=bool))
            # soft_channel_value += 1

            # Mask an array where a condition is met.
            # condition : array_like
            # print(np.shape(self.photons))
            # print(self.photons['detectors'])

            # m_ = np.ma.masked_where(self.photons['detectors']  != i, self.photons)
            # print(m_)
            # print(m_['detectors'])
            #
            # print(np.ma.getmask(m_))
            # test = np.ma.compressed(m_)
            # print(test)

            # Count the non-masked elements of the array along the given axis.
            # nbElementNonMasked = np.ma.MaskedArray.count(m_['detectors'])

            c = Channel()
            c.photons = photons
            c.update(self.expParam.mAcrotime_clickEquivalentIn_second)
            self.channels.append(c)
            # self.results.add_channel()
            # TODO  ???
            soft_channel_value += 1
            numChannel += 1

            # grande question : quand est-il des tableaux avec des masques, ici on garderait une version masquée
            # je prefere garder une copie.
            # c.photons.append[np.ma.masked_where(self.photons['detectors']  == i, self.photons)]

        # J'utilise d'abord np.unique sur le detector qui m'indique quelle valeur de detecteur sont présent dans le fichier et où.
        # Je peux par exemple savoir que trois detecteurs ont été utilisés : 0,1 et 3 et je connais l'index de leur tick respectif (unique_inverse).

        # for value in unique:
        #     m_ = np.ma.masked_where(unique_inverse == return_index[soft_channel_value], timestamps)
        #     self.timestamps.append(np.ma.compressed(m_))
        #
        #     m_ = np.ma.masked_where(unique_inverse == return_index[soft_channel_value], nanotimes)
        #     self.nanotimes.append(np.ma.compressed(m_))
        #
        #     self.isFiltered.append(np.ones(unique_counts[soft_channel_value], dtype=bool))
        #     soft_channel_value += 1

        return "OK"

    def new_generated_exp(self, type, params):
        """

        :param type:
        :param params:
        :return:
        """
        if type == "Poisson":
            self._del_data()
            del self.channels[:]

            self.expParam.nbOfMicrotimeChannel = 256
            # TODO ask user ?
            self.expParam.mAcrotime_clickEquivalentIn_second = 50E-9
            self.expParam.mIcrotime_clickEquivalentIn_second = 10E-12

            time_s = params[0]
            time_tick = time_s / self.expParam.mAcrotime_clickEquivalentIn_second

            count_per_second_s = params[1]
            count_per_tick = count_per_second_s * self.expParam.mAcrotime_clickEquivalentIn_second

            timeStamps = self.generate_poisson_noise(count_per_tick, 0, time_tick)
            photons = np.empty(np.size(timeStamps), self.photonDataType)

            photons['timestamps'] = timeStamps
            photons['nanotimes'] = 0

            c = Channel()
            c.photons = photons
            c.start_tick = c.photons['timestamps'][0]
            c.end_tick = c.photons['timestamps'][-1]
            c.nb_of_tick = c.photons['timestamps'].size
            c.CPS = float(c.nb_of_tick) / (c.end_tick - c.start_tick) / self.expParam.mAcrotime_clickEquivalentIn_second
            self.channels.append(c)

    def loadSPC_Bh_File(self, filePath):
        # find associate set
        path_set = os.path.splitext(filePath)[0] + '.set'
        if (os.path.isfile(path_set)):
            meta = bhreader.load_set(path_set)
            self.expParam.fill_with_SPC_meta_data(meta, None)
        else:
            return "default set File missing"

        # TODO fill ExpParam with set
        timestamps, detector, nanotime, timestamps_unit = bhreader.load_spc(filePath, spc_model="SPC-130")
        return timestamps, detector, nanotime, timestamps_unit, meta

    def searchIdxOfPhotonWithTimeSupTo_t1_and_InfTo_t2(self, array, t1, t2):
        # Find indices where elements should be inserted to maintain order
        return np.searchsorted(array, (t1, t2))
        pass

    def generate_poisson_noise(self, mean_rate_in_tick, t_start_click, t_end_click):
        """

        :param mean_rate_in_tick:
        :param t_start_click:
        :param t_end_click:
        :return:
        """

        # http://preshing.com/20111007/how-to-generate-random-timings-for-a-poisson-process/
        # ratePerClick = ratePerS / self.expParam.mAcrotime_clickEquivalentIn_second
        # TODO name of ratePerClick
        nb_of_tick_to_generate = int((t_end_click - t_start_click) * mean_rate_in_tick)
        arrival_times = t_start_click + np.cumsum(
            -(np.log(1.0 - np.random.random(nb_of_tick_to_generate)) / mean_rate_in_tick).astype(np.uint64))
        last_sample = np.searchsorted(arrival_times, t_end_click)
        return arrival_times[:last_sample]

    def filter_bin_and_threshold(self, num_channel, threshold, bin_in_tick, replacement_mode="nothing"):
        """

        :param num_channel:
        :param threshold:
        :param bin_in_tick:
        :param replacement_mode:
        :return:
        """
        time_stamps = self.channels[num_channel].photons['timestamps']
        # Binning
        num_bin_for_each_photons = (time_stamps / bin_in_tick).astype(np.int64)
        binned_timestamps = np.bincount(num_bin_for_each_photons)
        # Filter
        idx_bin_to_filter = np.where(binned_timestamps > threshold)
        is_photons_to_be_filtered = np.isin(num_bin_for_each_photons, idx_bin_to_filter)
        idx_photons_to_filter = np.nonzero(is_photons_to_be_filtered)

        # TODO Use mask ?

        if replacement_mode is "nothing":
            self.channels[num_channel].photons = np.delete(self.channels[num_channel].photons,
                                                           idx_photons_to_filter)
        elif replacement_mode is "glue":
            """
            Most of the times, this is a bad idea
            """
            pass

        elif replacement_mode is "poissonian_noise":
            # Strategy, put artificial poisson noise at the end of the photons list and sort the photon list
            old_cps_per_tick = self.channels[num_channel].CPS * self.expParam.mAcrotime_clickEquivalentIn_second

            self.channels[num_channel].photons = np.delete(self.channels[num_channel].photons,
                                                           idx_photons_to_filter)

            list_poisson_photons = []
            #TODO without a loop ?
            for bin_idx in idx_bin_to_filter[0]:
                poisson_signal = self.generate_poisson_noise(old_cps_per_tick, bin_idx*bin_in_tick, (bin_idx+1)*bin_in_tick )
                photons_poisson = np.empty(np.size(poisson_signal), self.photonDataType)

                photons_poisson['timestamps'] = poisson_signal
                photons_poisson['nanotimes'] = np.random.rand(poisson_signal.size) * self.expParam.nb_of_microtime_channel

                list_poisson_photons.append(photons_poisson)

            photons_poisson_all = np.concatenate(list_poisson_photons)
            self.channels[num_channel].photons = np.append(self.channels[num_channel].photons,
                                                           values=photons_poisson_all)

            np.sort(self.channels[num_channel].photons, order='timestamps')


    def filter_time_selection(self, num_channel, t1_tick, t2_tick, is_keep=True, replacement_mode="nothing"):
        time_stamps = self.channels[num_channel].photons['timestamps']

        if is_keep:
            idx_photons_to_be_filtered = np.where(np.logical_or(time_stamps < t1_tick, time_stamps > t2_tick))
        else:
            idx_photons_to_be_filtered = np.where(np.logical_and(time_stamps > t1_tick, time_stamps < t2_tick))

        if replacement_mode is "nothing":
            self.channels[num_channel].photons = np.delete(self.channels[num_channel].photons,
                                                           idx_photons_to_be_filtered)
        if replacement_mode is "glue":
            if is_keep:
                self.channels[num_channel].photons = np.delete(self.channels[num_channel].photons,
                                                               idx_photons_to_be_filtered)
                time_stamps -= t1_tick
            else:
                idx_t2_tick = np.searchsorted(time_stamps, t2_tick)
                time_stamps[idx_t2_tick:] -= np.int64(t2_tick - t1_tick)
                self.channels[num_channel].photons = np.delete(self.channels[num_channel].photons,
                                                               idx_photons_to_be_filtered)

        elif replacement_mode is "poissonian_noise":
            if is_keep:
                old_cps_per_tick = self.channels[num_channel].CPS * self.expParam.mAcrotime_clickEquivalentIn_second
                self.channels[num_channel].photons = np.delete(self.channels[num_channel].photons,
                                                               idx_photons_to_be_filtered)
                # Before t_1
                poisson_signal_t1 = self.generate_poisson_noise(old_cps_per_tick, 0, t1_tick)
                photons_poisson_t1 = np.empty(np.size(poisson_signal_t1), self.photonDataType)

                photons_poisson_t1['timestamps'] = poisson_signal_t1
                photons_poisson_t1['nanotimes'] = np.random.rand(poisson_signal_t1.size) * self.expParam.nb_of_microtime_channel
                self.channels[num_channel].photons = np.insert(self.channels[num_channel].photons, obj=0, values=photons_poisson_t1)

                # After t_2
                poisson_signal_t2 = self.generate_poisson_noise(old_cps_per_tick, t2_tick, self.channels[num_channel].end_tick)
                photons_poisson_t2 = np.empty(np.size(poisson_signal_t2), self.photonDataType)

                photons_poisson_t2['timestamps'] = poisson_signal_t2
                photons_poisson_t2['nanotimes'] = np.random.rand(poisson_signal_t2.size) * self.expParam.nb_of_microtime_channel
                self.channels[num_channel].photons = np.append(self.channels[num_channel].photons, values=photons_poisson_t2)

                np.sort(self.channels[num_channel].photons, order='timestamps')

            if is_keep is False:
                old_cps_per_tick = self.channels[num_channel].CPS * self.expParam.mAcrotime_clickEquivalentIn_second
                idx_t1_tick = np.searchsorted(time_stamps, t1_tick)
                self.channels[num_channel].photons = np.delete(self.channels[num_channel].photons,
                                                               idx_photons_to_be_filtered)
                poisson_signal = self.generate_poisson_noise(old_cps_per_tick, t1_tick, t2_tick)
                photons_poisson = np.empty(np.size(poisson_signal), self.photonDataType)

                photons_poisson['timestamps'] = poisson_signal
                photons_poisson['nanotimes'] = np.random.rand(photons_poisson.size) * self.expParam.nb_of_microtime_channel
                self.channels[num_channel].photons = np.insert(self.channels[num_channel].photons, obj=idx_t1_tick, values=photons_poisson)
                np.sort(self.channels[num_channel].photons, order='timestamps')

        self.channels[num_channel].update(self.expParam.mAcrotime_clickEquivalentIn_second)

    def filter_micro_time(self, num_channel, micro_t1, micro_t2, is_keep=True, replacement_mode="nothing"):
        nanotimes = self.channels[num_channel].photons['nanotimes']

        if is_keep:
            idx_photons_to_be_filtered = np.where(np.logical_or(nanotimes < micro_t1, nanotimes > micro_t2))
        else:
            idx_photons_to_be_filtered = np.where(np.logical_and(nanotimes > micro_t1, nanotimes < micro_t2))

        if replacement_mode is "nothing":
            self.channels[num_channel].photons = np.delete(self.channels[num_channel].photons,
                                                           idx_photons_to_be_filtered)
        elif replacement_mode is "poissonian_noise":
            """
            Est-ce faisable, les mircotimes ne sont pas à la suite...
            """
            pass
