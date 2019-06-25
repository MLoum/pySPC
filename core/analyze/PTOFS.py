import numpy as np
from scipy.constants import speed_of_light
from scipy.optimize import bisect

from .Measurement import Measurements



class PTOFS(Measurements):

    def __init__(self, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exp_param, num_channel, start_tick, end_tick, "PTOFS", name, comment)

    def create_microtime_wl_correspondence(self, microtimes_x, fiber_length, wl_calib, microtime_calib, wl_min, wl_max):
        """// Si on avait le temps de transit dans la fibre d'un photon le problème serait facile.
        // Il suffit d'inverser (par dichotomie) la formule donnant le temps de transit en fonction de lambda.
        //
        // Pour l'obtenir il faudrait faire une réponse percu avec une fibre courte (ou sans fibre) alors on aurait un pic à zero
        // ce pic ne serait pas forcement au microtime zero à cause de la taille des cables et de la propagation dans l'air.
        //
        // Par contre on peut se donner (comprendre mesure) une réference.
        // Par exemple, au microcanal 400 on à 0.530µm
        //
        // On a donc un point de reference à 400µc  on peut calculer le temps de transit.
        //
        // Maintenant pour un microtime donné. Deux cas de figure il est en avance ou en retard par rapport à la reference
        // Dans les deux cas on peut calculer l'écart en temps entre les deux transits -> Le Delay.
        // ATTENTION delay = TpsTransit_Ref - TpsTransit_photon_courant    l'ordre est IMPORTANT
        //
        // Après on trouve la longueur d'onde tel que son temps de transit moins le temps de transit de la reference soit égal au delay, ce à epsilon près.
        """

        self.fiber_length = fiber_length
        self.microtimes = microtimes_x
        self.wl_calib = wl_calib
        self.microtime_calib = microtime_calib
        self.wl_min = wl_min
        self.wl_max = wl_max

        # self.data = np.zeros(microtimes.size)
        # self.data[microtime_calib] = 0

        self.microtime_to_wl_tab = np.zeros(microtimes_x.size)


        # TODO use a polynomial fit of the experimental chromatic dispersion curve ?
        def calculate_index_and_derivative(wl):
            """
            SellMeir coefficient for fused Silica
            :param wl:
            :return:
            """
            index = np.sqrt(1 + (0.6961663 * wl * wl) / (wl * wl - 0.0684043 * 0.0684043)
                            + (0.4079426 * wl * wl) / (wl * wl - 0.1162414 * 0.1162414)
                            + (0.8974794 * wl * wl) / (wl * wl - 9.896161 * 9.896161)
                            )

            index_derivative =\
                (
                        - (1.79496 * wl * wl * wl) / (pow(-97.934 + wl * wl, 2))
                        + (1.79496 * wl) / (-97.934 + wl * wl)

                        - (0.815885 * wl * wl * wl) / (pow(-0.0135121 + wl * wl, 2))
                        + (0.815885 * wl) / (-0.0135121 + wl * wl)

                        - (1.39233 * wl * wl * wl) / (pow(-0.00467915 + wl * wl, 2))
                        + (1.39233 * wl) / (-0.00467915 + wl * wl)
                )\
                /\
                (2 * np.sqrt(
                    1
                    + (0.897479 * wl * wl) / (-97.934 + wl * wl)
                    + (0.407943 * wl * wl) / (-0.0135121 + wl * wl)
                    + (0.696166 * wl * wl) / (-0.00467915 + wl * wl)
                )
                 )
            return index, index_derivative

        def calculate_transit_time(wl):
            index, index_derivative = calculate_index_and_derivative(wl)
            group_velocity = speed_of_light / (index - wl * index_derivative)
            return self.fiber_length / group_velocity

        def get_diff_btw_delay_and_transit_time(wl, delay):
            return (calculate_transit_time(wl) - self.transit_time_calibration) - delay

        self.transit_time_calibration = calculate_transit_time(wl_calib)
        # TODO from exp_param.
        micro_channel_time_duration = 25E-9

        # IR photon arrive first.

        # NB : we don't take into account a possible wrapping of the spectra.
        # Photon with a shorter microtime than the calibration one are more red
        # (assuming that there is no fluorescence decay)

        delay_with_calib = (microtimes_x - microtime_calib) * micro_channel_time_duration

        i = 0
        for microtime in microtimes_x:
            delay_with_calib = (microtime - microtime_calib) * micro_channel_time_duration
            wl, r = bisect(f=get_diff_btw_delay_and_transit_time, a=0.38, b=1, args=(delay_with_calib,))
            self.microtime_to_wl_tab[i] = wl
            i += 1

    def test_conversion(self):
        # microtimes = np.linspace(0, 4095)
        # ptofs.create_microtime_wl_correspondence((microtimes, 100, 532, 1227, 350, 1000))
        import matplotlib.pyplot as plt
        plt.plot(microtimes, self.microtime_to_wl_tab)
        plt.show()

    def deconvolution_with_microtime(self):
        pass



ptofs = PTOFS()


