from snipset.other import winspec
import numpy as np
import matplotlib.pyplot as plt
import os
import colour

from colour.plotting import *
# plot_visible_spectrum()

dir_path = os.path.dirname(os.path.realpath(__file__))


is_normalize = True
is_colored_spectrum = True
is_plot_CIE_diagram = False
is_remove_bckgd = True

data_SPE = []
data_txt = []
#data_SPE.append(("name.spe", "label_name"))
# F_CN
# data_SPE.append(["230419_F_CN_crystal_laser.spe", "230419_F_CN_crystal_laser"])
# data_SPE.append(["230419_F_CN_crystal1.spe", "230419_F_CN_crystal1"])
# data_SPE.append(["230419_F_CN_crystal2.spe", "230419_F_CN_crystal2"])
# data_SPE.append(["230419_F_CN_crystal3.spe", "230419_F_CN_crystal3"])
# data_SPE.append(["230419_F_CN_crystal4.spe", "230419_F_CN_crystal4"])

# data_SPE.append(["230419_F_NO2_crystal1.spe", "230419_F_NO2_crystal1"])
# data_SPE.append(["230419_F_NO2_crystal2.spe", "230419_F_NO2_crystal2"])

# data_SPE.append(["230419_NO2_CN_crystal1.spe", "230419_NO2_CN_crystal1"])
# data_SPE.append(["230419_NO2_CN_big_crystal_2.spe", "230419_NO2_CN_big_crystal_2"])
# data_SPE.append(["230419_NO2_CN_big_crystal_3.spe", "230419_NO2_CN_big_crystal_3"])

# data_SPE.append(["230419_NO2_CN_crystal1.spe", "230419_NO2_CN_crystal1"])
# data_SPE.append(["230419_F_CN_crystal2.spe", "230419_F_CN_crystal2"])
# data_SPE.append(["230419_F_NO2_crystal2.spe", "230419_F_NO2_crystal2"])

# data_SPE.append(["2019_04_12_Ftype.spe", "2019_04_12_Ftype"])
# data_SPE.append(["230419_F_CN_crystal2.spe", "230419_F_CN_crystal2"])
# data_SPE.append(["230419_F_NO2_crystal2.spe", "230419_F_NO2_crystal2"])

data_txt.append(["230419_F_NO2_crystal2.spe", "230419_F_NO2_crystal2"])

class Spectrum:
    def __init__(self, file_name, type="spe", label=None):
        self.file_name = file_name
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.join(dir_path, file_name)
        self.label = label

        if type== "spe":
            self.data_from_spe()
        elif type == "txt":
            self.data_from_txt()

    def copute_color(self):
        self.sample_spd_data = dict(zip(self.raw_wl, self.raw_spec))
        self.spd = colour.SpectralDistribution(self.sample_spd_data, name=self.label)
        # Cloning the sample spectral power distribution.
        self.spd_interpolated = self.spd.clone()

        # Interpolating the cloned sample spectral power distribution.
        # Indeed, the algorithm for colour identification needs a spectra with a point each nanometer
        self.spd_interpolated.interpolate(colour.SpectralShape(400, 820, 1))

        self.cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        self.illuminant = colour.ILLUMINANTS_SDS['D65']
        self.XYZ = colour.sd_to_XYZ(self.spd_interpolated, self.cmfs, self.illuminant)
        self.RGB = colour.XYZ_to_sRGB(self.XYZ / 100)
        # TODO why can we get negative values and values > 1 ?
        self.RGB = np.abs(self.RGB)
        self.RGB = np.clip(self.RGB, 0, 1)
        self.xy = colour.XYZ_to_xy(self.XYZ)

    def data_from_spe(self):
        self.spectrum = winspec.Spectrum(self.file_name)
        self.raw_wl = self.spectrum.wavelen
        self.raw_spec = self.spectrum.lum

    def data_from_txt(self):
        raw_data = np.loadtxt(self.file_name)



spectra = []

for data in data_SPE:
    spectra.append(Spectrum(data[0], data[1]))

for spectrum in spectra:
    plt.plot(spectrum.spectrum.wavelen, spectrum.spectrum.lum)

plt.show()

# Plotting the *CIE 1931 Chromaticity Diagram*.
# The argument *standalone=False* is passed so that the plot doesn't get
# displayed and can be used as a basis for other plots.
plot_chromaticity_diagram_CIE1931(standalone=False)

for spectrum in spectra:
    x, y = spectrum.xy
    plt.plot(x, y, 'o-', color='white')
    # Annotating the plot.
    plt.annotate(spectrum.label,
                 xy=spectrum.xy,
                 xytext=(-50, 30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-0.2'))


# Displaying the plot.
render(
    standalone=True,
    limits=(-0.1, 0.9, -0.1, 0.9),
    x_tighten=True,
    y_tighten=True)

# for data in data_SPE:
#     data[0] = os.path.join(dir_path, data[0])
#
# for data in data_SPE:
#     spec = winspec.Spectrum(data[0])
#
#     if is_remove_bckgd:
#         #TODO better background estimation
#         spec.lum -= spec.lum[0]
#
#     if is_normalize:
#         spec.lum /= spec.lum.max()
#
#     data.append(spec.wavelen)
#     data.append(spec.lum)
#
# if is_colored_spectrum:
#     import colour
#
# for data in data_SPE:
#     if is_colored_spectrum:
#         sample_spd_data = dict(zip(data[2], data[3]))
#         spd = colour.SpectralDistribution(sample_spd_data, name='Sample')
#         # Cloning the sample spectral power distribution.
#         spd_interpolated = spd.clone()
#
#         # Interpolating the cloned sample spectral power distribution.
#         spd_interpolated.interpolate(colour.SpectralShape(400, 820, 1))
#
#         cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
#         illuminant = colour.ILLUMINANTS_SDS['D65']
#         XYZ = colour.sd_to_XYZ(spd_interpolated, cmfs, illuminant)
#         RGB = colour.XYZ_to_sRGB(XYZ / 100)
#         # TODO why can we get negative values and values > 1 ?
#         RGB = np.abs(RGB)
#         RGB = np.clip(RGB, 0, 1)
#         # print(XYZ)
#         print(RGB)
#         if is_plot_CIE_diagram:
#             # Computing *xy* chromaticity coordinates for the *neutral 5 (.70 D)* patch.
#             xy = colour.XYZ_to_xy(XYZ)
#
#             # Plotting the *CIE 1931 Chromaticity Diagram*.
#             # The argument *standalone=False* is passed so that the plot doesn't get
#             # displayed and can be used as a basis for other plots.
#             plot_chromaticity_diagram_CIE1931(standalone=False)
#
#             # Plotting the *xy* chromaticity coordinates.
#             x, y = xy
#             plt.plot(x, y, 'o-', color='white')
#
#             # Annotating the plot.
#             plt.annotate(data[1],
#                          xy=xy,
#                          xytext=(-50, 30),
#                          textcoords='offset points',
#                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-0.2'))
#
#
#         # single_colour_swatch_plot(ColourSwatch('Sample', RGB), text_size=32)
#         plt.plot(data[2], data[3], label=data[1], color=RGB)
#     else:
#         plt.plot(data[2], data[3], label=data[1])
#
# plt.legend()
# plt.title("Cristaux F comparison 230419")
# plt.xlabel("wavelength / nm")
# plt.ylabel("Relative intensity")
# plt.savefig("230419_F_comparison.png")
# plt.show()
#
# if is_plot_CIE_diagram:
#     # Displaying the plot.
#     render(
#         standalone=True,
#         limits=(-0.1, 0.9, -0.1, 0.9),
#         x_tighten=True,
#         y_tighten=True)

print("OK")