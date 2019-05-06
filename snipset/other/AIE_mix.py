from snipset.other import winspec
import numpy as np
import matplotlib.pyplot as plt
import os
import colour

from lmfit import Model, minimize, Parameters, fit_report

from colour.plotting import *
# plot_visible_spectrum()

dir_path = os.path.dirname(os.path.realpath(__file__))


is_normalize = True
is_colored_spectrum = True
is_plot_CIE_diagram = False
is_remove_bckgd = True


class Spectrum:
    def __init__(self, wl, spec, label=None):
        self.label = label

        self.raw_wl = wl
        self.raw_spec = spec

        self.compute_color()

    def compute_color(self):
        self.sample_spd_data = dict(zip(self.raw_wl, self.raw_spec))
        self.spd = colour.SpectralDistribution(self.sample_spd_data, name=self.label)
        # Cloning the sample spectral power distribution.
        self.spd_interpolated = self.spd.clone()

        # Interpolating the cloned sample spectral power distribution.
        # Indeed, the algorithm for colour identification needs a spectra with a point each nanometer
        self.spd_interpolated.interpolate(colour.SpectralShape(370, 680, 1))

        self.cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        self.illuminant = colour.ILLUMINANTS_SDS['D65']
        self.XYZ = colour.sd_to_XYZ(self.spd_interpolated, self.cmfs, self.illuminant)
        self.RGB = colour.XYZ_to_sRGB(self.XYZ / 100)
        """
        We then come to the next obstacle...  The RGB values that we get from this process are often out of range - 
        meaning that they are either greater than 1.0, or even that they are negative!  The first case is fairly 
        straightforward, it means that the color is too bright for the display.  The second case means that the color 
        is too saturated and vivid for the display.  The display must compose all colors from some combination of 
        positive amounts of the colors of its red, green and blue phosphors.  The colors of these phosphors are not 
        perfectly saturated, they are washed out, mixed with white, to some extent.  So not all colors can be displayed 
        accurately.  As an example, the colors of pure spectral lines, all have some negative component.  
        Something must be done to put these values into the 0.0 - 1.0 range that can actually be displayed, 
        known as color clipping.
        
        In the first case, values larger than 1.0, ColorPy scales the color so that the maximum component is 1.0.  
        This reduces the brightness without changing the chromaticity.  The second case requires some change in 
        chromaticity.  By default, ColorPy will add white to the color, just enough to make all of the components 
        non-negative.  (You can also have ColorPy clamp the negative values to zero.  My personal, qualitative, 
        assessment is that adding white produces somewhat better results.  There is also the potential to develop a 
        better clipping function.)
        """
        if min(self.RGB) < 0:
            self.RGB += min(self.RGB)
        if max(self.RGB) > 1:
            self.RGB /= max(self.RGB)

        # TODO why can we get negative values and values > 1 ?
        self.RGB = np.abs(self.RGB)
        self.RGB = np.clip(self.RGB, 0, 1)
        self.xy = colour.XYZ_to_xy(self.XYZ)



is_plot = True

# OF1530 -> Fluor  OF 155 -> CN OF 154 - NO2
print(plt.style.available)

plt.style.use('ggplot')
plt.rcParams['lines.linewidth'] = 4

label_Fluor = "F"
label_CN = "CN"
label_Nitro = "NO2"

# Absorption

raw_data = np.loadtxt("Absorption.dat", skiprows=3)
raw_data = np.transpose(raw_data)

idx_600 = np.searchsorted(raw_data[0], 600)
if is_plot:
    plt.plot(raw_data[0][:idx_600], (raw_data[1][:idx_600] - raw_data[1][idx_600])/raw_data[1][:idx_600].max(), label=label_Fluor, linestyle="solid")
    plt.plot(raw_data[0][:idx_600], (raw_data[2][:idx_600] - raw_data[2][idx_600])/raw_data[2][:idx_600].max(), label=label_CN, linestyle="dashed")
    plt.plot(raw_data[0][:idx_600], (raw_data[3][:idx_600] - raw_data[3][idx_600])/raw_data[3][:idx_600].max(), label=label_Nitro, linestyle="dotted")
    plt.ylabel("Relative Absorbance (u.a)")
    plt.xlabel("Wavelength")
    plt.title("Absorption spectra in DMSO (concentration = )")
    plt.vlines(405,0,0.9, alpha=0.5)
    plt.legend()
    plt.savefig("Abs_spectra_dmso.png", dpi=300)
    plt.show()

# Emission

raw_data = np.loadtxt("Emission.dat", skiprows=3)
raw_data = np.transpose(raw_data)
wl = raw_data[0]
dmso_Fluor = raw_data[1]
dmso_CN = raw_data[2]
dmso_NO2 = raw_data[3]

spec_fluor_dmso = Spectrum(wl, dmso_Fluor/dmso_Fluor.max(), label=label_Fluor)
spec_CN_dmso = Spectrum(wl, dmso_CN/dmso_CN.max(), label=label_CN)
spec_NO2_dmso = Spectrum(wl, dmso_NO2/dmso_NO2.max(), label=label_Nitro)
if is_plot:
    plt.plot(wl, dmso_Fluor/dmso_Fluor.max(), label=label_Fluor, linestyle="solid")
    plt.plot(wl, dmso_CN/dmso_CN.max(), label=label_CN, linestyle="dashed")
    plt.plot(wl, dmso_NO2/dmso_NO2.max(), label=label_Nitro, linestyle="dotted")
    plt.ylabel("Relative intensity (u.a)")
    plt.xlabel("Wavelength")
    plt.title("Emission spectra in DMSO (concentration = ?)")
    plt.legend()
    plt.savefig("Emission_dye_dmso.png", dpi=300)
    plt.show()


aggregate_Fluor = raw_data[4]
aggregate_CN = raw_data[5]
aggregate_NO2 = raw_data[6]

spec_fluor_aggregate = Spectrum(wl, aggregate_Fluor, label=label_Fluor)
spec_CN_aggregate = Spectrum(wl, aggregate_CN, label=label_CN)
spec_NO2_aggregate = Spectrum(wl, aggregate_NO2, label=label_Nitro)


if is_plot:
    plt.plot(wl, aggregate_Fluor/aggregate_Fluor.max(), label=label_Fluor, linestyle="solid")
    plt.plot(wl, aggregate_CN/aggregate_CN.max(), label=label_CN, linestyle="dashed")
    plt.plot(wl, aggregate_NO2/aggregate_NO2.max(), label=label_Nitro, linestyle="dotted")
    plt.ylabel("Relative intensity (u.a)")
    plt.xlabel("Wavelength")
    plt.title("Emission spectra of aggregates in H2O")
    plt.legend()
    plt.savefig("Emission_spectra_aggregates_h2o.png", dpi=300)
    plt.show()

F_CN_mix_of_dyes = raw_data[7]
F_CN_mix_of_aggregates = raw_data[8]
F_NO2_mix_of_dyes = raw_data[9]
F_NO2_mix_of_aggregates = raw_data[10]
CN_NO2_mix_of_dyes = raw_data[11]
CN_NO2_mix_of_aggregates = raw_data[12]
F_CN_NO2_mix_of_dyes = raw_data[13]
F_CN_NO2_mix_of_aggregates = raw_data[14]

if is_plot:
    plt.plot(wl, F_CN_mix_of_dyes/F_CN_mix_of_dyes.max(), label="F_CN", linestyle="solid")
    plt.plot(wl, F_NO2_mix_of_dyes/F_NO2_mix_of_dyes.max(), label="F_NO2", linestyle="dashed")
    plt.plot(wl, CN_NO2_mix_of_dyes/CN_NO2_mix_of_dyes.max(), label="CN_NO2", linestyle="dotted")
    plt.ylabel("Relative intensity (u.a)")
    plt.xlabel("Wavelength")
    plt.title("Emission spectra of aggregates, mix of dyes, in H2O")
    plt.legend()
    plt.savefig("Emission_aggregates_mix_of_dyes.png", dpi=300)
    plt.show()

if is_plot:
    plt.plot(wl, F_CN_mix_of_aggregates/F_CN_mix_of_aggregates.max(), label="F_CN", linestyle="solid")
    plt.plot(wl, F_NO2_mix_of_aggregates/F_NO2_mix_of_aggregates.max(), label="F_NO2", linestyle="dashed")
    plt.plot(wl, CN_NO2_mix_of_aggregates/CN_NO2_mix_of_aggregates.max(), label="CN_NO2", linestyle="dotted")
    plt.ylabel("Relative intensity (u.a)")
    plt.xlabel("Wavelength")
    plt.title("Emission spectra of aggregates, mix of aggregates, in H2O")
    plt.legend()
    plt.savefig("Emission_aggregates_mix_of_aggregates.png", dpi=300)
    plt.show()

if is_plot:
    plt.plot(wl, F_CN_mix_of_aggregates/F_CN_mix_of_aggregates.max(), label="F_CN m_agg", linestyle="solid")
    plt.plot(wl, F_CN_mix_of_dyes/F_CN_mix_of_dyes.max(), label="F_CN m_dyes", linestyle="dashed")
    plt.ylabel("Relative intensity (u.a)")
    plt.xlabel("Wavelength")
    plt.title("Emission spectra of aggregates, F-CN, in H2O")
    plt.legend()
    plt.savefig("Emission_aggregates_F_CN.png", dpi=300)
    plt.show()

if is_plot:
    plt.plot(wl, F_NO2_mix_of_aggregates/F_NO2_mix_of_aggregates.max(), label="F_NO2 m_agg")
    plt.plot(wl, F_NO2_mix_of_dyes/F_NO2_mix_of_dyes.max(), label="F_NO2 m_dyes", linestyle="dashed")
    plt.ylabel("Relative intensity (u.a)")
    plt.xlabel("Wavelength")
    plt.title("Emission spectra of aggregates, F-NO2 in H2O")
    plt.legend()
    plt.savefig("Emission_aggregates_F_NO2.png", dpi=300)
    plt.show()


if is_plot:
    plt.plot(wl, CN_NO2_mix_of_aggregates/CN_NO2_mix_of_aggregates.max(), label="CN_NO2 m_agg")
    plt.plot(wl, CN_NO2_mix_of_dyes/CN_NO2_mix_of_dyes.max(), label="CN_NO2 m_dyes", linestyle="dashed")
    plt.ylabel("Relative intensity (u.a)")
    plt.xlabel("Wavelength")
    plt.title("Emission spectra of aggregates, CN_NO2, in H2O")
    plt.legend()
    plt.savefig("Emission_aggregates_CN_NO2.png", dpi=300)
    plt.show()

if is_plot:
    plt.plot(wl, F_CN_NO2_mix_of_aggregates/F_CN_NO2_mix_of_aggregates.max(), label="F_CN_NO2 m_agg")
    plt.plot(wl, F_CN_NO2_mix_of_dyes/F_CN_NO2_mix_of_dyes.max(), label="F_CN_NO2 m_dyes", linestyle="dashed")
    plt.ylabel("Relative intensity (u.a)")
    plt.xlabel("Wavelength")
    plt.title("Emission spectra of aggregates, F-CN-NO2, in H2O")
    plt.legend()
    plt.savefig("Emission_aggregates_F_CN_NO2.png", dpi=300)
    plt.show()




def sum_spectra(pars, x, data=None, S1=None, S2=None):
    A = pars['A']
    B = pars['B']
    model = A*S1/S1.max() + B*S2/S2.max()
    return model - data


# F-CN
fit_params = Parameters()
fit_params.add('A', value=0.5)
fit_params.add('B', value=0.5)

out = minimize(sum_spectra, fit_params, args=(wl,), kws={'data': F_CN_mix_of_aggregates/F_CN_mix_of_aggregates.max(), "S1":aggregate_Fluor, "S2" : aggregate_CN})
A = out.params['A']
B = out.params['B']

plt.plot(wl, F_CN_mix_of_aggregates/F_CN_mix_of_aggregates.max(), label='exp')
# plt.plot(wl, A*aggregate_Fluor/aggregate_Fluor.max() + B*aggregate_CN/aggregate_CN.max(), label="sum " + str(A) + "% CN + " + str(B) + "% F")
plt.plot(wl, A*aggregate_Fluor/aggregate_Fluor.max() + B*aggregate_CN/aggregate_CN.max(), label="sum")
plt.plot(wl, A*aggregate_Fluor/aggregate_Fluor.max(), 'k--', label="F")
plt.plot(wl, B*aggregate_CN/aggregate_CN.max(), 'b--', label="CN")
plt.legend()
plt.ylabel("Relative intensity (u.a)")
plt.xlabel("Wavelength")
plt.title("F-CN mix of aggregates as the sum of F and CN spectra")
plt.savefig("F_CN_mix_aggregate_as_sum.png", dpi=300)
plt.show()

# CN-NO2
fit_params = Parameters()
fit_params.add('A', value=0.5)
fit_params.add('B', value=0.5)

out = minimize(sum_spectra, fit_params, args=(wl,), kws={'data': CN_NO2_mix_of_aggregates/CN_NO2_mix_of_aggregates.max(), "S1":aggregate_NO2, "S2" : aggregate_CN})
A = out.params['A']
B = out.params['B']



plt.plot(wl, CN_NO2_mix_of_aggregates/CN_NO2_mix_of_aggregates.max(), label='exp')
# plt.plot(wl, A*aggregate_Fluor/aggregate_Fluor.max() + B*aggregate_CN/aggregate_CN.max(), label="sum " + str(A) + "% CN + " + str(B) + "% F")
plt.plot(wl, A*aggregate_NO2/aggregate_NO2.max() + B*aggregate_CN/aggregate_CN.max(), label="sum")
plt.plot(wl, A*aggregate_NO2/aggregate_NO2.max(), 'k--', label="NO2")
plt.plot(wl, B*aggregate_CN/aggregate_CN.max(), 'b--', label="CN")
plt.legend()
plt.ylabel("Relative intensity (u.a)")
plt.xlabel("Wavelength")
plt.title("CN_NO2 mix of aggregates as the sum of CN and NO2 spectra")
plt.savefig("CN_NO2_mix_aggregate_as_sum.png", dpi=300)
plt.show()


# F-NO2
fit_params = Parameters()
fit_params.add('A', value=0.5)
fit_params.add('B', value=0.5)

out = minimize(sum_spectra, fit_params, args=(wl,), kws={'data': F_NO2_mix_of_aggregates/F_NO2_mix_of_aggregates.max(), "S1":aggregate_Fluor, "S2" : aggregate_NO2})
A = out.params['A']
B = out.params['B']

plt.plot(wl, F_NO2_mix_of_aggregates/F_NO2_mix_of_aggregates.max(), label='exp')
# plt.plot(wl, A*aggregate_Fluor/aggregate_Fluor.max() + B*aggregate_CN/aggregate_CN.max(), label="sum " + str(A) + "% CN + " + str(B) + "% F")
plt.plot(wl, A*aggregate_Fluor/aggregate_Fluor.max() + B*aggregate_NO2/aggregate_NO2.max(), label="sum")
plt.plot(wl, A*aggregate_Fluor/aggregate_Fluor.max(), 'k--', label="F")
plt.plot(wl, B*aggregate_NO2/aggregate_NO2.max(), 'b--', label="CN")
plt.legend()
plt.ylabel("Relative intensity (u.a)")
plt.xlabel("Wavelength")
plt.title("F_NO2 mix of aggregates as the sum of F and NO2 spectra")
plt.savefig("F_NO2_mix_aggregate_as_sum.png", dpi=300)
plt.show()


# gmodel = Model(sum_spectra)
# gmodel.set_param_hint('A', value=0.5, min=0, max=0.5)
# gmodel.set_param_hint('B', value=0.5, min=0, max=0.5)
# y = F_CN_mix_of_aggregates
# result = gmodel.fit(y, x=wl, A=0.5, B=0.5)
# print(result.params['A'])
# print(result.params['B'])




# Plotting the *CIE 1931 Chromaticity Diagram*.
# The argument *standalone=False* is passed so that the plot doesn't get
# displayed and can be used as a basis for other plots.
# plot_chromaticity_diagram_CIE1931(standalone=False)
#
# for spectrum in spectra:
#     x, y = spectrum.xy
#     plt.plot(x, y, 'o-', color='white')
#     # Annotating the plot.
#     plt.annotate(spectrum.label,
#                  xy=spectrum.xy,
#                  xytext=(-50, 30),
#                  textcoords='offset points',
#                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-0.2'))
#
#
# # Displaying the plot.
# render(
#     standalone=True,
#     limits=(-0.1, 0.9, -0.1, 0.9),
#     x_tighten=True,
#     y_tighten=True)

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