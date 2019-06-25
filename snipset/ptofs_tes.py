from scipy.constants import speed_of_light
import lmfit
import numpy as np
import matplotlib.pyplot as plt

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

    index_derivative = \
        (
                - (1.79496 * wl * wl * wl) / (pow(-97.934 + wl * wl, 2))
                + (1.79496 * wl) / (-97.934 + wl * wl)

                - (0.815885 * wl * wl * wl) / (pow(-0.0135121 + wl * wl, 2))
                + (0.815885 * wl) / (-0.0135121 + wl * wl)

                - (1.39233 * wl * wl * wl) / (pow(-0.00467915 + wl * wl, 2))
                + (1.39233 * wl) / (-0.00467915 + wl * wl)
        ) \
        / \
        (2 * np.sqrt(
            1
            + (0.897479 * wl * wl) / (-97.934 + wl * wl)
            + (0.407943 * wl * wl) / (-0.0135121 + wl * wl)
            + (0.696166 * wl * wl) / (-0.00467915 + wl * wl)
        )
         )
    return index, index_derivative


def calculate_index_and_derivative_sellmeier(wl, B1, C1, B2, C2, B3, C3):
    """
    SellMeir coefficient for fused Silica
    :param wl:
    B1 : 0.6961663
    B2 : 0.4079426
    B3 : 0.8974794
    C1 : 0.0684043**2= 0.00467914825849
    C2 : 0.1162414**2 = 0.01351206307396
    C3 : 9.896161**2 = 97.934002537921
    :return:
    """
    index = np.sqrt(1 + (B1 *wl**2) / (wl**2 - C1)
                    + (B2 * wl**2) / (wl**2 - C2)
                    + (B3 * wl**2) / (wl**2 - C3)
                    )

    index_derivative = \
        (
                (-2 * B1 * wl**3) / ((-C1 + wl**2)**2) + (2*B1 * wl) / (-C1 + wl**2)
                + (-2 * B2 * wl**3) / ((-C2 + wl**2)**2) + (2 * B2 * wl) / (-C2 + wl**2)
                + (-2 * B3 * wl**3) / ((-C3 + wl**2)**2) + (2 * B3 * wl) / (-C3 + wl**2) )/(2*index)
    return index, index_derivative



def calculate_transit_time(wl, fiber_length):
    index, index_derivative = calculate_index_and_derivative(wl)
    group_velocity = speed_of_light / (index - wl * index_derivative)
    return fiber_length / group_velocity

def calculate_transit_time_s(wl, fiber_length, B1, C1, B2, C2, B3, C3):
    index, index_derivative = calculate_index_and_derivative(wl, B1, C1, B2, C2, B3, C3)
    group_velocity = speed_of_light / (index - wl * index_derivative)
    return fiber_length / group_velocity


def get_diff_btw_delay_and_transit_time(wl, transit_time_calibration, delay, fiber_length):
    return (calculate_transit_time(wl, fiber_length) - transit_time_calibration) - delay

wl_calib = 0.532
microtime_calib = 1227
fiber_length = 100
microtimes_x = np.linspace(0, 4095)
microtime_to_wl_tab = np.zeros(microtimes_x.size)

transit_time_calibration = calculate_transit_time(wl_calib, fiber_length)
print("transit_time_calibration : ", transit_time_calibration)
# TODO from exp_param.
micro_channel_time_duration = 25E-9

# IR photon arrive first.

# NB : we don't take into account a possible wrapping of the spectra.
# Photon with a shorter microtime than the calibration one are more red
# (assuming that there is no fluorescence decay)

# delay_with_calib = (microtimes_x - microtime_calib) * micro_channel_time_duration
#
# i = 0
# for microtime in microtimes_x:
#     delay_with_calib = (microtime - microtime_calib) * micro_channel_time_duration
#     wl, r = bisect(f=get_diff_btw_delay_and_transit_time, a=0.38, b=1, args=(transit_time_calibration, delay_with_calib, fiber_length))
#     microtime_to_wl_tab[i] = wl
#     i += 1
#
# plt.plot(microtimes_x, microtime_to_wl_tab)
# plt.show()

wls = np.linspace(0.35, 1, 100)
index = np.zeros(wls.size)
index_derivative = np.zeros(wls.size)
index_s = np.zeros(wls.size)
index_derivative_s = np.zeros(wls.size)
transit_time = np.zeros(wls.size)
i=0
for wl in wls:
    index[i], index_derivative[i] = calculate_index_and_derivative(wl)
    index_s[i], index_derivative_s[i] = calculate_index_and_derivative_sellmeier(wl, B1=0.6961663, C1=0.00467914825849, B2=0.4079426, C2=0.01351206307396, B3=0.8974794, C3=97.934002537921)
    transit_time[i] = (calculate_transit_time(wl, fiber_length) - transit_time_calibration) * 1E9
    i+=1


def get_polynom(fiber_length, wl_calib, micro_calib, micro_time_duration_ns, deg=9):
    wls = np.linspace(0.35, 1, 100)
    delays_with_calib = np.zeros(wls.size)
    transit_time_calibration = calculate_transit_time(wl_calib, fiber_length)
    for wl in wls:
        delays_with_calib[i] = (calculate_transit_time(wl, fiber_length) - transit_time_calibration)
    delays_with_calib_microtime = delays_with_calib * 1E9/micro_time_duration_ns

    p_fit_inv = np.polyfit(delays_with_calib_microtime, wls, deg)
    print(p_fit_inv)
    polynomial_interpolation_inverse = np.poly1d(p_fit_inv)

# Interpolation of the theoric/experimental data ->  wavelength vs delay
p_fit = np.polyfit(wls, transit_time, deg=6)
print(p_fit)
polynomial_interpolation = np.poly1d(p_fit)

# Interpolation of the inverse -> delay vs wavelength
p_fit_inv = np.polyfit(transit_time, wls, deg=9)
print(p_fit_inv)
polynomial_interpolation_inverse = np.poly1d(p_fit_inv)

# plt.plot(wls, index)
# plt.plot(wls, index_s)
# plt.show()
#
# plt.plot(wls, index_derivative)
# plt.plot(wls, index_derivative_s)
# plt.show()

# plt.plot(wls*1E3, transit_time)
# plt.plot(wls*1E3, polynomial_interpolation(wls))
# plt.xlabel("wavelength in nm")
# plt.ylabel("delay in ns")
# plt.show()
#
# plt.plot(transit_time, wls)
# plt.plot(transit_time, polynomial_interpolation_inverse(transit_time))
# plt.ylabel("wavelength in nm")
# plt.xlabel("delay in ns")
# plt.show()

# Fit from experimental data
wl_calib = 0.531
y = [38.256,37.21,35.72,31.284,30.475,27.837,27.393,27.028,26.557,26.244,25.879,25.644,25.06,24.53,24.25,23.7,23.48,23.21,22.87,22.66,22.35,22.04,21.8,21.23,21.048,20.839,20.682,18.619]
x = [413.0,423.0,435,483,495,531,539,548,557,567,574,580,591,606,618,633,647,659,669,681,695,709,725,752,762,780,795,1072]

y = np.array(y) - 27.837
x = np.array(x)
x /= 1000.0
x -= 0.007


def delay_function(x, fiber_length, calib_error):
    global transit_time_calibration
    # x -= calib_error
    transit_time = (calculate_transit_time(x, fiber_length) - transit_time_calibration) * 1E9
    return transit_time

def delay_function_w_sellmeir(x, fiber_length, calib_error, B1, C1, B2, C2, B3, C3):
    global wl_calib
    transit_time_calibration = calculate_transit_time(wl_calib, fiber_length)
    # x -= calib_error
    transit_time = (calculate_transit_time(x, fiber_length) - transit_time_calibration) * 1E9
    return transit_time


gmodel = lmfit.Model(delay_function)
result = gmodel.fit(y, x=x, fiber_length=100, calib_error=0.000)

gmodel = lmfit.Model(delay_function_w_sellmeir)
result = gmodel.fit(y, x=x, fiber_length=100, calib_error=0.000,  B1=0.6961663, C1=0.00467914825849, B2=0.4079426, C2=0.01351206307396, B3=0.8974794, C3=97.934002537921)

print(result.fit_report())

plt.style.use('ggplot')
plt.rcParams['lines.linewidth'] = 4

plt.plot(x, y, 'bo', label="exp data")
plt.plot(x, result.best_fit, 'r-', label="Fit with L=121m", alpha=0.7)
plt.xlabel("wavelength (µm)")
plt.ylabel("Delay (ns)")
plt.title("Calibration of the GI50 Sedi Fibre (100m), effective length=125m")
plt.legend()
plt.savefig("Calib_fiber.png")
plt.show()

