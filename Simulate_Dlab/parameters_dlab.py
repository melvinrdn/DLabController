import numpy as np
# This script contains useful formulas to calculate relevant parameters

wl_red = 1.03  # microns
wl_green = 0.515  # microns

pulse_power_red = 3  # W
pulse_power_green = 0.350  # W
R = (4*pulse_power_green)/(pulse_power_red+4*pulse_power_green)

pulse_length_red = 180e-15  # s
pulse_length_green = pulse_length_red

rep_rate_red = 10e3  # Hz
rep_rate_green = rep_rate_red  # Hz

pulse_energy_red = pulse_power_red / rep_rate_red  # J
pulse_energy_green = pulse_power_green / rep_rate_green  # J

red_radius = 27e-6  # m
green_radius = 14e-6 # m
intensity_red = 0.94*2 * pulse_energy_red / ((pulse_length_red * np.pi) * (red_radius ** 2)) * 1e-4 * 1e-14
intensity_green = 0.94*2 * pulse_energy_green / (
        (pulse_length_green * np.pi) * (green_radius ** 2)) * 1e-4 * 1e-14

U_p_red = 9.337 * wl_red ** 2 * intensity_red  # eV
U_p_green = 9.337 * wl_green ** 2 * intensity_green  # eV

I_p_Argon = 15.7596  # eV
E_cut_red = U_p_red * 3.17 + I_p_Argon  # eV
E_cut_green = U_p_green * 3.17 + I_p_Argon  # eV

print(f"Ratio: {R}")
print(f"Peak Intensity red: {intensity_red} x 10e14 W/cm2")
print(f"Peak Intensity green: {intensity_green} x 10e14 W/cm2")
print(f"Up red: {U_p_red:} eV")
print(f"Up green: {U_p_green:} eV")
print(f"Cut_off red: {E_cut_red:} eV, {int(E_cut_red/1.2):}th order")
print(f"Cut_off green: {E_cut_green:} eV, {int(E_cut_green/1.2):}th order")





