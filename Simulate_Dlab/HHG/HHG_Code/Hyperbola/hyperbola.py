import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.interpolate import interp1d

# This code is calculating the hyperbola described in Nat Rev Phys 4, 713–722 (2022).

data_path_hyperbola = '//fysfile01/atomhome$/me2120re/My Documents/Doctorat/Projets/Pulse_duration/Hyperbola/'
data_sample_hyperbola = 'hyperbola_180fs_Imac.mat'
data_hyperbola = io.loadmat(data_path_hyperbola + data_sample_hyperbola)
CE = data_hyperbola['results']
CE[np.isnan(CE)] = 0

data_path_imac = '//fysfile01/atomhome$/me2120re/My Documents/Doctorat/Projets/Pulse_duration/'
data_sample_imac = 'PM_Window_q23_Ar_1030nm.txt'
data_imac = np.genfromtxt(data_path_imac + data_sample_imac, dtype=float, autostrip=True, skip_header=2)
interp_function_imac = interp1d(data_imac[:, 0], data_imac[:, 1], kind='cubic', fill_value="extrapolate")


def gamma_model(z, z_r, w_0, I_0, wavelength, I_p, q):
    w_z = w_0 * np.sqrt(1 + (z / z_r) ** 2)
    I_z = I_0 * (w_0 / w_z) ** 2
    c = 299792458  # Speed of light in vacuum (m/s)
    hbar = 1.054571800e-34  # Reduced Planck constant (Js)
    me = 9.10938356e-31  # Electron mass (kg)
    a = 0.0072973525693  # Fine structure
    omega = 2 * np.pi * c / wavelength   # Angular frequency (rad/s)

    y_s = 0.22 * c * me / (a * wavelength)
    beta_s = -y_s / I_z * (q * omega - I_p / hbar) ** 2

    y_l = -0.19 * c * me / (a * wavelength)
    a_l = -0.16 * a * wavelength ** 3 / (me * c ** 3)
    beta_l = a_l * I_z - y_l / I_z * (q * omega - I_p / hbar) ** 2
    return beta_s, beta_l


# Fundamental constants
c = 299792458  # Speed of light in vacuum (m/s)
eps0 = 8.854187817e-12  # Vacuum permittivity (C/V/m)
k_b = 1.380e-23  # Boltzmann constant (J/K)
hbar = 1.054571800e-34  # Reduced Planck constant (Js)
me = 9.10938356e-31  # Electron mass (kg)
a = 0.0072973525693  # Fine structure
e = 1.60217662e-19  # Electron charge (C)

# Parameters
I_0 = 1.5e18
w_0 = 30e-6
tau= 30e-15
q = 23
I_p = 15.7596 * 1.60218e-19  # Ar Ip in J
wavelength = 1030e-9  # Wavelength (m)
omega = 2 * np.pi * c / wavelength  # Angular frequency (rad/s)
z_r = 2.73e-3  # Rayleigh length (m)
coh_p = 3  # "Coherence" parameter
T = 288  # Temperature (K)
# Valid for q=23 @ 1030nm in Ar
sigma_abs = 1.98e-21  # Absorption cross section (m^2)
alpha_0 = 1.86e-40  # Polarizability of fundamental (C^2.m^2/J)
alpha_23 = -1.406e-40  # Polarizability of q=23 (C^2.m^2/J)
eta_mac = me * omega ** 2 / e ** 2 * (alpha_0 - alpha_23)
I_mic = me * omega ** 2 / (3.17 * 2 * np.pi * a) * (q * omega - I_p / hbar)
I_mac = interp_function_imac(tau)

# Lists
z_list = np.linspace(0, 0.2 * z_r, 1000)  # Medium length (m)
p_list = np.linspace(0, 1500, 100)
p_zr_list = p_list * z_r * 1e2  # mbar.cm

# Pressure and pressure length product
p_exp = 466  # Experimental pressure (mbar)
L_med_exp = 0.1 * z_r  # Experimental medium length (m)
p_zr_exp = p_exp * 100 * z_r  # Experimental pressure length product (mbar.cm)
pos_medium = L_med_exp/2   # L_med_exp/2 is the end of my medium
beta_s, beta_l = gamma_model(pos_medium, z_r, w_0, I_0, wavelength, I_p, q)
f_i_s = z_r ** 2 / (pos_medium ** 2 + z_r ** 2) * (
            1 + (2 * pos_medium * beta_s) / (q * z_r))  # Correction factor for short
f_i_l = z_r ** 2 / (pos_medium ** 2 + z_r ** 2) * (
            1 + (2 * pos_medium * beta_l) / (q * z_r))  # Correction factor for long
f_i_s_str = f'{f_i_s:.2f}'
f_i_l_str = f'{f_i_l:.2f}'
p0_s = 2 * eps0 * c * k_b * T * f_i_s / (omega * abs(alpha_0 - alpha_23))  # PM minimal pressure for eta=0, short
p0_l = 2 * eps0 * c * k_b * T * f_i_l / (omega * abs(alpha_0 - alpha_23))  # PM minimal pressure for eta=0, long
p0_s_zr = p0_s * z_r * 100  # PM minimum Pressure length product, short
p0_l_zr = p0_l * z_r  # PM minimum Pressure length product, long
p_zr_s = (coh_p * k_b * T / sigma_abs / z_list + p0_s) * z_r  # Pa.m = mbar.cm
p_zr_l = (coh_p * k_b * T / sigma_abs / z_list + p0_l) * z_r  # Pa.m = mbar.cm

print(f'I_0 = {I_0 * 1e-18:.2f}x1e14 W/cm2')
print(f'I_mic = {I_mic * 1e-18:.2f}x1e14 W/cm2')
print(f'eta_mac = {eta_mac * 100:.2f} %')
print(f'I_mac = {I_mac * 1e-14:.2f}x1e14 W/cm2')


# Plots
plt.figure(1)
extent = [z_list[0] / z_r, z_list[-1] / z_r, p_zr_list[0], p_zr_list[-1]]
plt.imshow(CE.T, extent=extent, cmap='turbo', aspect='auto', origin='lower', interpolation='None')
#plt.plot(z_list/z_r, p_zr_s, 'w--', linewidth=1)
#plt.axhline(y=p0_s_zr, color='w', linestyle='--', linewidth=1, label='p_0')
#plt.plot(z_list/z_r, p_zr_l, 'w--', linewidth=1)
#plt.plot(L_med_exp / z_r, p_zr_exp, 'bo', markersize=3)
#legend_text_exp = [f'$p$ = {p0_s:.1f} mbar']
#plt.legend(legend_text_exp)
plt.ylabel('$p z_R$ (mbar.cm)')
plt.xlabel('Medium length ($z_R$)')
plt.ylim((0, 400))
plt.xlim((0, 0.5))
plt.clim(0,1e-6)
plt.title(f'{data_sample_hyperbola}')
plt.colorbar(label='Conversion efficiency')
plt.show()

