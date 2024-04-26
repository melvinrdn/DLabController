import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Constants
c = 299792458  # Speed of light in vacuum (m/s)
eps0 = 8.854187817e-12  # Vacuum permittivity (C/V/m)
k_b = 1.380e-23  # Boltzmann constant (J/K)
hbar = 1.054571800e-34  # Reduced Planck constant (Js)
me = 9.10938356e-31  # Electron mass (kg)
a = 0.0072973525693  # Fine structure
e = 1.60217662e-19  # Electron charge (C)

# Load the ionization rate data
file_path = 'ionization-rate-tdse.txt'
data = np.genfromtxt(file_path, dtype=float, autostrip=True, skip_header=2)
intensity, ionization_rate = data[:, 0], data[:, 1]
G_interp = interp1d(intensity, ionization_rate, kind='cubic', fill_value="extrapolate")

# Parameters
wavelength = 1030e-9  # Wavelength (m)
q = 23
I_p = 15.7596 * e  # Ar Ip in J
omega = 2 * np.pi * c / wavelength  # Angular frequency (rad/s)
sigma_abs = 1.98e-21  # Absorption cross section (m^2)
alpha_0 = 1.86e-40  # Polarizability of fundamental (C^2.m^2/J)
alpha_23 = -1.406e-40  # Polarizability of q=23 (C^2.m^2/J)
eta_mac = me * omega ** 2 / e ** 2 * (alpha_0 - alpha_23)
print(f'eta_mac q=23 Ar 1030nm: {eta_mac} %')
I_mic = me * omega ** 2 / (3.17 * 2 * np.pi * a) * (q * omega - I_p / hbar)
print(f'I_mic: {I_mic / 1e18:.4f} x 1e14 W/cm^2')
w0 = 30e-6


def I(t, I0, sigma):
    return I0 * np.exp(-t ** 2 / (sigma ** 2))


def integrand(t, I0, sigma):
    return G_interp(I(t, I0, sigma))


tau_min = 10e-15
tau_max = 200e-15
tau_values = np.linspace(tau_min, tau_max, 5)
results = []

for tau in tau_values:
    sigma = tau / (2 * np.sqrt(np.log(2)))
    I_min, I_max = 1e14, 3e14
    I_list = np.linspace(I_min, I_max, 1000)
    found = False

    for I0 in I_list:
        result, error = quad(integrand, -4 * sigma, 0, args=(I0, sigma), limit=100)
        eta = 1 - np.exp(-result)

        if abs(eta - eta_mac) < 1e-3:
            print(f"For tau={tau * 1e15:.2e} fs, I_mac: {I0 / 1e14:.3f} x 1e14 W/cm^2")
            results.append((tau, I0, eta))
            found = True
            break

    if not found:
        print(f"For tau={tau * 1e15:.2e} fs, I_mac not found within the initial range")

taus, I0s, etas = zip(*results)
plt.figure(figsize=(10, 6))
plt.plot(np.array(taus) * 1e15, np.array(I0s) / 1e14, '-', color='royalblue', label='$I_{mac}$')
plt.axhline(y=I_mic / 1e18, color='royalblue', linestyle='--', label='$I_{mic}$')
plt.fill_between(np.array(taus) * 1e15, np.array(I0s) / 1e14, I_mic / 1e18,
                 where=(np.array(I0s) / 1e14 >= I_mic / 1e18), color='lightblue', alpha=0.5)
plt.xlabel("Pulse length (fs)")
plt.ylabel("Intensity (x 1e14 W/cm²)")
plt.title('Phase matching window for q=23 in Ar @ 1030nm')
plt.xlim(tau_min * 1e15, tau_max * 1e15)
plt.legend()
plt.show()

"""
results_array = np.array(results)
output_file_path = 'PM_Window_q23_Ar_1030nm.txt'
header_str = "#Phase matching window for q=23 in Ar @ 1030nm, eta_mac=0.038, tolerance=1e-3\nPulse Length (s)\tI_mac (W/cm^2)\tEta"
np.savetxt(output_file_path, results_array, fmt='%e', delimiter='\t', header=header_str, comments='#')
print(f"Results saved to '{output_file_path}'")
"""
