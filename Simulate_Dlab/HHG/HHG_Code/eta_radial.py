import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad,simps
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
file_path = 'ressources/ionization-rate-tdse.txt'
data = np.genfromtxt(file_path, dtype=float, autostrip=True, skip_header=2)
intensity, ionization_rate = data[:, 0], data[:, 1]
G = interp1d(intensity, ionization_rate, kind='cubic', fill_value="extrapolate")

# Parameters
w0 = 30e-6
tau = 30e-15
sigma = tau / (2 * np.sqrt(np.log(2)))
tmax= 3*sigma
lambda_0 = 1030e-9
zR = np.pi * w0 ** 2 / lambda_0
omega = 2 * np.pi * c / lambda_0 # Angular frequency (rad/s)
alpha_0 = 1.86e-40  # Polarizability of fundamental (C^2.m^2/J)
alpha_23 = -1.406e-40  # Polarizability of q=23 (C^2.m^2/J)
T = 288
pressure = 1277
sigma_abs = 1.98e-21  # Absorption cross section (m^2)
density = pressure *1e2/(k_b*T)

q = 23
I_p = 15.7596 * e  # Ar Ip in J
Imic = me * omega ** 2 / (3.17 * 2 * np.pi * a) * (q * omega - I_p / hbar)

def I(t, R, I_value, z):
    wz = w0 * np.sqrt(1 + (z / zR) ** 2)
    return I_value * np.exp(-t ** 2 / (sigma ** 2)) * np.exp(-2 * R ** 2 / (wz ** 2)) * (w0 / wz) ** 2

def calculate_results(I_value, z, R, t_borne):
    T = np.linspace(-tmax, t_borne, 200)
    intensity_values = I(T[:, None, None], R, I_value, z)
    ionization_rates = G(intensity_values)
    integral_results = np.trapz(ionization_rates, T, axis=0)
    return 1 - np.exp(-integral_results)


eta_mic = 1.55e-6
eta_mac = me * omega ** 2 / e ** 2 * (alpha_0 - alpha_23)

rmax = 1.2*w0
Imac = 1.77e14

x = np.linspace(-rmax, rmax, 128)
y = np.linspace(-rmax, rmax, 128)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X ** 2 + Y ** 2)

#intensity_multipliers = np.linspace(1,2,10)
#intensity_multipliers = [1,1.25,1.5,1.75,2]
from matplotlib.animation import PillowWriter, FuncAnimation

#radii_eta_mac = np.zeros(len(intensity_multipliers))
#radii_eta_mic = np.zeros(len(intensity_multipliers))
t_list = [-2*sigma,-1*sigma,-0.45*sigma,0]
#t_list = np.linspace(-3*sigma,0,30)
eta_r_values = np.zeros((len(t_list), *R.shape))

intensity = 1.77e14
for index, t_borne in enumerate(t_list):
    eta_r = calculate_results(intensity, 0.00*zR, R, t_borne)
    eta_r_values[index] = eta_r

fig, ax = plt.subplots()


im = ax.imshow(eta_r_values[3]*100, extent=(x.min()/w0, x.max()/w0, y.min()/w0, y.max()/w0), origin='lower', cmap='plasma')
cbar = fig.colorbar(im, ax=ax, label='$\eta_{fe}$')
#cbar.mappable.set_clim(eta_r_values.min()*100, eta_r_values.max()*100)
CS_mac = ax.contour(X/w0, Y/w0, eta_r_values[3]*100, levels=[eta_mac*100], colors='black', linewidths=1)
plt.title(f'Evolution of the ionization at t={t_list[3]/tau:.3f} tau')
plt.show()

def update(index):
    im.set_data(eta_r_values[index]*100)
    ax.set_title(f'Evolution of the ionization at t={t_list[index]/tau:.3f} tau')
    return [im]


#anim = FuncAnimation(fig, update, frames=len(t_list), blit=True)


#anim.save('ionization_evolution.gif', writer=PillowWriter(fps=2))

#plt.close()


"""
plt.figure(2)
index = 1
plt.imshow(eta_r_values[index]*100, extent=(x.min()/w0, x.max()/w0, y.min()/w0, y.max()/w0), origin='lower', cmap='plasma')
plt.colorbar(label='$\eta_{fe}$')
CS_mac1 = plt.contour(X/w0, Y/w0, eta_r_values[index]*100, levels=[eta_mac*100], colors='black', linewidths=1)
plt.clabel(CS_mac1, inline=True, fontsize=10, fmt=f'eta_mac = {eta_mac*100:.4f}')
plt.xlabel('x position ($w_0$)')
plt.ylabel('y position ($w_0$)')
plt.title(f'Evolution of the ionization at t={t_list[index]/tau:.3f} tau')
plt.show()
"""
"""
n_values = np.zeros((len(intensity_multipliers), *R.shape))
eta_r_values = np.zeros((len(intensity_multipliers), *R.shape))

for index, multiplier in enumerate(intensity_multipliers):
    print(f'Current I: {multiplier}')
    current_Imac = multiplier * Imac
    eta_r = calculate_results(current_Imac, 0, R)
    eta_r_values[index] = eta_r

    N_e = density * eta_r
    omega_p = np.sqrt(e ** 2 * N_e / (eps0 * me))
    n = np.sqrt(1 - (omega_p / omega) ** 2)

    n_values[index] = n

    diff_eta_mic = np.abs(eta_r - eta_mic)
    min_diff_eta_mic_idx = np.unravel_index(np.argmin(diff_eta_mic, axis=None), diff_eta_mic.shape)
    radius_at_eta_mic = R[min_diff_eta_mic_idx]

    diff_eta_mac = np.abs(eta_r - eta_mac)
    min_diff_eta_mac_idx = np.unravel_index(np.argmin(diff_eta_mac, axis=None), diff_eta_mac.shape)
    radius_at_eta_mac = R[min_diff_eta_mac_idx]

    radii_eta_mac[index] = radius_at_eta_mac/w0
    radii_eta_mic[index] = radius_at_eta_mic/w0
"""
"""
plt.figure(1)
plt.plot(intensity_multipliers, radii_eta_mic-radii_eta_mac,color='black', linestyle='--', label='R_eta_mic-R_eta_mac')
plt.xlabel('Intensity (Imac)')
plt.ylabel('Radius (w0)')
plt.grid(True)
plt.legend()
plt.tight_layout()


plt.figure(2)
index = -1
plt.imshow(eta_r_values[index]*100, extent=(x.min()/w0, x.max()/w0, y.min()/w0, y.max()/w0), origin='lower', cmap='plasma')
plt.colorbar(label='$\eta_{fe}$')
CS_mic = plt.contour(X/w0, Y/w0, eta_r_values[index]*100, levels=[eta_mic*100], colors='red', linewidths=2)
plt.clabel(CS_mic, inline=True, fontsize=10, fmt=f'eta_mic = {eta_mic*100:.4f}')
CS_mac1 = plt.contour(X/w0, Y/w0, eta_r_values[index]*100, levels=[eta_mac*100], colors='black', linewidths=1)
plt.clabel(CS_mac1, inline=True, fontsize=10, fmt=f'eta_mac = {eta_mac*100:.4f}')
plt.xlabel('x position ($w_0$)')
plt.ylabel('y position ($w_0$)')
plt.show()

plt.figure(3)
plt.imshow(I(0, R, intensity_multipliers[index]*Imac, 0), extent=(x.min()/w0, x.max()/w0, y.min()/w0, y.max()/w0), origin='lower', cmap='turbo')
plt.colorbar(label='Intensity')
plt.clim(0,2*1.77e14)
CS_mic = plt.contour(X/w0, Y/w0, eta_r_values[index], levels=[eta_mic], colors='red', linewidths=2)
plt.clabel(CS_mic, inline=True, fontsize=10, fmt=f'eta_mic = {eta_mic*100:.4f}')

CS_mac1 = plt.contour(X/w0, Y/w0, eta_r_values[index]*100, levels=[eta_mac*100], colors='black', linewidths=1)
plt.clabel(CS_mac1, inline=True, fontsize=10, fmt=f'eta_mac = {eta_mac*100:.4f}')

plt.xlabel('x position ($w_0$)')
plt.ylabel('y position ($w_0$)')

"""

