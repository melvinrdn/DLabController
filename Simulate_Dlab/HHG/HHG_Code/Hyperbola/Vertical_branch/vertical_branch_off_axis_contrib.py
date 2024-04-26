##
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.interpolate import interp1d
from scipy.integrate import quad
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

# Constants
c = 299792458  # Speed of light in vacuum (m/s)
eps0 = 8.854187817e-12  # Vacuum permittivity (C/V/m)
k_b = 1.380e-23  # Boltzmann constant (J/K)
hbar = 1.054571800e-34  # Reduced Planck constant (Js)
me = 9.10938356e-31  # Electron mass (kg)
a = 0.0072973525693  # Fine structure
e = 1.60217662e-19  # Electron charge (C)
wavelength = 1030e-9  # Wavelength (m)
I_p = 15.7596 * e  # Ar Ip in J
omega = 2 * np.pi * c / wavelength  # Angular frequency (rad/s)
sigma_abs = 1.98e-21  # Absorption cross section (m^2)
alpha_0 = 1.86e-40  # Polarizability of fundamental (C^2.m^2/J)
alpha_23 = -1.406e-40  # Polarizability of q=23 (C^2.m^2/J)
eta_mac = me * omega ** 2 / e ** 2 * (alpha_0 - alpha_23)
q = 23
I_mic = me * omega ** 2 / (3.17 * 2 * np.pi * a) * (q * omega - I_p / hbar)

data_path = '//fysfile01/atomhome$/me2120re/My Documents/Doctorat/Projets/Pulse_duration/Eq_and_E1/fine_scan/'
data_sample = 'fine_Eq_and_E1_tau_30_I_2.21.mat'
data = io.loadmat(data_path + data_sample)

CE = data['CE']
Eq = data['Eq']
nf_dq = data['nf_dq']
temp_dq = data['temp_dq']
E1 = data['E1']
nf_fundamental = data['nf_fundamental']
temp_fundamental = data['temp_fundamental']
param = data['param']

tmax = 1.6  # Range of temporal profile to include
tsteps = 300  # Number of steps in t
zmin = -0.015  # Minimum z in units of zR
zmax = 0.015  # Maximum z in units of zR
zsteps = 400  # Number of mesh steps in z
rmax = 4  # Maximum r in units of w0
rsteps = 500  # Number of mesh steps in r

z = np.linspace(zmin, zmax, zsteps)
t = np.linspace(-tmax, tmax, tsteps)
r = np.linspace(0, rmax, rsteps)

tau_str, I_str = data_sample.split('_tau_')[1].split('_I_')
tau = float(tau_str) * 1e-15
sig = tau / (2 * np.sqrt(np.log(2)))
I0 = float(I_str[:-4]) * 1e14

print(f'Pulse duration: {tau * 1e15:.2f} fs')
print(f'Intensity: {I0 * 1e-14:.2f} x1e14 W/cm2')

Imac = 1.77e14
##

file_path = '../../ressources/ionization-rate-tdse.txt'
data = np.genfromtxt(file_path, dtype=float, autostrip=True, skip_header=2)
intensity, ionization_rate = data[:, 0], data[:, 1]
G = interp1d(intensity, ionization_rate, kind='cubic', fill_value="extrapolate")


def I(t, I_value):
    return I_value * np.exp(-t ** 2 / (sig ** 2))


def calculate_results(I_value, t_borne):
    T = np.linspace(-tmax * sig, t_borne, 200)
    intensity_values = I(T, I_value)
    ionization_rates = G(intensity_values)
    integral_results = np.trapz(ionization_rates, T, axis=0)
    return 1 - np.exp(-integral_results)


eta = []
t_list = sig * t
for t_borne in t_list:
    eta.append(calculate_results(I0, t_borne))

plt.figure(1)
fig, ax1 = plt.subplots()

ax1.plot(t * sig * 1e15, temp_fundamental / np.max(temp_fundamental), color='black', alpha=0.3)
ax1.plot(t * sig * 1e15, temp_dq, '-o', markersize='2', label=f'I={I0 / Imac:.2f}xImac')
integrale_temp_dq = np.sum(temp_dq)
ax1.set_xlabel('Time (fs)')
ax1.set_ylabel('Power (Normalized)')
ax1.legend()
ax2 = ax1.twinx()
ax2.plot(t * sig * 1e15, eta, '--', color='blue', alpha=0.3)
ax2.axhline(y=eta_mac, color='blue', linestyle='-', alpha=0.3)
ax2.set_ylabel('$\eta$', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylim(0, 0.60)

plt.title(f'Normalized temporal shape of harmonic 23 vs fundamental, sum = {integrale_temp_dq}')
plt.tight_layout()
plt.show()

##
plt.figure(2)
plt.plot(r, nf_dq, '-o', markersize='2', label=f'I={I0 / Imac:.2f}xImac')
integrale_nf_dq = np.sum(nf_dq)
plt.title(f'Nearfield of harmonic 23, sum={integrale_nf_dq}')
plt.xlabel('r (w_0)')
plt.ylabel('Time-int.flux (Normalized)')
plt.legend()
plt.xlim((0, 2))
plt.tight_layout()
plt.show()

##
plt.figure(3)
plt.plot(r, nf_fundamental[0,:], '-o', markersize='2', label=f'I={I0 / Imac:.2f}xImac')
integrale_nf_fundamental = np.sum(nf_fundamental)
plt.title(f'Nearfield of fundamental, sum={integrale_nf_fundamental}')
plt.xlabel('r (w_0)')
plt.ylabel('Time-int.flux (Normalized)')
plt.legend()
plt.xlim((0, 2))
plt.tight_layout()
plt.show()

##
I_overall = np.abs(Eq[:, :, :]) ** 2
max_intensity = np.max(eps0 * c / 2 * 1e-4 * np.vstack([np.flipud(I_overall.T), I_overall.T]))
min_intensity = np.min(np.abs(Eq[:,:,:]) ** 2)
"""
t_slice = 150
real_t = sig*t[t_slice]*1e15
I_at_slice = np.abs(Eq[:, :, t_slice]) ** 2
I_at_slice_full = eps0 * c / 2 * 1e-4 * np.vstack([np.flipud(I_at_slice.T), I_at_slice.T])
r_plot = np.concatenate((np.flipud(r), -r))


sum_over_z = np.sum(I_at_slice_full, axis=1)

# Plotting
plt.figure(4)
plt.plot(r_plot, sum_over_z, '-o', markersize='2', label=f't={real_t}')
plt.xlim((0, 2))
plt.xlabel('r ($w_0$)')
plt.ylabel('Sum over z')
plt.title(f'Sum over z of intensity of harmonic 23 at I={I0 / Imac:.2f}xImac')
plt.legend()
plt.show()
"""


fig1, ax = plt.subplots()

t_slice = 180
real_t = sig*t[t_slice]*1e15

I_at_slice = np.abs(Eq[:, :, t_slice]) ** 2
I_at_slice_full = eps0 * c / 2 * 1e-4 * np.vstack([np.flipud(I_at_slice.T), I_at_slice.T])

r_plot = np.concatenate((np.flipud(r), -r))
im = ax.imshow(I_at_slice_full, cmap='turbo', extent=[z[0], z[-1], r_plot[0], r_plot[-1]], aspect='auto',vmax=max_intensity,vmin=min_intensity)
cbar = fig1.colorbar(im, ax=ax, label='Harmonic intensity')

plt.title(f'Intensity of harmonic 23 at t={real_t:.2f}fs')
plt.show()

def update(index):
    I_at_slice = np.abs(Eq[:, :, index]) ** 2
    I_at_slice_full = eps0 * c / 2 * 1e-4 * np.vstack([np.flipud(I_at_slice.T), I_at_slice.T])
    im.set_data(I_at_slice_full)
    ax.set_title(f'Intensity of harmonic 23 in time slice {index}/300')
    return [im]

#anim = FuncAnimation(fig1, update, frames=len(t_list), blit=True)
#anim.save(f'Evolution_dq_I={I0 / Imac:.2f}xImac.gif', writer=PillowWriter(fps=10))
#plt.close()
