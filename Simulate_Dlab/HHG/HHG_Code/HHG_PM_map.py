##
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
q = 23  # Harmonic order
p0 = 1277  # Pression (mbar)
rho = p0 * 1e2 / 1.380e-23 / 288  # Density
n_0 = np.sqrt(1 + rho * alpha_0 / eps0)  # Index of refraction
k = 2 * np.pi * n_0 / wavelength  # Wavenumber in the gas

# Macroscopic parameter
eta_mac = me * omega ** 2 / e ** 2 * (alpha_0 - alpha_23)
Imac = 1.77e14  # 1.77e14 for 30fs

# Laser parameters
tau = 30e-15  # Pulse length (s)
sig = tau / (2 * np.sqrt(np.log(2)))
w0 = 30e-6  # Waist (m)
I0 = 1.25 * Imac
zR = np.pi * w0 ** 2 / wavelength
L = 0.020 * zR  # Medium length

# Simulation box
tmax = 2 * sig  # Range of temporal profile to include
tsteps = 300  # Number of steps in t
zmin = -0.016 * zR  # Minimum z
zmax = 0.016 * zR  # Maximum z
zsteps = 400  # Number of mesh steps in z
rmax = 4 * w0  # Maximum r
rsteps = 500  # Number of mesh steps in r

a_l = -0.16 * a * wavelength ** 3 / (me * c ** 3)
y_s = 0.22 * c * me / (a * wavelength)
y_l = -0.19 * c * me / (a * wavelength)

print(f'Pulse duration: {tau * 1e15:.2f} fs')
print(f'Intensity: {I0 * 1e-14:.2f} x1e14 W/cm2')

file_path = 'ressources/ionization-rate-tdse.txt'
data = np.genfromtxt(file_path, dtype=float, autostrip=True, skip_header=2)
intensity, ionization_rate = data[:, 0], data[:, 1]
G = interp1d(intensity, ionization_rate, kind='cubic', fill_value="extrapolate")


def density1(z, L, p0):
    zcut0 = 1e-3
    zcut1 = zcut0

    rho = p0 * 1e2 / 1.380e-23 / 288

    term1 = 1/4 * (1 + np.tanh(2 * (L/2 + z) / zcut0))
    term2 = (1 + np.tanh(2 * (L/2 - z) / zcut1))
    term3 = ((1/2 + z/L) * rho + (1/2 - z/L) * rho)
    rho_gradient = term1 * term2 * term3


    return rho_gradient

def density(z, L, p0):
    z_range_min = -L/2
    z_range_max = L/2
    if z >= z_range_min and z <= z_range_max:
        return p0 * 1e2 / 1.380e-23 / 288  # Density
    else:
        return 0




def intensity(I0, r, z, t):
    wz = w0 * np.sqrt(1 + (z / zR) ** 2)
    return I0 * np.exp(-t ** 2 / (sig ** 2)) * np.exp(-2 * r ** 2 / (wz ** 2)) * (w0 / wz) ** 2


def eta_calculation(I0, r, z, t):
    T = np.linspace(-tmax, t, 200)
    intensity_values = intensity(I0, r, z, T[:, None, None])
    ionization_rates = G(intensity_values)
    integral_results = np.trapz(ionization_rates, T, axis=0)
    return 1 - np.exp(-integral_results)


t_list = np.linspace(-tmax, tmax, 30)
z_list = np.linspace(zmin, zmax, 100)
r_list = np.linspace(-rmax, rmax, 100)


eta_values = np.zeros((len(r_list), len(z_list), len(t_list)))
dk_matrix = np.zeros((len(r_list), len(z_list), len(t_list)))
print('---')
radial = 0
for t_index, t_borne in enumerate(t_list):
    for z_index, z_position in enumerate(z_list):
        for r_index, r_position in enumerate(r_list):
            print(f'z_position {z_index + 1}/{len(z_list)} : {z_position / zR:.3f} zR,',
                  f' r_position {r_index + 1}/{len(r_list)} : {r_position / w0:.3f} w0'
                  f' time_slice {t_index + 1}/{len(t_list)} : {t_borne/ sig:.3f} sig')

            eta_val = eta_calculation(I0, r_position, z_position, t_borne)
            eta_values[r_index, z_index, t_index] = eta_val

            w_z = w0 * np.sqrt(1 + (z_position / zR) ** 2)

            beta_l = a_l * intensity(I0, r_position, z_position, t_borne) - y_l / intensity(I0, r_position, z_position,
                                                                                            t_borne) * (
                             q * omega - I_p / hbar) ** 2
            beta_s = -y_s / intensity(I0, r_position, z_position, t_borne) * (q * omega - I_p / hbar) ** 2

            dk_at = q * omega * density(z_position, L, p0) / (2 * eps0 * c) * (alpha_0 - alpha_23) * (1 - eta_val)

            dk_fe = - q * omega * density(z_position, L, p0) / (2 * eps0 * c) * eta_val * e ** 2 / me * (
                    1 / omega ** 2 - 1 / (q ** 2 * omega ** 2))

            dk_s = -beta_s * ((4 * r_position / w_z ** 2)*radial  + 2 * z_position / (z_position ** 2 + zR ** 2) * (
                    1 + radial *(w0 / w_z) ** 2))
            dk_l = -beta_l * ((4 * r_position / w_z ** 2)*radial  + 2 * z_position / (z_position ** 2 + zR ** 2) * (
                    1 + radial *(w0 / w_z) ** 2))
            dk_i = dk_s + dk_l

            dk_foc = q * (-zR / (z_position ** 2 + zR ** 2) - radial *z_position / (z_position ** 2 + zR ** 2) * (
                    k * r_position ** 2 / 2 + k * r_position))

            dk = dk_at + dk_fe + dk_i + dk_foc
            dk_matrix[r_index, z_index, t_index] = abs(dk)
print('---')

index_plot = len(t_list) // 2

fig, ax = plt.subplots()
im = plt.imshow(dk_matrix[:, :, index_plot],
                extent=[z_list[0] / zR, z_list[-1] / zR, -r_list[-1] / w0, r_list[-1] / w0],
                aspect='auto',
                cmap='binary', vmin=np.min(dk_matrix), vmax=10000)
cbar = fig.colorbar(im, ax=ax, label='$\Delta k$')
plt.xlabel('z-axis ($z_R$)')
plt.ylabel('r-axis ($w_0$)')
plt.title(f'Evolution $\Delta k$ at t={t_list[index_plot]  / sig:.3f} sig')


def update(index):
    im.set_data(dk_matrix[:, :, index])
    ax.set_title(f'Evolution $\Delta k$ at t={t_list[index] / sig:.3f} sig')
    return [im]


#anim = FuncAnimation(fig, update, frames=len(t_list), blit=True)
#anim.save('dk_evolution.gif', writer=PillowWriter(fps=10))
plt.show()
