import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Constants
hbar = 1.054571800e-34  # Reduced Planck constant (Js)
e = 1.60217662e-19  # Electron charge (C)
c = 299792458  # Speed of light in vacuum (m/s)
a = 0.0072973525693  # Fine structure
me = 9.10938356e-31  # Electron mass (kg)
alpha_0 = 1.86e-40  # Polarizability of fundamental (C^2.m^2/J)
alpha_23 = -1.406e-40  # Polarizability of q=23 (C^2.m^2/J)
wavelength = 1030e-9  # Wavelength (m)
I_p = 15.7596 * e  # Ar Ip in J
omega = 2 * np.pi * c / wavelength  # Angular frequency (rad/s)
Ep = hbar * omega
tau_continuum = 0.6*3.47e-15 # time in continuum
sigma_abs = 1.98e-21  # Absorption cross section (m^2)
eta_mac = me * omega ** 2 / e ** 2 * (alpha_0 - alpha_23)  # Critical ionization
Eh = me * c ** 2 * a ** 2  # Hartree energy

data_path = '//fysfile01/atomhome$/me2120re/My Documents/Doctorat/Projets/DlabNumerical_data/Simulate_Dlab/Hyperbola/'
data_sample = 'Horizontal_branch/horizontal_branch_scan_large_I.mat'
data_hyperbola = io.loadmat(data_path + data_sample)

CE_matrix = np.array(data_hyperbola['CE_matrix'])
tau_values = np.array(data_hyperbola['tau_values'])
tau_values = tau_values[0,:] * 1e15
I0_values = np.array(data_hyperbola['I0_values'])
I0_values = I0_values[0,:]
pressure_values = np.array(data_hyperbola['pressure_values'])
pressure_values = pressure_values[0,:]

extent = [tau_values[0], tau_values[-1], I0_values[0], I0_values[-1]]

max_CE_matrix_all_L = np.max(CE_matrix, axis=2)
max_CE_all_L = I0_values[np.argmax(max_CE_matrix_all_L.T, axis=0)]
n=30

plt.figure(1)
plt.imshow(max_CE_matrix_all_L.T, aspect='auto', cmap='turbo', extent=extent, origin='lower',interpolation='None')
plt.colorbar()
#levels = np.linspace(np.min(max_CE_matrix_all_L), np.max(max_CE_matrix_all_L), 10)
#contour = plt.contour(max_CE_matrix_all_L.T, levels=levels, extent=extent, colors='black')
plt.xlabel('$\\tau$ (fs)')
plt.ylabel('$I_0$ (W/cm$^2$)')
plt.title('Maximum CE vs. Intensity and Tau, max for all P')
plt.xlim((20,200))
plt.ylim((1e14,4e14))
plt.clim(0,1.8e-6)
plt.legend()
plt.show()

plt.figure(2)

pressure_values_length = 7
max_CE_matrix = np.zeros((len(tau_values), len(I0_values)))
for i in range(len(tau_values)):
    for j in range(len(I0_values)):
        max_CE_matrix[i, j] = np.max(CE_matrix[i, j, pressure_values_length])
max_CE = I0_values[np.argmax(max_CE_matrix.T, axis=0)]
plt.imshow(CE_matrix[:,:,pressure_values_length].T, aspect='auto', cmap='turbo', extent=extent, origin='lower',interpolation='None')
plt.colorbar()
plt.xlabel('$\\tau$ (fs)')
plt.ylabel('$I_0$ (W/cm$^2$)')
plt.title(f'Maximum CE vs. Intensity and Tau, P={pressure_values[pressure_values_length]} mbar')
plt.xlim((20,200))
plt.ylim((1e14,2.5e14))


from matplotlib.animation import FuncAnimation

# Initialize your data and parameters
plt.figure(2)
pressure_values_length = 14  # Set the maximum value for pressure_values_length
fig, ax = plt.subplots()
max_CE_matrix = np.zeros((len(tau_values), len(I0_values)))
im = ax.imshow(CE_matrix[:,:,1].T, aspect='auto', cmap='turbo', extent=extent, origin='lower',interpolation='None')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Conversion efficiency')

def update(frame):
    ax.clear()
    im = ax.imshow(CE_matrix[:,:,frame].T, aspect='auto', cmap='turbo', extent=extent, origin='lower',interpolation='None')
    im.set_clim(vmin=6.3e-08, vmax=9.2e-07)
    levels = np.linspace(np.min(CE_matrix[:,:,frame].T), np.max(CE_matrix[:,:,frame].T), 10)
    contour = ax.contour(CE_matrix[:,:,frame].T, levels=levels, extent=extent, colors='black')
    ax.set_xlabel('$\\tau$ (fs)')
    ax.set_ylabel('$I_0$ (W/cm$^2$)')
    ax.set_title(f'CE vs. Intensity and Tau, P={pressure_values[frame]:.3f} mbar')
    ax.set_xlim((20,200))
    ax.set_ylim((1e14,2.5e14))


#ani = FuncAnimation(fig, update, frames=np.arange(0, pressure_values_length), interval=200)
#ani.save('horizontal_branch_contours.gif', writer='pillow', fps=1)