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
data_sample = 'Vertical_branch/vertical_branch_scan.mat'
data_hyperbola = io.loadmat(data_path + data_sample)

data_path_imac = '//fysfile01/atomhome$/me2120re/My Documents/Doctorat/Projets/DlabNumerical_data/Simulate_Dlab/ionization_tables/'
data_sample_imac = 'PM_Window_q23_Ar_1030nm.txt'
data_imac = np.genfromtxt(data_path_imac + data_sample_imac, dtype=float, autostrip=True, skip_header=2)

CE_matrix = np.array(data_hyperbola['CE_matrix'])
tau_values = np.array(data_hyperbola['tau_values'])
tau_values = tau_values[0,:] * 1e15
I0_values = np.array(data_hyperbola['I0_values'])
I0_values = I0_values[0,:]
medium_length_values = np.array(data_hyperbola['medium_length_values'])
medium_length_values = medium_length_values[0,:]

extent = [tau_values[0], tau_values[-1], I0_values[0], I0_values[-1]]

tau_fit = np.linspace(10, 200, 10000)
x_imac = data_imac[:, 0] * 1e15 #fs
y_imac = data_imac[:, 1] * 1e-14 #1e14 W/cm2

def model_robert(x, a):
    return a / x ** (0.2)


max_CE_matrix_all_L = np.max(CE_matrix, axis=2)
max_CE_all_L = I0_values[np.argmax(max_CE_matrix_all_L.T, axis=0)]
n=30

popt_fit_1, pcov_fit_1 = curve_fit(model_robert, x_imac, y_imac,p0=1)
max_CE_all_L_fit_1 = model_robert(tau_fit, *popt_fit_1)


plt.figure(1)
plt.imshow(max_CE_matrix_all_L.T, aspect='auto', cmap='turbo', extent=extent, origin='lower',interpolation='None')
plt.plot(tau_values[:n], max_CE_all_L[:n], marker='o',color='black', linestyle='-', linewidth=2,label='Max CE')
plt.plot(tau_fit, max_CE_all_L_fit_1*1e14, label='Imac', color='blue', linestyle='-')
plt.plot(tau_fit, max_CE_all_L_fit_1*1e14*1.25, label='Imac x1.25', color='blue', linestyle='--')
plt.colorbar()
plt.clim(1e-7,1.8e-6)
plt.xlabel('$\\tau$ (fs)')
plt.ylabel('$I_0$ (W/cm$^2$)')
plt.title('CE vs. Intensity and Tau, max for all L')
plt.xlim((20,200))
plt.ylim((1e14,4e14))
plt.clim(0,1.8e-6)
plt.legend()

plt.figure(2)
index_medium_length = 1
max_CE_matrix = np.zeros((len(tau_values), len(I0_values)))
for i in range(len(tau_values)):
    for j in range(len(I0_values)):
        max_CE_matrix[i, j] = np.max(CE_matrix[i, j, index_medium_length])
max_CE = I0_values[np.argmax(max_CE_matrix.T, axis=0)]
plt.imshow(CE_matrix[:,:,index_medium_length].T, aspect='auto', cmap='turbo', extent=extent, origin='lower',interpolation='None')
plt.plot(tau_values[:n], max_CE[:n], marker='o',color='black', linestyle='-', linewidth=2,label='Max CE')
plt.plot(tau_fit, max_CE_all_L_fit_1*1e14, label='Imac', color='blue', linestyle='-')
plt.plot(tau_fit, max_CE_all_L_fit_1*1e14*1.25, label='Imac x1.25', color='blue', linestyle='--')
plt.colorbar()
plt.xlabel('$\\tau$ (fs)')
plt.ylabel('$I_0$ (W/cm$^2$)')
plt.title(f'Maximum CE vs. Intensity and Tau, L={medium_length_values[index_medium_length]}')
plt.xlim((20,200))
plt.ylim((1e14,2.5e14))
plt.legend()
plt.show()


