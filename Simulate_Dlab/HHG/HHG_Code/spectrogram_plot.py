import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os
import re
from scipy.interpolate import interp1d
"""
# Importation des données
data_sample = 'Results_2024-04-16_10-10-32'
data = io.loadmat(data_sample)

spectrogram_data = data['spectrogram']
theta = data['theta']
freq = data['freq']
w0 = data['w0']
ohm = data['ohm']
center_freq = data['center_freq']
shift = data['shift']
df = data['df']
q = data['q']
CE = data['CE']
CE_central = data['CE_central']
param = data['param']
print(param)
# Constantes
h = 6.62607015e-34
c = 299792458
lambda_0 = 1030e-9

y_axis = 1e3 * 2 * c / (ohm * w0) * np.vstack([np.flipud(theta), -theta])
spectrogram = np.vstack([np.flipud(spectrogram_data.T), spectrogram_data.T])

"""
def plot_spectrogram(spectrogram, y_axis, x_axis_type='Harmonic', from_='13', to_='32'):
    E_0 = h * c / lambda_0
    E_q = h * freq

    if x_axis_type == 'Harmonic':
        x_axis = E_q / E_0
        x_lim = [float(from_), float(to_)]
        x_label = 'Harmonic order $q$'
    elif x_axis_type == 'Energy':
        x_axis = E_q / E_0 * 1.2
        x_lim = [float(from_), float(to_)]
        x_label = 'Energy (eV)'
    else:
        print("Select an x-axis, 'Harmonic' or 'Energy'")

    plt.figure()
    extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
    plt.imshow(spectrogram, extent=extent, cmap='turbo', aspect='auto')
    plt.ylabel('Angle from axis (mrad)')
    plt.xlabel(x_label)
    plt.title('Spectrogram')
    plt.ylim([-6, 6])
    plt.xlim(x_lim)
    plt.colorbar()
    plt.show()


h = 6.62607015e-34
c = 299792458
lambda_0 = 1030e-9
def plot_integrated_spectrogram(spectrogram, x_axis_type='Harmonic', from_='13', to_='33'):
    E_0 = h * c / lambda_0
    E_q = h * freq

    if x_axis_type == 'Harmonic':
        x_axis = E_q / E_0
        x_lim = [float(from_), float(to_)]
        x_label = 'Harmonic order $q$'
    elif x_axis_type == 'Energy':
        x_axis = E_q / E_0 * 1.2
        x_lim = [float(from_), float(to_)]
        x_label = 'Energy (eV)'
    else:
        print("Select an x-axis, 'Harmonic' or 'Energy'")

    sum_spec = np.sum(spectrogram, axis=0)

    plt.figure()
    plt.plot(x_axis, sum_spec)
    plt.title('Integrated spectrogram')
    plt.xlim(x_lim)
    plt.xlabel(x_label)
    plt.ylabel('Counts (arb.u)')

    # Find the indices corresponding to the x-axis limits
    a = np.argmin(np.abs(x_axis - x_lim[0]))
    b = np.argmin(np.abs(x_axis - x_lim[1]))

    return np.sum(sum_spec[a:b + 1])


#plot_spectrogram(spectrogram, y_axis, x_axis_type='Harmonic')
#plot_integrated_spectrogram(spectrogram, x_axis_type='Energy', from_='20', to_='30')
folder_path = 'Results_2024-04-16_12-32-00'
desired_I_value = 1.77
sum_spec_list = []
desired_num_points = 3000
for file_name in os.listdir(folder_path):
    if file_name.endswith('.mat'):
        match_I = re.search(r'_I_([\d.]+)_', file_name)
        match_tau = re.search(r'Spectrogram_tau_([\d.]+)_', file_name)
        if match_I and match_tau:
            I_value = float(match_I.group(1))
            tau_value = float(match_tau.group(1))

            if I_value == desired_I_value:
                print(tau_value)
                file_path = os.path.join(folder_path, file_name)
                data = io.loadmat(file_path)

                spectrogram_data = data['spectrogram']
                freq = data['freq']
                ohm = data['ohm']
                theta = data['theta']
                w0 = data['w0']

                spectrogram = np.vstack([np.flipud(spectrogram_data.T), spectrogram_data.T])
                y_axis = 1e3 * 2 * c / (ohm * w0) * np.vstack([np.flipud(theta), -theta])

                sum_spec = plot_integrated_spectrogram(spectrogram, x_axis_type='Harmonic', from_='18', to_='24')
                sum_spec_list.append(sum_spec)
                print(sum_spec.shape)

plt.imshow(sum_spec_list,aspect='auto')
plt.show()
