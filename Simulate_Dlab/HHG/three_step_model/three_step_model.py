import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from matplotlib.cm import ScalarMappable

def function(u, t):
    return u[1], E_omega * np.cos(t * w) + E_2_omega * np.cos(t * 2 * w + phase)

def velocity(u, t):
    return E_omega * np.cos(t * w) + E_2_omega * np.cos(t * 2 * w + phase)

def field(t):
    return E_omega * np.cos(t*2*np.pi) + E_2_omega * np.cos(2*t*2*np.pi + phase)

def find_zero_crossing(x, y):
    zero_crossing_index = None
    for i in range(1, len(y) - 1):
        if np.sign(y[i]) != np.sign(y[i + 1]):
            zero_crossing_index = i
            break
    return zero_crossing_index

def find_local_maxima(x, y):
    local_maxima_indices = []
    for i in range(1, len(y) - 1):
        if np.isnan(y[i - 1]) or np.isnan(y[i]) or np.isnan(y[i + 1]):
            continue
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            local_maxima_indices.append(i)
    return local_maxima_indices, x[local_maxima_indices], y[local_maxima_indices]

me = 9.1e-31
h = 6.62607015e-34
c = 299792458
qe = 1.60217662e-19
lam = 1030e-9
Eq = h * c / lam

e = qe
E0 = 2e11  # Example value for E0, V/m
w = c / lam * 2 * np.pi  # Example value for w
laser_cycle = 1 / (c / lam)
Up = e ** 2 * E0 ** 2 / (4 * me * w ** 2)

total = 1
sh_ratio = 0
f_ratio = total - sh_ratio
phase = 0

E_omega = np.sqrt(f_ratio) * E0 * e / me
E_2_omega = np.sqrt(sh_ratio) * E0 * e / me
y0 = [0, 0]

ionization_time_steps = 1000
time_axis_steps = 1000

time_axis = np.zeros([time_axis_steps, ionization_time_steps]) * np.nan
trajectories = np.zeros([time_axis_steps, ionization_time_steps]) * np.nan
return_energies = np.zeros([ionization_time_steps, 1]) * np.nan
travel_time = np.zeros([ionization_time_steps, 1]) * np.nan
T_i = np.linspace(-0.10, 0.99, ionization_time_steps)
t_end = 1.5

for time_ind, ti in enumerate(T_i):
    ts = np.linspace(ti, t_end, time_axis_steps) * laser_cycle
    Ts = ts / 3.435e-15
    us = odeint(function, y0, ts)
    ys = us[:, 0]
    ind = find_zero_crossing(Ts, ys)
    if ind is not None:
        start_time = ti * laser_cycle
        return_time = ts[ind]
        new_t = np.linspace(start_time, return_time, time_axis_steps)
        v = odeint(velocity, 0, new_t)
        E_kin_return = 0.5 * me * v[-1] ** 2
        return_energies[time_ind] = E_kin_return
        travel_time[time_ind] = return_time

    time_axis[:ind, time_ind] = Ts[:ind]
    trajectories[:ind, time_ind] = ys[:ind] / 1.5e-8

colors = plt.cm.jet(np.linspace(0, 1, 500))
max_energy = np.nanmax(return_energies)
color_indices = (return_energies / max_energy) * 500
color_indices = color_indices.astype(int)

ax1 = plt.subplot()
for i in np.arange(0, time_axis.shape[1]):
    col = color_indices[i]
    if col < 0:
        ax1.plot(time_axis[:, i], np.nan_to_num(trajectories[:, i]), 'k', alpha=0)
    else:
        col = colors[color_indices[i] - 1]
        ax1.plot(time_axis[:, i], np.nan_to_num(trajectories[:, i]), color=col, alpha=0.8)
ax1.set_ylabel(r'Electron Displacement (arb.u) / Electric Field (in $E_0$)')

scalarmappable = ScalarMappable(cmap=plt.cm.get_cmap('jet'))

scalarmappable.set_array(return_energies/Up)
# Create the colorbar
t=np.linspace(0,1.5,1000)
cbar = plt.colorbar(scalarmappable, ax=ax1, orientation='vertical')
cbar.set_label('Electron Kinetic Energy (in $U_p$)')
ax1.plot(t, field(t)/(E0 * e / me), color='black')
ax1.set_xlim([0,1.5])
ax1.set_ylim([-1.5,1.5])
plt.figure(2)
plt.plot(travel_time/laser_cycle,return_energies / Up , color='blue')
plt.xlabel('Recombination Time (in Field cycle)')
plt.ylabel('Electron Kinetic Energy (in $U_p$)')
plt.ylim([-0.1,4])
plt.show()

