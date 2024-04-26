import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

def find_zero_crossing(x, y):
    zero_crossing_index = None
    for i in range(1, len(y) - 1):
        if np.sign(y[i]) != np.sign(y[i + 1]):
            zero_crossing_index = i
            break
    return zero_crossing_index

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
ratio_min = 0
ratio_max = 0.5

phase_values = np.linspace(-np.pi, np.pi, 10)
ratios = np.linspace(ratio_min, ratio_max, 2)
phase_grid, ratio_grid = np.meshgrid(phase_values, ratios)
max_return_energies_2D_first = np.zeros((len(ratios), len(phase_values)))
max_return_energies_2D_second = np.zeros((len(ratios), len(phase_values)))

for ratio_idx, sh_ratio in enumerate(ratios):
    print(f'SH Ratio :', sh_ratio)
    f_ratio = total - sh_ratio

    E_omega = np.sqrt(f_ratio) * E0 * e / me
    E_2_omega = np.sqrt(sh_ratio) * E0 * e / me


    def function(u, t):
        return u[1], E_omega * np.cos(t * w) + E_2_omega * np.cos(t * 2 * w + phase)


    def velocity(u, t):
        return E_omega * np.cos(t * w) + E_2_omega * np.cos(t * 2 * w + phase)


    max_return_energies_first = []
    max_return_energies_second = []

    for phase_idx, phase in enumerate(phase_values):
        print(f'Phase :', phase)
        y0 = [0, 0]

        ionization_time_steps = 150
        time_axis_steps = 500

        # First half-cycle
        return_energies_first = np.zeros([ionization_time_steps, 1]) * np.nan
        travel_time_first = np.zeros([ionization_time_steps, 1]) * np.nan
        T_i_first = np.linspace(-0.05, 0.6, ionization_time_steps)
        t_end_first = 1
        for time_ind, ti in enumerate(T_i_first):
            ts = np.linspace(ti, t_end_first, time_axis_steps) * laser_cycle
            Ts = ts / 3.435e-15
            us = odeint(function, y0, ts)
            ys = us[:, 0]
            ind = find_zero_crossing(Ts, ys)
            if ind is not None:
                start_time = ti * laser_cycle
                return_time_first = ts[ind]
                new_t = np.linspace(start_time, return_time_first, time_axis_steps)
                v = odeint(velocity, 0, new_t)
                E_kin_return_first = 0.5 * me * v[-1] ** 2
                return_energies_first[time_ind] = E_kin_return_first
                travel_time_first[time_ind] = return_time_first
        max_return_energy_first = np.nanmax(return_energies_first)
        max_return_energies_first.append(max_return_energy_first)
        max_return_energies_2D_first[ratio_idx, phase_idx] = max_return_energy_first

        # Second half-cycle
        return_energies_second = np.zeros([ionization_time_steps, 1]) * np.nan
        travel_time_second = np.zeros([ionization_time_steps, 1]) * np.nan
        T_i_second = np.linspace(0.4, 1.5, ionization_time_steps)
        t_end_second = 1.5
        for time_ind, ti in enumerate(T_i_second):
            ts = np.linspace(ti, t_end_second, time_axis_steps) * laser_cycle
            Ts = ts / 3.435e-15
            us = odeint(function, y0, ts)
            ys = us[:, 0]
            ind = find_zero_crossing(Ts, ys)
            if ind is not None:
                start_time = ti * laser_cycle
                return_time_second = ts[ind]
                new_t = np.linspace(start_time, return_time_second, time_axis_steps)
                v = odeint(velocity, 0, new_t)
                E_kin_return_second = 0.5 * me * v[-1] ** 2
                return_energies_second[time_ind] = E_kin_return_second
                travel_time_second[time_ind] = return_time_second
        max_return_energy_second = np.nanmax(return_energies_second)
        max_return_energies_second.append(max_return_energy_second)
        max_return_energies_2D_second[ratio_idx, phase_idx] = max_return_energy_second

plt.figure(1)
plt.imshow(max_return_energies_2D_first/Up, extent=[-np.pi, np.pi, ratio_min, ratio_max], cmap='plasma', origin='lower',
           aspect='auto')
plt.colorbar(label='Electron Maximum Kinetic Energy (in $U_p$)')
plt.xlabel(r'Two-Color Phase $\phi$ (rad)')
plt.ylabel('SH Intensity Fraction')
#plt.savefig('./figures/first_half_cycle.png')

plt.figure(2)
plt.imshow(max_return_energies_2D_second/Up, extent=[-np.pi, np.pi, ratio_min, ratio_max], cmap='plasma', origin='lower',
           aspect='auto')
plt.colorbar(label='Electron Maximum Kinetic Energy (in $U_p$)')
plt.xlabel(r'Two-Color Phase $\phi$ (rad)')
plt.ylabel('SH Intensity Fraction')
#plt.savefig('./figures/second_half_cycle.png')

"""
plt.figure(3)
diff_cuttoffs = abs(max_return_energies_2D_first - max_return_energies_2D_second)/Up
plt.imshow(diff_cuttoffs,
           extent=[-np.pi, np.pi, ratio_min, ratio_max], cmap='plasma', origin='lower', aspect='auto')
plt.colorbar(label='Energy Difference (in $U_p$)')
plt.xlabel(r'Two-Color Phase $\phi$ (rad)')
plt.ylabel('SH Intensity Fraction')
#plt.savefig('./figures/diff_first_second.png')


plt.figure(4)
sum_diff_cutoffs = np.sum(diff_cuttoffs,axis=1)
plt.plot(sum_diff_cutoffs, ratios)
plt.ylim([0,0.5])
plt.xlim([0,np.max(sum_diff_cutoffs)+10])
plt.ylabel('SH Intensity Fraction')
plt.xlabel('Counts (arb.u)')
plt.title("Best fraction of SH: {:.2f}".format(ratios[np.argmax(sum_diff_cutoffs)]))
plt.savefig('./figures/best_sh_theorical.png')
"""
"""
plt.figure(10)
diff_cuttoffs = abs(max_return_energies_2D_first - max_return_energies_2D_second)/Up
sum_diff_cutoffs = np.sum(diff_cuttoffs,axis=1)
plt.plot(sum_diff_cutoffs, ratios)
plt.ylim([0,0.5])
plt.xlim([0,np.max(sum_diff_cutoffs)+1])
plt.ylabel('SH Intensity Fraction')
plt.xlabel('Counts (arb.u)')
plt.title("Best fraction of SH: {:.2f}".format(ratios[np.argmax(sum_diff_cutoffs)]))
#plt.savefig('./figures/best_sh_theorical.png')
"""
"""
plt.figure(5)
for ratio_idx, sh_harmonic_intensity_ratio in enumerate(ratios):
    plt.plot(phase_values, max_return_energies_2D_first[ratio_idx]/Up, color='red', label='First Half Cycle')
    plt.plot(phase_values, max_return_energies_2D_second[ratio_idx]/Up, color='blue', label='Second Half Cycle')
    plt.axhline(y=3.17, color='black', linestyle='--', label='One color')
    plt.xlabel(r'Two-Color Phase $\phi$ (rad)')
    plt.ylabel('Electron Maximum Kinetic Energy (in $U_p$)')
    plt.ylim([0,4])
    plt.xlim([-np.pi,np.pi])
    plt.legend()
    plt.savefig('./figures/phase_cycles.png')
    plt.savefig('./figures/phase_cycles.eps')
"""
plt.show()
