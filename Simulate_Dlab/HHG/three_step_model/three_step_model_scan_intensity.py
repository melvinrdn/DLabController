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
epsilon_0 = 8.854e-12
Eq = h * c / lam

e = qe
#E0 = 2e11  # Example value for E0, V/m
w = c / lam * 2 * np.pi  # Example value for w
laser_cycle = 1 / (c / lam)


total = 1
ratio_min = 0
ratio_max = 0.5

I_list = np.linspace(1e18, 5e18, 5) #W/m2
phase_values = np.linspace(-np.pi, np.pi, 15)
ratios = np.linspace(ratio_min, ratio_max, 40)

for I_idx, I_value in enumerate(I_list):
    print(f'I: {I_value:.2e}')
    E0_value = np.sqrt(2*I_value/(c*epsilon_0))
    phase_grid, ratio_grid = np.meshgrid(phase_values, ratios)
    max_return_energies_2D_first = np.zeros((len(ratios), len(phase_values)))
    max_return_energies_2D_second = np.zeros((len(ratios), len(phase_values)))
    Up = e ** 2 * E0_value ** 2 / (4 * me * w ** 2)
    for ratio_idx, sh_ratio in enumerate(ratios):
        print(f'SH Ratio :', sh_ratio)
        f_ratio = total - sh_ratio

        E_omega = np.sqrt(f_ratio) * E0_value * e / me
        E_2_omega = np.sqrt(sh_ratio) * E0_value * e / me


        def function(u, t):
            return u[1], E_omega * np.cos(t * w) + E_2_omega * np.cos(t * 2 * w + phase)


        def velocity(u, t):
            return E_omega * np.cos(t * w) + E_2_omega * np.cos(t * 2 * w + phase)


        max_return_energies_first = []
        max_return_energies_second = []

        for phase_idx, phase in enumerate(phase_values):
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

    diff_cuttoffs = abs(max_return_energies_2D_first - max_return_energies_2D_second)

    plt.figure(1)
    sum_diff_cutoffs = np.sum(diff_cuttoffs,axis=1)
    plt.plot(sum_diff_cutoffs*1e17, ratios, label=f'I={I_value*1e-18} TW/cm$^2$')
    plt.ylabel('SH Intensity Fraction')
    plt.xlabel('Counts (arb.u)')
    plt.legend(loc='lower right')

plt.savefig('./figures/best_sh_theorical_intensity.png')
plt.show()
