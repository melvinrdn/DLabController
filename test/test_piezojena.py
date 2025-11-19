import time
import numpy as np
import matplotlib.pyplot as plt
from dlab.hardware.wrappers.piezojena_controller import NV40


def generate_sawtooth(start, stop, N, cycles):
    up = np.linspace(start, stop, N)
    down = np.linspace(stop, start, N)
    waveform = []

    for _ in range(cycles):
        waveform.extend(up)
        waveform.extend(down)

    return np.array(waveform)


def run_sawtooth(device, waveform, dt):
    set_vals = []
    meas_vals = []
    t_vals = []

    t0 = time.time()

    for i, pos in enumerate(waveform):
        device.set_position(pos)
        time.sleep(dt)
        meas = device.get_position()
        t_now = time.time() - t0

        set_vals.append(pos)
        meas_vals.append(meas)
        t_vals.append(t_now)

        print(f"{i+1}/{len(waveform)} | Set={pos:.3f} | Measured={meas:.3f}")

    return np.array(t_vals), np.array(set_vals), np.array(meas_vals)


def analyze_calibration(set_arr, meas_arr):
    # Polynomial fit
    coeffs = np.polyfit(set_arr, meas_arr, deg=2)
    poly = np.poly1d(coeffs)

    fit_vals = poly(set_arr)
    residuals = meas_arr - fit_vals

    coeff_lin = np.polyfit(set_arr, meas_arr, deg=1)
    lin_poly = np.poly1d(coeff_lin)
    nonlin = np.max(np.abs(meas_arr - lin_poly(set_arr)))
    fullscale = set_arr.max() - set_arr.min()
    nonlin_pct = (nonlin / fullscale) * 100


    rms = np.sqrt(np.mean(residuals**2))

    return poly, fit_vals, residuals, nonlin_pct, rms


def analyze_hysteresis(set_arr, meas_arr):
    half = len(set_arr) // 2
    up_set = set_arr[:half]
    up_meas = meas_arr[:half]

    down_set = set_arr[half:]
    down_meas = meas_arr[half:]

    down_interp = np.interp(up_set, down_set[::-1], down_meas[::-1])
    hysteresis = np.max(np.abs(up_meas - down_interp))

    return up_set, up_meas, down_set, down_meas, hysteresis


if __name__ == "__main__":
    PORT = "COM6"
    START = 0
    STOP = 100       
    N_STEPS = 25
    CYCLES = 3
    DT = 0.1     
    dev = NV40(PORT, closed_loop=False)

    print("\nGenerating waveform…")
    waveform = generate_sawtooth(START, STOP, N_STEPS, CYCLES)

    print("Running sawtooth scan…")
    t, set_vals, meas_vals = run_sawtooth(dev, waveform, DT)

    dev.close()
    print("Scan completed.\n")

    poly, fit_vals, residuals, nonlin_pct, rms = analyze_calibration(set_vals, meas_vals)
    print("Calibration polynomial:", poly)
    print(f"Static nonlinearity: {nonlin_pct:.3f}%")
    print(f"RMS residual: {rms:.5f} V")

    up_set, up_meas, down_set, down_meas, hyst = analyze_hysteresis(set_vals, meas_vals)
    print(f"Hysteresis amplitude: {hyst:.4f} V\n")

    plt.figure(figsize=(12, 5))
    plt.plot(t, set_vals, "--", label="Set Voltage")
    plt.plot(t, meas_vals, label="Measured Voltage")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Piezo Sawtooth: Set vs Measured")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(set_vals, meas_vals, ".", label="Measured")
    plt.plot(set_vals, fit_vals, "-", label="Polynomial Fit")
    plt.xlabel("Set Voltage (V)")
    plt.ylabel("Measured Voltage (V)")
    plt.title("Calibration Curve")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 4))
    plt.plot(set_vals, residuals, label="Residual")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Set Voltage (V)")
    plt.ylabel("Error (V)")
    plt.title(f"Fit Residuals (RMS = {rms:.4f} V)")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(up_set, up_meas, label="Up Sweep")
    plt.plot(down_set, down_meas, label="Down Sweep")
    plt.xlabel("Set Voltage (V)")
    plt.ylabel("Measured Voltage (V)")
    plt.title(f"Hysteresis Loop (Amplitude = {hyst:.4f} V)")
    plt.grid(True)
    plt.legend()

    plt.show()
