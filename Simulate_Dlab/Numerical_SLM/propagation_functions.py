import numpy as np
from Simulate_Dlab.Numerical_SLM.utils import *
import Simulate_Dlab.Numerical_SLM.diagnostic_functions as diagfunc
import time
import Simulate_Dlab.Numerical_SLM.plot_functions as plotfunc
from scipy.special import genlaguerre
from scipy.special import jn


def define_slm_plan(slm_param, print_option=True):
    x_slm_max, y_slm_max, N_x, N_y = slm_param
    x_slm = np.arange(-N_x // 2, N_x // 2) * 2 * x_slm_max / N_x
    y_slm = np.arange(-N_y // 2, N_y // 2) * 2 * y_slm_max / N_y

    extent_slm = [x_slm[0], x_slm[-1], y_slm[0], y_slm[-1]]
    formatted_extent_slm = [f'{value * 1e3:.2f} mm' for value in extent_slm]
    if print_option is True:
        print(f'SLM plane coordinates: {formatted_extent_slm}')

    return x_slm, y_slm, extent_slm


def define_focus_plan(slm_param, beam_param, print_option=True):
    lambda_0, f, w_L, w_0 = beam_param
    x_slm_max, y_slm_max, N_x, N_y = slm_param

    k_0 = 2 * np.pi / lambda_0
    k_T_x = np.arange(-N_x // 2, N_x // 2) * np.pi / (2 * x_slm_max)
    k_T_y = k_T_x

    x_focus = f * k_T_x / k_0
    y_focus = f * k_T_y / k_0
    extent_focus = [x_focus[0], x_focus[-1], y_focus[0], y_focus[-1]]
    formatted_extent_focus = [f'{value * 1e6:.2f} µm' for value in extent_focus]
    if print_option is True:
        print(f'Focus plane coordinates: {formatted_extent_focus}')

    return x_focus, y_focus, extent_focus


def gaussian_2D(x_slm, y_slm, w_L):
    X, Y = np.meshgrid(x_slm, y_slm)
    R = np.sqrt(X ** 2 + Y ** 2)
    gaussian = np.exp(-2 * R ** 2 / w_L ** 2)
    indices = np.where(gaussian <= 1e-10)
    gaussian[indices] = 0
    return gaussian / np.sum(gaussian)

def s_gaussian_2D(x_slm, y_slm, w_L, order):
    X, Y = np.meshgrid(x_slm, y_slm)
    R = np.sqrt(X ** 2 + Y ** 2)
    sgaussian = np.exp(-2 * R ** order / w_L ** order)
    indices = np.where(sgaussian <= 1e-10)
    sgaussian[indices] = 0
    return sgaussian / np.sum(sgaussian)

def hgb_2D(x_slm, y_slm, w_L, n):
    X, Y = np.meshgrid(x_slm, y_slm)
    R = np.sqrt(X ** 2 + Y ** 2)
    g0 = 1
    E_hgb= g0 * (R ** 2 / w_L ** 2) ** n * np.exp(- R ** 2 / w_L ** 2)
    I_hgb = abs(E_hgb) ** 2
    indices = np.where(I_hgb <= 1e-10)
    I_hgb[indices] = 0
    return I_hgb / np.sum(I_hgb)

def flattop_2D(x_slm, y_slm, radius):
    X, Y = np.meshgrid(x_slm, y_slm)
    R = np.sqrt(X ** 2 + Y ** 2)
    circle = np.zeros_like(X)
    indices = np.where(R <= radius)
    circle[indices] = 1
    return circle / np.sum(circle)

def bessel_2D(x_slm, y_slm, radius):

    X, Y = np.meshgrid(x_slm, y_slm)
    R = np.sqrt(X ** 2 + Y ** 2)
    n = 0
    J = abs(jn(n, R/4e-3))**2
    return J


def laguerre_gaussian_2D(x_slm, y_slm, w_L, p, l):
    X, Y = np.meshgrid(x_slm, y_slm)
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)

    radial_component = ((np.sqrt(2) * R) / w_L) ** abs(l)
    laguerre_polynomial = genlaguerre(p, abs(l))(2 * R ** 2 / w_L ** 2)
    amplitude = radial_component * laguerre_polynomial * np.exp(-R ** 2 / w_L ** 2)
    azimuthal_component = np.exp(1j * l * Theta)
    intensity = np.abs(amplitude * azimuthal_component) ** 2
    return intensity / np.sum(intensity)

def ifft2c(array):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array), norm='ortho'))


def fft2c(array):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array), norm='ortho'))


def propagate(u, L, method, wavelength, dz=None, dq=None, bandlimit=True):
    if method == 'fourier':
        if dz > 0:
            # forward propagation
            u_new = fft2c(u)
        else:
            # backward propagation
            u_new = ifft2c(u)

        return u_new

    elif method == 'aspw':
        k = 2 * np.pi / wavelength

        # source coordinates
        N = u.shape[-1]
        f = np.arange(-N / 2, N / 2) / L
        Fx, Fy = np.meshgrid(f, f)

        if bandlimit is True:
            f_max = L / (wavelength * np.sqrt(L ** 2 + 4 * dz ** 2))
            W = circ(Fx, Fy, 2 * f_max)

        else:
            print('Bandlimit is not activated')
            W = 1

        # Free space propagator with bandlimit factor W
        H = np.exp(1.j * k * dz * np.sqrt(1 - (Fx * wavelength) ** 2 - (Fy * wavelength) ** 2)) * W
        U = fft2c(u)
        u_new = ifft2c(U * H)

        return u_new

    elif method == 'fresnel':
        # not recommanded
        k = 2 * np.pi / wavelength

        # source coordinates
        N = u.shape[-1]
        dx = L / N
        x = np.arange(-N // 2, N // 2) * dx
        [Y, X] = np.meshgrid(x, x)

        # target coordinates
        dq = wavelength * dz / L
        q = np.arange(-N // 2, N // 2) * dq
        [Qy, Qx] = np.meshgrid(q, q)

        Q1 = np.exp(1j * k / (2 * dz) * (X ** 2 + Y ** 2))
        Q2 = np.exp(1j * k / (2 * dz) * (Qx ** 2 + Qy ** 2))

        # pre-factor
        A = 1 / (1j * wavelength * dz)

        # Fresnel-Kirchhoff integral
        u_new = A * Q2 * fft2c(u * Q1)

        return u_new


    elif method == 'saspw':
        # not fully working
        k = 2 * np.pi / wavelength
        dz += 1e-10

        N = u.shape[-1]
        dx = L / N
        f_max = L / (wavelength * np.sqrt(L ** 2 + 4 * dz ** 2))

        # source plane coordinates
        x1 = np.arange(-N // 2, N // 2) * dx
        X1, Y1 = np.meshgrid(x1, x1)
        r1sq = X1 ** 2 + Y1 ** 2
        f = np.arange(-N // 2, N // 2) / (N * dx)
        FX, FY = np.meshgrid(f, f)
        W = circ(FX, FY, 2 * f_max)
        fsq = FX ** 2 + FY ** 2

        # scaling parameter
        # dq = wavelength * dz / L
        m = dq / dx

        # quadratic phase factors
        Q1 = np.exp(1.j * (k / 2) * ((1 - m) / dz) * r1sq)
        Q2 = np.exp(1.j * (np.pi ** 2) * (2 * (-dz) / (m * k)) * fsq) * W
        # Q1 = np.exp(1.j * k / 2 * (1 - m) / dz * r1sq)
        # Q2 = np.exp(-1.j * np.pi ** 2 * 2 * dz / m / k * fsq)

        if bandlimit:
            if m != 1:
                r1sq_max = wavelength * dz / (2 * dx * (1 - m))
                Wr = np.array(circ(X1, Y1, 2 * r1sq_max))
                Q1 = Q1 * Wr

            fsq_max = m / (2 * dz * wavelength * (1 / (N * dx)))
            Wf = np.array(circ(FX, FY, 2 * fsq_max))
            Q2 = Q2 * Wf

        # note: to be analytically correct, add Q3
        # if only intensities matter, leave it out
        x2 = np.arange(-N / 2, N / 2) * dq
        X2, Y2 = np.meshgrid(x2, x2)
        r2sq = X2 ** 2 + Y2 ** 2
        Q3 = np.exp(1.j * k / 2 * (m - 1) / (m * (dz)) * r2sq)

        # compute the propagated field
        if dz > 0:
            # u_new = ifft2c(Q2 * fft2c(Q1 * u))
            u_new = Q3 * ifft2c(Q2 * fft2c(Q1 * u))
        else:
            # u_new = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u))
            u_new = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u * np.conj(Q3)))

        return u_new


def propagation_scan(E_focus, deltaz, z_steps, beam_param, slm_param, factor=2, plot_option=True):

    x_focus, y_focus, extent_focus = define_focus_plan(slm_param, beam_param, print_option=False)
    Y_focus, X_focus = np.meshgrid(x_focus, y_focus)
    lambda_0, f, w_L, w_0 = beam_param
    _, _, N_x, N_y = slm_param
    k_0 = 2 * np.pi / lambda_0
    z_R = np.pi * w_0 ** 2 / lambda_0
    L = 2 * x_focus[-1]
    dx = L / N_x

    som_x_focus, som_y_focus = diagfunc.get_som(abs(E_focus) ** 2, dx)
    som_max = np.max([som_x_focus, som_y_focus])
    E_focus_cut = cut_focus(E_focus, x_focus, y_focus, som_max, factor=2)

    # Defining propagation range
    zmin =  deltaz[0] * z_R
    zmax = deltaz[1] * z_R
    z = np.linspace(zmin, zmax, z_steps)

    # Preparing the arrays for the loop
    E_z = np.zeros((N_y, N_x, z_steps), dtype=complex)
    som_x = np.zeros_like(z, dtype=float)
    som_y = np.zeros_like(z, dtype=float)
    E_z_cut = np.zeros_like(E_z, dtype=complex)

    # Propagation loop
    t_start = time.time()
    for i, e in enumerate(z):
        # Actual propagation
        E_z[:, :, i] = np.exp(-1j * k_0 * e) * propagate(E_focus_cut, L=L, method='aspw', wavelength=lambda_0, dz=e)
        # Calculate second order moments
        som_x[i], som_y[i] = diagfunc.get_som(abs(E_z[:, :, i]) ** 2, dx)
        print(f'z{i}: {e / z_R:.3f}z_R; som_x: {som_x[i] * 1e6:.3f} µm; som_y: {som_y[i] * 1e6:.3f} µm')

        # Cut the array
        circle_radius = som_x[i] * factor
        mask = (X_focus ** 2 + Y_focus ** 2) <= circle_radius ** 2
        E_z_cut[:, :, i][mask] = E_z[:, :, i][mask]

        # Plot in the focus plane
        if plot_option is True:
            plotfunc.plot_focus_plane(E_z_cut[:, :, i], som_x[i], som_y[i], extent_focus, unwrap=False)

    t_end = time.time()
    print(f"Loop time: {t_end - t_start:.5f} s")
    print('-----')

    return E_z_cut,z, som_x, som_y
