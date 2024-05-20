import numpy as np
import matplotlib.pyplot as plt
from prysm import coordinates, geometry, polynomials, propagation, fttools
from scipy import optimize
from diagnostic_board.beam_treatment_functions import process_image


class NLOptimizer():
    def __init__(self, max_zn_coef, F):
        self.N = 2 ** 8  # Number of points
        self.wL = 1  # Diameter of the beam
        self.size = 15  # Size of the SLM plane in wL
        self.wvl = 1.030  # Wavelength in µm
        self.f = 200  # Focal length in mm
        self.wL_SI = 4e-3 * 2
        self.w0_SI = (self.wvl * 1e-6) * self.f / (np.pi * self.wL_SI) * 4

        self.js = np.arange(3, max_zn_coef)

        self.D, _, _ = process_image(F)
        plt.imshow(self.D)
        plt.show()

        x, y = coordinates.make_xy_grid(self.N, diameter=self.size)
        r, t = coordinates.cart_to_polar(x, y)
        self.dx = x[0, 1] - x[0, 0]
        self.amp = geometry.gaussian(self.wL, x, y)
        self.mask_slm = geometry.rectangle(1.92 / 2, x, y, height=1.2 / 2)
        self.amp[~self.mask_slm] = 0

        nms = [polynomials.noll_to_nm(j) for j in self.js]

        self.basis = list(polynomials.zernike_nm_sequence(nms, r, t))
        self.cost_denom = np.sum(self.D ** 2)

    def forward_model(self, coefs):
        phs = polynomials.sum_of_2d_modes(self.basis, coefs)
        wf = propagation.Wavefront.from_amp_and_phase(self.amp, phs, self.wvl, self.dx)
        psf = wf.focus(self.f, Q=1)
        return abs(psf.data) ** 2

    def cost_function(self, coefs):
        M = self.forward_model(coefs)
        return np.sum((self.D - M) ** 2) / self.cost_denom

    def perform_optimization(self):
        result = optimize.minimize(self.cost_function, x0=np.zeros_like(self.js), method='L-BFGS-B',
                                   options={'eps': 1e-5, 'gtol': 1e-20, 'ftol': 1e-15, 'maxiter': 500})
        return result

    def display_results(self, result):
        phsr = polynomials.sum_of_2d_modes(self.basis, result.x)

        phsr[~self.mask_slm] = 0

        wfr = propagation.Wavefront.from_amp_and_phase(self.amp, phsr, self.wvl, self.dx)
        psfr = wfr.focus(self.f, Q=1)
        psfr = abs(psfr.data) ** 2

        extent_slm = [-self.size * 8 / 2, self.size * 8 / 2, -self.size * 8 / 2, self.size * 8 / 2]
        k0 = 2 * np.pi / (self.wvl * 1e-6)  # Wavenumber in m^-1
        x_focus = self.f * 1e-3 * np.arange(-self.N // 2, self.N // 2 + 1) * np.pi / (self.wL_SI * self.size) / k0
        extent_focus = [x_focus[0] * 1e6, x_focus[-1] * 1e6, x_focus[0] * 1e6, x_focus[-1] * 1e6]
        size_slm_plot_x = 15.36  # in mm, only for plotting
        size_slm_plot_y = 9.6  # in mm, only for plotting
        size_focus_plot = 50  # in microns, only for plotting

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot initial PSF
        axs[0, 0].imshow(self.D, cmap='turbo', extent=extent_focus)
        axs[0, 0].set_title('Initial PSF - $|F|^2$')
        axs[0, 0].set_xlabel('x (µm)')
        axs[0, 0].set_ylabel('y (µm)')
        axs[0, 0].set(xlim=(-size_focus_plot, size_focus_plot), ylim=(-size_focus_plot, size_focus_plot))

        # Plot retrieved PSF
        axs[0, 1].imshow(psfr, cmap='turbo', extent=extent_focus)
        axs[0, 1].set_title('L-BFGS-B PSF - $|G|^2$')
        axs[0, 1].set_xlabel('x (µm)')
        axs[0, 1].set_ylabel('y (µm)')
        axs[0, 1].set(xlim=(-size_focus_plot, size_focus_plot), ylim=(-size_focus_plot, size_focus_plot))

        # Plot phase distribution
        axs[1, 0].imshow(phsr, cmap='hsv', extent=extent_slm)
        axs[1, 0].set_title('L-BFGS-B - $arg(g)$')
        axs[1, 0].set_xlabel('x (mm)')
        axs[1, 0].set_ylabel('y (mm)')
        axs[1, 0].set(xlim=(-size_slm_plot_x / 2, size_slm_plot_x / 2),
                      ylim=(-size_slm_plot_y / 2, size_slm_plot_y / 2))

        # Plot Zernike coefficients
        axs[1, 1].bar(self.js, result.x, color='b', alpha=0.8)
        axs[1, 1].set_xlabel('Zernike Mode (j)')
        axs[1, 1].set_ylabel('Coefficient Value (nm RMS)')
        axs[1, 1].set_title('Optimized Zernike Coefficients')

        plt.show()

    def phase_retrieval(self):
        result = self.perform_optimization()
        self.display_results(result)


