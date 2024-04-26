import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit
from scipy import ndimage
import utils_tools as tool
import os
from PIL import Image
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150
#file = '/2024-04-04/2024-04-04-camera_focus-14.h5'
#data_filename = 'G:/Atto/Data/LASC/dlab/dlabharmonics' + file

#print(f'Opening {data_filename}')
#hfr = h5py.File(data_filename, 'r')
#images = np.asarray(hfr.get('images'))
#positions = np.asarray(hfr.get('positions'))
#print(positions)


def plot_all_images(images_dict):
    if not images_dict:
        print("No images to plot.")
        return

    num_images = len(images_dict)
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols

    vmin = min(image_array.min() for image_array in images_dict.values())
    vmax = max(image_array.max() for image_array in images_dict.values())

    fig, axes = plt.subplots(num_rows, num_cols)

    reference_center_of_mass = None

    for idx, (filename, image_array) in enumerate(images_dict.items()):
        if idx == 0:
            reference_center_of_mass = ndimage.center_of_mass(image_array)
        else:
            current_center_of_mass = ndimage.center_of_mass(image_array)
            x_shift = int(reference_center_of_mass[1] - current_center_of_mass[1])
            y_shift = int(reference_center_of_mass[0] - current_center_of_mass[0])
            image_array = np.roll(image_array, [y_shift, x_shift], axis=(0, 1))

        extent = (-image_array.shape[1] * 3.45 / 2, image_array.shape[1] * 3.45 / 2,
                  -image_array.shape[0] * 3.45 / 2, image_array.shape[0] * 3.45 / 2)
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]

        ax.imshow(image_array, cmap='turbo', extent=extent, vmin=vmin, vmax=vmax)

    for idx in range(len(images_dict), num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def plot_single_image(images_dict, key, save_image=False):
    if key not in images_dict:
        print(f"Key '{key}' not found in the dictionary.")
        return

    image_array = images_dict[key]

    vmin = image_array.min()
    vmax = image_array.max()

    extent = (
        -image_array.shape[1] * 3.45 / 2,
        image_array.shape[1] * 3.45 / 2,
        -image_array.shape[0] * 3.45 / 2,
        image_array.shape[0] * 3.45 / 2
    )
    plt.figure(figsize=(5, 3))
    plt.imshow(image_array, cmap='grey', extent=extent, vmin=vmin, vmax=vmax)
    plt.ylabel('y-axis (µm)', fontsize=15)
    plt.xlabel('x-axis (µm)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)  # Adjust size of tick labels
    plt.tight_layout()

    return image_array

def plot_all_integrated_cross_sections(images_dict, pixel_shift):
    data_x = []
    data_y = []
    reference_center_of_mass = None

    for idx, (filename, image_array) in enumerate(images_dict.items()):
        if idx == 0:
            reference_center_of_mass = ndimage.center_of_mass(image_array)
        else:
            current_center_of_mass = ndimage.center_of_mass(image_array)
            x_shift = int(reference_center_of_mass[1] - current_center_of_mass[1])
            y_shift = int(reference_center_of_mass[0] - current_center_of_mass[0])
            image_array = np.roll(image_array, [y_shift, x_shift], axis=(0, 1))

        x_position = int(reference_center_of_mass[1])
        y_position = int(reference_center_of_mass[0])

        x_min = (x_position - pixel_shift)
        x_max = (x_position + pixel_shift)
        y_min = 0
        y_max = image_array.shape[0]

        integration_x = np.sum(image_array[y_min:y_max, x_min:x_max], axis=1)
        data_x.append(integration_x)

        x_min = 0
        x_max = image_array.shape[1]
        y_min = (y_position - pixel_shift)
        y_max = (y_position + pixel_shift)

        integration_y = np.sum(image_array[y_min:y_max, x_min:x_max], axis=0)
        data_y.append(integration_y)

    fig, axes = plt.subplots(2, 1, figsize=(4, 8))
    extent = (-image_array.shape[1] * 3.45 / 2, image_array.shape[1] * 3.45 / 2,
              positions[0],positions[-1])

    axes[0].imshow(data_x,extent=extent, aspect='auto',cmap='turbo')
    axes[0].set_title('yz plane')
    axes[0].set_ylabel('dz')
    axes[0].set_xlabel('y [µm]')

    axes[1].imshow(data_y,extent=extent, aspect='auto',cmap='turbo')
    axes[1].set_title('xy plane')
    axes[1].set_ylabel('dz')
    axes[1].set_xlabel('x [µm]')

    plt.tight_layout()
    plt.show()

    return data_x, data_y


def get_M_sq(som_x, som_y, z, lambda_0, dx, plot=True):
    def beam_quality_factor_fit(z, w0, M2, z0):
        return w0 * np.sqrt(1 + (z - z0) ** 2 * (M2 * lambda_0 / (np.pi * w0 ** 2)) ** 2)

    p0 = [dx, 1, 0]

    params_x, _ = curve_fit(beam_quality_factor_fit, z, som_x, p0=p0)
    w0_x_fit, M_sq_x_fit, z0_x_fit = params_x
    print(f'M_sq_x: {abs(M_sq_x_fit):.4f},w0_x: {w0_x_fit * 1e6:.4f} µm, z0_x: {z0_x_fit * 1e3:.4f} mm')

    params_y, _ = curve_fit(beam_quality_factor_fit, z, som_y, p0=p0)
    w0_y_fit, M_sq_y_fit, z0_y_fit = params_y
    print(f'M_sq_y: {abs(M_sq_y_fit):.4f},w0_y: {w0_y_fit * 1e6:.4f} µm, z0_y: {z0_y_fit * 1e3:.4f} mm')

    z_fit = np.linspace(z[0], z[-1], 100)

    if plot is True:
        fig_focus, axs_focus = plt.subplots(figsize=(12, 6))

        axs_focus.plot(z, som_x, linestyle='None', marker='x', color='blue')
        axs_focus.plot(z, som_y, linestyle='None', marker='x', color='red')
        axs_focus.plot(z_fit, beam_quality_factor_fit(z_fit, w0_y_fit, M_sq_y_fit, z0_y_fit),
                       label=f'M_sq_y: {abs(params_y[1]):.4f}, '
                             f'w0_y: {params_y[0] * 1e6:.4f} µm, '
                             f'z0_y: {params_y[2] * 1e3:.4f} mm', color='red')
        axs_focus.plot(z_fit, beam_quality_factor_fit(z_fit, w0_x_fit, M_sq_x_fit, z0_x_fit),
                       label=f'M_sq_x: {abs(params_x[1]):.4f}, '
                             f'w0_x: {params_x[0] * 1e6:.4f} µm, '
                             f'z0_x: {params_x[2] * 1e3:.4f} mm', color='blue')
        axs_focus.set_ylabel('z [m]')
        axs_focus.set_xlabel('x [m]')
        axs_focus.legend()
        plt.tight_layout()

        plt.show()

    return params_x, params_y


def process_images_dict(images_array):
    processed_images = {}
    dz = images_array.shape[2]
    print(f'Number of steps: {dz}')
    som_x = np.zeros(dz, dtype=float)
    som_y = np.zeros(dz, dtype=float)

    for i in range(dz):
        processed_image, som_x[i], som_y[i] = tool.process_image(images_array[:, :, i], dx=3.45e-6)
        processed_images[f'processed_image_{i}'] = processed_image

    return processed_images, som_x, som_y


#processed_images_dict, som_x, som_y = process_images_dict(images)
#zero_position = 0
#zmin = (positions[0] - zero_position) * 1e-3
#zmax = (positions[-1] - zero_position) * 1e-3

#z = np.linspace(zmin, zmax, len(positions))
#get_M_sq(som_x, som_y, z, 515e-9, 20e-6)

#plot_all_images(processed_images_dict)

def process_and_plot_images(start_focus=19, end_focus=19):
    for focus in range(start_focus, end_focus + 1):
        file = f'/2024-04-04/2024-04-04-camera_focus-{focus}.h5'
        data_filename = 'G:/Atto/Data/LASC/dlab/dlabharmonics' + file
        print(f'Opening {data_filename}')

        with h5py.File(data_filename, 'r') as hfr:
            images = np.asarray(hfr.get('images'))
            positions = np.asarray(hfr.get('positions'))
            processed_images_dict, som_x, som_y = process_images_dict(images)

            print("Keys in the file:")
            for key in hfr.keys():
                print(key)

            for i, image in enumerate(processed_images_dict.values()):
                image_array = np.array(image)
                tif_image = Image.fromarray(image_array)
                tif_image.save(f"HGB_{positions[i]:.3f}.tif", compression=None)

        plt.show()

process_and_plot_images()


#plt.imshow(focus)
#plt.show()
#focus_image = Image.fromarray((focus * 255).astype(np.uint8))

# Convert the image to grayscale
#focus_gray = focus_image.convert('L')

# Save the grayscale image as BMP
#focus_gray.save("processed_image.bmp")


#data_x,data_y = plot_all_integrated_cross_sections(processed_images_dict,2)
