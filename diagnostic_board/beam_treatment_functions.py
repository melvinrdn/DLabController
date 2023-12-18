import numpy as np
from scipy import ndimage
from numpy.fft import fftshift, fft2, ifft2


def find_second_order_moment(image):
    num_rows, num_cols = image.shape

    # Calculate the center of mass of the input image
    center_of_mass = ndimage.center_of_mass(image)
    center_col = int(center_of_mass[1])
    center_row = int(center_of_mass[0])

    # Compute the deviation of each pixel from the center of mass
    col_vals = np.arange(num_cols)
    row_vals = np.arange(num_rows)
    col_grid, row_grid = np.meshgrid(col_vals, row_vals)
    col_deviation = col_grid - center_col
    row_deviation = row_grid - center_row

    # Compute the second order moments and sum of pixel values
    som_x = np.sum(col_deviation ** 2 * image)
    som_y = np.sum(row_deviation ** 2 * image)
    pixel_sum = np.sum(image)

    # Compute the x and y second order moments
    x_som = 2 * np.sqrt(som_x / pixel_sum)
    y_som = 2 * np.sqrt(som_y / pixel_sum)

    return x_som, y_som


def set_outside_circle_to_zero(image, radius):
    max_pixel = np.max(image)
    max_coord = np.where(image == max_pixel)
    max_coord_tuple = (max_coord[0][0], max_coord[1][0])

    radius = int(radius)
    center_col = int(max_coord_tuple[1])
    center_row = int(max_coord_tuple[0])

    #center_of_mass = ndimage.center_of_mass(image)
    #center_col = int(center_of_mass[1])
    #center_row = int(center_of_mass[0])

    x_start = max(0, center_col - radius)
    x_end = min(image.shape[1], center_col + radius)
    y_start = max(0, center_row - radius)
    y_end = min(image.shape[0], center_row + radius)

    # Extract a sub-array containing only the pixels within the circle
    new_image = image[y_start:y_end, x_start:x_end]

    # Generate a grid of x and y values for computing the distance from the center
    nx, ny = new_image.shape
    x, y = np.meshgrid(np.arange(ny), np.arange(nx))

    # Compute the distance from the center for each pixel in the sub-array
    distance_from_center = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)

    # Create a Boolean mask for pixels outside the circle
    outside_circle = distance_from_center > radius

    # Set all pixels outside the circle to zero
    new_image[outside_circle] = 0

    # Generate arrays of x and y values corresponding to cuts along the x and y axes of the circle
    cut_x = np.linspace(-radius, radius, 2 * radius)
    cut_y = cut_x

    return new_image, cut_x, cut_y


def process_image(image):
    # Remove the background noise, #TODO - back reflection treatment
    image[image <= np.min(image) + 1] = 0

    # Hot pixels treatment
    # Create four shifted versions of the input image, one for each direction
    shifted_right = np.roll(image, 1, axis=1)
    shifted_left = np.roll(image, -1, axis=1)
    shifted_up = np.roll(image, -1, axis=0)
    shifted_down = np.roll(image, 1, axis=0)

    # Calculate a boolean mask where the value is True if the corresponding pixel has all neighbors equal to 0
    neighbors = (shifted_right == 0) & (shifted_left == 0) & (shifted_up == 0) & (shifted_down == 0)

    # Set the value of all pixels that are identified as hot pixels (neighbors is True) and have a value greater than
    # 0 to 0
    image[neighbors & (image > 0)] = 0

    # Calculate the second order moment (variance)
    #old_som = find_second_order_moment(image)
    #print(f'old som {old_som}')

    # Calculate the radius such as the number is the next 2**n
    radius = 32
    #radius = 2 ** np.ceil(np.log2((np.max(som[1]) + np.max(som[0])) / 2))

    # Cut the value outside the circle of chosen radius
    new_image, cut_x, cut_y = set_outside_circle_to_zero(image, radius)

    # Normalize to the area
    new_image = new_image / np.sum(new_image)

    new_som = find_second_order_moment(new_image)
    #print(f'new som {new_som}')
    som_x,som_y = new_som

    return new_image, som_x, som_y


def k_transverse_calc(cut_x, cut_y, delta_xy):
    k_transverse_max = np.pi / delta_xy
    k_transverse = (cut_x * k_transverse_max / (len(cut_x) / 2), cut_y * k_transverse_max / (len(cut_y) / 2))

    return k_transverse, k_transverse_max


def k_transverse_filter(tf_image, k_transverse, k_transverse_max):
    # Calculate the magnitudes of all possible vectors using broadcasting
    value = np.sqrt(k_transverse[0][:, np.newaxis] ** 2 + k_transverse[1][np.newaxis, :] ** 2)

    # Use boolean mask to set the corresponding elements of tf_image to 0.0 where the magnitude exceeds k_transverse_max
    tf_image[value > k_transverse_max] = 0.0
    return tf_image


def utils_plot(cut_x, cut_y, delta_xy, lambda_0, focal_length):
    k_0 = 2 * np.pi / lambda_0

    k_transverse_max = np.pi / delta_xy
    k_transverse_x = cut_x * k_transverse_max / (len(cut_x) / 2)
    k_transverse_y = cut_y * k_transverse_max / (len(cut_y) / 2)

    X_plot = cut_x * delta_xy
    Y_plot = cut_y * delta_xy

    XY_plot = (X_plot, Y_plot)

    X_prime_plot = k_transverse_x * focal_length / k_0
    Y_prime_plot = k_transverse_y * focal_length / k_0

    XY_prime_plot = (X_prime_plot, Y_prime_plot)

    return XY_plot, XY_prime_plot


def initial_process(initial_image_raw, delta_xy=3.45e-6):
    # Processing the raw image to improve image quality and remove noise
    initial_image_processed, cut_x, cut_y = process_image(initial_image_raw)

    # Calculating k_T of the image using the cut_x and cut_y values obtained from the processed image
    k_T, k_T_max = k_transverse_calc(cut_x, cut_y, delta_xy)

    # Applying a filter to the Fourier transform of the processed image using the calculated k_T and k_T_max values
    tf_initial_image_processed = fftshift(fft2(initial_image_processed, norm='ortho'))
    tf_initial_image_processed_filtered = k_transverse_filter(tf_initial_image_processed, k_T, k_T_max)

    # Performing an inverse Fourier transform on the filtered image to obtain the treated image.
    initial_image_treated = np.abs(ifft2(tf_initial_image_processed_filtered, norm='ortho'))

    return initial_image_treated, cut_x, cut_y

