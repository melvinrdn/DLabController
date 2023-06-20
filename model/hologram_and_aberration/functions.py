import numpy as np
from scipy import ndimage
from numpy.fft import fftshift, fft2, ifft2
from time import time


def find_second_order_moment(image):
    """
    Calculate the second order moments (also known as variances) of a 2D array.

    Parameters
    ----------
    image : ndarray
        The input 2D array.

    Returns
    -------
    x_som, y_som : float
        The x and y second order moments of the input array, respectively.
    """
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
    """
    Set all pixels outside a given radius around the maximum value pixel to zero.

    Parameters
    ----------
    image : ndarray
        The input 2D array.
    radius : float or int
        The radius around the maximum pixel, beyond which all pixels will be set to zero.

    Returns
    -------
    new_image : ndarray
        The input array with all pixels outside the circle set to zero.
    cut_x : ndarray
        An array of x-values corresponding to the cuts along the x-axis of the circle.
    cut_y : ndarray
        An array of y-values corresponding to the cuts along the y-axis of the circle.
    """
    max_pixel = np.max(image)
    max_coord = np.where(image == max_pixel)
    max_coord_tuple = (max_coord[0][0], max_coord[1][0])

    radius = int(radius)
    center_col = int(max_coord_tuple[1])
    center_row = int(max_coord_tuple[0])

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
    """
    Process an image to remove noise and hot pixels and set it within a circular boundary.

    Parameters
    ----------
    image : numpy.ndarray
        The input image to process.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - new_image : numpy.ndarray
            The processed image with noise and hot pixels removed and set within a circular boundary.
        cut_x : ndarray
            An array of x-values corresponding to the cuts along the x-axis of the circle.
        cut_y : ndarray
            An array of y-values corresponding to the cuts along the y-axis of the circle.
    """
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
    # som = find_second_order_moment(image)

    # Calculate the radius such as the number is the next 2**n
    # radius = 2 ** np.ceil(np.log2((np.max(som[1]) + np.max(som[0])) / 2))
    radius = 128  # With the current setup, it's for some reason always 128

    # Cut the value outside the circle of chosen radius
    new_image, cut_x, cut_y = set_outside_circle_to_zero(image, radius)

    # Normalize to the area
    new_image = new_image / np.sum(new_image)

    return new_image, cut_x, cut_y


def k_transverse_calc(cut_x, cut_y, delta_xy):
    """
    Calculate the transverse momentum k_transverse

    Parameters
    ----------
    cut_x : ndarray
        An array of x-values corresponding to the cuts along the x-axis of the circle.
    cut_y : ndarray
        An array of y-values corresponding to the cuts along the y-axis of the circle.
    delta_xy : float
        The distance between adjacent points on the grid.

    Returns
    -------
    k_transverse : tuple
        Tuple containing k_transverse_x and k_transverse_y

    k_transverse_max : float
        Maximum value of the k_transverse vector
    """
    k_transverse_max = np.pi / delta_xy
    k_transverse = (cut_x * k_transverse_max / (len(cut_x) / 2), cut_y * k_transverse_max / (len(cut_y) / 2))

    return k_transverse, k_transverse_max


def k_transverse_filter(tf_image, k_transverse, k_transverse_max):
    """
    Filter an array based on the transverse momentum (k_T_red) magnitudes.

    Parameters
    ----------
    tf_image : np.ndarray
        2D array to be filtered.
    k_transverse : tuple
        The x and y-components of the transverse momentum vectors.
    k_transverse_max : float
        The maximum allowable magnitude of the transverse momentum vectors.

    Returns
    -------
    tf_image : ndarray
        The filtered array with elements set to 0.0 where the corresponding
        k_transvere magnitude exceeds k_transverse_max.
    """
    # Calculate the magnitudes of all possible vectors using broadcasting
    value = np.sqrt(k_transverse[0][:, np.newaxis] ** 2 + k_transverse[1][np.newaxis, :] ** 2)

    # Use boolean mask to set the corresponding elements of tf_image to 0.0 where the magnitude exceeds k_transverse_max
    tf_image[value > k_transverse_max] = 0.0
    return tf_image


def utils_plot(cut_x, cut_y, delta_xy, lambda_0, focal_length):
    """Calculate the plot coordinates of two points, XY_plot and XY_prime_plot.

    Parameters
    ----------
    cut_x : ndarray
        An array of x-values corresponding to the cuts along the x-axis of the circle.
    cut_y : ndarray
        An array of y-values corresponding to the cuts along the y-axis of the circle.
    delta_xy : float
        The distance between adjacent points on the grid.
    lambda_0 : int or float
        The wavelength of the signal.
    focal_length : int or float
        The focal length of the lens.

    Returns
    -------
    tuple
        A tuple containing the coordinates of two points, XY_plot and XY_prime_plot.
        XY_plot : tuple
            A tuple containing the x and y coordinates of the point XY.
        XY_prime_plot : tuple
            A tuple containing the x and y coordinates of the point XY_prime.
    """
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


def check_corr(arr1, arr2):
    """
    Calculates the Pearson correlation coefficient between two arrays, arr1 and arr2.

    Parameters
    ----------
    arr1 : numpy.ndarray
        A 2D numpy array representing the first input array.
    arr2 : numpy.ndarray
        A 2D numpy array representing the second input array.

    Returns
    -------
    correlation_coefficient : float
        A scalar value representing the Pearson correlation coefficient between the two input arrays.

    Notes
    -----
    The Pearson correlation coefficient is a measure of the linear correlation between two variables.
    It ranges between -1 and 1, where 1 indicates a perfect positive correlation, 0 indicates no correlation,
    and -1 indicates a perfect negative correlation.
    The correlation coefficient is calculated by dividing the covariance of the two arrays
    by the product of their standard deviations.
    """
    arr1 = np.abs(arr1)
    arr2 = np.abs(arr2)

    return np.sum(np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1])


def initial_process(initial_image_raw, delta_xy):
    """
    Applies a series of image processing steps to an initial image.

    Parameters
    ----------
    initial_image_raw : numpy.ndarray
        A 2D numpy array representing the initial raw image to be processed.
    delta_xy : float
        The pixel size of the image in microns.

    Returns
    -------
    initial_image_treated : numpy.ndarray
        A 2D numpy array representing the treated image obtained after applying the processing steps.
    cut_x : int
        The value of the cut position along the x-axis obtained from processing the raw image.
    cut_y : int
        The value of the cut position along the y-axis obtained from processing the raw image.
    """
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


def wgs_algo(initial_image_focus, target_image_focus, max_iter=1000, crit_convergence=None):
    """
    Implements the Weighted Gerchberg-Saxton algorithm for phase retrieval.

    Parameters
    ----------
    initial_image_focus : numpy.ndarray
        A 2D numpy array representing the initial image in the focus plane.
    target_image_focus : numpy.ndarray
        A 2D numpy array representing the target image in the focus plane.
    max_iter : int, optional
        The maximum number of iterations to perform the algorithm. Default is 1000.
    crit_convergence : float, optional
        The convergence criterion for the algorithm.
        The algorithm stops when the correlation coefficient between the final reconstructed image and the target image
        is greater than or equal to this value.
        Default is None, which means the algorithm will perform the maximum number of iterations specified by max_iter.

    Returns
    -------
    np.abs(E_focus) : numpy.ndarray
        A 2D numpy array representing the final reconstructed image in the focus plane.
    np.abs(E_slm) : numpy.ndarray
        A 2D numpy array representing the final reconstructed image in the SLM plane.
    phi_focus : numpy.ndarray
        A 2D numpy array representing the phase in the focus plane at the end of the algorithm.
    phi_slm : numpy.ndarray
        A 2D numpy array representing the phase in the SLM plane at the end of the algorithm.

    Notes
    -----
    The Gerchberg-Saxton algorithm is an iterative algorithm for phase retrieval in optical imaging.
    It uses an initial estimate of the phase in the SLM plane to calculate the complex field at the focus plane,
    and then imposes the modulus of the target image at the focus plane.
    This complex field is then propagated back to the SLM plane, where the phase is updated based on the target modulus
    and the previous estimate of the phase. The algorithm continues for a maximum number of iterations or until
    a convergence criterion is met. The convergence criterion is based on the correlation coefficient between the final
    reconstructed image and the target image.
    """
    # Calculate both fields in the SLM plane
    initial_image_slm = np.abs(fftshift(ifft2(initial_image_focus, norm='ortho')))
    target_image_slm = np.abs(fftshift(ifft2(target_image_focus, norm='ortho')))

    phi_slm = np.ones_like(initial_image_slm, dtype=np.float64)
    t1 = time()
    for i in np.linspace(0, max_iter - 1, max_iter, dtype=np.int32):

        # Impose initial image modulus in SLM plane and apply weight
        B = initial_image_slm * np.exp(1j * phi_slm)
        B *= target_image_slm / np.abs(B)

        # Propagate to focus plane
        E_focus = fftshift(fft2(fftshift(B), norm='ortho'))

        # Impose target module in the focus plane and apply weight
        D = np.abs(target_image_focus) * np.exp(1j * np.angle(E_focus))
        D *= np.abs(target_image_focus) / np.abs(D)

        # Invert propagation back to SLM plane
        E_slm = fftshift(ifft2(fftshift(D), norm='ortho'))

        # Update phase in SLM plane
        phi_slm = np.angle(E_slm)
        phi_focus = np.angle(E_focus)

        # Correlation calculation
        corr = check_corr(E_focus, target_image_focus)

        if crit_convergence is not None and corr >= crit_convergence:
            t2 = time()
            print(f'Correlation coefficient = {corr} reached \nwith {i + 1} iterations in {t2 - t1} seconds')
            break

    return np.abs(E_focus), np.abs(E_slm), phi_focus, phi_slm
