import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import sys
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

me = 9.1e-31
h = 6.62607015e-34
c = 299792458
qe = 1.60217662e-19
lam = 1030e-9
Eq = h * c / lam

def shear_image(image_old, val):
    T = np.float32([[1, val / 100, 0], [0, 1, 0]])
    size_T = (image_old.shape[1], image_old.shape[0])
    image_new = cv2.warpAffine(image_old, T, size_T)
    return image_new
