import numpy as np
from PIL import Image
import matplotlib.pyplot as plt




tif_path = '../HGB_6.571.tif'
image = Image.open(tif_path)
image_array = np.array(image)
plt.imshow(image_array)
plt.show()



