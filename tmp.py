import helper
import numpy as np
from scipy import ndimage

image = np.pad([[1]], 2, constant_values=0).astype(np.float)
print(image)
gauss_sigma = 0.21
# gauss_sigma = 0.56
output = ndimage.gaussian_filter1d(image, gauss_sigma, 0)
output = ndimage.gaussian_filter1d(output, gauss_sigma, 1)
print(output)