# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

# %%
np.random.seed(1)
image = np.random.random((1000)).astype(np.float32) * 256

plt.figure(figsize=(10, 5))
plt.plot(image)

# %%
tmp_img = np.copy(image)

hist,bins = np.histogram(tmp_img, bins=256)
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.figure(figsize=(10, 5))
plt.hist(tmp_img, bins=256)
plt.plot(cdf_normalized, color = 'r')

print(f'mean: { np.mean(image) }')
print(f'std: { np.std(image) }')

# %%
sigma = 10
porosity = 0.4
gauss_image = ndimage.gaussian_filter(image, sigma=sigma, truncate=4)
plt.figure(figsize=(10, 5))
plt.plot(gauss_image)

plt.figure(figsize=(10, 5))
plt.plot(gauss_image > (256 * porosity))

tmp_img = np.copy(gauss_image)
hist,bins = np.histogram(tmp_img, bins=256)
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.figure(figsize=(10, 5))
plt.hist(tmp_img, bins=256)
plt.plot(cdf_normalized, color = 'r')

print(f'mean: { np.mean(gauss_image) }')
print(f'std: { np.std(gauss_image) }')

# %%
# https://docs.opencv.org/3.4/d5/daf/tutorial_py_histogram_equalization.html
# https://en.wikipedia.org/wiki/Histogram_equalization

img = cv2.normalize(gauss_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
img = cv2.equalizeHist(img)
# img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.figure(figsize=(10, 5))
plt.plot(img)

plt.figure(figsize=(10, 5))
plt.plot(img > (256 * porosity))

tmp_img = np.copy(img)
hist,bins = np.histogram(tmp_img, bins=256)
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.figure(figsize=(10, 5))
plt.hist(tmp_img, bins=256)
plt.plot(cdf_normalized, color = 'r')

print(f'mean: { np.mean(img) }')
print(f'std: { np.std(img) }')

# %%
sigmas = [1, 3, 10, 30, 100]
porosity = 0.5

plt.figure(figsize=(10, 5))

for sigma in sigmas:
    
    gauss_image = ndimage.gaussian_filter(image, sigma=sigma, truncate=4)

    tmp_img = np.copy(gauss_image)
    hist,bins = np.histogram(tmp_img, bins=256)
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    plt.hist(tmp_img, bins=256)
    plt.plot(cdf_normalized)


# %%
