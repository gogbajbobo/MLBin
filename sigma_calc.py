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
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


# %%
def sigma_estimate(size=10_000_000, sigma=1):

    range = np.arange(size)
    noise_image = np.random.random(size)
    image = ndimage.gaussian_filter(noise_image, sigma=sigma, truncate=4)
    bin_image = image >= 0.5

    boarders = bin_image[1:] != bin_image[:-1]
    boarders = np.append(boarders, True)
    indexes = np.where(boarders)[0] + 1
    line_elements = np.split(boarders, indexes)
    element_lengths = np.array([len(elem) for elem in line_elements])[:-1]

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'same') / w

    hist, edges = np.histogram(element_lengths, bins=np.max(element_lengths))
    ma_size = (np.max(element_lengths) // 100) or 1
    ma = moving_average(hist, ma_size)
    max_indicies_hist = np.where(hist == np.max(hist))
    max_indicies_ma = np.where(ma == np.max(ma))
    max_x_hist = np.round(edges[max_indicies_hist[0]])
    max_x_ma = np.round(edges[max_indicies_ma[0]])

    period_hist = np.mean(max_x_hist) * 2
    calc_sigma_hist = period_hist // (2 * np.pi)
    period_ma = np.mean(max_x_ma) * 2
    calc_sigma_ma = period_ma // (2 * np.pi)
    
#     csh = np.round(calc_sigma_hist * np.sqrt(2))
#     csm = np.round(calc_sigma_ma * np.sqrt(2)) # why np.sqrt(2)?
    
#     csh = np.round(calc_sigma_hist * np.sqrt(2 * np.log(2)))
#     csm = np.round(calc_sigma_ma * np.sqrt(2 * np.log(2))) # why np.sqrt(2 * np.log(2))?

    return calc_sigma_hist, calc_sigma_ma#, csh, csm


# %%
x = []
y_h = []
y_m = []
y_h_1 = []
y_m_1 = []
y_h_2 = []
y_m_2 = []
k1 = np.sqrt(2 * np.log(2))
k2 = np.sqrt(2)


def processing_sigma(sigma):
    sigma_e = sigma_estimate(sigma=sigma)
    x.append(sigma)
    y_h.append(sigma_e[0])
    y_m.append(sigma_e[1])
    y_h_1.append(sigma_e[0] * k1)
    y_m_1.append(sigma_e[1] * k1)
    y_h_2.append(sigma_e[0] * k2)
    y_m_2.append(sigma_e[1] * k2)
    print(f'sigma: {sigma}, calc: {sigma_e}')


for sigma in np.arange(1, 10, 1, dtype=np.int):
    processing_sigma(sigma)
    
for sigma in np.arange(10, 101, 10, dtype=np.int):
    processing_sigma(sigma)

# %%
plt.figure(figsize=(10, 10))
plt.plot(x, x, color='gray')
plt.scatter(x, y_h, color='blue')
plt.scatter(x, y_m, color='red')
plt.scatter(x, y_h_1, color='blue', marker='+')
plt.scatter(x, y_m_1, color='red', marker='+')
plt.scatter(x, y_h_2, color='blue', marker='x')
plt.scatter(x, y_m_2, color='red', marker='x')

# %%
