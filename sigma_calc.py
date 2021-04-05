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
from scipy import stats
import matplotlib.pyplot as plt
import cv2


# %%
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


# %%
def hist_fit(element_lengths, edges):
    fit_func = stats.invgauss
    params = fit_func.fit(element_lengths)
    fit_values = fit_func.pdf(edges, *params)
    max_fit_value_index = np.where(fit_values == np.max(fit_values))[0]
    return max_fit_value_index


def calc_sigma(element_lengths):
    if len(element_lengths) == 0:
        print(f'empty element_lengths')
        return 0, 0
    max_value = np.max(element_lengths)
    hist, edges = np.histogram(element_lengths, bins=max_value)
    ma_size = (np.max(element_lengths) // 100) or 1
    ma = moving_average(hist, ma_size)
    max_indicies_hist = np.where(hist == np.max(hist))
    max_indicies_ma = np.where(ma == np.max(ma))
    max_x_hist = np.round(edges[max_indicies_hist[0]])
    max_x_ma = np.round(edges[max_indicies_ma[0]])
    
    max_x_fit = hist_fit(element_lengths, edges)

    period_hist = np.mean(max_x_hist)
    calc_sigma_hist = period_hist // (2 * np.pi)
    period_ma = np.mean(max_x_ma)
    calc_sigma_ma = period_ma // (2 * np.pi)
    period_fit = np.mean(max_x_fit)
    calc_sigma_fit = period_fit // (2 * np.pi)
    
    return calc_sigma_hist, calc_sigma_ma, calc_sigma_fit


def sigma_estimate_2(size=10_000_000, sigma=1, porosity=0.5):

    range = np.arange(size)
    noise_image = np.random.random(size)
    image = ndimage.gaussian_filter(noise_image, sigma=sigma, truncate=4)
    
    eq_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    eq_image = cv2.equalizeHist(eq_image)
    eq_image = eq_image / np.max(eq_image)

    bin_image = eq_image >= porosity

    borders = bin_image[1:] != bin_image[:-1]
    borders = np.append(borders, True)
    indexes = np.where(borders)[0] + 1
    line_elements = np.split(bin_image, indexes)
    line_elements = np.array([elem.flatten() for elem in line_elements], dtype=object)[:-1]

    true_elements = filter(lambda x: x[0] == True, line_elements)
    true_element_lengths = np.array([len(elem) for elem in true_elements])

    false_elements = filter(lambda x: x[0] == False, line_elements)
    false_element_lengths = np.array([len(elem) for elem in false_elements])
    
    if len(true_element_lengths) == 0 or len(false_element_lengths) == 0:
        print(f'line_elements: {line_elements}')
        print(f'true_elements: {true_elements}')
        print(f'false_elements: {false_elements}')

    true_sigma_h, true_sigma_ma, true_sigma_fit = calc_sigma(true_element_lengths)
    false_sigma_h, false_sigma_ma, false_sigma_fit = calc_sigma(false_element_lengths)
    
    return true_sigma_h + false_sigma_h, true_sigma_ma + false_sigma_ma, true_sigma_fit + false_sigma_fit


# %%
k = np.sqrt(2 * np.log(np.e))

def processing_sigma_2(sigma, porosity, x, y_h, y_m, y_f):
    sigma_e = sigma_estimate_2(sigma=sigma, porosity=porosity, size=10_000_000)
    x.append(sigma)
    y_h.append(sigma_e[0] * k)
    y_m.append(sigma_e[1] * k)
    y_f.append(sigma_e[2] * k)
    print(f'porosity: {porosity}, sigma: {sigma}, calc: {sigma_e}')

def plot_scatter(x, y_h, y_m, y_f):
    plt.figure(figsize=(10, 10))
    plt.plot(x, x, color='gray')
    plt.scatter(x, y_h, color='blue', marker='x')
    plt.scatter(x, y_m, color='red', marker='x')
    plt.scatter(x, y_f, color='green', marker='x')
    
for porosity in np.arange(0.1, 1.0, 0.2):
    x, y_h, y_m, y_f = [], [], [], []
    for sigma in np.arange(10, 101, 10, dtype=np.int):
        processing_sigma_2(sigma, porosity, x, y_h, y_m, y_f)
    plot_scatter(x, y_h, y_m, y_f)

# %%
