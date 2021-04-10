# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from scipy import stats
from scipy.signal import convolve2d


# %%
def get_line_elements(row):
    borders = row[1:] != row[:-1]
    borders = np.append(borders, True)
    indices = np.where(borders)[0] + 1
    line_elements = np.split(row, indices)
    line_elements = np.array([elem.flatten() for elem in line_elements], dtype=object)[:-1]
    line_elements = line_elements[1:-1] # rm lines on edges
    return line_elements


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


# %%
def calc_sigma(element_lengths):
    if len(element_lengths) == 0:
        print(f'empty element_lengths')
        return 0, 0, 0
    max_value = np.max(element_lengths)
    hist, edges = np.histogram(element_lengths, bins=max_value)    
    
    print(f'max_value {max_value}')
    plt.plot(hist)
    plt.semilogx()
    
    ma_size = (np.max(element_lengths) // 100) or 1
    ma = moving_average(hist, ma_size)
    max_indices_hist = np.where(hist == np.max(hist))
    max_indices_ma = np.where(ma == np.max(ma))
    max_x_hist = np.round(edges[max_indices_hist[0]])
    max_x_ma = np.round(edges[max_indices_ma[0]])

    max_x_fit = hist_fit(element_lengths, edges)

    period_hist = np.mean(max_x_hist)
    calc_sigma_hist = period_hist / (2 * np.pi)
    period_ma = np.mean(max_x_ma)
    calc_sigma_ma = period_ma / (2 * np.pi)
    period_fit = np.mean(max_x_fit)
    calc_sigma_fit = period_fit / (2 * np.pi)
    
    return calc_sigma_hist, calc_sigma_ma, calc_sigma_fit


# %%
def get_sigma_from_bin_image(bin_img):
    y_range = bin_img.shape[0]
    line_elements = np.array([])

    for i in np.arange(y_range):
        row = bin_img[i, :]
        line_elements = np.append(line_elements, get_line_elements(row))

    true_elements = filter(lambda x: x[0] == True, line_elements)
    true_element_lengths = np.array([len(elem) for elem in true_elements])

    false_elements = filter(lambda x: x[0] == False, line_elements)
    false_element_lengths = np.array([len(elem) for elem in false_elements])

    print(f'true_element_lengths {true_element_lengths}')
    print(f'false_element_lengths {false_element_lengths}')

    true_sigma_h, true_sigma_ma, true_sigma_fit = calc_sigma(true_element_lengths)
    false_sigma_h, false_sigma_ma, false_sigma_fit = calc_sigma(false_element_lengths)
    
    print('true_sigma_h, true_sigma_ma, true_sigma_fit')
    print(true_sigma_h, true_sigma_ma, true_sigma_fit)
    
    print('false_sigma_h, false_sigma_ma, false_sigma_fit')
    print(false_sigma_h, false_sigma_ma, false_sigma_fit)

    k = np.sqrt(2 * np.log(np.e))

    sigma_by_max_value = k * (true_sigma_h + false_sigma_h)
    sigma_by_max_of_ma = k * (true_sigma_ma + false_sigma_ma)
    sigma_by_max_of_fit = k * (true_sigma_fit + false_sigma_fit)

    print(sigma_by_max_value, sigma_by_max_of_ma, sigma_by_max_of_fit)


# %%
path = '/Users/grimax/Downloads/sample.h5'
with h5py.File(path, 'r') as h5f:
    img2d = np.array(h5f['layer1'][:])

plt.imshow(img2d)
print(np.max(img2d), np.min(img2d))

# %%
thresh = threshold_otsu(img2d)
bin_img = img2d >= thresh
plt.imshow(bin_img)

# %%
get_sigma_from_bin_image(bin_img)


# %%
def calc_filtered_image(img2d, filter_width):

    filtered_image = convolve2d(img2d, np.ones((filter_width, filter_width)), mode='same')
    thresh = threshold_otsu(filtered_image)
    bin_img = filtered_image >= thresh
    print(np.max(filtered_image), np.min(filtered_image))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(filtered_image)
    axes[1].imshow(bin_img)

    plt.figure()
    get_sigma_from_bin_image(bin_img)


# %%
calc_filtered_image(img2d, 3)

# %%
calc_filtered_image(img2d, 5)

# %%
calc_filtered_image(img2d, 7)

# %%
calc_filtered_image(img2d, 9)

# %%
