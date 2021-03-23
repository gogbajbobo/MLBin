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
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import interpolate

# %%
size = 100001
range = np.arange(size)

# %%
sigma = 10
delta_image = np.zeros(size)
delta_image[size//2] = 1
image = ndimage.gaussian_filter(delta_image, sigma=sigma, truncate=4)
plt.figure(figsize=(10, 5))
plt.plot(image)
# image = np.sin(range/np.pi)
# plt.figure(figsize=(10, 5))
# plt.plot(image)

# %%
fft = np.fft.fftn(image)
fft_abs = np.fft.fftshift(np.log(1 + np.abs(fft)))
fft_phase = np.fft.fftshift(np.angle(fft))

half_range = (range - size//2) / size
half_range = half_range[size//2:]
fft_abs = fft_abs[size//2:]

fft_abs_norm = fft_abs / np.max(fft_abs)

plt.figure(figsize=(10, 5))
plt.plot(half_range, fft_abs_norm)
plt.semilogx()

cut_off_f = 1 / (2 * np.pi * sigma)
cut_off_period = 1 // cut_off_f
print(f'cut_off_f: {cut_off_f:.3}, cut_off_period: {cut_off_period}, cut_off_period half: {cut_off_period // 2}')
cut_off_f_fwhm = np.sqrt(2 * np.log(2)) / (2 * np.pi * sigma)
cut_off_period_fwhm = 1 // cut_off_f_fwhm
print(f'cut_off_f_fwhm: {cut_off_f_fwhm:.3}, cut_off_period_fwhm: {cut_off_period_fwhm}')

plt.axvline(x=cut_off_f, color='gray')
plt.axvline(x=cut_off_f_fwhm, color='green')

# %%
noise_image = np.random.random(size)

plt.figure(figsize=(10, 5))
plt.plot(noise_image)

# %%
image = ndimage.gaussian_filter(noise_image, sigma=sigma, truncate=4)
plt.figure(figsize=(10, 5))
plt.plot(image)

# %%
bin_image = image >= 0.5
plt.figure(figsize=(10, 5))
plt.plot(bin_image)

# %%
boarders = bin_image[1:] != bin_image[:-1]
boarders = np.append(boarders, True)
indexes = np.where(boarders)[0] + 1
line_elements = np.split(boarders, indexes)
element_lengths = np.array([len(elem) for elem in line_elements])[:-1]
# element_lengths = np.array(list(filter(lambda el: el, element_lengths)))
# el_lengths = np.append(el_lengths, element_lengths)


print(f'element_lengths: {element_lengths}, sum: {np.sum(element_lengths)}')


# %%
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


# %%
plt.figure(figsize=(15, 5))
hist, edges, bars = plt.hist(element_lengths, bins=np.max(element_lengths))

# x_range = np.arange(np.max(element_lengths))
# interpolate_f = interpolate.interp1d(x_range, hist, kind='cubic')
# plt.plot(interpolate_f(x_range))

ma_size = np.max(element_lengths) // 10
ma = moving_average(hist, ma_size)
plt.plot(ma)

# tck = interpolate.splrep(x_range, hist, s=0)
# ynew = interpolate.splev(x_range, tck, der=0)
# plt.plot(ynew, color='red')

max_indicies_hist = np.where(hist == np.max(hist))
max_indicies_ma = np.where(ma == np.max(ma))
max_x_hist = np.round(edges[max_indicies_hist[0]])
max_x_ma = np.round(edges[max_indicies_ma[0]])
print(f'max_x_hist: {max_x_hist}, max_x_ma: {max_x_ma}')

# %%
period = max_x_ma * 2
calc_sigma = period // (2 * np.pi)

print(f'calc_sigma: {calc_sigma}')

# %%
