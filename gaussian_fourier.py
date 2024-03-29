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
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import interpolate
from scipy import stats
from scipy import odr
import cv2
import seaborn as sns

# %%
size = 10_000_000
range = np.arange(size)

# %%
sigma = 20
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

# %%
plt.figure(figsize=(10, 5))
plt.plot(half_range, fft_abs_norm)
plt.semilogx()

cut_off_f = 1 / (2 * np.pi * sigma)
cut_off_period = 1 // cut_off_f
print(f'cut_off_f: {cut_off_f:.3}, cut_off_period: {cut_off_period}, half: {cut_off_period // 2}')
cut_off_f_fwhm = np.sqrt(2 * np.log(2)) / (2 * np.pi * sigma)
cut_off_period_fwhm = 1 // cut_off_f_fwhm
print(f'cut_off_f_fwhm: {cut_off_f_fwhm:.3}, cut_off_period_fwhm: {cut_off_period_fwhm}, half: {cut_off_period_fwhm // 2}')

cut_off_f_e = np.sqrt(2 * np.log(np.e)) / (2 * np.pi * sigma)

plt.axvline(x=cut_off_f, color='gray')
plt.axvline(x=cut_off_f_fwhm, color='green')
plt.axvline(x=cut_off_f_e, color='black')

# %%
noise_image = np.random.random(size)

plt.figure(figsize=(15, 5))
plt.plot(noise_image[:500])

# %%
image = ndimage.gaussian_filter(noise_image, sigma=sigma, truncate=4)
plt.figure(figsize=(15, 5))
plt.plot(image[:500])

# %%
bin_image = image >= 0.5
plt.figure(figsize=(15, 5))
plt.plot(bin_image[:500])

# %%
borders = bin_image[1:] != bin_image[:-1]
borders = np.append(borders, True)
indexes = np.where(borders)[0] + 1
line_elements = np.split(borders, indexes)
element_lengths = np.array([len(elem) for elem in line_elements])[:-1]
# element_lengths = np.array(list(filter(lambda el: el, element_lengths)))
# el_lengths = np.append(el_lengths, element_lengths)

print(f'element_lengths: {element_lengths}, sum: {np.sum(element_lengths)}')


# %%
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


# %%
plt.figure(figsize=(15, 5))
plt.xlim([0, 500])
# plt.semilogx()
hist, edges, bars = plt.hist(element_lengths, bins=np.max(element_lengths))

# x_range = np.arange(np.max(element_lengths))
# interpolate_f = interpolate.interp1d(x_range, hist, kind='cubic')
# plt.plot(interpolate_f(x_range))

ma_size = np.max(element_lengths) // 100
ma = moving_average(hist, ma_size)
plt.plot(ma, 'r-', linewidth=3)

# tck = interpolate.splrep(x_range, hist, s=0)
# ynew = interpolate.splev(x_range, tck, der=0)
# plt.plot(ynew, color='red')

max_indicies_hist = np.where(hist == np.max(hist))
max_indicies_ma = np.where(ma == np.max(ma))
max_x_hist = np.round(edges[max_indicies_hist[0]])
max_x_ma = np.round(edges[max_indicies_ma[0]])
print(f'max_x_hist: {max_x_hist}, max_x_ma: {max_x_ma}')

# %%
plt.figure(figsize=(15, 5))
plt.xlim([0, 500])
sns.histplot(element_lengths, bins=np.max(element_lengths), kde=True)

# %%
fit_func = stats.invgamma

params = fit_func.fit(element_lengths)
print(params)
fit_values = fit_func.pdf(edges, *params)
max_fit_value_index = np.where(fit_values == np.max(fit_values))
print(max_fit_value_index)

plt.figure(figsize=(15, 5))
plt.plot(hist/np.sum(hist))
plt.plot(ma/np.sum(ma))
plt.plot(edges, fit_values)
plt.semilogx()

# %%
x = edges[:-1]
y = hist

poly_model = odr.polynomial(3)
data = odr.Data(x, y)
odr_obj = odr.ODR(data, poly_model)
output = odr_obj.run()

poly = np.poly1d(output.beta[::-1])
poly_y = poly(x)

plt.figure(figsize=(15, 5))
plt.plot(x, y)
plt.plot(x, poly_y)
plt.semilogx()

# %%
# https://en.wikipedia.org/wiki/Gaussian_filter
# use FWHM

period = np.mean(max_x_ma) * 2
freq = 1 / period
freq_sigma = freq / np.sqrt(2 * np.log(2))
calc_sigma =  1 // (freq_sigma * 2 * np.pi)

print(f'calc_sigma: {calc_sigma}')


# short version

period = np.mean(max_x_ma) * 2
calc_sigma = period * np.sqrt(2 * np.log(2)) // (2 * np.pi)
# calc_sigma = period * np.sqrt(2 * np.log(np.e)) // (2 * np.pi) # np.e is better as 2

print(f'calc_sigma: {calc_sigma}')

# %%
eq_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
eq_image = cv2.equalizeHist(eq_image)
eq_image = eq_image / np.max(eq_image)

plt.figure(figsize=(15, 5))
plt.plot(eq_image[:500])

bin_image = eq_image >= 0.3
plt.figure(figsize=(15, 5))
plt.plot(bin_image[:500])

# %%
borders = bin_image[1:] != bin_image[:-1]
borders = np.append(borders, True)
indexes = np.where(borders)[0] + 1

line_elements = np.split(bin_image, indexes) # use bin_image instead of borders earlier
element_lengths = np.array([len(elem) for elem in line_elements])[:-1]

print(f'element_lengths: {element_lengths}, sum: {np.sum(element_lengths)}')

line_elements = np.array([elem.flatten() for elem in line_elements], dtype=object)[:-1]

true_elements = filter(lambda x: x[0] == True, line_elements)
true_element_lengths = np.array([len(elem) for elem in true_elements])

false_elements = filter(lambda x: x[0] == False, line_elements)
false_element_lengths = np.array([len(elem) for elem in false_elements])

print(f'true_element_lengths: {true_element_lengths}, sum: {np.sum(true_element_lengths)}')
print(f'false_element_lengths: {false_element_lengths}, sum: {np.sum(false_element_lengths)}')

# %%
plt.figure(figsize=(15, 5))
max_value = np.max(true_element_lengths)
hist, edges, bars = plt.hist(true_element_lengths, bins=max_value)

ma_size = max_value // 100
ma = moving_average(hist, ma_size)
plt.plot(ma)

max_indicies_hist = np.where(hist == np.max(hist))
max_indicies_ma = np.where(ma == np.max(ma))
max_x_hist = np.round(edges[max_indicies_hist[0]])
max_x_ma = np.round(edges[max_indicies_ma[0]])
print(f'max_x_hist: {max_x_hist}, max_x_ma: {max_x_ma}')

# %%
fit_func = stats.gengamma

params = fit_func.fit(true_element_lengths)
a, c, loc, scale = params
print(a, c, loc, scale)
fit_values = fit_func.pdf(edges, *params)
max_fit_value_index = np.where(fit_values == np.max(fit_values))
print(max_fit_value_index)

plt.figure(figsize=(15, 5))
plt.plot(hist/np.sum(hist))
plt.plot(ma/np.sum(ma))
plt.plot(edges, fit_values)
plt.semilogx()

# %%
plt.figure(figsize=(15, 5))
max_value = np.max(false_element_lengths)
hist, edges, bars = plt.hist(false_element_lengths, bins=max_value)

ma_size = max_value // 100
ma = moving_average(hist, ma_size)
plt.plot(ma)

max_indicies_hist = np.where(hist == np.max(hist))
max_indicies_ma = np.where(ma == np.max(ma))
max_x_hist = np.round(edges[max_indicies_hist[0]])
max_x_ma = np.round(edges[max_indicies_ma[0]])
print(f'max_x_hist: {max_x_hist}, max_x_ma: {max_x_ma}')

# %%
fit_func = stats.gengamma

params = fit_func.fit(false_element_lengths)
a, c, loc, scale = params
print(a, c, loc, scale)
fit_values = fit_func.pdf(edges, a, c, loc, scale)

max_fit_value_index = np.where(fit_values == np.max(fit_values))
print(max_fit_value_index)

plt.figure(figsize=(15, 5))
plt.plot(hist/np.sum(hist))
plt.plot(ma/np.sum(ma))
plt.plot(edges, fit_values)
plt.semilogx()

# %%
bin_image = eq_image >= 0.7
plt.figure(figsize=(15, 5))
plt.plot(bin_image[:500])

# %%
borders = bin_image[1:] != bin_image[:-1]
borders = np.append(borders, True)
indexes = np.where(borders)[0] + 1

line_elements = np.split(bin_image, indexes) # use bin_image instead of borders earlier
element_lengths = np.array([len(elem) for elem in line_elements])[:-1]

print(f'element_lengths: {element_lengths}, sum: {np.sum(element_lengths)}')

line_elements = np.array([elem.flatten() for elem in line_elements], dtype=object)[:-1]

true_elements = filter(lambda x: x[0] == True, line_elements)
true_element_lengths = np.array([len(elem) for elem in true_elements])

false_elements = filter(lambda x: x[0] == False, line_elements)
false_element_lengths = np.array([len(elem) for elem in false_elements])

print(f'true_element_lengths: {true_element_lengths}, sum: {np.sum(true_element_lengths)}')
print(f'false_element_lengths: {false_element_lengths}, sum: {np.sum(false_element_lengths)}')

# %%
plt.figure(figsize=(15, 5))
max_value = np.max(true_element_lengths)
hist, edges, bars = plt.hist(true_element_lengths, bins=max_value)

ma_size = max_value // 100
ma = moving_average(hist, ma_size)
plt.plot(ma)

max_indicies_hist = np.where(hist == np.max(hist))
max_indicies_ma = np.where(ma == np.max(ma))
max_x_hist = np.round(edges[max_indicies_hist[0]])
max_x_ma = np.round(edges[max_indicies_ma[0]])
print(f'max_x_hist: {max_x_hist}, max_x_ma: {max_x_ma}')

# %%
plt.figure(figsize=(15, 5))
max_value = np.max(false_element_lengths)
hist, edges, bars = plt.hist(false_element_lengths, bins=max_value)

ma_size = max_value // 100
ma = moving_average(hist, ma_size)
plt.plot(ma)

max_indicies_hist = np.where(hist == np.max(hist))
max_indicies_ma = np.where(ma == np.max(ma))
max_x_hist = np.round(edges[max_indicies_hist[0]])
max_x_ma = np.round(edges[max_indicies_ma[0]])
print(f'max_x_hist: {max_x_hist}, max_x_ma: {max_x_ma}')

# %%
bin_image = eq_image >= 0.1
plt.figure(figsize=(15, 5))
plt.plot(bin_image[:500])

# %%
borders = bin_image[1:] != bin_image[:-1]
borders = np.append(borders, True)
indexes = np.where(borders)[0] + 1

line_elements = np.split(bin_image, indexes) # use bin_image instead of borders earlier
element_lengths = np.array([len(elem) for elem in line_elements])[:-1]

print(f'element_lengths: {element_lengths}, sum: {np.sum(element_lengths)}')

line_elements = np.array([elem.flatten() for elem in line_elements], dtype=object)[:-1]

true_elements = filter(lambda x: x[0] == True, line_elements)
true_element_lengths = np.array([len(elem) for elem in true_elements])

false_elements = filter(lambda x: x[0] == False, line_elements)
false_element_lengths = np.array([len(elem) for elem in false_elements])

print(f'true_element_lengths: {true_element_lengths}, sum: {np.sum(true_element_lengths)}')
print(f'false_element_lengths: {false_element_lengths}, sum: {np.sum(false_element_lengths)}')

# %%
plt.figure(figsize=(15, 5))
max_value = np.max(true_element_lengths)
hist, edges, bars = plt.hist(true_element_lengths, bins=max_value)

ma_size = max_value // 100
ma = moving_average(hist, ma_size)
plt.plot(ma)

max_indicies_hist = np.where(hist == np.max(hist))
max_indicies_ma = np.where(ma == np.max(ma))
max_x_hist = np.round(edges[max_indicies_hist[0]])
max_x_ma = np.round(edges[max_indicies_ma[0]])
print(f'max_x_hist: {max_x_hist}, max_x_ma: {max_x_ma}')

# %%
plt.figure(figsize=(15, 5))
max_value = np.max(false_element_lengths)
hist, edges, bars = plt.hist(false_element_lengths, bins=max_value)

ma_size = max_value // 100
ma = moving_average(hist, ma_size)
plt.plot(ma)

max_indicies_hist = np.where(hist == np.max(hist))
max_indicies_ma = np.where(ma == np.max(ma))
max_x_hist = np.round(edges[max_indicies_hist[0]])
max_x_ma = np.round(edges[max_indicies_ma[0]])
print(f'max_x_hist: {max_x_hist}, max_x_ma: {max_x_ma}')

# %%
