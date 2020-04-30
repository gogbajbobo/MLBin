# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import helper
import anisotropic_volume_generator as avg
from scipy import ndimage

# %%
recalculate = False

# %%
sample = helper.get_data_from_file('/Users/grimax/Desktop/tmp/porous sample/sample.h5', 'Reconstruction')

bin_sample_path = '/Users/grimax/Desktop/tmp/porous sample/bin_sample.h5'

if recalculate:
    bin_sample = helper.binarize_image(sample)
    helper.save_data_to_file(bin_sample_path, 'Binarized', bin_sample)

bin_sample = helper.get_data_from_file(bin_sample_path, 'Binarized', 'bool')
porosity = helper.calc_porosity(bin_sample)

bin_sample_filled = helper.fill_floating_solids_and_closed_pores(bin_sample)
helper.calc_porosity(bin_sample_filled)

sample_filtered_path = '/Users/grimax/Desktop/tmp/porous sample/sample_filtered.h5'

if recalculate:
    sample_filtered = helper.filter_image(sample, 9)
    helper.save_data_to_file(sample_filtered_path, 'Filtered', sample_filtered)

# %%
sample_filtered_path = '/Users/grimax/Desktop/tmp/porous sample/sample_filtered_median_9.h5'

sample_filtered = helper.get_data_from_file(sample_filtered_path, 'Filtered')

sample_diff = sample - sample_filtered

bin_sample_filtered_path = '/Users/grimax/Desktop/tmp/porous sample/bin_sample_filtered.h5'

if recalculate:
    bin_sample_filtered = helper.binarize_image(sample_filtered)
    helper.save_data_to_file(bin_sample_filtered_path, 'Binarized', bin_sample_filtered)

bin_sample_filtered = helper.get_data_from_file(bin_sample_filtered_path, 'Binarized', 'bool')
helper.calc_porosity(bin_sample_filtered)

bin_sample_filtered_filled = helper.fill_floating_solids_and_closed_pores(bin_sample_filtered)
helper.calc_porosity(bin_sample_filtered_filled)

# %%
sample_shape, sample_min, sample_max, sample_mean, sample_std = helper.get_stats(sample)
sample_f_shape, sample_f_min, sample_f_max, sample_f_mean, sample_f_std = helper.get_stats(sample_filtered)
sample_diff_shape, sample_diff_min, sample_diff_max, sample_diff_mean, sample_diff_std = helper.get_stats(sample_diff)
helper.get_stats(bin_sample)
helper.get_stats(bin_sample_filled)
helper.get_stats(bin_sample_filtered)
helper.get_stats(bin_sample_filtered_filled)

# %%
helper.show_histogram(sample, xmin=sample_min, xmax=sample_max, log=True)
helper.show_histogram(sample_filtered, xmin=sample_f_min, xmax=sample_f_max, log=True)
helper.show_histogram(sample_diff, xmin=sample_diff_min, xmax=sample_diff_max, log=True)

# %%
x_slice = 150
y_slice = 25
z_slice = 25

# %%
helper.show_2d_sections(sample, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(sample_filtered, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(sample_diff, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(bin_sample, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(bin_sample_filled, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(bin_sample_filtered, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(bin_sample_filtered_filled, x=x_slice, y=y_slice, z=z_slice)

# %%
helper.plot_column(sample)
helper.plot_column(sample_filtered)
helper.plot_column(sample_diff)
helper.plot_column(bin_sample)
helper.plot_column(bin_sample_filled)
helper.plot_column(bin_sample_filtered)
helper.plot_column(bin_sample_filtered_filled)

# %%
axes_elements_lengths = helper.get_full_lengths_stats(bin_image=bin_sample_filtered_filled)
helper.get_stats(axes_elements_lengths)

# %%
helper.show_histogram(axes_elements_lengths, log=False)

# %%
import numpy as np
el_len_bins = np.bincount(axes_elements_lengths.astype(np.uint8))
helper.plot_line(el_len_bins[1:] - el_len_bins[:-1])

# %%
test_sample = np.copy(sample[0:99, 0:99, 0:99])
threshold = 3.5
test_sample[test_sample < threshold] = 0
test_sample[test_sample >= threshold] = 1
print(np.sum(test_sample))
helper.show_3d_image(test_sample)

# %%
sample_stones = np.copy(sample)[bin_sample_filled == True]
sample_stones_shape, sample_stones_min, sample_stones_max, sample_stones_mean, sample_stones_std = helper.get_stats(sample_stones)

sample_pores = np.copy(sample)[bin_sample_filled == False]
sample_pores_shape, sample_pores_min, sample_pores_max, sample_pores_mean, sample_pores_std = helper.get_stats(sample_pores)

print(f'Diff: {np.prod(bin_sample_filtered_filled.shape) - (sample_stones_shape[0] + sample_pores_shape[0])}')

# %%
stones_hist = np.histogram(sample_stones, bins=256)
stones_peak = stones_hist[1][np.argmax(stones_hist[0])]
print(f'stones peak: {stones_peak}')
helper.show_histogram_with_vline(sample_stones, [sample_stones_mean, stones_peak], log=True)

pores_hist = np.histogram(sample_pores, bins=256)
pores_peak = pores_hist[1][np.argmax(pores_hist[0])]
print(f'pores peak: {pores_peak}')
helper.show_histogram_with_vline(sample_pores, [sample_pores_mean, pores_peak], log=True)

# %%
bin_sample_erosion = helper.erosion(bin_sample_filled)
bin_sample_dilation = helper.dilation(bin_sample_filled)

helper.calc_porosity(bin_sample_erosion)
helper.calc_porosity(bin_sample_dilation)

helper.show_2d_sections(bin_sample, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(bin_sample_erosion, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(bin_sample_dilation, x=x_slice, y=y_slice, z=z_slice)

# %%
stones_erosion = np.copy(sample)[bin_sample_erosion == True]
se_shape, se_min, se_max, se_mean, se_std = helper.get_stats(stones_erosion)

pores_dilation = np.copy(sample)[bin_sample_dilation == False]
pd_shape, pd_min, pd_max, pd_mean, pd_std = helper.get_stats(pores_dilation)

print(f'Diff: {np.prod(sample.shape) - (se_shape[0] + pd_shape[0])}')
print(f'Borders: {(np.prod(sample.shape) - (se_shape[0] + pd_shape[0])) / np.prod(sample.shape)}')

# %%
se_hist, se_edges = np.histogram(stones_erosion, bins=256)
se_peak = se_edges[np.argmax(se_hist)]
print(f'stones erosion peak: {se_peak}')
helper.show_histogram_with_vline(stones_erosion, [se_mean, se_peak], log=True)

pd_hist, pd_edges = np.histogram(pores_dilation, bins=256)
pd_peak = pd_edges[np.argmax(pd_hist)]
print(f'pores dilation peak: {pd_peak}')
helper.show_histogram_with_vline(pores_dilation, [pd_mean, pd_peak], log=True)

# %%
widths = se_edges[1:] - se_edges[:-1]
se_values = np.array([v + widths[i]/2 for i, v in np.ndenumerate(se_edges[:-1])])

widths = pd_edges[1:] - pd_edges[:-1]
pd_values = np.array([v + widths[i]/2 for i, v in np.ndenumerate(pd_edges[:-1])])

stones_pdf = se_hist / np.sum(se_hist)
pores_pdf = pd_hist / np.sum(pd_hist)

stones_pdf_test = np.random.choice(se_values, size=8457901, p=stones_pdf)
pores_pdf_test = np.random.choice(pd_values, size=3220295, p=pores_pdf)

helper.show_histogram(stones_pdf_test, log=True)
helper.show_histogram(pores_pdf_test, log=True)

# %%
phantom = avg.blobs(shape=sample.shape, porosity=porosity, blobiness=0.8)
helper.show_2d_sections(bin_sample_filtered, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom)

# %%
phantom_with_noise = np.zeros(phantom.shape)
phantom_with_noise[phantom == True] = np.random.choice(se_values, size=(phantom[phantom == True]).shape[0], p=stones_pdf)
phantom_with_noise[phantom == False] = np.random.choice(pd_values, size=(phantom[phantom == False]).shape[0], p=pores_pdf)

# %%
helper.show_2d_sections(sample, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_with_noise, vmin=-0.8, vmax=3.8)

# %%
helper.show_histogram(sample, log=True)
helper.show_histogram(phantom_with_noise, log=True)

# %%
phantom_with_noise_binarized = helper.binarize_image(phantom_with_noise)

# %%
phantom_with_noise_binarized_filled = helper.fill_floating_solids_and_closed_pores(phantom_with_noise_binarized)

# %%
helper.show_2d_sections(bin_sample, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_with_noise_binarized)

# %%
dim = len(phantom.shape)
kern_shape = tuple(1 for _ in range(dim))
kern = np.zeros(kern_shape)
kern = np.pad(kern, 1, constant_values=1)
kern /= 3 ** dim - 1

# %%
sample_convolute = ndimage.convolve(sample, kern)
phantom_convolute = ndimage.convolve(phantom_with_noise, kern)
helper.show_2d_sections(sample_convolute, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_convolute, vmin=-0.8, vmax=3.8)

# %%
im_lenght = np.prod(sample.shape)
coeff = 100
sample_lenght = (im_lenght / coeff).astype(np.int)
indices = (np.random.rand(sample_lenght) * im_lenght).astype(np.int)

# %%
x = sample.flatten()
xlim = (np.min(x), np.max(x))
y = sample_convolute.flatten()
ylim = (np.min(y), np.max(y))
origin = bin_sample.flatten()
helper.scatter_plot_values(x, y, origin, 'exp data', indices, xlim=xlim, ylim=ylim)

# %%
x = phantom_with_noise.flatten()
xlim = (np.min(x), np.max(x))
y = phantom_convolute.flatten()
ylim = (np.min(y), np.max(y))
origin = phantom_with_noise_binarized.flatten()
helper.scatter_plot_values(x, y, origin, 'phantom data', indices, xlim=xlim, ylim=ylim)

# %%
