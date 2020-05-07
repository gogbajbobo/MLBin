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
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import anisotropic_volume_generator as avg

# %%
sample = helper.get_data_from_file('/Users/grimax/Desktop/tmp/porous sample/sample.h5', 'Reconstruction')
sample_shape, sample_min, sample_max, sample_mean, sample_std = helper.get_stats(sample)

# %%
sample_fft = np.fft.fftn(sample)
sample_fft_abs = np.fft.fftshift(np.log(1 + np.abs(sample_fft)))
sample_fft_phase = np.fft.fftshift(np.angle(sample_fft))

# %%
helper.show_2d_sections(sample_fft_abs, x=125, y=125, z=125)
helper.show_2d_sections(sample_fft_phase, x=125, y=125, z=125)

# %%
helper.plot_column(sample_fft_abs, x=125, y=125)

# %%
sigma = 25
gauss_filter = avg.kernel_basis_3d(250, [sigma, sigma, sigma], [0, 0, 0])
gauss_filter /= np.max(gauss_filter)
# helper.show_2d_sections(gauss_filter, x=125, y=125, z=125)
# helper.plot_column(gauss_filter, x=125, y=125)

# %%
gauss_filter_inverse = 1 - gauss_filter
# helper.show_2d_sections(gauss_filter_inverse, x=125, y=125, z=125)
# helper.plot_column(gauss_filter_inverse, x=125, y=125)

# %%
sample_fft_filtered = sample_fft * np.fft.ifftshift(gauss_filter_inverse)
sample_fft_filterd_abs = np.fft.fftshift(np.log(1 + np.abs(sample_fft_filtered)))
sample_fft_filtered_phase = np.fft.fftshift(np.angle(sample_fft_filtered))
# helper.show_2d_sections(sample_fft_filterd_abs, x=125, y=125, z=125)
# helper.show_2d_sections(sample_fft_filtered_phase, x=125, y=125, z=125)
# helper.plot_column(sample_fft_filterd_abs, x=125, y=125)

# %%
sample_ifft = np.fft.ifftn(sample_fft_filtered).real

# %%
helper.show_2d_sections(sample, x=125, y=125, z=125)
helper.show_2d_sections(sample_ifft, x=125, y=125, z=125)

helper.show_2d_sections(sample)
helper.show_2d_sections(sample_ifft)

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
helper.show_2d_image(sample[125:250, 125:250, 125], fig, axes[0], None, None, None)
helper.show_2d_image(sample_ifft[125:250, 125:250, 125], fig, axes[1], None, None, None)

# %%
sample_diff = sample - sample_ifft
helper.show_2d_sections(sample_diff, x=125, y=125, z=125)

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
helper.show_2d_image(sample[125, 0:125, 0:125], fig, axes[0], None, None, None)
helper.show_2d_image(sample_diff[125, 0:125, 0:125], fig, axes[1], None, None, None)

# %%
helper.save_data_to_file('/Users/grimax/Desktop/tmp/porous sample/sample_noise.h5', 'Noise', sample_ifft)

# %%
