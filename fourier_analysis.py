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

# %%
sample = helper.get_data_from_file('/Users/grimax/Desktop/tmp/porous sample/sample.h5', 'Reconstruction')
sample_shape, sample_min, sample_max, sample_mean, sample_std = helper.get_stats(sample)

# %%
sample_fft = np.fft.fftn(sample)

# %%
sample_fft_abs = np.fft.fftshift(np.log(1 + np.abs(sample_fft)))
sample_fft_phase = np.fft.fftshift(np.angle(sample_fft))

# %%
helper.show_2d_sections(sample_fft_abs, x=125, y=125, z=125)
helper.show_2d_sections(sample_fft_phase, x=125, y=125, z=125)

# %%
b, a = signal.butter(40, 0.5, 'high')
w, h = signal.freqz(b, a)
plt.plot(w, np.abs(h))

# %%
sample_ifft = np.fft.ifftn(sample_fft_abs).real

# %%
helper.show_2d_sections(sample, x=125, y=125, z=125)
helper.show_2d_sections(sample_ifft, x=125, y=125, z=125)

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
helper.show_2d_image(sample[125, 0:125, 0:125], fig, axes[0], None, None, None)
helper.show_2d_image(sample_ifft[125, 0:125, 0:125], fig, axes[1], None, None, None)

# %%
