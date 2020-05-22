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
import model_of_experiment as moe
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# %%
size = 250
dim = 3
shape = tuple(size for _ in range(dim))
phantom = avg.blobs(shape=shape, porosity=0.5, random_seed=3)
phantom = helper.fill_floating_solids_and_closed_pores(phantom)

# %%
sample_shape, sample_min, sample_max, sample_mean, sample_std = helper.get_stats(phantom)

# %%
x_slice = size // 2
y_slice = size // 2
z_slice = size // 2

helper.show_2d_sections(phantom, x=x_slice, y=y_slice, z=z_slice)

# %%
phantom_recon_180 = moe.process_image(phantom, 180)
helper.show_2d_sections(phantom_recon_180, x=x_slice, y=y_slice, z=z_slice)

# %%
phantom_recon_180_blurring = moe.process_image(phantom, 180, detector_blurring=True)
helper.show_2d_sections(phantom_recon_180_blurring, x=x_slice, y=y_slice, z=z_slice)

# %%
noise_method = 'poisson'
noise_parameter = 30

# %%
phantom_recon_180_noise_30 = moe.process_image(phantom, 180, noise_parameter=noise_parameter, noise_method=noise_method)
helper.show_2d_sections(phantom_recon_180_noise_30, x=x_slice, y=y_slice, z=z_slice)


# %%
phantom_recon_180_noise_30_blurring = moe.process_image(phantom, 180, noise_parameter=noise_parameter, noise_method=noise_method, detector_blurring=True)
helper.show_2d_sections(phantom_recon_180_noise_30_blurring, x=x_slice, y=y_slice, z=z_slice)

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
helper.show_2d_image(phantom_recon_180_noise_30[0:125, 125, 0:125], fig, axes[0], None, None, None)
helper.show_2d_image(phantom_recon_180_noise_30_blurring[0:125, 125, 0:125], fig, axes[1], None, None, None)

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
helper.show_2d_image(phantom_recon_180_noise_30[0:75, 125, 0:75], fig, axes[0], None, None, None)
helper.show_2d_image(phantom_recon_180_noise_30_blurring[0:75, 125, 0:75], fig, axes[1], None, None, None)

# %%
helper.show_histogram(phantom_recon_180_noise_30, log=True)
helper.show_histogram(phantom_recon_180_noise_30_blurring, log=True)

# %%
recon_shape, recon_min, recon_max, recon_mean, recon_std = helper.get_stats(phantom_recon_180_noise_30)
recon_blurring_shape, recon_blurring_min, recon_blurring_max, recon_blurring_mean, recon_blurring_std = helper.get_stats(phantom_recon_180_noise_30_blurring)

# %%
noise_parameter = 100
phantom_recon_180_noise_30 = moe.process_image(phantom, 180, noise_parameter=noise_parameter, noise_method=noise_method)
helper.show_2d_sections(phantom_recon_180_noise_30, x=x_slice, y=y_slice, z=z_slice)

# %%
phantom_recon_180_noise_30_blurring = moe.process_image(phantom, 180, noise_parameter=noise_parameter, noise_method=noise_method, detector_blurring=True)
helper.show_2d_sections(phantom_recon_180_noise_30_blurring, x=x_slice, y=y_slice, z=z_slice)

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
helper.show_2d_image(phantom_recon_180_noise_30[0:125, 125, 0:125], fig, axes[0], None, None, None)
helper.show_2d_image(phantom_recon_180_noise_30_blurring[0:125, 125, 0:125], fig, axes[1], None, None, None)

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
helper.show_2d_image(phantom_recon_180_noise_30[0:75, 125, 0:75], fig, axes[0], None, None, None)
helper.show_2d_image(phantom_recon_180_noise_30_blurring[0:75, 125, 0:75], fig, axes[1], None, None, None)

# %%
helper.show_histogram(phantom_recon_180_noise_30, log=True)
helper.show_histogram(phantom_recon_180_noise_30_blurring, log=True)

# %%
recon_shape, recon_min, recon_max, recon_mean, recon_std = helper.get_stats(phantom_recon_180_noise_30)
recon_blurring_shape, recon_blurring_min, recon_blurring_max, recon_blurring_mean, recon_blurring_std = helper.get_stats(phantom_recon_180_noise_30_blurring)

# %%
