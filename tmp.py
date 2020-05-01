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
import matplotlib.pyplot as plt

# %%
sample = helper.get_data_from_file('/Users/grimax/Desktop/tmp/porous sample/tomo_rec_cut.h5', 'Reconstruction')

# %%
sample_shape, sample_min, sample_max, sample_mean, sample_std = helper.get_stats(sample)

# %%
helper.show_histogram(sample, xmin=sample_min, xmax=sample_max, log=True)

# %%
x_slice = 250
y_slice = 250
z_slice = 850

# %%
helper.show_2d_sections(sample, x=x_slice, y=y_slice, z=z_slice)

# %%
fig, axes = plt.subplots(1, 1, figsize=(20, 20))
helper.show_2d_image(sample[850, 250:500, 250:500], fig, axes, None, None, None)

# %%
phantom = helper.get_data_from_file('/Users/grimax/Desktop/tmp/porous sample/phantom_with_noise.h5', 'Phantom')

# %%
fig, axes = plt.subplots(1, 1, figsize=(20, 20))
helper.show_2d_image(phantom[125, :, :], fig, axes, None, None, None)

# %%
