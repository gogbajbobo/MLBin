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
import numpy as np
import model_of_experiment as moe
from scipy import ndimage
from sklearn.metrics import jaccard_score
import logreg

# %%
size = 250
dim = 3
shape = tuple(size for _ in range(dim))
phantom = avg.blobs(shape=shape, porosity=0.5, random_seed=3)

# %%
x_slice = size // 2
y_slice = size // 2
z_slice = size // 2

helper.show_2d_sections(phantom, x=x_slice, y=y_slice, z=z_slice)

# %%
floating_solids = helper.get_floating_solids(phantom)
print(f'Floating solids: {np.sum(floating_solids)}')
phantom[floating_solids] = False

# %%
phantom_recon = moe.process_image(phantom, 180, reconstruct_filter='ramp', noise_parameter=30, noise_method='poisson', projection_blurring=True)
phantom_recon_shape, phantom_recon_min, phantom_recon_max, phantom_recon_mean, phantom_recon_std = helper.get_stats(phantom_recon)
helper.show_2d_sections(phantom_recon, x=x_slice, y=y_slice, z=z_slice)
helper.show_histogram(phantom_recon, xmin=phantom_recon_min, xmax=phantom_recon_max, log=True)

# %%
dim = len(phantom.shape)
kern_shape = tuple(1 for _ in range(dim))
kern = np.ones(kern_shape)
kern = np.pad(kern, 1, constant_values=1)
kern /= 3 ** dim - 1

# %%
im_lenght = np.prod(phantom.shape)
coeff = 100
sample_lenght = (im_lenght / coeff).astype(np.int)
indices = (np.random.rand(sample_lenght) * im_lenght).astype(np.int)

# %%
figsize = (5, 5)
x = phantom_recon.flatten()
xlim = (np.min(x), np.max(x))
y = ndimage.convolve(phantom_recon, kern).flatten()
ylim = (np.min(y), np.max(y))
origin = phantom.flatten()
helper.scatter_plot_values(x, y, origin, 'phantom_recon vs phantom', indices, xlim=xlim, ylim=ylim, figsize=figsize)

# %%
phantom_recon_bin = helper.binarize_image(phantom_recon)
floating_solids = helper.get_floating_solids(phantom_recon_bin)
print(f'Floating solids: {np.sum(floating_solids)}')
# helper.show_2d_sections(phantom_recon_bin, x=x_slice, y=y_slice, z=z_slice)

# %%
origin = phantom_recon_bin.flatten()
helper.scatter_plot_values(x, y, origin, 'phantom_recon vs phantom_recon_bin', indices, xlim=xlim, ylim=ylim, figsize=figsize)

# %%
phantom_recon_bin_filled = helper.fill_floating_solids_and_closed_pores(phantom_recon_bin)

floating_solids_bin_filled = helper.get_floating_solids(phantom_recon_bin_filled)
print(f'Floating solids: {np.sum(floating_solids_bin_filled)}')

phantom_recon_bin_erosion = helper.erosion(phantom_recon_bin_filled)
phantom_recon_bin_dilation = helper.dilation(phantom_recon_bin_filled)

helper.show_2d_sections(phantom, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_bin_filled, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_bin_erosion, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_bin_dilation, x=x_slice, y=y_slice, z=z_slice)

# %%
origin = phantom_recon_bin_erosion.flatten()
helper.scatter_plot_values(x, y, origin, 'phantom_recon vs phantom_recon_bin_erosion', indices, xlim=xlim, ylim=ylim, figsize=figsize)

origin = phantom_recon_bin_dilation.flatten()
helper.scatter_plot_values(x, y, origin, 'phantom_recon vs phantom_recon_bin_dilation', indices, xlim=xlim, ylim=ylim, figsize=figsize)

# %%
phantom_recon_convolve = ndimage.convolve(phantom_recon, kern)

logreg.train_logreg_model(phantom_recon, phantom_recon_convolve, phantom)

# %%
phantom_recon_filtered = ndimage.convolve(phantom_recon, kern)
helper.get_stats(phantom_recon_filtered)
helper.show_2d_sections(phantom_recon_filtered, x=x_slice, y=y_slice, z=z_slice)
helper.show_histogram(phantom_recon_filtered, xmin=phantom_recon_min, xmax=phantom_recon_max, log=True)

# %%
x = phantom_recon_filtered.flatten()
xlim = (np.min(x), np.max(x))
y = ndimage.convolve(phantom_recon_filtered, kern).flatten()
ylim = (np.min(y), np.max(y))
origin = phantom.flatten()
helper.scatter_plot_values(x, y, origin, 'phantom_recon_filtered vs phantom', indices, xlim=xlim, ylim=ylim)

# %%
phantom_recon_filtered_bin = helper.binarize_image(phantom_recon_filtered)
floating_solids = helper.get_floating_solids(phantom_recon_filtered_bin)
print(f'Floating solids: {np.sum(floating_solids)}')
helper.show_2d_sections(phantom_recon_filtered_bin, x=x_slice, y=y_slice, z=z_slice)

# %%
origin = phantom_recon_filtered_bin.flatten()
helper.scatter_plot_values(x, y, origin, 'phantom_recon_filtered vs phantom_recon_filtered_bin', indices, xlim=xlim, ylim=ylim)

# %%

# %%
