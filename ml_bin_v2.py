# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.1
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
phantom_recon = moe.process_image(phantom, 180, reconstruct_filter='ramp', noise_parameter=30, noise_method='poisson', detector_blurring=True)
phantom_recon_shape, phantom_recon_min, phantom_recon_max, phantom_recon_mean, phantom_recon_std = helper.get_stats(phantom_recon)
helper.show_2d_sections(phantom_recon, x=x_slice, y=y_slice, z=z_slice)
helper.show_histogram(phantom_recon, xmin=phantom_recon_min, xmax=phantom_recon_max, log=True)

# %%
helper.scatter_plot(phantom_recon, phantom, 'phantom_recon vs phantom')

# %%
phantom_recon_bin, _ = helper.binarize_image(phantom_recon)
floating_solids = helper.get_floating_solids(phantom_recon_bin)
print(f'Floating solids: {np.sum(floating_solids)}')
# helper.show_2d_sections(phantom_recon_bin, x=x_slice, y=y_slice, z=z_slice)

# %%
helper.scatter_plot(phantom_recon, phantom_recon_bin, 'phantom_recon vs phantom_recon_bin')

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
helper.scatter_plot(phantom_recon, phantom_recon_bin_erosion, 'phantom_recon vs phantom_recon_bin_erosion')
helper.scatter_plot(phantom_recon, phantom_recon_bin_dilation, 'phantom_recon vs phantom_recon_bin_dilation')

# %%
phantom_recon_convolve = ndimage.convolve(phantom_recon, helper.convolve_kern(phantom_recon))

lr, x_, y_ = logreg.train_logreg_model(phantom_recon, phantom_recon_convolve, phantom)
logreg.logreg_predict(lr, x_, y_)

# %%
helper.scatter_plot_with_logreg_line(phantom_recon, phantom, 'phantom_recon vs phantom', lr)

# %%
lr, x_, y_ = logreg.train_logreg_model(phantom_recon, phantom_recon_convolve, phantom_recon_bin_erosion)
logreg.logreg_predict(lr, x_, y_)
helper.scatter_plot_with_logreg_line(phantom_recon, phantom_recon_bin_erosion, 'phantom_recon vs phantom_recon_bin_erosion', lr)

# %%
lr, x_, y_ = logreg.train_logreg_model(phantom_recon, phantom_recon_convolve, phantom_recon_bin_dilation)
logreg.logreg_predict(lr, x_, y_)
helper.scatter_plot_with_logreg_line(phantom_recon, phantom_recon_bin_dilation, 'phantom_recon vs phantom_recon_bin_dilation', lr)

# %%
