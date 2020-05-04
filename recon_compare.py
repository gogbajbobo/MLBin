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
phantom_recon_180_no_filter = moe.process_image(phantom, 180)
helper.show_2d_sections(phantom_recon_180_no_filter, x=x_slice, y=y_slice, z=z_slice)

reconstruct_filter='hamming'
phantom_recon_180_hamming = moe.process_image(phantom, 180, reconstruct_filter=reconstruct_filter)
helper.show_2d_sections(phantom_recon_180_hamming, x=x_slice, y=y_slice, z=z_slice)

reconstruct_sart=True
phantom_recon_180_sart = moe.process_image(phantom, 180, reconstruct_sart=reconstruct_sart)
helper.show_2d_sections(phantom_recon_180_sart, x=x_slice, y=y_slice, z=z_slice)

# %%
noise_method = 'poisson'
noise_parameter = 30

# %%
phantom_recon_180_no_filter_noise_30 = moe.process_image(phantom, 180, noise_parameter=noise_parameter, noise_method=noise_method)
helper.show_2d_sections(phantom_recon_180_no_filter_noise_30, x=x_slice, y=y_slice, z=z_slice)

phantom_recon_180_hamming_noise_30 = moe.process_image(
    phantom, 180, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)
helper.show_2d_sections(phantom_recon_180_hamming_noise_30, x=x_slice, y=y_slice, z=z_slice)

phantom_recon_180_sart_noise_30 = moe.process_image(
    phantom, 180, reconstruct_sart=reconstruct_sart, noise_parameter=noise_parameter, noise_method=noise_method
)
helper.show_2d_sections(phantom_recon_180_sart_noise_30, x=x_slice, y=y_slice, z=z_slice)


# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
helper.show_2d_image(phantom_recon_180_hamming_noise_30[0:125, 125, 0:125], fig, axes[0], None, None, None)
helper.show_2d_image(phantom_recon_180_sart_noise_30[0:125, 125, 0:125], fig, axes[1], None, None, None)

# %%
helper.show_histogram(phantom_recon_180_hamming_noise_30, log=True)
helper.show_histogram(phantom_recon_180_sart_noise_30, log=True)

# %%
recon_shape, recon_min, recon_max, recon_mean, recon_std = helper.get_stats(phantom_recon_180_hamming_noise_30)
recon_sart_shape, recon_sart_min, recon_sart_max, recon_sart_mean, recon_sart_std = helper.get_stats(phantom_recon_180_sart_noise_30)

# %%
helper.plot_column(phantom_recon_180_hamming_noise_30)
helper.plot_column(phantom_recon_180_sart_noise_30)

# %%
bin_recon = helper.binarize_image(phantom_recon_180_hamming_noise_30)
bin_recon_sart = helper.binarize_image(phantom_recon_180_sart_noise_30)

# %%
helper.show_2d_sections(bin_recon, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(bin_recon_sart, x=x_slice, y=y_slice, z=z_slice)

# %%
recon_fs = helper.get_floating_solids(bin_recon)
recon_sart_fs = helper.get_floating_solids(bin_recon_sart)
helper.show_2d_sections(recon_fs, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(recon_sart_fs, x=x_slice, y=y_slice, z=z_slice)

# %%
