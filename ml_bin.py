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
import h5py
import matplotlib.pyplot as plt
import numpy as np

# %%
sample_path = '/Users/grimax/Desktop/tmp/porous sample/tomo_rec_cut.h5'
sample_file = h5py.File(sample_path, mode='r')
sample = sample_file['Reconstruction']
sample_min = np.min(sample)
sample_max = np.max(sample)
sample_mean = np.mean(sample)
sample_std = np.std(sample)
print(f'shape: {sample.shape}')
print(f'min: {sample_min:.2f}')
print(f'max: {sample_max:.2f}')
print(f'mean: {sample_mean:.2f}')
print(f'std: {sample_std:.2f}')


# %%
def show_histogram(image):
    img = np.ravel(image)
    plt.figure(figsize=(15, 5))
    plt.hist(img, bins=255, color='lightgray')
    plt.figure(figsize=(15, 5))
    plt.hist(img, bins=255, color='lightgray', log=True)


# %%
def show_2d_image(image, fig, axis, vmin, vmax, title):
    im = axis.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    axis.set_title(title)
    fig.colorbar(im, ax=axis)


def show_2d_sections(image, x=0, y=0, z=0):
    z_section = image[z, :, :]
    y_section = image[:, y, :]
    x_section = image[:, :, x]
    sections_array = np.vstack((x_section, y_section, z_section))
    vmin, vmax = np.min(sections_array), np.max(sections_array)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    show_2d_image(z_section, fig, axes[0], vmin, vmax, 'z')
    show_2d_image(y_section, fig, axes[1], vmin, vmax, 'y')
    show_2d_image(x_section, fig, axes[2], vmin, vmax, 'x')


# %%
show_histogram(sample)

# %%
show_2d_sections(sample, x=150, z=25)

# %%
