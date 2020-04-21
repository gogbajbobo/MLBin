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
def show_2d_sections(image, x=None, y=None, z=None):
    shape = image.shape
    x = x or shape[0]//2
    y = y or shape[1]//2
    z = z or shape[2]//2
    vmin = np.min(image)
    vmax = np.max(image)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(image[x, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(image[:, y, :], cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(image[:, :, z], cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(im2, ax=axes[2])


# %%
show_histogram(sample)

# %%
show_2d_sections(sample)

# %%
