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
from skimage import filters
from skimage import morphology

# %%
sample_path = '/Users/grimax/Desktop/tmp/porous sample/tomo_rec_cut.h5'
sample_file = h5py.File(sample_path, mode='r')
sample = sample_file['Reconstruction']


# %%
def image_stats(image):
    im_shape = image.shape
    im_min = np.min(image)
    im_max = np.max(image)
    im_mean = np.mean(image)
    im_std = np.std(image)
    print(f'shape: {im_shape}')
    print(f'min: {im_min:.2f}')
    print(f'max: {im_max:.2f}')
    print(f'mean: {im_mean:.2f}')
    print(f'std: {im_std:.2f}')
    return im_shape, im_min, im_max, im_mean, im_std


# %%
sample_shape, sample_min, sample_max, sample_mean, sample_std = image_stats(sample)


# %%
def plot_histogram(data, xmin=None, xmax=None, log=False):
    plt.figure(figsize=(15, 5))
    plt.hist(data, bins=255, color='lightgray', log=log)
    xmin and xmax and plt.xlim(xmin, xmax)


def get_histogram(image, xmin=None, xmax=None):
    img = np.ravel(image)
    plot_histogram(img, xmin, xmax)
    plot_histogram(img, xmin, xmax, True)


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
def plot_column(image, x=0, y=0):
    plt.figure(figsize=(15, 5))
    plt.plot(image[:, y, x])


# %%
get_histogram(sample, xmin=sample_min, xmax=sample_max)

# %%
x_slice = 150
y_slice = 0
z_slice = 25
show_2d_sections(sample, x=x_slice, y=y_slice, z=z_slice)

# %%
plot_column(sample)

# %%
sample_filtered = filters.median(sample, morphology.cube(5))
sample_f_shape, sample_f_min, sample_f_max, sample_f_mean, sample_f_std = image_stats(sample_filtered)

# %%
get_histogram(sample_filtered, xmin=sample_f_min, xmax=sample_f_max)

# %%
show_2d_sections(sample_filtered, x=x_slice, y=y_slice, z=z_slice)

# %%
plot_column(sample_filtered)

# %%
sample_diff = sample - sample_filtered
sample_diff_shape, sample_diff_min, sample_diff_max, sample_diff_mean, sample_diff_std = image_stats(sample_diff)

# %%
get_histogram(sample_diff, xmin=sample_diff_min, xmax=sample_diff_max)

# %%
show_2d_sections(sample_diff, x=x_slice, y=y_slice, z=z_slice)

# %%
plot_column(sample_diff)

# %%
