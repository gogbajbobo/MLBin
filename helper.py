import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import morphology


def get_data_from_file(file_path, group_name):
    file = h5py.File(file_path, mode='r')
    return file[group_name]


def save_data_to_file(file_path, group_name, data):
    file = h5py.File(file_path, mode='w')
    file.create_dataset(group_name, data=data, compression='lzf')


def filter_image(image):
    return filters.median(image, morphology.cube(3))


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


def plot_histogram(data, xmin=None, xmax=None, log=False):
    plt.figure(figsize=(15, 5))
    plt.hist(data, bins=255, color='lightgray', log=log)
    xmin and xmax and plt.xlim(xmin, xmax)


def show_histogram(image, xmin=None, xmax=None):
    img = np.ravel(image)
    plot_histogram(img, xmin, xmax)
    plot_histogram(img, xmin, xmax, True)


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


def plot_column(image, x=0, y=0):
    plt.figure(figsize=(15, 5))
    plt.plot(image[:, y, x])
