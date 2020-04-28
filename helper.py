import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import morphology
import porespy as ps


def get_data_from_file(file_path, group_name, type=None):
    file = h5py.File(file_path, mode='r')
    result = file[group_name][()].astype(np.bool) if type == 'bool' else file[group_name][()]
    return result


def save_data_to_file(file_path, group_name, data):
    file = h5py.File(file_path, mode='w')
    file.create_dataset(group_name, data=data, compression='lzf')


def filter_image(image, width=3):
    return filters.median(image, morphology.cube(width))


def erosion(image):
    return morphology.binary_erosion(image, morphology.cube(3))


def dilation(image):
    return morphology.binary_dilation(image, morphology.cube(3))


def get_stats(image):
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
    plt.hist(data, bins=256, color='lightgray', log=log)
    xmin and xmax and plt.xlim(xmin, xmax)
    return plt


def show_histogram(image, xmin=None, xmax=None, log=False):
    img = np.ravel(image)
    plot_histogram(img, xmin, xmax)
    if log:
        plot_histogram(img, xmin, xmax, log)


def show_histogram_with_vline(image, vlines, xmin=None, xmax=None, log=False):
    img = np.ravel(image)
    _plt = plot_histogram(img, xmin, xmax)
    for vline in vlines:
        _plt.axvline(x=vline, color='blue')
    if log:
        plot_histogram(img, xmin, xmax, log)
        for vline in vlines:
            _plt.axvline(x=vline, color='blue')


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


def show_3d_image(image, title=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.voxels(image, edgecolor='k')
    plt.title(title or '3d preview')


def plot_line(line):
    plt.figure(figsize=(15, 5))
    plt.plot(line)


def mod_otsu(histogram):
    n_bins = np.size(histogram)
    zero_class_count = 0
    zero_class_sum = 0
    total_count = np.sum(histogram)
    total_sum = np.sum(np.arange(n_bins) * histogram)
    total_mean = total_sum / total_count
    total_variance = np.sum((np.arange(n_bins) - total_mean) ** 2
                            * histogram) / total_count
    criteria = np.zeros(np.size(histogram) - 1, dtype=np.float64)
    for bin_index, bin_value in enumerate(histogram[:-1]):
        zero_class_count += bin_value
        zero_class_sum += bin_index * bin_value
        zero_class_probability = zero_class_count / total_count
        zero_class_mean = zero_class_sum / zero_class_count
        first_class_count = total_count - zero_class_count
        first_class_sum = total_sum - zero_class_sum
        first_class_probability = 1 - zero_class_probability
        first_class_mean = first_class_sum / first_class_count
        variance_between = zero_class_probability * first_class_probability \
                           * (zero_class_mean - first_class_mean) ** 2
        variance_within = total_variance - variance_between
        criteria[bin_index] \
            = zero_class_probability * np.log(zero_class_probability) \
              + first_class_probability * np.log(first_class_probability) \
              - np.log(variance_within)
    otsu_level = np.argmax(criteria) + 1

    return otsu_level, criteria


def binarize_image(image):

    result = np.copy(image)
    result -= np.min(result)
    result /= np.max(result)
    result = (result * 255).astype(np.uint8)

    hist, _ = np.histogram(result.ravel(), bins=256)
    otsu_level, _ = mod_otsu(hist)
    print(f'Otsu level: {otsu_level}')

    result[result < otsu_level] = False
    result[result >= otsu_level] = True

    calc_porosity(result)

    return result


def calc_porosity(image):
    porosity = 1 - np.count_nonzero(image)/image.size
    print(f'Porosity: {porosity}')
    return porosity


def fill_floating_solids_and_closed_pores(image):
    image_filled = np.copy(image)
    floating_solids = ps.filters.find_disconnected_voxels(image_filled)
    closed_pores = ps.filters.find_disconnected_voxels(~image_filled)
    print(f'Floating solids: {np.sum(floating_solids)}')
    print(f'Closed pores: {np.sum(closed_pores)}')
    image_filled[floating_solids] = False
    image_filled[closed_pores] = True
    return image_filled


def get_full_lengths_stats(bin_image):
    axes_el_lengths = np.array([])
    for axis in range(len(bin_image.shape)):
        axes_el_lengths = np.append(axes_el_lengths, get_lengths_stats(bin_image, axis))
    return axes_el_lengths


def get_lengths_stats(bin_image, axis):
    if 0 > axis > 2:
        raise ValueError('incorrect value of axis')
    el_lengths = np.array([])
    z, y, x = bin_image.shape
    range_1 = range(y) if axis == 0 else range(x) if axis == 1 else range(z)
    range_2 = range(x) if axis == 0 else range(z) if axis == 1 else range(y)
    for i in range_1:
        for j in range_2:
            stripe = get_stripe(bin_image, axis, i, j)
            boarders = stripe[1:] != stripe[:-1]
            boarders = np.append(boarders, True)
            indexes = np.where(boarders)[0] + 1
            line_elements = np.split(boarders, indexes)
            element_lengths = np.array([len(elem) for elem in line_elements])
            element_lengths = np.array(list(filter(lambda el: el, element_lengths)))
            el_lengths = np.append(el_lengths, element_lengths)
    return el_lengths


def get_stripe(image, axis, i ,j):
    if 0 > axis > 2:
        raise ValueError('incorrect value of axis')
    return np.copy(image[:, i, j]) if axis == 0 else np.copy(image[i, :, j]) if axis == 1 else np.copy(image[i, j, :])
