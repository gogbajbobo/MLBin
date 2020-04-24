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

    result[result < otsu_level] = False
    result[result >= otsu_level] = True

    porosity = 1 - np.count_nonzero(result)/result.size

    print(f'Otsu level: {otsu_level}')
    print(f'Porosity: {porosity}')

    return result
