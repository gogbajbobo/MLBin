import numpy as np
from scipy import special
from scipy import ndimage
import cv2


def blobs(
        shape: tuple = (50, 50, 50), porosity: float = 0.5, filt_size: int = 4,
        anisotropy_value: float = 1.0, anisotropy_vector: tuple = (0, 0, 0),
        random_seed: int = None, blobiness: float = 1.0,
        **kwargs
        ):
    """
    Function generate 2D or 3D image of porous structure of blobls
    Parameters:
    shape - (array_like) output image shape
    porosity - (float) measure of the void spaces in a material
    filt_size - (int) number of sigmas in filter size
    anisotropy_value — (float) anisotropy coefficient, 1 means no anisotropy
    anisotropy_vector — (tuple) anisotropy direction, (0, 0, 0) means no anisotropy
    random_seed - (int) is a number used to initialize a pseudorandom number generator
    """

    shape, dimensions = check_shape_param(shape)

    np.random.seed(random_seed)
    image = np.random.random(shape).astype(np.float32)

    sigma = np.mean(shape[:-1] if dimensions == 2 else shape) / (40 * blobiness)
    anisotropy_vector, anisotropy_vector_length = check_anisotropy_vector(anisotropy_vector, dimensions)

    if anisotropy_value == 1.0 or anisotropy_vector_length == 0:  # TODO: it is not good to compare floats
        image = ndimage.gaussian_filter(image, sigma=sigma, truncate=filt_size)

    elif anisotropy_vector.count(0) == dimensions - 1:
        av_array = np.array(anisotropy_vector)
        anisotropy_axis = int(np.where(av_array != 0)[0])
        for axis in range(dimensions):
            gauss_sigma = sigma * anisotropy_value if axis == anisotropy_axis else sigma
            output = ndimage.gaussian_filter1d(image, gauss_sigma, axis, truncate=filt_size)
            image = output

    else:
        gaussian_kernel = kernel_func(filt_size, sigma, dimensions, anisotropy_value, anisotropy_vector)

        # TODO: check the following "if" — seems it's useless
        if image.shape[-1] == 1:
            image = image[:, :, 0]
            gaussian_kernel = gaussian_kernel[:, :, 0]

        image = ndimage.convolve(image, gaussian_kernel)

    return porous_image(image, porosity)


def check_shape_param(shape):

    dimensions = np.size(shape)

    if dimensions == 1:
        shape = tuple(int(shape) for _ in range(3))
        dimensions = 3

    if 2 >= dimensions >= 3:
        raise ValueError('Image shape error')

    shape = np.array(shape)

    return shape, dimensions


def check_anisotropy_vector(anisotropy_vector, dimensions):

    vector_dim = np.size(anisotropy_vector)

    if dimensions == 2 and 2 <= vector_dim <= 3:
        anisotropy_vector = anisotropy_vector if vector_dim == 2 else anisotropy_vector[:-1]
        avx, avy = anisotropy_vector
        anisotropy_vector_length = np.sqrt(avx ** 2 + avy ** 2)
        return anisotropy_vector, anisotropy_vector_length

    if dimensions == 3 and vector_dim == 3:
        avx, avy, avz = anisotropy_vector
        anisotropy_vector_length = np.sqrt(avx ** 2 + avy ** 2 + avz ** 2)
        return anisotropy_vector, anisotropy_vector_length

    raise ValueError('Anisotropy vector values error')


def porous_image(image, porosity):

    if porosity:
        # image = norm_to_uniform(image, scale=[0, 1])
        image = histogram_equalization(image)
        image = image > porosity

    return image


def norm_to_uniform(image, scale=None):

    if scale is None:
        scale = [image.min(), image.max()]

    # TODO: have to check np.std(image) is not zero to avoid RuntimeWarning: invalid value encountered in true_divide
    image = (image - np.mean(image)) / np.std(image)
    image = 0.5 * special.erfc(-image/np.sqrt(2))
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (scale[1] - scale[0]) + scale[0]

    return image


def histogram_equalization(im):
    _im = np.copy(im)
    dim = np.size(_im.shape)
    if dim == 2:
        return histogram_equalization_2d(_im)
    elif dim == 3:
        for i in range(_im.shape[0]):
            im_slice = _im[i, :, :]
            im_slice = histogram_equalization_2d(im_slice)
            _im[i, :, :] = im_slice
        return _im
    else:
        raise ValueError('image dimension error')


def histogram_equalization_2d(image):
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.equalizeHist(img)
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img


def kernel_basis_2d(shape, sigma_arr, angle):
    semiwidth = shape // 2
    x, y = np.meshgrid(
            np.linspace(-1 * semiwidth, semiwidth, shape),
            np.linspace(-1 * semiwidth, semiwidth, shape)
        )

    x, y = rotate_space_2d(angle, [x, y])

    g_exp = np.exp(-1./2. * ((x / sigma_arr[0])**2. + (y / sigma_arr[1])**2.))
    g_denominator = 2. * np.pi * sigma_arr[0] * sigma_arr[1]

    gaussian_kernel = g_exp / g_denominator

    return gaussian_kernel


def kernel_basis_3d(shape, sigma_arr, angles):

    semiwidth = shape // 2
    x, y, z = np.meshgrid(
            np.linspace(-1 * semiwidth, semiwidth, shape),
            np.linspace(-1 * semiwidth, semiwidth, shape),
            np.linspace(-1 * semiwidth, semiwidth, shape)
        )
    if len(sigma_arr) == 2:
        x, y, z = np.meshgrid(
            np.linspace(-1 * semiwidth, semiwidth, shape),
            np.linspace(-1 * semiwidth, semiwidth, shape),
            np.linspace(-1 * semiwidth, semiwidth, 1)
        )
        sigma_arr = np.append(sigma_arr, 1)
        angles = np.append([angles], [0, 0])

    x, y, z = rotate_space_3d(angles, [x, y, z])

    g_exp = np.exp(-1./2. * ((x / sigma_arr[0])**2. + (y / sigma_arr[1])**2. + (z / sigma_arr[2])**2.))
    g_denominator = np.sqrt((2. * np.pi)**3) * sigma_arr[0] * sigma_arr[1] * sigma_arr[2]

    gaussian_kernel = g_exp / g_denominator

    return gaussian_kernel


def rotate_space_2d(angle, space):

    x, y = space

    r = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    x_r = r[0, 0] * x + r[1, 0] * y
    y_r = r[0, 1] * x + r[1, 1] * y

    return [x_r, y_r]


def rotate_space_3d(angles, space):

    x, y, z = space

    rx = [
        [1, 0, 0],
        [0, np.cos(angles[2]), -np.sin(angles[2])],
        [0, np.sin(angles[2]), np.cos(angles[2])]
        ]
    ry = [
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ]
    rz = [
        [np.cos(angles[0]), -np.sin(angles[0]), 0],
        [np.sin(angles[0]), np.cos(angles[0]), 0],
        [0, 0, 1]
        ]

    rot_mtrx = np.matmul(np.matmul(rz, ry), rx)

    x_r = rot_mtrx[0, 0] * x + rot_mtrx[1, 0] * y + rot_mtrx[2, 0] * z
    y_r = rot_mtrx[0, 1] * x + rot_mtrx[1, 1] * y + rot_mtrx[2, 1] * z
    z_r = rot_mtrx[0, 2] * x + rot_mtrx[1, 2] * y + rot_mtrx[2, 2] * z
    
    return [x_r, y_r, z_r]


def kernel_func(
        filt_size: int = 4,
        sigma: float = 2.5,
        dimensions: int = 3,
        anisotropy_value: float = 1.0,
        anisotropy_vector: tuple = (0, 0, 0)
        ):

    if dimensions == 2:
        x, y = anisotropy_vector
        angle = np.arccos(x / np.sqrt(x**2 + y**2))
        shape, sigma_arr = kernel_shape_and_sigmas(filt_size, sigma, anisotropy_value, dimensions)
        return kernel_basis_2d(shape, sigma_arr, angle)

    if dimensions == 3:
        x, y, z = anisotropy_vector
        angle_x = np.arccos(y / np.sqrt(y ** 2 + z ** 2))
        angle_y = np.arccos(z / np.sqrt(z ** 2 + x ** 2))
        angle_z = np.arccos(x / np.sqrt(x ** 2 + y ** 2))
        angles = (angle_x, angle_y, angle_z)
        shape, sigma_arr = kernel_shape_and_sigmas(filt_size, sigma, anisotropy_value, dimensions)
        return kernel_basis_3d(shape, sigma_arr, angles)

    raise ValueError('Kernel_func parameters error')


def kernel_shape_and_sigmas(filt_size, sigma, anisotropy_value, dimensions):
    shape = 2 * np.uint8(np.ceil(filt_size * sigma) * (anisotropy_value if anisotropy_value > 1.0 else 1)) + 1
    sigma_arr = np.ones(dimensions, dtype=int) * sigma
    sigma_arr[0] = sigma * anisotropy_value
    return shape, sigma_arr
