import numpy as np
from skimage.transform import radon, iradon, iradon_sart
from helper import crop
from scipy import ndimage


def process_image(
        image,
        num_of_angles,
        noise_parameter=0,
        noise_method=None,
        reconstruct_sart=False,
        reconstruct_filter=None,
        detector_blurring=False,
        source_blurring=False
):
    sim = create_sinogram(num_of_angles, image)
    print(f'sinogram shape: {sim.shape}')
    if noise_method is None:
        pass
    elif noise_method == 'poisson':
        # sim = add_poisson_noise(sim, noise_parameter)
        sim, _ = add_poisson_noise_physical(sim, nominal_intensity=noise_parameter)
    else:
        raise ValueError('unknown noise_method param')
    if source_blurring:
        sim = sinogram_source_blurring(sim)
    if detector_blurring:
        sim = sinogram_detector_blurring(sim)
    rec = reconstruct(sim, reconstruct_sart, reconstruct_filter)
    print(f'reconstruction shape: {rec.shape}')
    return crop(rec, image.shape)


def create_sinogram(num_of_angles, img):
    set_of_angles = np.linspace(0, 180, num_of_angles, endpoint=False)
    if len(img.shape) == 3:
        sim = np.asarray([radon(img_slice, theta=set_of_angles, circle=False) for img_slice in img])
    else:
        sim = radon(img, theta=set_of_angles, circle=False) 
    
    return sim


def reconstruct(sinogram, reconstruct_sart=False, filt_name=None):

    def iradon_recon(s, angles, filt):
        if reconstruct_sart:
            return iradon_sart(s, angles)
        else:
            return iradon(s, angles, filter=filt)

    if len(sinogram.shape) == 3:
        set_of_angles = np.linspace(0, 180, sinogram[0].shape[1], endpoint=False)
        image = [iradon_recon(s, set_of_angles, filt_name) for s in sinogram]
    else:
        set_of_angles = np.linspace(0, 180, sinogram.shape[1], endpoint=False)
        image = iradon_recon(sinogram, set_of_angles, filt_name)

    image = np.asarray(image)

    return image


def add_poisson_noise(sinogram, intensity):

    sinogram /= 100.0
    I_0 = intensity * np.ones(sinogram.shape)
    I_0 = np.random.poisson(lam=I_0).astype('float32')
    sinogram = np.exp(-sinogram) * I_0

    exposition = 10.0

    I_0_arr = intensity * exposition * np.ones(sinogram.shape)
    I_0_arr = np.random.poisson(lam=I_0_arr).astype('float32') / exposition

    sinogram = np.log(I_0_arr / sinogram)
    sinogram *= 100.0
    sinogram = np.ceil(sinogram)

    return sinogram


def add_poisson_noise_physical(sinogram, atten_coef=0.25, pixel_size=0.01, nominal_intensity=100, frame_count=5):

    I0_empty = np.zeros(sinogram.shape)
    for i in range(frame_count):
        I0_empty += np.random.poisson(lam=nominal_intensity, size=sinogram.shape).astype('float32')
    I0_empty /= frame_count

    I0 = np.random.poisson(lam=nominal_intensity, size=sinogram.shape).astype('float32')
    material_length = sinogram * pixel_size
    experimental_sinogram = (I0 * np.exp(-atten_coef * material_length)).astype(int)
    experimental_sinogram[experimental_sinogram < 1] = 1

    noised_radon_sinogram = np.log(I0_empty / experimental_sinogram) / pixel_size / atten_coef

    return noised_radon_sinogram, experimental_sinogram


def sinogram_source_blurring(sinograms):
    _sinograms = np.copy(sinograms).astype(np.float)
    dim = len(_sinograms.shape)
    num_of_angles = _sinograms.shape[2] if dim == 3 else _sinograms.shape[1]
    source_object_distance = 1  # all values in meters
    object_detector_distance = 5e-2
    source_size = 100e-6
    pixel_size = 10e-6
    source_image_size = source_size * object_detector_distance / source_object_distance
    source_image_size_in_pixels = source_image_size / pixel_size
    gauss_sigma = source_image_size_in_pixels / (2 * np.sqrt(2 * np.log(2)))
    print(f'source blurring sigma: {gauss_sigma}')
    result = np.empty_like(_sinograms)
    for angle in np.arange(num_of_angles):
        projection = _sinograms[:, :, angle] if dim == 3 else _sinograms[:, angle]
        projection_dim = len(projection.shape)
        for axis in np.arange(projection_dim):
            output = ndimage.gaussian_filter1d(projection, gauss_sigma, axis)
            projection = output
        result[:, :, angle] = projection
    return result


def sinogram_detector_blurring(sinograms):
    _sinograms = np.copy(sinograms).astype(np.float)
    dim = len(_sinograms.shape)
    num_of_angles = _sinograms.shape[2] if dim == 3 else _sinograms.shape[1]
    gauss_sigma = 0.56
    print(f'detector blurring sigma: {gauss_sigma}')
    result = np.empty_like(_sinograms)
    for angle in np.arange(num_of_angles):
        projection = _sinograms[:, :, angle] if dim == 3 else _sinograms[:, angle]
        projection_dim = len(projection.shape)
        for axis in np.arange(projection_dim):
            output = ndimage.gaussian_filter1d(projection, gauss_sigma, axis)
            projection = output
        result[:, :, angle] = projection
    return result
