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
        projection_blurring=False
):
    sim = create_sinogram(num_of_angles, image)
    print(f'sinogram shape: {sim.shape}')
    if noise_method is None:
        pass
    elif noise_method == 'poisson':
        sim = add_poisson_noise(sim, noise_parameter)
    else:
        raise ValueError('unknown noise_method param')
    if projection_blurring:
        sim = sinogram_blurring(sim)
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


def sinogram_blurring(sinograms):
    _sinograms = np.copy(sinograms).astype(np.float)
    dim = len(_sinograms.shape)
    num_of_angles = _sinograms.shape[2] if dim == 3 else _sinograms.shape[1]
    gauss_sigma = 0.56
    result = np.empty_like(_sinograms)
    for angle in np.arange(num_of_angles):
        projection = _sinograms[:, :, angle] if dim == 3 else _sinograms[:, angle]
        print(projection)
        projection_dim = len(projection.shape)
        for axis in np.arange(projection_dim):
            output = ndimage.gaussian_filter1d(projection, gauss_sigma, axis)
            projection = output
        print(projection)
        result[:, :, angle] = projection
    return result