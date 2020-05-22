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
import numpy as np
import matplotlib.pyplot as plt

# %%
size = 250
dim = 3
shape = tuple(size for _ in range(dim))
random_seed = 3
number_of_angles = 180
noise_method = 'poisson'
source_blurring = False
detector_blurring = True

x_slice = size // 2
y_slice = size // 2
z_slice = size // 2


def show_phantoms_stats(phantom, processed_phantom, otsu_level=None):
    
    porous = (phantom == 1).sum()
    stones = (phantom == 0).sum()

    ph = np.ravel(phantom)
    p_ph = np.ravel(processed_phantom)
    p_ph_porous = p_ph[ph == 1]
    p_ph_stones = p_ph[ph == 0]
    
    plt.figure(figsize=(20, 5))
    p_range = [p_ph.min(), p_ph.max()]
    plt.hist(p_ph, 255, p_range, color='lightgray')
    plt.hist(p_ph_porous, 255, p_range, histtype='step', color='red')
    plt.hist(p_ph_stones, 255, p_range, histtype='step', color='blue')
    if otsu_level:
        plt.axvline(x=otsu_level, color='gray')


def generate_phantom(porosity, noise_parameter, threshold_levels=None):
    
    print(f'Porosity: {porosity:.2f}')
    print(f'Noise parameter: {noise_parameter}')
    
    phantom = avg.blobs(shape=shape, porosity=porosity, random_seed=random_seed)
    print(f'Generated porosity: {1 - np.sum(phantom)/phantom.size:.2f}')

    phantom[helper.get_floating_solids(phantom)] = False    
    helper.show_2d_sections(phantom, x=x_slice, y=y_slice, z=z_slice)

    phantom_recon = moe.process_image(
        phantom, 
        number_of_angles, 
        reconstruct_filter='ramp', 
        noise_parameter=noise_parameter, 
        noise_method=noise_method, 
        source_blurring=source_blurring, 
        detector_blurring=detector_blurring
    )
    helper.show_2d_sections(phantom_recon, x=x_slice, y=y_slice, z=z_slice)
    
#     plt.figure(figsize=(20, 20))
#     plt.imshow(phantom_recon[0, :, :], cmap='gray')

    phantom_recon -= np.min(phantom_recon)
    phantom_recon /= np.max(phantom_recon)
    phantom_recon = (phantom_recon * 255).astype(np.uint8)

    phantom_recon_bin, otsu_level = helper.binarize_image(phantom_recon)
    helper.show_2d_sections(phantom_recon_bin, x=x_slice, y=y_slice, z=z_slice)
    recon_bin_porosity = 1 - np.sum(phantom_recon_bin)/phantom_recon_bin.size
        
    show_phantoms_stats(phantom, phantom_recon, otsu_level)

    floating_solids = helper.get_floating_solids(phantom_recon_bin)
    closed_pores = helper.get_closed_pores(phantom_recon_bin)
    floating_solids_count = np.sum(floating_solids)
    closed_pores_count = np.sum(closed_pores)
    
    print(f'Floating solids: {floating_solids_count}')
    print(f'Closed pores: {closed_pores_count}')
    print('\n')
    
    if threshold_levels is not None:
        for threshold_level in threshold_levels:
            phantom_recon_bin[phantom_recon < threshold_level] = False
            phantom_recon_bin[phantom_recon >= threshold_level] = True
            helper.show_2d_sections(phantom_recon_bin, x=x_slice, y=y_slice, z=z_slice)
            floating_solids = helper.get_floating_solids(phantom_recon_bin)
            closed_pores = helper.get_closed_pores(phantom_recon_bin)
            print(f'Threshold level: {threshold_level}')
            print(f'Binarized porosity: {1 - np.sum(phantom_recon_bin)/phantom_recon_bin.size:.2f}')
            print(f'Floating solids: {np.sum(floating_solids)}')
            print(f'Closed pores: {np.sum(closed_pores)}')
            print('\n')
    
    plt.show()
    return recon_bin_porosity, floating_solids_count, closed_pores_count


# porosities = [0.1, 0.3, 0.5, 0.7, 0.9]
# intensities = [10, 30, 100, 300, 1000]
# threshold_levels = None

porosities = [0.3]
intensities = [30]
threshold_levels = np.arange(100, 152, 2)

porosities_array = np.array([])
intensities_array = np.array([])
binarized_porosities_array = np.array([])
floating_solids_count_array = np.array([])
closed_pores_count_array = np.array([])

for p in porosities:
    for i in intensities:
        recon_bin_porosity, floating_solids_count, closed_pores_count = generate_phantom(p, i, threshold_levels)
        porosities_array = np.append(porosities_array, p)
        intensities_array = np.append(intensities_array, i)
        binarized_porosities_array = np.append(binarized_porosities_array, recon_bin_porosity)
        floating_solids_count_array = np.append(floating_solids_count_array, floating_solids_count)
        closed_pores_count_array = np.append(closed_pores_count_array, closed_pores_count)

data = np.array((porosities_array, intensities_array, binarized_porosities_array, floating_solids_count_array, closed_pores_count_array))
data = np.transpose(data)
np.savetxt(
    'floating_stones_evaluation.csv', 
    data, 
    delimiter=','
)


# %%
