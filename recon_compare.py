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

# %%
size = 250
dim = 3
shape = tuple(size for _ in range(dim))
phantom = avg.blobs(shape=shape, porosity=0.7)

# %%
sample_shape, sample_min, sample_max, sample_mean, sample_std = helper.get_stats(phantom)

# %%
x_slice = 125
y_slice = 125
z_slice = 125

helper.show_2d_sections(phantom, x=x_slice, y=y_slice, z=z_slice)

# %%
phantom_recon_45 = moe.process_image(phantom, 45)
phantom_recon_90 = moe.process_image(phantom, 90)
phantom_recon_180 = moe.process_image(phantom, 180)

# %%
helper.show_2d_sections(phantom_recon_45, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_90, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_180, x=x_slice, y=y_slice, z=z_slice)

# %%
reconstruct_filter='ramp'
phantom_recon_45_ramp = moe.process_image(phantom, 45, reconstruct_filter=reconstruct_filter)
phantom_recon_90_ramp = moe.process_image(phantom, 90, reconstruct_filter=reconstruct_filter)
phantom_recon_180_ramp = moe.process_image(phantom, 180, reconstruct_filter=reconstruct_filter)

# %%
helper.show_2d_sections(phantom_recon_45_ramp, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_90_ramp, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_180_ramp, x=x_slice, y=y_slice, z=z_slice)

# %%
noise_method = 'poisson'
noise_parameter = 100
phantom_recon_45_ramp_noise_100 = moe.process_image(
    phantom, 45, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_90_ramp_noise_100 = moe.process_image(
    phantom, 90, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_180_ramp_noise_100 = moe.process_image(
    phantom, 180, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)

helper.show_2d_sections(phantom_recon_45_ramp_noise_100, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_90_ramp_noise_100, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_180_ramp_noise_100, x=x_slice, y=y_slice, z=z_slice)


reconstruct_filter='shepp-logan'
phantom_recon_45_shepp_logan = moe.process_image(phantom, 45, reconstruct_filter=reconstruct_filter)
phantom_recon_90_shepp_logan = moe.process_image(phantom, 90, reconstruct_filter=reconstruct_filter)
phantom_recon_180_shepp_logan = moe.process_image(phantom, 180, reconstruct_filter=reconstruct_filter)

helper.show_2d_sections(phantom_recon_45_shepp_logan, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_90_shepp_logan, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_180_shepp_logan, x=x_slice, y=y_slice, z=z_slice)

phantom_recon_45_shepp_logan_noise_100 = moe.process_image(
    phantom, 45, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_90_shepp_logan_noise_100 = moe.process_image(
    phantom, 90, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_180_shepp_logan_noise_100 = moe.process_image(
    phantom, 180, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)

helper.show_2d_sections(phantom_recon_45_shepp_logan_noise_100, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_90_shepp_logan_noise_100, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_180_shepp_logan_noise_100, x=x_slice, y=y_slice, z=z_slice)


reconstruct_filter='hamming'
phantom_recon_45_hamming = moe.process_image(phantom, 45, reconstruct_filter=reconstruct_filter)
phantom_recon_90_hamming = moe.process_image(phantom, 90, reconstruct_filter=reconstruct_filter)
phantom_recon_180_hamming = moe.process_image(phantom, 180, reconstruct_filter=reconstruct_filter)

helper.show_2d_sections(phantom_recon_45_hamming, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_90_hamming, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_180_hamming, x=x_slice, y=y_slice, z=z_slice)

phantom_recon_45_hamming_noise_100 = moe.process_image(
    phantom, 45, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_90_hamming_noise_100 = moe.process_image(
    phantom, 90, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_180_hamming_noise_100 = moe.process_image(
    phantom, 180, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)

helper.show_2d_sections(phantom_recon_45_hamming_noise_100, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_90_hamming_noise_100, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_180_hamming_noise_100, x=x_slice, y=y_slice, z=z_slice)


# %%
reconstruct_sart=True

phantom_recon_45_sart = moe.process_image(phantom, 45, reconstruct_sart=reconstruct_sart)
phantom_recon_90_sart = moe.process_image(phantom, 90, reconstruct_sart=reconstruct_sart)
phantom_recon_180_sart = moe.process_image(phantom, 180, reconstruct_sart=reconstruct_sart)

helper.show_2d_sections(phantom_recon_45_sart, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_90_sart, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_180_sart, x=x_slice, y=y_slice, z=z_slice)

phantom_recon_45_sart_noise_100 = moe.process_image(
    phantom, 45, reconstruct_sart=reconstruct_sart, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_90_sart_noise_100 = moe.process_image(
    phantom, 90, reconstruct_sart=reconstruct_sart, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_180_sart_noise_100 = moe.process_image(
    phantom, 180, reconstruct_sart=reconstruct_sart, noise_parameter=noise_parameter, noise_method=noise_method
)

helper.show_2d_sections(phantom_recon_45_sart_noise_100, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_90_sart_noise_100, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_180_sart_noise_100, x=x_slice, y=y_slice, z=z_slice)

# %%
noise_parameter = 30

phantom_recon_45_hamming_noise_30 = moe.process_image(
    phantom, 45, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_90_hamming_noise_30 = moe.process_image(
    phantom, 90, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_180_hamming_noise_30 = moe.process_image(
    phantom, 180, reconstruct_filter=reconstruct_filter, noise_parameter=noise_parameter, noise_method=noise_method
)

helper.show_2d_sections(phantom_recon_45_hamming_noise_30, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_90_hamming_noise_30, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_180_hamming_noise_30, x=x_slice, y=y_slice, z=z_slice)


phantom_recon_45_sart_noise_30 = moe.process_image(
    phantom, 45, reconstruct_sart=reconstruct_sart, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_90_sart_noise_30 = moe.process_image(
    phantom, 90, reconstruct_sart=reconstruct_sart, noise_parameter=noise_parameter, noise_method=noise_method
)
phantom_recon_180_sart_noise_30 = moe.process_image(
    phantom, 180, reconstruct_sart=reconstruct_sart, noise_parameter=noise_parameter, noise_method=noise_method
)

helper.show_2d_sections(phantom_recon_45_sart_noise_30, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_90_sart_noise_30, x=x_slice, y=y_slice, z=z_slice)
helper.show_2d_sections(phantom_recon_180_sart_noise_30, x=x_slice, y=y_slice, z=z_slice)


# %%
