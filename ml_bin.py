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

# %%
sample = helper.get_data_from_file('/Users/grimax/Desktop/tmp/porous sample/tomo_rec_cut.h5', 'Reconstruction')

# %%
sample_shape, sample_min, sample_max, sample_mean, sample_std = helper.image_stats(sample)

# %%
helper.show_histogram(sample, xmin=sample_min, xmax=sample_max)

# %%
x_slice = 150
y_slice = 0
z_slice = 25
helper.show_2d_sections(sample, x=x_slice, y=y_slice, z=z_slice)

# %%
helper.plot_column(sample)

# %%
sample_filtered_path = '/Users/grimax/Desktop/tmp/porous sample/sample_filtered.h5'

# sample_filtered = helper.filter_image(sample)
# helper.save_data_to_file(sample_filtered_path, 'Filtered', sample_filtered)

sample_filtered = helper.get_data_from_file(sample_filtered_path, 'Filtered')


sample_f_shape, sample_f_min, sample_f_max, sample_f_mean, sample_f_std = helper.image_stats(sample_filtered)

# %%
helper.show_histogram(sample_filtered, xmin=sample_f_min, xmax=sample_f_max)

# %%
helper.show_2d_sections(sample_filtered, x=x_slice, y=y_slice, z=z_slice)

# %%
helper.plot_column(sample_filtered)

# %%
sample_diff = sample - sample_filtered
sample_diff_shape, sample_diff_min, sample_diff_max, sample_diff_mean, sample_diff_std = helper.image_stats(sample_diff)

# %%
helper.get_histogram(sample_diff, xmin=sample_diff_min, xmax=sample_diff_max)

# %%
helper.show_2d_sections(sample_diff, x=x_slice, y=y_slice, z=z_slice)

# %%
helper.plot_column(sample_diff)
