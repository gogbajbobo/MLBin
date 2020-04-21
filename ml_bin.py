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
import numpy as np
# %%
sample_path = '/Users/grimax/Desktop/tmp/porous sample/tomo_rec_cut.h5'
sample_file = h5py.File(sample_path, mode='r')
sample = sample_file['Reconstruction']
sample_min = np.min(sample)
sample_max = np.max(sample)
sample_mean = np.mean(sample)
sample_std = np.std(sample)
print(f'shape: {sample.shape}')
print(f'min: {sample_min:.2f}')
print(f'max: {sample_max:.2f}')
print(f'mean: {sample_mean:.2f}')
print(f'std: {sample_std:.2f}')


# %%
