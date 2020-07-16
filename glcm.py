# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% id="qTvP5IEis3BU" colab_type="code" colab={}
import h5py
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as skimf
import scipy.stats
import helper

# %% id="OBfxNFzutGvq" colab_type="code" colab={}
file_dir = '/Users/grimax/Desktop/tmp/porous sample 2/'
file = h5py.File(f'{file_dir}sample.h5', mode='r')

# %% id="aq9V9ivMuDa7" colab_type="code" colab={}
data = file['Reconstruction'][()]

# %% id="5PiB93tFxl3k" colab_type="code" colab={}
data_int = data - np.min(data)
data_int = 255 * data_int / np.max(data_int)
data_int = data_int.astype(np.uint8)

# %% id="Ij2yrjqCyJZ1" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="10c0e1d6-e287-4e19-b6c5-db169460dbcf"
file_to_save = h5py.File(f'{file_dir}sample_int.h5', mode='w')
file_to_save.create_dataset('Reconstruction', data=data_int, compression='lzf')

# %% id="Tlan_ddGy4JE" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 611} outputId="0b4c5713-3ff9-4a19-8bef-fafe347e9325"
plt.figure(figsize=(10, 10))
plt.imshow(data_int[:, :, 0])

# %%
data_bin, _ = helper.binarize_image(data_int)
plt.figure(figsize=(10, 10))
plt.imshow(data_bin[:, :, 0])

# %%
data_bin_erosion = helper.erosion(data_bin)
plt.figure(figsize=(10, 10))
plt.imshow(data_bin_erosion[:, :, 0])

data_bin_dilation = helper.dilation(data_bin)
plt.figure(figsize=(10, 10))
plt.imshow(data_bin_dilation[:, :, 0])

# %%
data_stones = np.copy(data_int)
data_stones[data_bin_erosion == False] = 0
plt.figure(figsize=(10, 10))
plt.imshow(data_stones[:, :, 0])

# %%
data_pores = np.copy(data_int)
data_pores[data_bin_dilation == True] = 255
plt.figure(figsize=(10, 10))
plt.imshow(data_pores[:, :, 0])

# %% id="_9w49PKE0DmQ" colab_type="code" colab={}
result = np.zeros((256, 256))

for i in np.arange(250):
  result += skimf.greycomatrix(data_int[:, :, i], [1], [0], symmetric=True)[:, :, 0, 0]

# %% id="0Yw8aEgK2OrU" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 611} outputId="c23599f2-9c4d-4dcc-d38d-d9f9aa10f43f"
plt.figure(figsize=(10, 10))
plt.imshow(result)

# %% id="wgheTDjO37Yf" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="e7fc9fb6-f714-4112-f8a2-98a7dedc5692"
# for i in np.arange(256):
#   print(i, np.sum(result[i, :]), np.sum(result[:, i]))

# %% id="nuUhzQ-J64Ms" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 609} outputId="d096b4c3-138f-4aab-e22f-78f9667597fa"
plt.figure(figsize=(10, 10))
plt.yscale('log')
plt.plot(result[114, :])

# %% id="fShgZQwO76Qi" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="bfbabdc6-aef4-4112-9710-8216a936c6d7"
stds = []

for i in np.arange(256):
  res = result[i, :]
  hist_dist = scipy.stats.rv_histogram((res, np.arange(257)))
  stds.append(hist_dist.std())
#   print(i, hist_dist.mean(), hist_dist.std())

# %% id="oHA7VhLkAUAN" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 609} outputId="fc654f3f-90a7-46d3-d941-9fbbad3ebf88"
plt.figure(figsize=(10, 10))
plt.plot(stds)

# %%
result_stones = np.zeros((255, 255))

for i in np.arange(250):
  result_stones += skimf.greycomatrix(data_stones[:, :, i], [1], [0], symmetric=True)[1:, 1:, 0, 0]

plt.figure(figsize=(10, 10))
plt.imshow(result_stones)

# %%
result_pores = np.zeros((255, 255))

for i in np.arange(250):
  result_pores += skimf.greycomatrix(data_pores[:, :, i], [1], [0], symmetric=True)[:255, :255, 0, 0]

plt.figure(figsize=(10, 10))
plt.imshow(result_pores)


# %%
def h_glcm(arr, levels=256, symmetric=True, normed=True, cut=None):
    size = levels-1 if cut in ['start', 'end'] else levels
    h_glcm = np.zeros((size, size))
    for i in np.arange(arr.shape[0]):
        _a = arr[i, :, :]
        _glcm = skimf.greycomatrix(_a, [1], [0], levels=levels, symmetric=symmetric, normed=normed)
        _glcm = _glcm[:, :, 0, 0]
        h_glcm += _glcm[1:, 1:] if cut == 'start' else _glcm[:size, :size] if cut == 'end' else _glcm
    return h_glcm

def v_glcm(arr, levels=256, symmetric=True, normed=True, cut=None):
    size = levels-1 if cut in ['start', 'end'] else levels
    v_glcm = np.zeros((size, size))
    for i in np.arange(arr.shape[1]):
        _a = arr[:, i, :].T
        _glcm = skimf.greycomatrix(_a, [1], [0], levels=levels, symmetric=symmetric, normed=normed)
        _glcm = _glcm[:, :, 0, 0]
        v_glcm += _glcm[1:, 1:] if cut == 'start' else _glcm[:size, :size] if cut == 'end' else _glcm
    return v_glcm

def d_glcm(arr, levels=256, symmetric=True, normed=True, cut=None):
    size = levels-1 if cut in ['start', 'end'] else levels
    d_glcm = np.zeros((size, size))
    for i in np.arange(arr.shape[2]):
        _a = arr[:, :, i]
        _glcm = skimf.greycomatrix(_a, [1], [0], levels=levels, symmetric=symmetric, normed=normed)
        _glcm = _glcm[:, :, 0, 0]
        d_glcm += _glcm[1:, 1:] if cut == 'start' else _glcm[:size, :size] if cut == 'end' else _glcm
    return d_glcm

def get_glcm(arr, levels=256, symmetric=True, normed=True, cut=None):
    kwargs = {'levels': levels, 'symmetric': symmetric, 'normed': normed, 'cut': cut}
    return h_glcm(arr, **kwargs), v_glcm(arr, **kwargs), d_glcm(arr, **kwargs)


# %%
a = np.array(
    [[[0, 0],
      [0, 0]],
     [[2, 2],
      [1, 1]]])
get_glcm(a, levels=3)

# %%
hcm, vcm, dcm = get_glcm(data_stones, cut='start')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(hcm)
axes[1].imshow(vcm)
axes[2].imshow(dcm)


# %%
get_glcm(data_pores, cut='end')

# %%