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

# %% id="qTvP5IEis3BU" colab_type="code" colab={}
import h5py

# %% id="OBfxNFzutGvq" colab_type="code" colab={}
file = h5py.File('/content/sample.h5', mode='r')

# %% id="aq9V9ivMuDa7" colab_type="code" colab={}
data = file['Reconstruction'][()]

# %% id="X-YbgxC-w_uS" colab_type="code" colab={}
import numpy as np

# %% id="5PiB93tFxl3k" colab_type="code" colab={}
data_int = data - np.min(data)
data_int = 255 * data_int / np.max(data_int)
data_int = data_int.astype(np.uint8)

# %% id="Ij2yrjqCyJZ1" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="10c0e1d6-e287-4e19-b6c5-db169460dbcf"
file_to_save = h5py.File('/content/sample_int.h5', mode='w')
file_to_save.create_dataset('Reconstruction', data=data_int, compression='lzf')

# %% id="3Ev5taFIy0cu" colab_type="code" colab={}
import matplotlib.pyplot as plt

# %% id="Tlan_ddGy4JE" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 611} outputId="0b4c5713-3ff9-4a19-8bef-fafe347e9325"
plt.figure(figsize=(10, 10))
plt.imshow(data_int[:, :, 0])

# %% id="MlwIJa1Pzkq-" colab_type="code" colab={}
import skimage.feature as skimf

# %% id="_9w49PKE0DmQ" colab_type="code" colab={}
result = np.zeros((256, 256))

for i in np.arange(250):
  result += skimf.greycomatrix(data_int[:, :, i], [1], [0])[:, :, 0, 0]

# %% id="0Yw8aEgK2OrU" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 611} outputId="c23599f2-9c4d-4dcc-d38d-d9f9aa10f43f"
plt.figure(figsize=(10, 10))
plt.imshow(result)

# %% id="wgheTDjO37Yf" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="e7fc9fb6-f714-4112-f8a2-98a7dedc5692"
for i in np.arange(256):
  print(i, np.sum(result[i, :]), np.sum(result[:, i]))

# %% id="nuUhzQ-J64Ms" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 609} outputId="d096b4c3-138f-4aab-e22f-78f9667597fa"
plt.figure(figsize=(10, 10))
plt.yscale('log')
plt.plot(result[114, :])

# %% id="Ay5BrYyt9zsP" colab_type="code" colab={}
import scipy.stats

# %% id="fShgZQwO76Qi" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="bfbabdc6-aef4-4112-9710-8216a936c6d7"
stds = []

for i in np.arange(256):
  res = result[i, :]
  hist_dist = scipy.stats.rv_histogram((res, np.arange(257)))
  stds.append(hist_dist.std())
  print(i, hist_dist.mean(), hist_dist.std())

# %% id="oHA7VhLkAUAN" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 609} outputId="fc654f3f-90a7-46d3-d941-9fbbad3ebf88"
plt.figure(figsize=(10, 10))
plt.plot(stds)
