# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% id="qTvP5IEis3BU" colab_type="code" colab={}
# %load_ext autoreload
# %autoreload 2

# %%
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
plt.figure(figsize=(5, 5))
plt.imshow(data_int[:, :, 0])

# %%
data_bin, _ = helper.binarize_image(data_int)
plt.figure(figsize=(5, 5))
plt.imshow(data_bin[:, :, 0])

# %%
data_bin_erosion = helper.erosion(data_bin)
data_bin_dilation = helper.dilation(data_bin)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(data_bin_erosion[:, :, 0])
axes[1].imshow(data_bin_dilation[:, :, 0])

# %%
data_stones = np.copy(data_int)
data_stones[data_bin_erosion == False] = 0
plt.figure(figsize=(5, 5))
plt.imshow(data_stones[:, :, 0])

# %%
data_pores = np.copy(data_int)
data_pores[data_bin_dilation == True] = 255
plt.figure(figsize=(5, 5))
plt.imshow(data_pores[:, :, 0])

# %% id="_9w49PKE0DmQ" colab_type="code" colab={}
result = np.zeros((256, 256))

for i in np.arange(250):
  result += skimf.greycomatrix(data_int[:, :, i], [1], [0], symmetric=True)[:, :, 0, 0]

# %% id="0Yw8aEgK2OrU" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 611} outputId="c23599f2-9c4d-4dcc-d38d-d9f9aa10f43f"
plt.figure(figsize=(5, 5))
plt.imshow(result)

# %% id="wgheTDjO37Yf" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="e7fc9fb6-f714-4112-f8a2-98a7dedc5692"
# for i in np.arange(256):
#   print(i, np.sum(result[i, :]), np.sum(result[:, i]))

# %% id="nuUhzQ-J64Ms" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 609} outputId="d096b4c3-138f-4aab-e22f-78f9667597fa"
plt.figure(figsize=(10, 5))
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
plt.figure(figsize=(10, 5))
plt.plot(stds)

# %%
result_stones = np.zeros((255, 255))

for i in np.arange(250):
  result_stones += skimf.greycomatrix(data_stones[:, :, i], [1], [0], symmetric=True)[1:, 1:, 0, 0]

plt.figure(figsize=(5, 5))
plt.imshow(result_stones)

# %%
result_pores = np.zeros((255, 255))

for i in np.arange(250):
  result_pores += skimf.greycomatrix(data_pores[:, :, i], [1], [0], symmetric=True)[:255, :255, 0, 0]

plt.figure(figsize=(5, 5))
plt.imshow(result_pores)


# %%
def handle_glcm(image, levels=256, symmetric=True, cut=None, size=256):
    _glcm = skimf.greycomatrix(image, [1], [0], levels=levels, symmetric=symmetric)
    _glcm = _glcm[:, :, 0, 0]
    _glcm = _glcm[1:, 1:] if cut == 'start' else _glcm[:size, :size] if cut == 'end' else _glcm
    return _glcm
    
def sum_norm(glcm):
    glcm -= np.min(glcm)
    glcm /= np.sum(glcm)
    return glcm

def calc_glcm(arr, levels=256, symmetric=True, normed=True, cut=None, type='h'):
    size = levels-1 if cut in ['start', 'end'] else levels
    glcm = np.zeros((size, size))
    axis = 0 if type == 'h' else 1 if type == 'v' else 2 if type == 'd' else ValueError('incorrect type')
    for i in np.arange(arr.shape[axis]):
        if type == 'h':
            _a = arr[i, :, :]
        elif type == 'v':
            _a = arr[:, i, :].T 
        elif type == 'd':
            _a = arr[:, :, i]
        else:
            ValueError('incorrect type')
        glcm += handle_glcm(_a, levels=levels, symmetric=symmetric, cut=cut, size=size)
    return sum_norm(glcm) if normed else glcm

def h_glcm(arr, **kwargs):
    return calc_glcm(arr, **kwargs, type='h')

def v_glcm(arr, **kwargs):
    return calc_glcm(arr, **kwargs, type='v')

def d_glcm(arr, **kwargs):
    return calc_glcm(arr, **kwargs, type='d')

def get_glcm(arr, levels=256, symmetric=True, normed=True, cut=None):
    kwargs = {'levels': levels, 'symmetric': symmetric, 'normed': normed, 'cut': cut}
    return h_glcm(arr, **kwargs), v_glcm(arr, **kwargs), d_glcm(arr, **kwargs)


# %%
a = np.array(
    [[[0, 0],
      [0, 0]],
     [[2, 2],
      [1, 0]]])

print(f'z0\n{a[0, :, :]}')
print(f'y0\n{a[:, 0, :]}')
print(f'x0\n{a[:, :, 0]}')

h, v, d = get_glcm(a, levels=3, normed=False, symmetric=False)
print(f'h\n{h}')
print(f'v\n{v}')
print(f'd\n{d}')

# %%
hcm, vcm, dcm = get_glcm(data_stones, normed=True, cut='start')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(hcm)
axes[1].imshow(vcm)
axes[2].imshow(dcm)


# %%
def calc_euclidian_distance(x, y):
    return np.sqrt(np.sum(np.power((x - y), 2)))

hv_diff = calc_euclidian_distance(hcm, vcm)
hd_diff = calc_euclidian_distance(hcm, dcm)
vd_diff = calc_euclidian_distance(vcm, dcm)
print(hv_diff, hd_diff, vd_diff)

# %%
stones_hist, bins = np.histogram(data_stones, bins=255)
stones_hist[0] = 0

# %%
helper.plot_bars(stones_hist, bins, log=True)

# %%
stones_pdf = stones_hist / np.sum(stones_hist)
helper.plot_bars(stones_pdf, bins, log=True)

# %%
# %%time
stones_pdf_test = np.random.choice(255, size=250*250*2, p=stones_pdf)
plt.hist(stones_pdf_test, bins=256, log=True)
plt.xlim(0, 255)

# %%
stones_pdf_image = stones_pdf_test.reshape((250, 250, 2))
plt.figure(figsize=(5, 5))
plt.imshow(stones_pdf_image[:, :, 0], vmin=0, vmax=255)

# %%
hcm_g, vcm_g, dcm_g = get_glcm(stones_pdf_image, normed=True, cut='start')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(hcm_g)
axes[1].imshow(vcm_g)
axes[2].imshow(dcm_g)

# %%
h_diff = calc_euclidian_distance(hcm, hcm_g)
v_diff = calc_euclidian_distance(vcm, vcm_g)
d_diff = calc_euclidian_distance(dcm, dcm_g)
print(h_diff, v_diff, d_diff)

origin_glcm = hcm + vcm + dcm
generated_glcm = hcm_g + vcm_g + dcm_g
print(calc_euclidian_distance(origin_glcm, generated_glcm))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(origin_glcm)
axes[1].imshow(generated_glcm)


# %%
def get_props_of_glcm(glcm):
    _glcm = np.copy(glcm)[..., np.newaxis, np.newaxis]
    contrast = skimf.greycoprops(_glcm, 'contrast')
    dissimilarity = skimf.greycoprops(_glcm, 'dissimilarity')
    homogeneity = skimf.greycoprops(_glcm, 'homogeneity')
    energy = skimf.greycoprops(_glcm, 'energy')
    correlation = skimf.greycoprops(_glcm, 'correlation')
    asm = skimf.greycoprops(_glcm, 'ASM')
    print('---- glcm props ----')
    print(f'contrast: {contrast}')
    print(f'dissimilarity: {dissimilarity}')
    print(f'homogeneity: {homogeneity}')
    print(f'energy: {energy}')
    print(f'cocorrelationntrast: {correlation}')
    print(f'asm: {asm}')
    print('\n')
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm])


# %%
origin_glcm_props = get_props_of_glcm(origin_glcm)
generated_glcm_props = get_props_of_glcm(generated_glcm)
origin_glcm_props - generated_glcm_props

# %%
object_image = data_int[:, :, 0]
stones_image = data_stones[:, :, 0]
pores_image = data_pores[:, :, 0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(object_image)
axes[1].imshow(stones_image)
axes[2].imshow(pores_image)

# %%
h, _, _ = get_glcm(object_image[np.newaxis, ...], normed=True)
hs, _, _ = get_glcm(stones_image[np.newaxis, ...], normed=True, cut='start')
hp, _, _ = get_glcm(pores_image[np.newaxis, ...], normed=True, cut='end')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(h)
axes[1].imshow(hs)
axes[2].imshow(hp)

# %%
object_hist, bins = np.histogram(object_image, bins=255, range=(0, 255))
helper.plot_bars(object_hist, bins, log=True)

stones_hist, bins = np.histogram(stones_image, bins=255, range=(0, 255))
stones_hist[0] = 0
helper.plot_bars(stones_hist, bins, log=True)

pores_hist, bins = np.histogram(pores_image, bins=255, range=(0, 255))
pores_hist[-1] = 0
helper.plot_bars(pores_hist, bins, log=True)

sum_hist = stones_hist + pores_hist
helper.plot_bars(sum_hist, bins, log=True)

diff_hist = object_hist - sum_hist
helper.plot_bars(diff_hist, bins, log=True)

# %%
stones_pdf = stones_hist / np.sum(stones_hist)
pores_pdf = pores_hist / np.sum(pores_hist)

# %%
# %%time
stones_pdf_data = np.random.choice(255, size=250*250, p=stones_pdf)
plt.hist(stones_pdf_data, bins=256, log=True)
plt.xlim(0, 255)

pores_pdf_data = np.random.choice(255, size=250*250, p=pores_pdf)
plt.hist(pores_pdf_data, bins=256, log=True)
plt.xlim(0, 255)

# %%
stones_pdf_image = stones_pdf_data.reshape((250, 250))
pores_pdf_image = pores_pdf_data.reshape((250, 250))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(stones_pdf_image, vmin=0, vmax=255)
axes[1].imshow(pores_pdf_image, vmin=0, vmax=255)

# %%
hsg, _, _ = get_glcm(stones_pdf_image[np.newaxis, ...], normed=True, cut='start')
hpg, _, _ = get_glcm(pores_pdf_image[np.newaxis, ...], normed=True, cut='end')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(hs)
axes[1].imshow(hp)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(hsg)
axes[1].imshow(hpg)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(hs-hsg)
axes[1].imshow(hp-hpg)

abs_diff_hs = np.absolute(hs-hsg)
abs_diff_hp = np.absolute(hp-hpg)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(abs_diff_hs)
axes[1].imshow(abs_diff_hp)

# %%
print(np.unravel_index(np.argmax(abs_diff_hs), abs_diff_hs.shape))
print(np.unravel_index(np.argmax(abs_diff_hp), abs_diff_hp.shape))


# %%
def plot_glcms(glcm, glcmg, diff, generated_image):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(glcm)
    axes[1].imshow(glcmg)
    axes[2].imshow(diff)
    axes[3].imshow(generated_image, vmin=0, vmax=255)
    print(np.unravel_index(np.argmax(diff), diff.shape))
    
plot_glcms(hs, hsg, abs_diff_hs, stones_pdf_image)

# %%
rx1, rx2, ry1, ry2 = np.random.randint(0, 250, 4)
stones_pdf_image[rx1, ry1]
stones_pdf_image[rx2, ry2]

# %%
test_image = np.copy(stones_pdf_image)
hs, _, _ = get_glcm(stones_image[np.newaxis, ...], normed=True, cut='start')
hsg, _, _ = get_glcm(test_image[np.newaxis, ...], normed=True, cut='start')
abs_diff_hs = np.absolute(hs-hsg)
err = np.sqrt(np.sum(abs_diff_hs ** 2))
success_count = 0
unsuccess_count = 0
print(f'initial error: {err}')

for i in range(1000):
    
#     max_diff_value, _ = np.unravel_index(np.argmax(abs_diff_hs), abs_diff_hs.shape)
#     result = np.where(stones_pdf_image == max_diff_value)
#     listOfCoordinates = list(zip(result[0], result[1]))
#     max_value_coord = listOfCoordinates[0]

    coord1 = np.random.randint(0, 250, 2)
    coord2 = np.random.randint(0, 250, 2)
    
    new_image = np.copy(test_image)
    v1 = test_image[coord1]
    v2 = test_image[coord2]
    new_image[coord1] = v2
    new_image[coord2] = v1
    hsg, _, _ = get_glcm(new_image[np.newaxis, ...], normed=True, cut='start')
    abs_diff_hs = np.absolute(hs-hsg)
    new_err = np.sqrt(np.sum(abs_diff_hs ** 2))
    
    if new_err >= err:
        unsuccess_count += 1
        if unsuccess_count % 100 == 0:
            print(f'unsuccess_count {unsuccess_count}')
        continue
    
    success_count += 1
    if success_count % 100 == 0:
        print(f'success_count {success_count}')
    test_image = new_image
    err = new_err
    
    if i % 100 == 0:
        print(f'current error: {err}')
        plot_glcms(hs, hsg, abs_diff_hs, test_image)

print(f'success_count: {success_count}')
print(f'unsuccess_count: {unsuccess_count}')
plot_glcms(hs, hsg, abs_diff_hs, test_image)

# %%
