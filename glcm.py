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
with h5py.File(f'{file_dir}sample.h5', mode='r') as file:
    data = file['Reconstruction'][()]

# %% id="5PiB93tFxl3k" colab_type="code" colab={}
bits = 4
max_v = 2**bits
data_int = helper.image_digitize(data, bits)

# %% id="Ij2yrjqCyJZ1" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="10c0e1d6-e287-4e19-b6c5-db169460dbcf"
with h5py.File(f'{file_dir}sample_int.h5', mode='w') as file_to_save:
    file_to_save.create_dataset('Reconstruction', data=data_int, compression='lzf')

# %% id="Tlan_ddGy4JE" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 611} outputId="0b4c5713-3ff9-4a19-8bef-fafe347e9325"
plt.figure(figsize=(5, 5))
plt.imshow(data_int[:, :, 0])
plt.colorbar()

# %%
data_bin, _ = helper.binarize_image(data_int)
plt.figure(figsize=(5, 5))
plt.imshow(data_bin[:, :, 0])

# %%
data_bin_erosion = helper.erosion(data_bin)
data_bin_dilation = helper.dilation(data_bin)
data_bin_edges = np.ones(data_bin.shape) - data_bin_erosion - ~data_bin_dilation

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(data_bin_erosion[:, :, 0])
axes[1].imshow(data_bin_dilation[:, :, 0])
axes[2].imshow(data_bin_edges[:, :, 0])

# %%
data_stones = np.copy(data_int)
data_pores = np.copy(data_int)
data_edges = np.copy(data_int)

data_stones[data_bin_erosion == False] = 0
data_pores[data_bin_dilation == True] = max_v - 1
data_edges[data_bin_edges == False] = 0

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(data_stones[:, :, 0])
axes[1].imshow(data_pores[:, :, 0])
axes[2].imshow(data_edges[:, :, 0])

# %% id="_9w49PKE0DmQ" colab_type="code" colab={}
result = np.zeros((max_v, max_v))

for i in np.arange(250):
  result += skimf.greycomatrix(data_int[:, :, i], [1], [0], levels=max_v, symmetric=True)[:, :, 0, 0]

# %% id="0Yw8aEgK2OrU" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 611} outputId="c23599f2-9c4d-4dcc-d38d-d9f9aa10f43f"
plt.figure(figsize=(5, 5))
plt.imshow(result)
plt.colorbar()

# %%
result_stones = np.zeros((max_v - 1, max_v - 1))
result_pores = np.zeros((max_v - 1, max_v - 1))
result_edges = np.zeros((max_v - 1, max_v - 1))

for i in np.arange(250):

    result_stones += skimf.greycomatrix(
        data_stones[:, :, i], 
        [1], 
        [0], 
        levels=max_v, 
        symmetric=True
    )[1:, 1:, 0, 0]

    result_pores += skimf.greycomatrix(
        data_pores[:, :, i], 
        [1], 
        [0], 
        levels=max_v, 
        symmetric=True
    )[:max_v - 1, :max_v - 1, 0, 0]

    result_edges += skimf.greycomatrix(
        data_edges[:, :, i], 
        [1], 
        [0], 
        levels=max_v, 
        symmetric=True
    )[1:, 1:, 0, 0]


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].imshow(result_stones)
fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(result_pores)
fig.colorbar(im1, ax=axes[1])
im2 = axes[2].imshow(result_edges)
fig.colorbar(im2, ax=axes[2])


# %%
def handle_glcm(image, levels=256, symmetric=True, cut=None, size=256):
    _glcm = skimf.greycomatrix(image, [1], [0], levels=levels, symmetric=symmetric)
    _glcm = _glcm[:, :, 0, 0]
    if cut == 'start':
        _glcm[0, :] = 0
        _glcm[:, 0] = 0
    if cut == 'end':
        _glcm[-1, :] = 0
        _glcm[:, -1] = 0
#     _glcm = _glcm[1:, 1:] if cut == 'start' else _glcm[:size, :size] if cut == 'end' else _glcm
    return _glcm
    
def sum_norm(glcm):
    glcm -= np.min(glcm)
    glcm /= np.sum(glcm)
    return glcm

def calc_glcm(arr, levels=256, symmetric=True, normed=True, cut=None, type='h'):
#     size = levels-1 if cut in ['start', 'end'] else levels
    size = levels
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
def switch_pixels(img, p1, p2):
    _img = np.copy(img)
    v1 = img[p1]
    v2 = img[p2]
    _img[p1] = v2
    _img[p2] = v1
    return _img


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
# print(f'v\n{v}')
# print(f'd\n{d}')

# %%
b = switch_pixels(a, (1, 0, 1), (0, 0, 0))
print(b)
h, _, _ = get_glcm(b, levels=3, normed=False, symmetric=False)
print(f'h\n{h}')

# %%
a1 = np.array(
    [[0, 1],
    [0, 2]]
)
print(f'y0\n{a1[0, :]}')
print(f'x0\n{a1[:, 0]}')
h, _, _ = get_glcm(a1[np.newaxis, ...], levels=3, normed=False, symmetric=False)
print(f'h\n{h}')

b1 = switch_pixels(a1, (0, 0), (0, 1))
print(b1)
h, _, _ = get_glcm(b1[np.newaxis, ...], levels=3, normed=False, symmetric=False)
print(f'h\n{h}')

# %%
hcm, vcm, dcm = get_glcm(data_stones, levels=max_v, normed=True, cut='start')
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
stones_hist, bins = np.histogram(data_stones, bins=max_v - 1)
stones_hist[0] = 0

# %%
helper.plot_bars(stones_hist, bins, log=True)

# %%
stones_pdf = stones_hist / np.sum(stones_hist)
helper.plot_bars(stones_pdf, bins, log=True)

# %%
# %%time
stones_pdf_test = np.random.choice(max_v - 1, size=250*250*2, p=stones_pdf)
plt.hist(stones_pdf_test, bins=max_v, log=True)
plt.xlim(0, max_v - 1)

# %%
stones_pdf_image = stones_pdf_test.reshape((250, 250, 2))
plt.figure(figsize=(5, 5))
plt.imshow(stones_pdf_image[:, :, 0], vmin=0, vmax=max_v - 1)
plt.colorbar()

# %%
hcm_g, vcm_g, dcm_g = get_glcm(stones_pdf_image, levels=max_v, normed=True, cut='start')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].imshow(hcm_g)
fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(vcm_g)
fig.colorbar(im1, ax=axes[1])
im2 = axes[2].imshow(dcm_g)
fig.colorbar(im2, ax=axes[2])

# %%
h_diff = calc_euclidian_distance(hcm, hcm_g)
v_diff = calc_euclidian_distance(vcm, vcm_g)
d_diff = calc_euclidian_distance(dcm, dcm_g)
print(h_diff, v_diff, d_diff)

origin_glcm = hcm + vcm + dcm
generated_glcm = hcm_g + vcm_g + dcm_g
print(calc_euclidian_distance(origin_glcm, generated_glcm))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
im0 = axes[0].imshow(origin_glcm)
im1 = axes[1].imshow(generated_glcm)
fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])


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
edges_image = data_edges[:, :, 0]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(object_image)
axes[1].imshow(stones_image)
axes[2].imshow(pores_image)
axes[3].imshow(edges_image)

# %%
h, _, _ = get_glcm(object_image[np.newaxis, ...], levels=max_v, normed=True)
hs, _, _ = get_glcm(stones_image[np.newaxis, ...], levels=max_v, normed=True, cut='start')
hp, _, _ = get_glcm(pores_image[np.newaxis, ...], levels=max_v, normed=True, cut='end')
he, _, _ = get_glcm(edges_image[np.newaxis, ...], levels=max_v, normed=True, cut='start')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(h)
axes[1].imshow(hs)
axes[2].imshow(hp)
axes[3].imshow(he)

# %%
bins = max_v - 1

object_hist, bins = np.histogram(object_image, bins=bins, range=(0, bins))
helper.plot_bars(object_hist, bins, log=True)

stones_hist, bins = np.histogram(stones_image, bins=bins, range=(0, bins))
stones_hist[0] = 0
helper.plot_bars(stones_hist, bins, log=True)

pores_hist, bins = np.histogram(pores_image, bins=bins, range=(0, bins))
pores_hist[-1] = 0
helper.plot_bars(pores_hist, bins, log=True)

edges_hist, bins = np.histogram(edges_image, bins=bins, range=(0, bins))
edges_hist[0] = 0
helper.plot_bars(edges_hist, bins, log=True)

sum_hist = stones_hist + pores_hist + edges_hist
helper.plot_bars(sum_hist, bins, log=True)

diff_hist = object_hist - sum_hist
helper.plot_bars(diff_hist, bins, log=True)

# %%
stones_pdf = stones_hist / np.sum(stones_hist)
pores_pdf = pores_hist / np.sum(pores_hist)
edges_pdf = edges_hist / np.sum(edges_hist)

# %%
test_image_size = 50

# %%
# %%time
stones_pdf_data = np.random.choice(max_v - 1, size=test_image_size*test_image_size, p=stones_pdf)
plt.hist(stones_pdf_data, bins=max_v, log=True)
plt.xlim(0, max_v - 1)

pores_pdf_data = np.random.choice(max_v - 1, size=test_image_size*test_image_size, p=pores_pdf)
plt.hist(pores_pdf_data, bins=max_v, log=True)
plt.xlim(0, max_v - 1)

edges_pdf_data = np.random.choice(max_v - 1, size=test_image_size*test_image_size, p=edges_pdf)
plt.hist(edges_pdf_data, bins=max_v, log=True)
plt.xlim(0, max_v - 1)

# %%
stones_pdf_image = stones_pdf_data.reshape((test_image_size, test_image_size))
pores_pdf_image = pores_pdf_data.reshape((test_image_size, test_image_size))
edges_pdf_image = edges_pdf_data.reshape((test_image_size, test_image_size))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(stones_pdf_image, vmin=0, vmax=max_v - 1)
axes[1].imshow(pores_pdf_image, vmin=0, vmax=max_v - 1)
axes[2].imshow(edges_pdf_image, vmin=0, vmax=max_v - 1)

# %%
hsg, _, _ = get_glcm(stones_pdf_image[np.newaxis, ...], levels=max_v, normed=True, cut='start')
hpg, _, _ = get_glcm(pores_pdf_image[np.newaxis, ...], levels=max_v, normed=True, cut='end')
heg, _, _ = get_glcm(edges_pdf_image[np.newaxis, ...], levels=max_v, normed=True, cut='start')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(hs)
axes[1].imshow(hp)
axes[2].imshow(he)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(hsg)
axes[1].imshow(hpg)
axes[2].imshow(heg)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(hs-hsg)
axes[1].imshow(hp-hpg)
axes[2].imshow(he-heg)

abs_diff_hs = np.absolute(hs-hsg)
abs_diff_hp = np.absolute(hp-hpg)
abs_diff_he = np.absolute(he-heg)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(abs_diff_hs)
axes[1].imshow(abs_diff_hp)
axes[2].imshow(abs_diff_he)

# %%
print(np.unravel_index(np.argmax(abs_diff_hs), abs_diff_hs.shape))
print(np.unravel_index(np.argmax(abs_diff_hp), abs_diff_hp.shape))
print(np.unravel_index(np.argmax(abs_diff_he), abs_diff_he.shape))


# %%
def plot_glcms(glcm, glcmg, diff, generated_image):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    im0 = axes[0].imshow(glcm)
    im1 = axes[1].imshow(glcmg)
    im2 = axes[2].imshow(diff)
    im3 = axes[3].imshow(generated_image, vmin=0, vmax=max_v - 1)
    fig.colorbar(im0, ax=axes[0])
    fig.colorbar(im1, ax=axes[1])
    fig.colorbar(im2, ax=axes[2])
    fig.colorbar(im3, ax=axes[3])
    print(f'diff max: {np.unravel_index(np.argmax(diff), diff.shape)}')
    
plot_glcms(hs, hsg, abs_diff_hs, stones_pdf_image)

# %%
image_to_test = np.copy(stones_pdf_image)


hsg, _, _ = get_glcm(
    image_to_test[np.newaxis, ...], 
    levels=max_v, 
    normed=False, 
    symmetric=False, 
#     cut='start'
)

rx1, rx2, ry1, ry2 = np.random.randint(3, test_image_size-3, 4)
new_image = switch_pixels(image_to_test, (ry1, rx1), (ry2, rx2))

# %time
new_h_glcm, _, _ = get_glcm(
    new_image[np.newaxis, ...], 
    levels=max_v, 
    normed=False, 
    symmetric=False, 
#     cut='start'
)

test_hsg = np.copy(hsg)

print((rx1, ry1), (rx2, ry2))

v1 = image_to_test[ry1, rx1]
v1_left = image_to_test[ry1, rx1 - 1]
v1_right = image_to_test[ry1, rx1 + 1]

v2 = image_to_test[ry2, rx2]
v2_left = image_to_test[ry2, rx2 - 1]
v2_right = image_to_test[ry2, rx2 + 1]

print(image_to_test[ry1-2:ry1+3, rx1-2:rx1+3])
print(image_to_test[ry2-2:ry2+3, rx2-2:rx2+3])

print(v1_left, v1, v1_right)
print(v2_left, v2, v2_right)

# v1 = new_image[ry1, rx1]
# v1_left = new_image[ry1, rx1 - 1]
# v1_right = new_image[ry1, rx1 + 1]
# v2 = new_image[ry2, rx2]
# v2_left = new_image[ry2, rx2 - 1]
# v2_right = new_image[ry2, rx2 + 1]

# print(v1_left, v1, v1_right)
# print(v2_left, v2, v2_right)

# print(f'hsg: \n{hsg.astype(np.int)}')
# print(f'new_h_glcm: \n{new_h_glcm.astype(np.int)}')

def test_speed():
    test_hsg[v1_left, v1] -= 1
    test_hsg[v1, v1_right] -= 1
    test_hsg[v2_left, v2] -= 1
    test_hsg[v2, v2_right] -= 1

    test_hsg[v1_left, v2] += 1
    test_hsg[v2, v1_right] += 1
    test_hsg[v2_left, v1] += 1
    test_hsg[v1, v2_right] += 1

# %time
test_speed()
    
# print((new_h_glcm - hsg).astype(np.int))
# print((test_hsg - hsg).astype(np.int))

print(hsg.astype(np.int))
print(new_h_glcm.astype(np.int))
print(test_hsg.astype(np.int))

print((test_hsg - new_h_glcm).astype(np.int))

print(np.sqrt(np.sum((test_hsg - new_h_glcm) ** 2)))

# %%
# %%time
test_image = np.copy(stones_pdf_image)
hs, _, _ = get_glcm(stones_image[np.newaxis, ...], levels=max_v, normed=False, cut='start')
hsg, _, _ = get_glcm(test_image[np.newaxis, ...], levels=max_v, normed=False)
abs_diff_hs = np.absolute(hs-hsg)
err = np.sqrt(np.sum(abs_diff_hs ** 2))
success_count = 0
unsuccess_count = 0
trans_count = 0
print(f'initial error: {err}')
plot_glcms(hs, hsg, abs_diff_hs, test_image)

num_of_iters = 1_000_000

for i in range(num_of_iters):
    
    if i % (num_of_iters//10) == 0:
        print(f'current error: {err}')
        plot_glcms(hs, hsg, abs_diff_hs, test_image)

#     max_diff_value, _ = np.unravel_index(np.argmax(abs_diff_hs), abs_diff_hs.shape)
#     result = np.where(stones_pdf_image == max_diff_value)
#     listOfCoordinates = list(zip(result[0], result[1]))
#     max_value_coord = listOfCoordinates[0]

    coord1 = np.random.randint(1, 249, 2)
    coord2 = np.random.randint(1, 249, 2)
    
    ry1, rx1 = coord1
    ry2, rx2 = coord2
    
    v1 = test_image[ry1, rx1]
    v1_left = test_image[ry1, rx1 - 1]
    v1_right = test_image[ry1, rx1 + 1]
    v2 = test_image[ry2, rx2]
    v2_left = test_image[ry2, rx2 - 1]
    v2_right = test_image[ry2, rx2 + 1]

    test_hsg = np.copy(hsg)
    
    test_hsg[v1_left, v1] -= 1
    test_hsg[v1, v1_right] -= 1
    test_hsg[v2_left, v2] -= 1
    test_hsg[v2, v2_right] -= 1

    test_hsg[v1_left, v2] += 1
    test_hsg[v2, v1_right] += 1
    test_hsg[v2_left, v1] += 1
    test_hsg[v1, v2_right] += 1

    abs_diff_hs = np.absolute(hs - test_hsg)
    new_err = np.sqrt(np.sum(abs_diff_hs ** 2))
    
    if new_err >= err:
#         p = 1 / (1 + np.exp(new_err/(num_of_iters - i + 100)))
        k = new_err / (new_err * (num_of_iters - i) / num_of_iters)
        p = 1 / (1 + np.exp(k))

        if p > np.random.uniform(0, 1):
            test_image = switch_pixels(test_image, coord1, coord2)
            hsg = test_hsg
            err = new_err
            trans_count += 1
            if trans_count % (num_of_iters//10) == 0:
                print(f'trans_count {trans_count}')
            continue

        unsuccess_count += 1
        if unsuccess_count % (num_of_iters//10) == 0:
            print(f'unsuccess_count {unsuccess_count}')
            print(f'p {p}')
        continue
    
    success_count += 1
    if success_count % (num_of_iters//10) == 0:
        print(f'success_count {success_count}')

    test_image = switch_pixels(test_image, coord1, coord2)
    hsg = test_hsg
    err = new_err
    
print(f'success_count: {success_count}')
print(f'unsuccess_count: {unsuccess_count}')
print(f'trans_count: {trans_count}')
print(f'final_err: {err}')
plot_glcms(hs, hsg, abs_diff_hs, test_image)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].imshow(stones_pdf_image)
axes[1].imshow(test_image)

# %%
