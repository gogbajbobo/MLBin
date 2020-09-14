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

# %%
# %load_ext autoreload
# %autoreload 2

# %matplotlib widget
import matplotlib.pyplot as plt

import numpy as np

# %%
import time

# %%
x = np.arange(100)
y = np.power(x, 2)

plt.plot(x, y)

# %%
