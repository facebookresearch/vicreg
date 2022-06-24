#%%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

#%%
# Load the data
data_dir = '/nas/softechict-nas-1/rbenaglia/vicreg/stats.txt'
results = np.loadtxt(data_dir)
print(results)


