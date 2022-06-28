# %%
import stat
import pandas as pd
import matplotlib.pyplot as plt

stats = open('stats.txt').read().splitlines()[0]

PosixPath = lambda x: x

stats = eval(stats)
# %%
for r in stats['results']:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i, k in enumerate(stats['results'][r]):
        ax[i].plot(range(len(stats['results'][r][k][:5])), stats['results'][r][k][:5], label=k)
    plt.title(r)
# %%
# 4 o 8-NN

# import os
# with open('file_to_transfer.txt', 'w') as tf:
#     for f in os.listdir('/mnt/ext/IMAGENET/train'):
#         for i, im in enumerate(os.listdir('/mnt/ext/IMAGENET/train/' + f)):
#             if i < 3:
#                 tf.write('/mnt/ext/IMAGENET/train/' + f + '/' + im + '\n')

# %%
