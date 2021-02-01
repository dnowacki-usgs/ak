import numpy as np
import glob
import skimage.io as sio
import os
from shutil import move
# %%
filbase = '/Volumes/Backstaff/field/unk/proc/rect/snap/'
fildir = list(glob.iglob(filbase + '*.png'))

# %%
"""
Loop through files and copy only images brighter than a threshold
"""
goods = []
i = 0
for n in fildir:
    basename = os.path.basename(n)
    im = sio.imread(n)

    # print (n, np.mean(im))
    if np.mean(im) < 15:
        print(i/len(fildir), n, f'*** DARK {np.mean(im)}')
        move(n, filbase + 'dark/')
    i += 1
