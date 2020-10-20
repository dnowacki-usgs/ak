import numpy as np
import glob
import skimage.io as sio
import os
from shutil import copyfile
# %%

fildir = list(glob.iglob('../spypoint/DCIM/101DSCIM/*.JPG'))

# %%
"""
Loop through files and copy only images brighter than a threshold
"""
goods = []

for n in fildir:
    basename = os.path.basename(n)
    im = sio.imread(n)

    print (n, np.mean(im))
    if np.mean(im) >= 50:
        copyfile(n,
                 '../spypoint/DCIM/daylight/' + basename)
