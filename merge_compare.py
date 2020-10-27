# compare images
import matplotlib.pyplot as plt
import imageio
import os
import glob
import numpy as np
%config InlineBackend.figure_format='retina'
# %%
ts1 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/bti/products/*c1.snap.jpg')]
ts2 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/bti/products/*c2.snap.jpg')]

camera = 'both'
if camera is 'both':
    ts = list(set(ts1) & set(ts2))


camera = 'both'

fildir = '/Volumes/Backstaff/field/bti/'
ts = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'proc/rect/*' + camera + '*snap*png')]




# %%
t = ts[200]


plt.figure(figsize=(14,8))
plt.subplot(1,3,1)
ifile = fildir + 'proc/rect/' + t + '.' + camera + '.snap.rect.png'
imgboth = imageio.imread(ifile)
plt.imshow(imgboth)
plt.title(t)
plt.axvline(1000)
plt.axhline(1100)
plt.subplot(1,3,2)
ifile = fildir + 'proc/rect/' + t + '.c1.snap.rect.png'
imgc1 = imageio.imread(ifile)
plt.imshow(imgc1)
plt.subplot(1,3,3)
ifile = fildir + 'proc/rect/' + t + '.c2.snap.rect.png'
imgc2 = imageio.imread(ifile)
plt.imshow(imgc2)
plt.show()
# %%
plt.figure(figsize=(10,8))
mboth = np.mean(imgboth[:,1000], axis=-1)
mc1 = np.mean(imgc1[:,1000], axis=-1)
mc2 = np.mean(imgc2[:,1000], axis=-1)
plt.plot(mc1, label='c1')
plt.plot(mc2, label='c2')
plt.plot(mboth, label='both')
plt.legend()
# plt.plot(mc1-mboth)
# plt.plot(mc2-mboth)
plt.axvspan(957,1132, color='grey')
plt.ylim(120,256)
# plt.xlim(700,1400)
plt.xlim(940,1160)
plt.show()
# %%
