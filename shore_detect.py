import matplotlib.pyplot as plt
import imageio
import os
import glob
import numpy as np
import datetime
import pandas as pd
import pytz
from skimage.color import rgb2hsv
import matplotlib.colors
%config InlineBackend.figure_format='retina'
# %%
camera = 'both'
product = 'timex'

fildir = '/Volumes/Backstaff/field/bti/'
ts = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'proc/rect/*' + camera + '*' + product + '*png')]
#
t = ts[195]
print(datetime.datetime.utcfromtimestamp(int(t)).isoformat())
print(datetime.datetime.fromtimestamp(int(t), tz=pytz.timezone('US/Alaska')).isoformat())
dt = pd.to_datetime(ts, unit='s').tz_localize(pytz.utc).tz_convert(pytz.timezone('US/Alaska'))
gb = dt.groupby(dt.hour)
hours = []
pics = []
for name in gb:
    hours.append(name)
    pics.append(len(gb[name]))
plt.plot(hours, pics)
plt.xticks(np.arange(0,25,2))
plt.xlim(0,24)
plt.grid()
noons = (gb[12].tz_convert(pytz.utc).tz_convert(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
tens = (gb[10].tz_convert(pytz.utc).tz_convert(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
eights = (gb[8].tz_convert(pytz.utc).tz_convert(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
sixteens = (gb[16].tz_convert(pytz.utc).tz_convert(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# %%
t = str(sixteens[0])

ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
imgboth = imageio.imread(ifile)
hsv = rgb2hsv(imgboth)
rboth = imgboth[:,:,0]
gboth = imgboth[:,:,1]
bboth = imgboth[:,:,2]
# %%
plt.figure(figsize=(14,8))
plt.subplot(1,3,1)
plt.imshow(imgboth)
plt.title(t)
plt.axvline(1000)
plt.axhline(1100)
# plt.subplot(1,3,2)
# ifile = fildir + 'proc/rect/' + t + '.c1.' + product + '.rect.png'
# imgc1 = imageio.imread(ifile)
# plt.imshow(imgc1)
# plt.subplot(1,3,3)
# ifile = fildir + 'proc/rect/' + t + '.c2.' + product + '.rect.png'
# imgc2 = imageio.imread(ifile)
# plt.imshow(imgc2)
plt.show()
# %%


plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
plt.imshow(rboth, cmap=plt.cm.gray)
plt.clim(0,255)
plt.subplot(1,3,2)
plt.imshow(gboth, cmap=plt.cm.gray)
plt.clim(0,255)
plt.subplot(1,3,3)
plt.imshow(bboth, cmap=plt.cm.gray)
plt.clim(0,255)
# %%
for t in sixteens:
    t = str(t)

    ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
    imgboth = imageio.imread(ifile)
    hsv = rgb2hsv(imgboth)
    rboth = imgboth[:,:,0]
    gboth = imgboth[:,:,1]
    bboth = imgboth[:,:,2]

    yloc = 800

    plt.figure(figsize=(10,10))
    # plt.subplot(2,1,1)
    diff = bboth[yloc,:]-rboth[yloc,:]
    plt.imshow(imgboth)
    plt.axhline(yloc, c='grey')
    plt.plot(1500-diff, c='purple')
    plt.text(1500,1480, 'B-R')
    # plt.plot(500-np.abs(np.gradient(rboth[yloc,:])))
    # plt.figure()
    plt.plot(1100-bboth[yloc,:], c='blue')
    plt.plot(1100-rboth[yloc,:], c='red')
    plt.plot(1100-gboth[yloc,:], c='green')
    plt.text(1500,1100, 'RGB')
    # plt.subplot(2,1,2)
    plt.plot(1750-200*hsv[yloc,:,0])
    plt.plot(1750-200*hsv[yloc,:,1])
    plt.plot(1750-200*hsv[yloc,:,2])
    plt.text(1500,1750, 'HSV')
    plt.title(f"{pd.to_datetime(t, unit='s').tz_localize(pytz.utc)} --- {pd.to_datetime(t, unit='s').tz_localize(pytz.utc).tz_convert(pytz.timezone('US/Alaska'))}")
    plt.savefig(t+'.png', bbox_inches='tight', dpi=150)
    plt.show()
# plt.plot(diff)
# plt.plot(np.gradient(diff))
# plt.plot(diffup*10, '*')
# plt.colorbar()
# %%
plt.figure(figsize=(16,6))
plt.subplot(1,3,1)
plt.imshow(hsv[:,:,0], cmap=plt.cm.gray)
plt.clim(0,1)
plt.subplot(1,3,2)
plt.imshow(hsv[:,:,1], cmap=plt.cm.gray)
plt.clim(0,1)
plt.subplot(1,3,3)
plt.imshow(hsv[:,:,2], cmap=plt.cm.gray)
plt.clim(0,1)
# %%

plt.figure()
plt.hist(np.ravel(hsv[:,:,2]), bins=np.linspace(0,1,50))
plt.show()
plt.figure()
plt.hist2d(np.ravel(hsv[:,:,0]), np.ravel(hsv[:,:,1]), bins=(np.linspace(0,1,50), np.linspace(0,1,50)), norm=matplotlib.colors.LogNorm())
plt.show()
# plt.imshow(hsv[:,:,2])
