import matplotlib.pyplot as plt
import imageio
import os
import glob
import numpy as np
import datetime
import pandas as pd
import pytz
import sys
import cv2
sys.path.append('/Users/dnowacki/Documents/python')
import matlabtools
import djn
import scipy
import skimage.filters
from skimage.color import rgb2hsv, rgb2lab
import skimage
import matplotlib.colors
%config InlineBackend.figure_format='retina'
import xarray as xr
import warnings
n9468333 = xr.load_dataset('/Volumes/Backstaff/field/unk/n9468333.nc')
wavesunk = xr.load_dataset('/Volumes/Backstaff/field/unk/ak_4m.nc')

camera = 'both'
product = 'timex'

fildir = '/Volumes/Backstaff/field/unk/'
# %%

def rotz(ang):
    # https://www.mathworks.com/help/phased/ref/rotz.html
    gamma = np.deg2rad(ang)
    return np.array([[np.cos(gamma), -np.sin(gamma), 0],
                     [np.sin(gamma), np.cos(gamma), 0],
                     [0, 0, 1]])
R = rotz(165)
Origin = np.array([411796.031, 7084427.476, 6.121]); # [E,N,Z] all in m.

# topo = pd.read_csv(fildir + 'gis/Unk19_ForDN/topo/asc/unk19_topo_argusFOV.csv')
# vec = np.array([topo['Easting']-Origin[0], topo['Northing']-Origin[1], topo['Elevation']-0])
# Rvec = np.matmul(R,vec)
# topo['x'] = Rvec[0,:]
# topo['y'] = Rvec[1,:]
# topo.to_csv(fildir + 'gis/Unk19_ForDN/topo/asc/unk19_topo_argusFOV_with_local.csv')

# bathy = pd.read_csv(fildir + 'gis/Unk19_ForDN/bathy/csv/UNK19_bathy.csv')
# vec = np.array([bathy['Easting']-Origin[0], bathy['Northing']-Origin[1], bathy['OrthometricHeight']-0])
# Rvec = np.matmul(R,vec)
# bathy['x'] = Rvec[0,:]
# bathy['y'] = Rvec[1,:]
# bathy.to_csv(fildir + 'gis/Unk19_ForDN/bathy/csv/UNK19_bathy_with_local.csv')

topo = pd.read_csv(fildir + 'gis/Unk19_ForDN/topo/asc/unk19_topo_argusFOV_with_local.csv')
bathy = pd.read_csv(fildir + 'gis/Unk19_ForDN/bathy/csv/UNK19_bathy_with_local.csv')

x = np.arange(0, 250, .1)
y = np.arange(-150,40,.1)
""" load all rectified images and plot availability by hour """
ts = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'proc/rect/*' + camera + '.' + product + '.rect.png')]

t = ts[0]
ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
img = np.rot90(imageio.imread(ifile))
# %%


mat = matlabtools.loadmat(fildir + 'unalakleet/GCPs_20180825_10beachtargets_UTM.mat')
for n in mat['gcp'][0]._fieldnames:
dir(mat['gcp'][0])
tE = np.array([x.x for x in mat['gcp']])# 411752.494000000
tN = np.array([x.y for x in mat['gcp']])# 411752.494000000
tZ = np.array([x.z for x in mat['gcp']])# 411752.494000000
vec = np.array([tE-Origin[0], tN-Origin[1], tZ-0]) # I like to keep orign z NAVD88=0

Rvec = np.matmul(R,vec);  # This is a rotated vector of relative distances 3 high by super wide
# first row of Rvec is new xs, R




# to go from utm to local
# from local to utm, rotate, then translate
# from utm to local, translate then rotate
# i.e. you always want to rotate around zero


# %%

plt.figure(figsize=(10,10))
plt.imshow(np.rot90(img, -1), extent=(0, 250, -150, 40, ))
# plt.scatter(Rvec[0,:], Rvec[1,:], c=Rvec[2,:], cmap=plt.cm.gist_earth)
plt.scatter(topo['x'][495:537], topo['y'][495:537], c=topo['Elevation'][495:537], cmap=plt.cm.gist_earth)
plt.scatter(topo['x'][544:610], topo['y'][544:610], c=topo['Elevation'][544:610], cmap=plt.cm.gist_earth)
plt.colorbar(fraction=0.034)
plt.axhline(-40)
# plt.savefig('backpack_merged.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
xsecs = np.array([[200, 252], [262, 304], [312, 365], [388, 420], [428, 478], [495, 537], [544, 610]])
# %%
""" these are the coords that bound the 800 line [495, 537], [544, 610] """
# we pick out at y = -40, 1100
idx=[610]
plt.figure(figsize=(10,10))
plt.scatter(topo['Easting'], topo['Northing'], c=topo['Elevation'])
plt.plot(topo['Easting'][idx], topo['Northing'][idx], '*', color='r')
plt.colorbar()
plt.show()
# %%

for xy in xsecs:
    plt.plot(np.arange(xy[1]-xy[0]), topo['Elevation'][xy[0]:xy[1]])
# %%
plt.figure(figsize=(10,8))
# zval = topo['Elevation'][537:495:-1].values
# xval = Rvec[0,537:495:-1]
zval = topo['Elevation'][544:610].values
xval = topo['x'][544:610]

plt.plot(xval, zval)
lr = scipy.stats.linregress(xval, zval)
plt.plot(xval, xval*lr.slope+lr.intercept)
lr
plt.show()
# plt.plot(Rvec[0,495:537], Rvec[1,495:537], c=Rvec[2,495:537], cmap=plt.cm.gist_earth)
# %%


goods = (bathy['Northing'] > 7.0843e6) & (bathy['Northing'] < 7.0845e6) & (bathy['Easting'] < 412000) & (bathy['Easting'] > 411400)
plt.figure(figsize=(10,8))
plt.scatter(topo['Easting'][495:537], topo['Northing'][495:537], c=topo['Elevation'][495:537])
plt.scatter(bathy['Easting'][goods], bathy['Northing'][goods], c=bathy['OrthometricHeight'][goods])
plt.axis('equal')
plt.show()
# %%
goods = (bathy['x'] > 0) & (bathy['x'] < 120) & (bathy['y'] > -60) & (bathy['y'] < 0)

print(np.where(goods))
plt.figure(figsize=(10,8))
plt.scatter(bathy['x'][goods], bathy['y'][goods], c=bathy['OrthometricHeight'][goods])
# idx =123555:123255
idx =123255
plt.plot(bathy['x'][idx], bathy['y'][idx], 'rs')
plt.axis('equal')
plt.colorbar()
# %%
plt.figure(figsize=(10,8))
plt.scatter(bathy['x'][123255:123555], bathy['y'][123255:123555], c=bathy['OrthometricHeight'][123255:123555])
# idx =123555:123255
