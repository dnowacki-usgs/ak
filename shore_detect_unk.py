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
import xarray as xr
n9468333 = xr.load_dataset('/Volumes/Backstaff/field/unk/n9468333.nc')
# %%
camera = 'both'
product = 'timex'

fildir = '/Volumes/Backstaff/field/unk/'
ts = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'proc/rect/*' + camera + '.' + product + '.rect.png')]
# ts
# t = ts[0]
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

def to_epoch(dtime):
    return (dtime.tz_convert(pytz.utc).tz_convert(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

elevens = to_epoch(gb[11])
noons = to_epoch(gb[12])
tens = to_epoch(gb[10])
sixteens = to_epoch(gb[16])
# %%
t = str(sixteens[0])

ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
imgboth = imageio.imread(ifile)
hsv = rgb2hsv(imgboth)
rboth = imgboth[:,:,0]
gboth = imgboth[:,:,1]
bboth = imgboth[:,:,2]
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

def movavg(data, window_width=5):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

wls = []
ylocs = []
sylocs = []
times = []
for t in np.random.choice(np.hstack([elevens, noons]), 25):
    if (pd.Timestamp(t, unit='s').month > 10) or (pd.Timestamp(t, unit='s').month < 5):
        continue
    t = str(t)
    ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
    imgboth = np.rot90(imageio.imread(ifile))
    hsv = rgb2hsv(imgboth)
    rboth = imgboth[:,:,0]
    gboth = imgboth[:,:,1]
    bboth = imgboth[:,:,2]


    plt.figure(figsize=(12,8))

    plt.imshow(imgboth)
    sstart = 1500
    send = 1800
    plt.axhline(sstart, c='grey')
    plt.axhline(send, c='grey')
    for xloc in [800]:#np.arange(100,1801,100):
        h = hsv[:,xloc, 0].copy()
        s = movavg(hsv[:,xloc, 1].copy())
        v = hsv[:,xloc, 2].copy()
        h[0:sstart] = np.nan
        h[send:] = np.nan
        s[0:sstart] = np.nan
        s[send:] = np.nan
        v[0:sstart] = np.nan
        v[send:] = np.nan


        try:
            lm = np.argwhere(v < np.nanmean(v))[0][0]
            if np.nanmean(s[sstart:sstart+10]) < np.nanmean(s[send-10:send]):
                mins = np.argwhere(s > np.nanmean(s))[0][0]
            else:
                mins = np.argwhere(s < np.nanmean(s))[0][0]
        except IndexError:
            continue
        # plt.plot(lm,yloc,  'c*') # 8 is first value that isn't one of the info lines
        gv = np.gradient(v)
        gv[0:lm-40] = np.nan
        gv[lm+40:] = np.nan

        gs = np.gradient(s)
        gs[0:mins-25] = np.nan
        gs[mins+25:] = np.nan
        if np.nanmean(s[sstart:sstart+10]) < np.nanmean(s[send-10:send]):
            gsloc = np.nanargmax(gs)
        else:
            gsloc = np.nanargmin(gs)
        try:
            gvloc = np.nanargmin(gv)
            plt.plot(xloc, gvloc, 'rs')
            plt.plot(xloc-10, gsloc, 'bd')
            if xloc == 800:
                plt.plot(xloc-400+200*h,np.arange(len(v)))

                plt.plot(xloc+200*v,np.arange(len(v)))
                plt.plot(xloc+200*v[lm],lm, '.')
                plt.plot(xloc+200*v[lm],gvloc, '.')

                plt.plot(xloc-200+200*s, np.arange(len(s)))
                plt.plot(xloc-200+200*s[mins],mins, '.')
                plt.plot(xloc-200+200*s[mins],gsloc, '.')

                ylocs.append(gvloc)
                sylocs.append(gsloc)
        except ValueError:
            if xloc == 800:
                ylocs.append(np.nan)
                sylocs.append(np.nan)
            continue
    wl = n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))]
    wls.append(wl.values)
    times.append(t)
    plt.ylim(2000,1000)
    plt.title(f"{t} --- {pd.to_datetime(t, unit='s').tz_localize(pytz.utc).tz_convert(pytz.timezone('US/Alaska'))}, WL = {wl.values} m NAVD88")
    print(t)
    # plt.savefig('unk' + t + '.png', bbox_inches='tight', dpi=150)
    plt.show()
# %%
imgboth.shape
x.shape
y.shape
x = np.arange(0, 250, .1)
y = np.arange(-150,40,.1)
times = np.array(times)
ylocs = np.array(ylocs)
sylocs = np.array(sylocs)
wls = np.array(wls)
# goodtimes = [1536868801, 1537041601, 1537128001, 1557606601, 1557779401, 1561840201, 1563827401, 1566676801, 1570566601, 1571428801]
# goods = np.in1d(times, [str(x) for x in goodtimes])
goods = np.arange(len(times))
# plt.plot(wls[goods], xlocs[goods], 'o', label='Jul–Oct 2019')
# plt.plot(wls[goods], ylocs[goods], 's', label='Aug–Oct 2018')
plt.scatter(wls[goods], x[sylocs[goods]], c=pd.DatetimeIndex([pd.Timestamp(x, unit='s') for x in times[goods].astype(int)]), label='Aug–Oct 2018')
plt.scatter(wls[goods], x[ylocs[goods]], c=pd.DatetimeIndex([pd.Timestamp(x, unit='s') for x in times[goods].astype(int)]), label='Aug–Oct 2018')
cbar = plt.colorbar()
cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%Y-%m-%d'))
plt.xlabel('Unalakleet water level [m NAVD88]')
plt.ylabel('Cross-shore position [m]')
xs = np.arange(1, 2.5, .1)
ys = 10*xs + 150
plt.plot(xs, ys, '--', lw=1, c='grey')

# plt.savefig('pixelpos.png', dpi=150)
# plt.ylim(ymin=600)
