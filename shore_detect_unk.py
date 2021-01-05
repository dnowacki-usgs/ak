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
""" load all rectified images and plot availability by hour """
ts = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'proc/rect/*' + camera + '.' + product + '.rect.png')]
# pd.Timestamp('7/11/2019 8:30:00 PM', tz='US/Alaska').tz_convert('utc').timestamp()
dt = pd.to_datetime(ts, unit='s').tz_localize(pytz.utc).tz_convert(pytz.timezone('US/Alaska'))
gb = dt.groupby(dt.hour)
hours = []
pics = []
for name in gb:
    hours.append(name)
    pics.append(len(gb[name]))
plt.plot(hours, pics, '*-')
plt.xticks(np.arange(0,25,2))
plt.xlim(0,24)
plt.grid()

def to_epoch(dtime):
    return (dtime.tz_convert(pytz.utc).tz_convert(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

nines = to_epoch(gb[9])
tens = to_epoch(gb[10])
elevens = to_epoch(gb[11])
noons = to_epoch(gb[12])
thirteens = to_epoch(gb[13])
fourteens = to_epoch(gb[14])
fifteens = to_epoch(gb[15])
sixteens = to_epoch(gb[16])
seventeens = to_epoch(gb[17])
# %%
""" process and detect shorelines for given images """
def movavg(data, window_width=5):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
x = np.arange(0, 250, .1)
y = np.arange(-150,40,.1)

ts = np.hstack([nines, tens, elevens, noons, thirteens, fourteens, fifteens, sixteens, seventeens])
# ENSURE ONLY 2018 DATA
# goods = np.array([pd.Timestamp(x, unit='s').isocalendar()[0] == 2018 for x in ts])# need the funny isocalendar for week of year computations later
# ENSURE ONLY 2019 DATA
[pd.Timestamp(x, unit='s') for x in ts]
goods = np.array([pd.Timestamp(x, unit='s').isocalendar()[0] == 2019 for x in ts]) & np.array([pd.Timestamp(x, unit='s').year == 2019 for x in ts])
ts = ts[goods]
goods = (np.array([pd.Timestamp(x, unit='s').weekofyear for x in ts]) <= 44) & (np.array([pd.Timestamp(x, unit='s').month for x in ts]) >= 5)
ts = ts[goods]

wvs = xr.Dataset()
_, index = np.unique(wavesunk['valid_time'], return_index=True)
wvs['time'] = wavesunk.valid_time.values[index]
wvs['swh'] = xr.DataArray(wavesunk['swh'].values[index], dims='time')

xlocs = np.arange(300,1301,100)

aux = xr.Dataset()
aux['time'] = xr.DataArray([pd.Timestamp(int(x), unit='s') for x in ts], dims='time')
aux['xlocs'] = xr.DataArray(xlocs, dims='xlocs')
aux['timestamp'] = xr.DataArray([int(x) for x in ts], dims='time')
aux['wl'] = n9468333['v'].reindex_like(aux, method='nearest', tolerance='5min')
aux['hs'] =  wvs['swh'].reindex_like(aux, method='nearest', tolerance='6hours')

aux['sylocs'] = xr.full_like(aux['wl'], np.nan, np.int)
aux['stds'] = xr.full_like(aux['wl'], np.nan)
aux['snow'] = xr.full_like(aux['wl'], np.nan)
# QAQC the values very close to the edge of the allowable limit
aux['min_ys'] = aux['wl'] * 8 + 151 - 150*.1
aux['sycoords'] = xr.full_like(aux['wl'], np.nan)
aux['linestds'] = xr.full_like(aux['wl'], np.nan)
aux['lvs'] = xr.DataArray(np.full((len(aux['time']), len(aux['xlocs'])), np.nan), dims=['time', 'xlocs'])

warnings.filterwarnings("ignore", category=RuntimeWarning)

n = 0
doplot = False

for t in ts:#np.random.choice(ts, 800, replace=False):
    t = str(t)
    ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
    img = np.rot90(imageio.imread(ifile))

    hsv = cv2.cvtColor(np.flip(img,2), cv2.COLOR_BGR2HSV).astype(float) / 255

    wl = aux['wl'].sel(time=pd.Timestamp(int(t), unit='s'))
    hs = aux['hs'].sel(time=pd.Timestamp(int(t), unit='s'))
    firstguess = np.where(x == int(8*wl.values + 151))[0][0] # eyeballed slope from a first pass
    xstart = firstguess - 125
    xend = firstguess + 125
    if doplot:
        plt.figure(figsize=(16,8))
        plt.imshow(img)
        plt.axhline(xstart, c='grey')
        plt.axhline(xend, c='grey')
    linevals = []#{xloc: np.nan for xloc in xlocs}
    snowval = hsv[1800:2000,750:1000,1].mean()
    for xloc in xlocs:
        h = movavg(hsv[:,xloc, 0].copy())
        s = movavg(hsv[:,xloc, 1].copy())
        v = hsv[:,xloc, 2].copy()
        h[0:xstart] = np.nan
        h[xend:] = np.nan
        s[0:xstart] = np.nan
        s[xend:] = np.nan
        v[0:xstart] = np.nan
        v[xend:] = np.nan

        if np.nanmean(s[xstart:xstart+10]) < np.nanmean(s[xend-10:xend]):
            mins = np.argwhere(s >= np.nanmean(s))[0][0]
        else:
            mins = np.argwhere(s <= np.nanmean(s))[0][0]
        if np.nanmean(h[xstart:xstart+10]) < np.nanmean(h[xend-10:xend]):
            minh = np.argwhere(h >= np.nanmean(h))[0][0]
        else:
            minh = np.argwhere(h <= np.nanmean(h))[0][0]

        gs = np.gradient(s)
        gs[0:mins-25] = np.nan
        gs[mins+25:] = np.nan

        gh = np.gradient(h)
        gh[0:minh-25] = np.nan
        gh[minh+25:] = np.nan
        if np.nanmean(s[xstart:xstart+10]) < np.nanmean(s[xend-10:xend]):
            gsloc = np.nanargmax(gs)
        else:
            gsloc = np.nanargmin(gs)
        linevals.append(gsloc)

        if np.nanmean(h[xstart:xstart+10]) < np.nanmean(h[xend-10:xend]):
            ghloc = np.nanargmax(gh)
        else:
            ghloc = np.nanargmin(gh)
        if doplot:
            plt.plot(xloc, gsloc, 's', c='C0')
            plt.plot(xloc+200*s, np.arange(len(s)), 'r')

        if xloc == 800:
            aux.sylocs[aux.timestamp == int(t)] = np.int(gsloc)
            aux.stds[aux.timestamp == int(t)] = np.nanstd(s)
            # times.append(t)
            aux.snow[aux.timestamp == int(t)] = snowval

    aux['lvs'].loc[dict(time=pd.Timestamp(int(t), unit='s'))] = linevals
    if doplot:
        plt.ylim(2000,1200)
        plt.text(1250,1850, f"{snowval:.3f}", color='white', va='top', fontsize=14)
        if snowval < 0.1:
            plt.text(1250, 1900, 'probable snow detected', color='white', va='top', fontsize=14)
        # plt.text(1250, 1900, f"week of year: {pd.to_datetime(t, unit='s').isocalendar()[1]}\nWL: {aux['wl'][aux['timestamp'] == int(t)].values[0]:.2f}\nHs: {aux['hs'][aux['timestamp'] == int(t)].values[0]:.2f}", color='white', va='top', fontsize=14)
        plt.title(f"{t} --- {pd.to_datetime(t, unit='s').tz_localize(pytz.utc).tz_convert(pytz.timezone('US/Alaska'))}, WL = {wl.values} m NAVD88, Hs = {hs.values:.2} m, % Hs = {100*hs.values/aux['hs'].mean().values:.2f}")
        plt.savefig('unk' + t + '.png', bbox_inches='tight')
        plt.show()
    print(t)
    print(f"{100*n/len(ts):.2f}")
    n+=1
# %%
""" WRITE TO NETCDF """
# aux.to_netcdf('aux_output.nc')
aux = xr.load_dataset('aux_output.nc')
# %%
# len(lvs)
# lvs = np.array(lvs)
# times = np.array(times)
# len(times)
# ylocs = np.array(ylocs)
# sylocs = np.array(sylocs)
# snow = np.array(snow)
# syotsulocs = np.array(syotsulocs)
# hylocs = np.array(hylocs)
# stds = np.array(stds)
# wls = np.array(wls)
# times.shape
# np.unique(times).shape

# df = xr.Dataset()
df = aux.where(~np.isnan(aux['stds']), drop=True)
# df['time'] = pd.DatetimeIndex([pd.Timestamp(int(x), unit='s') for x in times])
# df['timestamp'] = xr.DataArray([int(x) for x in times], dims='time')
# df['sylocs'] = aux['sylocs'].reindex_like(df)
# df['stds'] = aux['stds'].reindex_like(df)
# df['wls'] = aux['wl'].reindex_like(df)
# df['snow'] = aux['snow'].reindex_like(df)
# QAQC the values very close to the edge of the allowable limit
# df['min_ys'] = aux['min_ys'].reindex_like(df)
df.sylocs.values = df.sylocs.values.astype(int)
df['sycoords'] = xr.DataArray(x[::-1][df['sylocs'].values], dims='time')
df['linestds'] = df['lvs'].std(dim='xlocs')
xs = np.arange(df['wl'].min(), df['wl'].max(), .1)
ys = 8*xs + 151
# %%
# plt.hist(df.linestds, bins=np.arange(0, 100, 10))
# plt.plot(df.snow, df.linestds, '.')
# df.stds.plot(marker='.', ls='none')

goods = (np.abs(x[df['sylocs'].values] - df['min_ys'].values) > 6) & (df['stds'].values >= 0.04) & (df['snow'].values > 0.1) #& (df['linestds'].values < 15)
sum(goods)
# goods = np.abs(x[df['sylocs']] - df['min_ys']) > 4
# goods = stds >= 0.04
sum(goods)/len(df['wl'])
df
plt.scatter(df['wl'], x[df['sylocs'].values], c=df['stds'])
plt.plot(df['wl'][goods], x[df['sylocs'].values][goods], 'r.')
# plt.plot(df['wls'][stds < 0.04], x[df['sylocs']][stds < 0.04], 'gs')
plt.fill_between(xs, ys-150*.1, ys+150*.1, color='lightgrey', zorder=0)
plt.colorbar()

# %%
goods = (np.abs(x[df['sylocs'].values] - df['min_ys'].values) > 6) & (df['stds'].values >= 0.04) & (df['snow'].values > 0.1) #& (df['linestds'].values < 15)

plt.figure(figsize=(8,6))

# plt.subplot(1,3,1)
# plt.scatter(wls[goods], x[hylocs[goods]], c=pd.DatetimeIndex([pd.Timestamp(x, unit='s') for x in times[goods].astype(int)]), label='Aug–Oct 2018', marker='s')
# cbar = plt.colorbar()
# cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%Y-%m-%d'))

# plt.plot(xs, ys, '--', lw=1, c='grey')

# plt.subplot(1,3,2)
plt.scatter(df['wl'][goods], df['sycoords'][goods], c=df['time'][goods])
# plt.scatter(wls[goods], x[syotsulocs[goods]], c=pd.DatetimeIndex([pd.Timestamp(x, unit='s') for x in times[goods].astype(int)]), label='Aug–Oct 2018', marker='s')
cbar = plt.colorbar()
cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%Y-%m-%d'))
plt.plot(xs, ys, '--', lw=1, c='grey')
plt.xlabel('Unalakleet water level [m NAVD88]')
siegel = djn.siegel(df['wl'][goods], x[df['sylocs'].values[goods]])
plt.plot(xs, siegel[0]*xs + siegel[1])
plt.title(f"{siegel[0]:.3f} {siegel[1]:.3f}")
plt.fill_between(xs, ys-150*.1, ys+150*.1, color='lightgrey', zorder=0)
plt.ylabel('Cross-shore position [m]')

plt.show()#
# %%
# %%
plt.figure(figsize=(24,8))
n = 1
slopes = []
weeks = []
r2s = []
theils = []
ns = []
doplot = True

for month, gb in df.groupby(df['time'].dt.week):
    print(month)
    weeks.append(month)
    goods = (np.abs(gb['sycoords'] - gb['min_ys']) > 6) & (gb['stds'] >= 0.04) & (gb['snow'] > 0.1) # & (df['linestds'] < 15)
    xs = np.arange(df['wl'].min(), df['wl'].max(), .1)
    ys = 8*xs + 151
    if sum(goods) > 1:
        theil = scipy.stats.theilslopes(gb['wl'][goods], gb['sycoords'][goods])
        theils.append(theil)
        siegel = djn.siegel(gb['sycoords'][goods], gb['wl'][goods] )
        slopes.append(siegel)
    else:
        theils.append((np.nan, np.nan, np.nan, np.nan))
        slopes.append((siegel, siegel))

    r2s.append(np.corrcoef(gb['sycoords'][goods], gb['wl'][goods])[0,1]**2)
    ns.append(np.sum(goods))

    if doplot:
        plt.subplot(4,8,n)
        plt.scatter(gb['sycoords'][goods], gb['wl'][goods], c=gb['time'][goods], label='Aug–Oct 2018')
        if n == 28:
            plt.xlabel('Cross-shore position [m]')
        plt.plot(ys, siegel[0]*ys + siegel[1])
        plt.title(f"{month}: {siegel[0]:.3f}")
        # plt.fill_between(xs, ys-150*.1, ys+150*.1, color='lightgrey', zorder=0)
        xlims = plt.xlim(65, 110)
        ylims = plt.ylim(0,5)
        # plt.plot(np.array([80, 100]), -0.08992005276821789*np.array([80, 100])+ 9.418220795056857)
        if (n != 1) and (n != 9) and (n != 17) and (n != 25):
            plt.gca().set_yticklabels([])
        if n < 25:
            plt.gca().set_xticklabels([])
        if n == 9:
            plt.ylabel('Unalakleet water level [m NAVD88]')
        plt.text(xlims[0] + 0.1*np.diff(xlims), ylims[0] + 0.9*np.diff(ylims),
                 f"N = {sum(goods.values)}, r2 = {r2s[-1]:.2f}\nWL: {n9468333.v.sel(time=slice(gb['time'][0], gb['time'][-1])).mean().values:.2f} m, Hs: {wvs['swh'].sel(time=slice(gb['time'][0], gb['time'][-1])).mean().values:.2f} m", va='top')


    n += 1
if doplot:
    plt.suptitle(f"{df['time'][0].values} — {df['time'][-1].values}")
    # plt.savefig(f'week_of_year_{df.time[0].dt.year.values}.png', dpi=150, bbox_inches='tight')
    plt.show()
# %%
df.time[0]
print(df.timestamp[0].values)
# df['localtime'] = xr.DataArray(pd.DatetimeIndex(df['time'].values, tz='utc').tz_convert('US/Alaska').tz_localize(None), dims='time')
# df= df.drop('localtime')
goods = (df.time.dt.week == 28) & (np.abs(df['sycoords'] - df['min_ys']) > 4) & (df['stds'] >= 0.04) & (df['linestds'] < 15)
plt.figure(figsize=(12,8))
df['time'][goods]
goodloctime = pd.DatetimeIndex(df['time'][goods].values, tz='utc').tz_convert('US/Alaska').tz_localize(None)

bounds = mdates.date2num(pd.date_range(goodloctime.min().date(), goodloctime.max().date()+pd.Timedelta('1d'), freq='1d'))
norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
plt.scatter(df['sycoords'][goods], df['wl'][goods], c=mdates.date2num(goodloctime), label='Auto-picked shoreline', norm=norm, cmap=plt.cm.plasma)
topo = pd.read_csv(fildir + 'gis/Unk19_ForDN/topo/asc/unk19_topo_argusFOV.csv')

plt.plot(np.array([84.64998139, 83.34286574, 81.93244546, 80.49556624, 78.99894828,
       77.62351934, 76.42739224, 75.17229157, 73.85486761, 72.47395137,
       71.08193445, 69.82481399, 68.50898509, 67.19738887, 65.95328867,
       64.73967702, 63.56128335, 61.9854817 , 60.48641425, 59.31981475,
       58.39288996, 57.29982095, 56.13251799, 55.53489228, 54.81238208,
       53.93530782, 53.10355508, 52.15583683, 51.18809928, 49.92436964,
       48.65321357, 47.28473059, 45.86469818, 44.34012278, 43.04489787,
       41.70704712, 40.58187956, 39.32785631, 38.26467525, 37.62736594,
       37.21211209, 36.86540908]),
       np.array([1.634, 1.842, 2.024, 2.158, 2.404, 2.578, 2.664, 2.795, 2.853,
       3.014, 3.189, 3.309, 3.353, 3.509, 3.516, 3.457, 3.484, 3.532,
       3.716, 4.079, 4.215, 4.311, 4.445, 4.556, 4.591, 4.546, 4.62 ,
       4.7  , 4.759, 4.772, 4.895, 5.006, 5.056, 5.137, 5.268, 5.464,
       5.544, 6.025, 6.019, 6.2  , 6.475, 6.77 ]), '*-', label='Survey line 1')
plt.plot(np.array([83.93740293, 82.82702754, 81.65197362, 80.52749956, 79.47963826,
78.36083102, 77.32731128, 76.21401224, 75.0541543 , 74.15115467,
73.07802699, 72.17498654, 71.24757496, 70.74084649, 70.74409102,
70.73715865, 70.70369302, 70.35191784, 69.46209575, 68.40731204,
67.49012448, 66.59527333, 65.76084082, 64.86594984, 63.88844651,
63.26323635, 62.42584169, 61.31380456, 60.22776082, 59.50374483,
58.83034587, 58.24000292, 57.7445052 , 57.26210936, 56.5969654 ,
56.22686849, 55.54193283, 55.0996639 , 54.25365887, 53.16207342,
52.00724454, 50.88132542, 49.89084762, 48.92482257, 48.04412218,
47.09727828, 46.34220533, 45.35526937, 44.41815906, 43.42207152,
42.50598494, 41.66157132, 40.68764433, 39.84419166, 39.08973788,
38.25875796, 38.07074419, 37.56686637, 37.53644718, 37.28233933,
37.01196439, 36.35991734, 35.66203842, 35.38083886, 35.10639945,
34.70512911]),
np.array([1.782, 1.94 , 2.027, 2.232, 2.374, 2.442, 2.631, 2.737, 2.765,
       2.923, 3.02 , 3.12 , 3.227, 3.276, 3.275, 3.28 , 3.262, 3.288,
       3.429, 3.483, 3.47 , 3.47 , 3.486, 3.397, 3.358, 3.598, 3.565,
       3.674, 3.793, 3.937, 3.995, 4.138, 4.31 , 4.405, 4.547, 4.536,
       4.699, 4.695, 4.702, 4.625, 4.64 , 4.707, 4.778, 4.844, 4.928,
       5.025, 5.141, 5.182, 5.333, 5.408, 5.448, 5.525, 5.541, 5.732,
       5.756, 6.026, 6.255, 6.234, 6.205, 6.224, 6.516, 6.698, 7.044,
       7.342, 7.316, 7.383]),
        '*-', label='Survey line 2')
import matplotlib.dates as mdates
cb = plt.colorbar(label='Local date (black line indicates time of topo survey)')
loc = mdates.AutoDateLocator()
cbylims = cb.ax.get_ylim()
print(cbylims)
meantime = pd.DatetimeIndex(topo['GNSS Vector Observation.Start Time'], tz='US/Alaska').tz_localize(None).mean()
cb.ax.plot([cbylims[0], cbylims[1]], mdates.date2num([meantime, meantime]),c='k', lw=1)#
cb.ax.yaxis.set_major_locator(loc)
cb.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


plt.xlim(70,100)
plt.ylim(1,4.5)
plt.xlabel('Cross-shore distance [m]')
plt.ylabel('Elevation [m NAVD88]')
plt.legend()

""" NOTE all times in this plot are LOCAL """
djn.set_fontsize(plt.gcf(), 14)
plt.savefig('beach_slope_with_topo_survey.png', dpi=300, bbox_inches='tight')
plt.show()
# plt.plot(np.array([80, 100]), -0.08992005276821789*np.array([80, 100])+ 9.418220795056857)
# %%
n_points = 10
aa = np.linspace(-5, 5, n_points)
bb = np.linspace(-1.5, 1.5, n_points)

def cost(a, b):
    return a + b

z = []
for a in aa:
    for b in bb:
        z.append(cost(a, b))

z = np.reshape(z, [len(aa), len(bb)])

fig, ax = plt.subplots()
im = ax.pcolormesh(aa, bb, z)
cb = fig.colorbar(im)
cb.ax.plot([-5, 5], [0.5, 0.5],'w')

ax.axis('tight')
plt.show()
# %%


goods = (df.time.dt.week == 28) & (np.abs(df['sycoords'] - df['min_ys']) > 4) & (df['stds'] >= 0.04) & (df['linestds'] < 15)
n = 1
plt.figure(figsize=(15,10))
for t in df.timestamp[goods]:
    plt.subplot(np.ceil(np.sqrt(sum(goods.values))).astype(int), np.ceil(np.sqrt(sum(goods.values))).astype(int), n)
    n+=1
    ifile = fildir + 'proc/rect/' + str(int(t.values)) + '.' + camera + '.' + product + '.rect.png'
    imgboth = np.rot90(imageio.imread(ifile))
    plt.imshow(imgboth)
    plt.ylim(2000,1000)
    plt.plot(800, df['sylocs'][df['timestamp']==t.values], 'rs')
plt.show()

# %%
ns = np.array(ns)
slopes = np.array(slopes)
weeks = np.array(weeks)
r2s = np.array(r2s)
theils = np.array(theils)
#
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
goods = np.isfinite(weeks)#weeks < 46#r2s > 0.
# plt.plot(weeks[goods], slopes[:,0][goods], '.')
# plt.plot(weeks[goods], theils[:,0][goods], '.')
plt.errorbar(weeks[goods], theils[goods,0], np.vstack([theils[goods,0]-theils[goods,2], theils[goods,3]-theils[goods,0]]))
plt.fill_between(weeks[goods], theils[goods,2], theils[goods,3], color='lightgrey')

for x in np.where(goods)[0]:
    plt.text(weeks[x], theils[x,3]+.005, ns[x], ha='center')

plt.ylabel('Foreshore slope')
plt.xlabel('Week of year ')
# plt.ylim(0,.15)
plt.ylim(0.02,.15)
plt.grid()
# plt.subplot(2,1,2)
# 175*theils[goods, 0]+ theils[goods,1]
# elev = 175*theils[goods, 0]+ theils[goods,1]
# elevlo = 175*theils[goods, 2]+ theils[goods,1]
# elevhi = 175*theils[goods, 3]+ theils[goods,1]
#
# plt.errorbar(weeks[goods], elev, np.vstack([elev-elevlo, elevhi-elev]))
# plt.ylabel('')
# plt.ylim(-20,20)
plt.grid()
# %%


pltdict = {a: b.mean().values for a, b in wvs.swh.groupby(wvs.time.dt.week)}

theils.shape
plt.plot(theils[:,0], [pltdict[a] for a in weeks], '.')
# for w, s, r, ls, us in zip(weeks[goods], theils[:,0][goods], r2s[goods], theils[:,2][goods],theils[:,3][goods],):
    # plt.text(w, s, f"{s:.2f}pm{(s-ls):.2f}", ha='center')
# %%



def imadjust(img):
    lower = np.percentile(img, 1)
    upper = np.percentile(img, 99)
    out = (img - lower) * (255 / (upper - lower))
    return np.clip(out, 0, 255, out) # in-place clipping
# def imhist(im):
#   # calculates normalized histogram of an image
# 	m, n = im.shape
# 	h = [0.0] * 256
# 	for i in range(m):
# 		for j in range(n):
# 			h[im[i, j]]+=1
# 	return np.array(h)/(m*n)
#
# def cumsum(h):
# 	# finds cumulative sum of a numpy array, list
# 	return [sum(h[:i+1]) for i in range(len(h))]
#
# def histeq(im):
# 	#calculate Histogram
# 	h = imhist(im)
# 	cdf = np.array(cumsum(h)) #cumulative distribution function
# 	sk = np.uint8(255 * cdf) #finding transfer function values
# 	s1, s2 = im.shape
# 	Y = np.zeros_like(im)
# 	# applying transfered values for each pixels
# 	for i in range(0, s1):
# 		for j in range(0, s2):
# 			Y[i, j] = sk[im[i, j]]
# 	H = imhist(Y)
# 	#return transformed image, original and new istogram,
# 	# and transform function
# 	return Y , h, H, sk

ts = np.hstack([tens, elevens, noons, thirteens, fourteens, fifteens])
# ENSURE ONLY 2018 DATA
goods = np.array([pd.Timestamp(x, unit='s').isocalendar()[0] == 2018 for x in ts])# need the funny isocalendar for week of year computations later
# ENSURE ONLY 2019 DATA
# goods = np.array([pd.Timestamp(x, unit='s').isocalendar()[0] == 2019 for x in ts])
ts = ts[goods]
goods = (np.array([pd.Timestamp(x, unit='s').weekofyear for x in ts]) <= 48) & (np.array([pd.Timestamp(x, unit='s').month for x in ts]) >= 5)
ts = ts[goods]
cannyloc = []
from skimage.filters import threshold_otsu




for t in np.random.choice(ts, 10): #ts:
    t = str(t)
    ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
    img = np.rot90(imageio.imread(ifile))
    rowlo = 1200
    rowhi = 2000
    collo = 400
    colhi = 1000
    gray = cv2.cvtColor(np.flip(img,2), cv2.COLOR_BGR2GRAY)[rowlo:rowhi,collo:colhi]
    plt.figure(figsize=(10,8))
    # plt.imshow(gray)
    gray2 = imadjust(gray).astype(np.uint8)
    # plt.imshow(cv2.equalizeHist(gray))
    # plt.imshow(cv2.equalizeHist(gray2))
    eq = cv2.equalizeHist(gray2)
    thresh = threshold_otsu(eq)*.9
    binary = gray2 > thresh
    canny = skimage.feature.canny(binary, sigma=3)
    # plt.imshow(canny)
    # np.argwhere(canny)
    r, c = np.where(canny)

    loc = []
    for n in range(canny.shape[1]):
        try:
            loc.append(np.where(canny[:,n])[0][0])
        except:
            loc.append(np.nan)
    loc = np.array(loc)
    # get the 800 value from the original image size

    # plt.imshow(skimage.morphology.thin(canny))
    plt.imshow(gray, cmap=plt.cm.gray)
    plt.plot(range(canny.shape[1]), loc, 'r')
    plt.plot(800-collo, loc[800-collo], 'bs')
    cannyloc.append(loc[800-collo])
    plt.show()
    # plt.imshow(gray)
    # plt.colorbar()
    # plt.imshow(imadjust(gray))
    # plt.colorbar()
    #
    # # edges = skimage.feature.canny(hsv[:,:,1], sigma=3)
    # # plt.imshow(edges)
    # plt.show()
# %%
    hsv = cv2.cvtColor(np.flip(img,2), cv2.COLOR_BGR2HSV)[rowlo:rowhi,collo:colhi]
    rgb = img[rowlo:rowhi,collo:colhi].astype(float)
    RmB = rgb[:,400,0]-rgb[:,400,2]
    kde = scipy.stats.gaussian_kde(RmB)
    x = np.linspace(RmB.min(), RmB.max(), 100)
    p = kde(x)
    plt.plot(x, p)
    peakidx = scipy.signal.find_peaks(p)
    threshold_otsu(RmB)
    plt.plot(RmB, range(len(RmB)), )
    plt.imshow(gray, cmap=plt.cm.gray)
    plt.plot()
# %%
""" look at single image and inspect different channels """
for t in np.random.choice(ts, 1):
    t = str(t)
    ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
    imgboth = np.rot90(imageio.imread(ifile))

    hsv = cv2.cvtColor(np.flip(imgboth,2), cv2.COLOR_BGR2HSV).astype(float) / 255
    r = imgboth[:,:,0].astype(float)
    g = imgboth[:,:,1].astype(float)
    b = imgboth[:,:,2].astype(float)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    plt.figure(figsize=(16,8))
    plt.imshow(imgboth)
    plt.ylim(2000,1000)
    # plt.plot(r[:,800]/b[:,800])
    print(s[1000:1200,750:1000].mean())
    print(s[1800:2000,750:1000].mean(), )
    plt.plot(200 +r[:,800], range(len(h[:,800])),'r')
    plt.plot(400+ g[:,800], range(len(h[:,800])),'g')
    plt.plot(600+ b[:,800], range(len(h[:,800])),'b')
    plt.plot(800 + (r[:,800]-b[:,800]), range(len(h[:,800])),'r')
    plt.plot(1000+ 200*h[:,800], range(len(h[:,800])),'r')
    plt.plot(1200+ 200*s[:,800], range(len(h[:,800])),'r')
    plt.plot(1400+ 200*v[:,800], range(len(h[:,800])),'r')
