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
import matplotlib.dates as mdates
from joblib import Parallel, delayed
import multiprocessing

fildir = '/Volumes/Backstaff/field/unk/'

n9468333 = xr.load_dataset(fildir + 'n9468333.nc')
wavesunk = xr.load_dataset(fildir + 'ak_4m.nc')
paun = xr.load_dataset(fildir + 'paun_timeseries.nc')
topo = pd.read_csv(fildir + 'gis/Unk19_ForDN/topo/asc/unk19_topo_argusFOV_with_local.csv')
bathy = pd.read_csv(fildir + 'gis/Unk19_ForDN/bathy/csv/UNK19_bathy_with_local.csv')
camera = 'both'
product = 'timex'


# %%
""" load all rectified images and plot availability by hour """
ts = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'proc/rect/' + product + '/*' + camera + '.' + product + '.rect.png')]
# pd.Timestamp('7/11/2019 8:30:00 PM', tz='US/Alaska').tz_convert('utc').timestamp()
dt = pd.to_datetime(ts, unit='s').tz_localize(pytz.utc).tz_convert(pytz.timezone('US/Alaska'))
gb = dt.groupby(dt.hour)
hours = []
pics = []
for name in gb:
    hours.append(name)
    pics.append(len(gb[name]))
plt.bar(np.array(hours)+.5, pics)
plt.xticks(np.arange(0,25,2))
plt.xlabel('Hour of day')
plt.ylabel('Number of photos')
plt.xlim(0,24)
plt.grid()
plt.show()

def to_epoch(dtime):
    return (dtime.tz_convert(pytz.utc).tz_convert(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# eights = to_epoch(gb[8]) # restrict to summer months
# nines = to_epoch(gb[9])
# tens = to_epoch(gb[10])
# elevens = to_epoch(gb[11])
# noons = to_epoch(gb[12])
# thirteens = to_epoch(gb[13])
# fourteens = to_epoch(gb[14])
# fifteens = to_epoch(gb[15])
# sixteens = to_epoch(gb[16])
# seventeens = to_epoch(gb[17])
# eighteens = to_epoch(gb[18]) # restrict to summer months
# nineteens = to_epoch(gb[19])
# twenties = to_epoch(gb[20])
print(len(ts))
# %%
""" pre-compute HSV images """
# def dohsv(t):
#     t = str(t)
#     ifile = fildir + 'proc/rect/' + product + '/' + t + '.' + camera + '.' + product + '.rect.png'
#     img = np.rot90(imageio.imread(ifile))
#     hsv = cv2.cvtColor(np.flip(img,2), cv2.COLOR_BGR2HSV)
#     ofile = fildir + 'proc/rect/' + product + '/hsv/' + t + '.' + camera + '.' + product + '.rect.png'
#     print(ofile)
#     imageio.imwrite(ofile, hsv, format='png', optimize=True)
#     return ofile
#
# for t in ts:
#     dohsv(t)
# %%
""" process and detect shorelines for given images """
def movavg(data, window_width=5):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

x = np.linspace(250,0,2501)
y = np.linspace(40,-150,1901)

# we can just use ts now that we don't have any super dark images left
# ts = np.hstack([nines, tens, elevens, noons, thirteens, fourteens, fifteens, sixteens, seventeens, eighteens, nineteens, twenties])
# len(ts)
# ENSURE ONLY 2018 DATA
# goods = np.array([pd.Timestamp(x, unit='s').isocalendar()[0] == 2018 for x in ts])# need the funny isocalendar for week of year computations later
# ENSURE ONLY 2019 DATA
# goods = np.array([pd.Timestamp(x, unit='s').isocalendar()[0] == 2019 for x in ts]) & np.array([pd.Timestamp(x, unit='s').year == 2019 for x in ts])
# ts = ts[goods]
ts = np.array([int(x) for x in ts])
goods = (np.array([pd.Timestamp(int(x), unit='s').weekofyear for x in ts]) <= 44) & (np.array([pd.Timestamp(int(x), unit='s').month for x in ts]) >= 5)
ts = ts[goods]

wvs = xr.Dataset()
_, index = np.unique(wavesunk['valid_time'], return_index=True)
wvs['time'] = wavesunk.valid_time.values[index]
for var in ['swh', 'perpw']:
    wvs[var] = xr.DataArray(wavesunk[var].values[index], dims='time')

xlocs = np.arange(300,1301,50)

aux = xr.Dataset()
aux['time'] = xr.DataArray([pd.Timestamp(int(x), unit='s') for x in ts], dims='time')
aux['xlocs'] = xr.DataArray(xlocs, dims='xlocs')
aux['timestamp'] = xr.DataArray([int(x) for x in ts], dims='time')
aux['wl'] = n9468333['v'].reindex_like(aux, method='nearest', tolerance='5min')
aux['hs'] =  wvs['swh'].reindex_like(aux, method='nearest', tolerance='6hours')
aux['tp'] =  wvs['perpw'].reindex_like(aux, method='nearest', tolerance='6hours')
aux['wind_speed'] = paun['wind_speed'].reindex_like(aux, method='nearest', tolerance='5min')

aux['sylocs'] = xr.full_like(aux['wl'], np.nan, np.int)
aux['hsobloc'] = xr.full_like(aux['wl'], np.nan, np.int)
aux['ssobloc'] = xr.full_like(aux['wl'], np.nan, np.int)
aux['vsobloc'] = xr.full_like(aux['wl'], np.nan, np.int)
aux['hotsuloc'] = xr.full_like(aux['wl'], np.nan, np.int)
aux['sotsuloc'] = xr.full_like(aux['wl'], np.nan, np.int)
aux['votsuloc'] = xr.full_like(aux['wl'], np.nan, np.int)
aux['stds'] = xr.full_like(aux['wl'], np.nan)
aux['snow'] = xr.full_like(aux['wl'], np.nan)
# QAQC the values very close to the edge of the allowable limit
aux['min_ys'] = aux['wl'] * 8 + 155 - 155*.1
# aux['sycoords'] = xr.full_like(aux['wl'], np.nan)
aux['linestds'] = xr.full_like(aux['wl'], np.nan)
aux['lvs'] = xr.DataArray(np.full((len(aux['time']), len(aux['xlocs'])), np.nan), dims=['time', 'xlocs'])
aux['hsoblvs'] = xr.DataArray(np.full((len(aux['time']), len(aux['xlocs'])), np.nan), dims=['time', 'xlocs'])
aux['ssoblvs'] = xr.DataArray(np.full((len(aux['time']), len(aux['xlocs'])), np.nan), dims=['time', 'xlocs'])
aux['vsoblvs'] = xr.DataArray(np.full((len(aux['time']), len(aux['xlocs'])), np.nan), dims=['time', 'xlocs'])
warnings.filterwarnings("ignore", category=RuntimeWarning)

def procimg(t, n):
    t = str(t)
    ifile = fildir + 'proc/rect/' + product + '/' + t + '.' + camera + '.' + product + '.rect.png'
    img = np.rot90(imageio.imread(ifile))

    hsv = cv2.cvtColor(np.flip(img,2), cv2.COLOR_BGR2HSV).astype(float) / 255
    # ihsv = fildir + 'proc/rect/' + product + '/hsv/' + t + '.' + camera + '.' + product + '.rect.png'
    # hsv = imageio.imread(ihsv) / 255
    # print('mean', (hsv - hsv2).mean())
    # print('median', (hsv - hsv2).std())

    wl = aux['wl'].sel(time=pd.Timestamp(int(t), unit='s'))
    hs = aux['hs'].sel(time=pd.Timestamp(int(t), unit='s'))
    firstguess = np.where(x == int(-9.12*wl.values + 97.77))[0][0] # eyeballed slope from a first pass
    xstart = firstguess - 110
    xend = firstguess + 100
    if doplot:
        plt.figure(figsize=(16,8))
        plt.imshow(img)
        plt.axhline(xstart, c='grey')
        plt.axhline(xend, c='grey')
    linevals = []
    hsoblinevals = []
    ssoblinevals = []
    vsoblinevals = []
    snowval = hsv[1800:2000,750:1000,1].mean()
    for xloc in xlocs:
        # used to do moving average, switch to gaussian blur
        # h = movavg(hsv[:,xloc, 0].copy())
        # s = movavg(hsv[:,xloc, 1].copy())
        # v = hsv[:,xloc, 2].copy()
        h = scipy.ndimage.gaussian_filter1d(hsv[:,xloc, 0], 7)
        s = scipy.ndimage.gaussian_filter1d(hsv[:,xloc, 1], 7)
        v = scipy.ndimage.gaussian_filter1d(hsv[:,xloc, 2], 7)

        hsob = np.abs(scipy.ndimage.sobel(h))
        ssob = np.abs(scipy.ndimage.sobel(s))
        vsob = np.abs(scipy.ndimage.sobel(v))

        hsob[0:xstart] = np.nan
        hsob[xend:] = np.nan
        ssob[0:xstart] = np.nan
        ssob[xend:] = np.nan
        vsob[0:xstart] = np.nan
        vsob[xend:] = np.nan

        hsobloc = np.nanargmax(hsob)
        ssobloc = np.nanargmax(ssob)
        vsobloc = np.nanargmax(vsob)

        hsoblinevals.append(hsobloc)
        ssoblinevals.append(ssobloc)
        vsoblinevals.append(vsobloc)

        h[0:xstart] = np.nan
        h[xend:] = np.nan
        s[0:xstart] = np.nan
        s[xend:] = np.nan
        v[0:xstart] = np.nan
        v[xend:] = np.nan

        # otsu will fail when all values are the same (this happens on very dark images)
        try:
            hotsuloc = np.where(np.diff(h > skimage.filters.threshold_otsu(h[xstart:xend])))[0][0] # find first otsu threshold breakpoint
        except ValueError:
            hotsuloc = np.nan
        except IndexError:
            hotsuloc = np.nan
        try:
            sotsuloc = np.where(np.diff(s > skimage.filters.threshold_otsu(s[xstart:xend])))[0][0] # find first otsu threshold breakpoint
        except ValueError:
            sotsuloc = np.nan
        except IndexError:
            sotsuloc = np.nan
        try:
            votsuloc = np.where(np.diff(v > skimage.filters.threshold_otsu(v[xstart:xend])))[0][0] # find first otsu threshold breakpoint
        except ValueError:
            votsuloc = np.nan

        if np.nanmean(h[xstart:xstart+10]) <= np.nanmean(h[xend-10:xend]):
            try:
                minh = np.argwhere(h >= np.nanmean(h))[0][0]
            except IndexError:
                minh = xstart
        else:
            try:
                minh = np.argwhere(h <= np.nanmean(h))[0][0]
            except IndexError:
                minh = xstart

        if np.nanmean(s[xstart:xstart+10]) <= np.nanmean(s[xend-10:xend]):
            try:
                mins = np.argwhere(s >= np.nanmean(s))[0][0]
            except IndexError:
                mins = xstart
        else:
            try:
                mins = np.argwhere(s <= np.nanmean(s))[0][0]
            except IndexError:
                mins = xstart

        gh = np.gradient(h)
        gh[0:minh-25] = np.nan
        gh[minh+25:] = np.nan

        gs = np.gradient(s)
        gs[0:mins-25] = np.nan
        gs[mins+25:] = np.nan

        if np.nanmean(h[xstart:xstart+10]) < np.nanmean(h[xend-10:xend]):
            ghloc = np.nanargmax(gh)
        else:
            ghloc = np.nanargmin(gh)

        if np.nanmean(s[xstart:xstart+10]) < np.nanmean(s[xend-10:xend]):
            gsloc = np.nanargmax(gs)
        else:
            gsloc = np.nanargmin(gs)
        linevals.append(gsloc)

        if doplot:
            # plt.plot(xloc, gsloc, 's', c='C0')
            plt.plot(xloc, ssobloc, 's', c='C0')
            plt.plot(xloc, vsobloc, 'D', c='C1')
            plt.plot(xloc+200*s, np.arange(len(s)), 'r')

        if xloc == 800:
            aux.hsobloc[aux.timestamp == int(t)] = np.int(hsobloc)
            aux.ssobloc[aux.timestamp == int(t)] = np.int(ssobloc)
            aux.vsobloc[aux.timestamp == int(t)] = np.int(vsobloc)
            aux.hotsuloc[aux.timestamp == int(t)] = np.int(hotsuloc)
            aux.sotsuloc[aux.timestamp == int(t)] = np.int(sotsuloc)
            aux.votsuloc[aux.timestamp == int(t)] = np.int(votsuloc)
            aux.sylocs[aux.timestamp == int(t)] = np.int(gsloc)
            aux.stds[aux.timestamp == int(t)] = np.nanstd(s)
            # times.append(t)
            aux.snow[aux.timestamp == int(t)] = snowval

    aux['lvs'].loc[dict(time=pd.Timestamp(int(t), unit='s'))] = linevals
    aux['hsoblvs'].loc[dict(time=pd.Timestamp(int(t), unit='s'))] = hsoblinevals
    aux['ssoblvs'].loc[dict(time=pd.Timestamp(int(t), unit='s'))] = ssoblinevals
    aux['vsoblvs'].loc[dict(time=pd.Timestamp(int(t), unit='s'))] = vsoblinevals
    if doplot:
        plt.ylim(2000,1200)
        plt.text(1250,1850, f"snowval: {snowval:.3f}\nSsigma = {np.nanstd(ssoblinevals):.3f}\nVsigma = {np.nanstd(vsoblinevals):.3f}", color='white', va='top', fontsize=14)
        if snowval < 0.1:
            plt.text(1250, 1900, 'probable snow detected', color='white', va='top', fontsize=14)
        # plt.text(1250, 1900, f"week of year: {pd.to_datetime(t, unit='s').isocalendar()[1]}\nWL: {aux['wl'][aux['timestamp'] == int(t)].values[0]:.2f}\nHs: {aux['hs'][aux['timestamp'] == int(t)].values[0]:.2f}", color='white', va='top', fontsize=14)
        plt.title(f"{t} --- {pd.to_datetime(t, unit='s').tz_localize(pytz.utc).tz_convert(pytz.timezone('US/Alaska'))}, WL = {wl.values} m NAVD88, Hs = {hs.values:.2} m, % Hs = {100*hs.values/aux['hs'].mean().values:.2f}")
        plt.savefig(fildir + 'proc/shore/' + t + '.png', bbox_inches='tight', dpi=300)
        plt.show()
    print(f"{t}: {100*n/len(ts):.2f}%")
    # n+=1

n = 0
doplot = True
for t in np.random.choice(ts, 100, replace=False):
    procimg(t, n)
    n += 1
#
# Parallel(n_jobs=4, backend='multiprocessing')(
#     delayed(procimg)(t, n) for t in ts[0:4])

aux = aux.sortby('time') # because of the concat, time is not monotonically increasing.
# %%
""" WRITE TO NETCDF """
# aux.to_netcdf(fildir + 'aux_output.nc')
aux = xr.load_dataset(fildir + 'aux_output.nc')
# %%
""" make a new photo bar graph with only good images """
dt = pd.to_datetime(aux.time.values).tz_localize(pytz.utc).tz_convert(pytz.timezone('US/Alaska'))
gb = dt.groupby(dt.hour)
hours = []
pics = []
for name in gb:
    hours.append(name)
    pics.append(len(gb[name]))
plt.bar(np.array(hours)+.5, pics)
plt.xticks(np.arange(0,25,2))
plt.xlabel('Hour of day')
plt.ylabel('Number of photos')
plt.xlim(0,24)
plt.grid()
plt.show()

# %%
df = aux.where(~np.isnan(aux['stds']), drop=True)
df.sylocs.values = df.sylocs.values.astype(int)
df.ssobloc.values = df.ssobloc.values.astype(int)
df['sycoords'] = xr.DataArray(x[df['sylocs'].values], dims='time')
df['ssobcoord'] = xr.DataArray(x[df['ssobloc'].values], dims='time')
df['linestds'] = df['lvs'].std(dim='xlocs')
xs = np.arange(df['wl'].min(), df['wl'].max(), .1)
# ys = 8.8*xs + 155
ys = -9.126*xs + 97.77
# %%
# THE OLD WAY WITH SYLOCS
# goods = (np.abs(x[df['sylocs'].values] - df['min_ys'].values) > 6) & (df['stds'].values >= 0.04) & (df['snow'].values > 0.1) #& (df['linestds'].values < 15)
# plt.scatter(df['wl'], x[df['sylocs'].values], c=df['stds'])
# plt.plot(df['wl'][goods], x[df['sylocs'].values][goods], 'r.')

# THE NEW WAY WITH SSOBLOC
goods = (np.abs(x[df['ssobloc'].values] - df['min_ys'].values) > 6) & (df['stds'].values >= 0.04) & (df['snow'].values > 0.1) #& (df['linestds'].values < 15)
plt.scatter(df['wl'], x[df['ssobloc'].values], c=df['stds'])
plt.plot(df['wl'][goods],x[df['ssobloc'].values][goods],   'r.')
plt.xlabel('Measured water level [m NAVD88]')
plt.ylabel('x-location of detected shoreline [m]')
scipy.stats.theilslopes(x[df['ssobloc'].values][goods],df['wl'][goods],)
plt.fill_between(xs, ys-155*.1, ys+155*.1, color='lightgrey', zorder=0)
# plt.fill_between(ys, 1/8.8*xs-155/8.8*.1, 1/8.8*xs+155/8.8*.1, color='lightgrey', zorder=0)
# plt.plot(ys, 1/8.8*xs-155/8.8)
plt.colorbar()

# %%
goods = (np.abs(x[df['ssobloc'].values] - df['min_ys'].values) > 6) & (df['stds'].values >= 0.04) & (df['snow'].values > 0.1) #& (df['linestds'].values < 15)

plt.figure(figsize=(8,6))

# plt.subplot(1,3,1)
# plt.scatter(wls[goods], x[hylocs[goods]], c=pd.DatetimeIndex([pd.Timestamp(x, unit='s') for x in times[goods].astype(int)]), label='Aug–Oct 2018', marker='s')
# cbar = plt.colorbar()
# cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%Y-%m-%d'))

# plt.plot(xs, ys, '--', lw=1, c='grey')

# plt.subplot(1,3,2)
plt.scatter(df['wl'][goods], df['ssobcoord'][goods], c=df['time'][goods])
# plt.scatter(wls[goods], x[syotsulocs[goods]], c=pd.DatetimeIndex([pd.Timestamp(x, unit='s') for x in times[goods].astype(int)]), label='Aug–Oct 2018', marker='s')
cbar = plt.colorbar()
cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%Y-%m-%d'))
# plt.plot(xs, ys, '--', lw=1, c='grey')
plt.xlabel('Unalakleet water level [m NAVD88]')
# siegel = djn.siegel(df['wl'][goods], x[::-1][df['ssobloc'].values[goods]])
theil = scipy.stats.theilslopes(x[df['ssobloc'].values[goods]], df['wl'][goods])
plt.plot(xs, theil[0]*xs + theil[1])
plt.fill_between(xs, theil[2]*xs + theil[1], theil[3]*xs + theil[1], color='none', edgecolor='black', ls='--')
plt.title(f"cross-shore position = {theil[0]:.3f}*wl + {theil[1]:.3f}")
# plt.fill_between(xs, ys-150*.1, ys+150*.1, color='lightgrey', zorder=0)
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
# goods = df['time'].dt.year == 2018
df2 = df.sel(time=df.time.dt.year.isin([2019]))
df2 = df2.sel(time=df2.time.dt.week > 1)
# for month, gb in df2.groupby(df2.time.dt.dayofyear):
for month, gb in df2.groupby_bins(df2.time, pd.date_range(df2.time[0].values, df2.time[-1].values, freq='3d')):
    # print(month)
    weeks.append(month)
    # goods = (np.abs(gb['sycoords'] - gb['min_ys']) > 6) & (gb['stds'] >= 0.04) & (gb['snow'] > 0.1) # & (df['linestds'] < 15)
    goods = (np.abs(gb['ssobcoord'] - gb['min_ys']) > 6) & (gb['stds'] >= 0.04) & (gb['snow'] > 0.1) # & (df['linestds'] < 15)
    xs = np.arange(df['wl'].min(), df['wl'].max(), .1)
    ys = -12.5*xs + 62.5
    if sum(goods) > 1:
        # theil = scipy.stats.theilslopes(gb['wl'][goods], gb['sycoords'][goods])
        theil = scipy.stats.theilslopes(gb['wl'][goods], gb['ssobcoord'][goods])
        theils.append(theil)
        # siegel = djn.siegel(gb['sycoords'][goods], gb['wl'][goods] )
        siegel = djn.siegel(gb['ssobcoord'][goods], gb['wl'][goods] )
        slopes.append(siegel)
    else:
        theils.append((np.nan, np.nan, np.nan, np.nan))
        slopes.append((siegel, siegel))

    # r2s.append(np.corrcoef(gb['sycoords'][goods], gb['wl'][goods])[0,1]**2)
    r2s.append(np.corrcoef(gb['ssobcoord'][goods], gb['wl'][goods])[0,1]**2)
    ns.append(np.sum(goods))

    if doplot:
        plt.subplot(4,8,n)
        # plt.scatter(gb['sycoords'][goods], gb['wl'][goods], c=gb['time'][goods], label='Aug–Oct 2018')
        plt.scatter(gb['ssobcoord'][goods], gb['wl'][goods], c=gb['time'][goods], label='Aug–Oct 2018')
        if n == 20:
            plt.xlabel('Cross-shore position [m]')
        # plt.plot(ys, siegel[0]*ys + siegel[1])
        # plt.plot([gb['sycoords'][goods].min(), gb['sycoords'][goods].max()], siegel[0]*np.array([gb['sycoords'][goods].min(), gb['sycoords'][goods].max()]) + siegel[1])
        plt.plot([gb['ssobcoord'][goods].min(), gb['ssobcoord'][goods].max()], siegel[0]*np.array([gb['ssobcoord'][goods].min(), gb['ssobcoord'][goods].max()]) + siegel[1])
        plt.title(f"{month.left}: {siegel[0]:.3f}")
        # plt.fill_between(xs, ys-150*.1, ys+150*.1, color='lightgrey', zorder=0)
        xlims = plt.xlim(65, 110)
        ylims = plt.ylim(0,5)
        # plt.plot(np.array([80, 100]), -0.08992005276821789*np.array([80, 100])+ 9.418220795056857)
        if (n != 1) and (n != 9) and (n != 17) and (n != 25):
            plt.gca().set_yticklabels([])
        if n < 17:
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

# df['localtime'] = xr.DataArray(pd.DatetimeIndex(df['time'].values, tz='utc').tz_convert('US/Alaska').tz_localize(None), dims='time')
# df= df.drop('localtime')

for week in np.arange(18,43):
    for year in [2018, 2019]:
        goods = (df.time.dt.year == year) & (df.time.dt.week == week) &  (np.abs(df['ssobcoord'] - df['min_ys']) > 6) & (df['stds'] >= 0.04) & (df['snow'] > 0.1) & (df['linestds'] < 15)
        # goods = (df.time.dt.year == year) & (df.time.dt.week == week) &  (np.abs(df['sycoords'] - df['min_ys']) > 6) & (df['stds'] >= 0.04) & (df['snow'] > 0.1) & (df['linestds'] < 15)
        if not np.sum(goods):
            continue
        plt.figure(figsize=(12,8))
        plt.subplot2grid((3,1),(0,0))
        df['hs'].plot()
        df['hs'][goods].plot(marker='.', ls='none')

        plt.subplot2grid((3,1),(1,0), rowspan=2)
        [x.timestamp() for x in pd.DatetimeIndex(df['time'][goods].values)]
        goodloctime = pd.DatetimeIndex(df['time'][goods].values, tz='utc').tz_convert('US/Alaska').tz_localize(None)

        bounds = mdates.date2num(pd.date_range(goodloctime.min().date(), goodloctime.max().date()+pd.Timedelta('1d'), freq='1d'))
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        # plt.scatter(df['sycoords'][goods], df['wl'][goods], c=mdates.date2num(goodloctime), label='Auto-picked shoreline', norm=norm, cmap=plt.cm.plasma)
        plt.scatter(df['ssobcoord'][goods], df['wl'][goods], c=mdates.date2num(goodloctime), label='Auto-picked shoreline', norm=norm, cmap=plt.cm.plasma)
        # f = scipy.interpolate.interp1d(df['ssobcoord'][goods], df['wl'][goods])
        # plt.plot(85, f(85), 'rs')
        # interpy = scipy.interpolate.interp1d(df['ssobcoord'][goods].values, df['wl'][goods].values, 85)
        # ordered = np.argsort(df['ssobcoord'][goods].values)
        # spl = scipy.interpolate.splrep(df['ssobcoord'][goods][ordered], df['wl'][goods][ordered])
        # plt.plot(85, scipy.interpolate.splev(85, spl), 'bd')
        # plt.plot(85, interpy, 'rs')
        cb = plt.colorbar(label='Local date (black line indicates time of topo survey)')
        loc = mdates.AutoDateLocator()
        cbylims = cb.ax.get_ylim()
        print(cbylims)
        meantime = pd.DatetimeIndex(topo['GNSS Vector Observation.Start Time'], tz='US/Alaska').tz_localize(None).mean()
        cb.ax.plot([cbylims[0], cbylims[1]], mdates.date2num([meantime, meantime]),c='k', lw=1)#
        cb.ax.yaxis.set_major_locator(loc)
        cb.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


               # plt.scatter(topo['x'][495:537], topo['y'][495:537], c=topo['Elevation'][495:537], cmap=plt.cm.gist_earth)
               # plt.scatter(topo['x'][544:610], topo['y'][544:610], c=topo['Elevation'][544:610], cmap=plt.cm.gist_earth)
        plt.plot(topo['x'][495:537], topo['Elevation'][495:537], '*-', label='Topo survey line 1')
        plt.plot(topo['x'][544:610], topo['Elevation'][544:610], '*-', label='Topo survey line 2')

        plt.plot(bathy['x'][123255:123555], bathy['OrthometricHeight'][123255:123555], '--', label='Bathy survey')



        plt.xlim(70,100)
        plt.ylim(0.5,4.5)
        plt.xlabel('Cross-shore distance [m]')
        plt.ylabel('Elevation [m NAVD88]')
        plt.title(f"{np.sum(goods.values)} points")
        plt.legend()


        """ NOTE all times in this plot are LOCAL """
        djn.set_fontsize(plt.gcf(), 14)
        plt.subplots_adjust(hspace=.4)
        plt.savefig(f'beach_slope_with_topo_survey_{year}_{week}.png', dpi=300, bbox_inches='tight')
        plt.show()
# plt.plot(np.array([80, 100]), -0.08992005276821789*np.array([80, 100])+ 9.418220795056857)
# %%
df['hs'].plot()
plt.twinx()
df['tp'].plot(c='C1')
# %%

# pd.Timestamp('2019-07-07').dayofyear
# goods = (df.time.dt.dayofyear == 156) & (np.abs(df['sycoords'] - df['min_ys']) > 6) & (df['stds'] >= 0.04) & (df['snow'] > 0.1) # & (df['linestds'] < 15)
goods = (df.time.dt.dayofyear == 153) & (np.abs(df['ssobcoord'] - df['min_ys']) > 6) & (df['stds'] >= 0.04) & (df['snow'] > 0.1) # & (df['linestds'] < 15)
n = 1

plt.figure(figsize=(18,10))
n = 18
for t in df.timestamp[goods][n:n+1]:
    # plt.subplot(np.ceil(np.sqrt(sum(goods.values))).astype(int), np.ceil(np.sqrt(sum(goods.values))).astype(int), n)
    n+=1
    ifile = fildir + 'proc/rect/' + product + '/' + str(int(t.values)) + '.' + camera + '.' + product + '.rect.png'
    imgboth = np.rot90(imageio.imread(ifile))
    plt.imshow(imgboth)
    plt.title(f"{int(t.values)} --- {pd.Timestamp(int(t.values), unit='s')} --- {product} --- WL: {df.wl[df.timestamp == t].values}")
    plt.ylim(1800,1400)
    plt.xlim(250,1500)
    # plt.plot(800, df['sylocs'][df['timestamp']==t.values], 'cs')
    df['lvs'][df['timestamp']==t.values,:].plot(marker='.', ls='none', color='r')

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
goods = np.isfinite(r2s)#weeks < 46#r2s > 0.
# plt.plot(weeks[goods], slopes[:,0][goods], '.')
# plt.plot(weeks[goods], theils[:,0][goods], '.')

plt.errorbar([x.mid for x in weeks[goods]], theils[goods,0], np.vstack([theils[goods,0]-theils[goods,2], theils[goods,3]-theils[goods,0]]))
plt.fill_between([x.mid for x in weeks[goods]], theils[goods,2], theils[goods,3], color='lightgrey')

for x in np.where(goods)[0]:
    plt.text(weeks[x].mid, theils[x,3]+.005, ns[x], ha='center')

plt.ylabel('Foreshore slope')
plt.xlabel('Week of year ')
# plt.ylim(0,.15)
# plt.ylim(0.02,.15)
plt.ylim(-0.2,.1)
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
    plt.text(200, 1100, 'R', fontweight='bold', fontsize=14, color='r', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(400+ g[:,800], range(len(h[:,800])),'g')
    plt.text(400, 1100, 'G', fontweight='bold', fontsize=14, color='g', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(600+ b[:,800], range(len(h[:,800])),'b')
    plt.text(600, 1100, 'B', fontweight='bold', fontsize=14, color='b', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(800 + (r[:,800]-b[:,800]), range(len(h[:,800])),'r')
    plt.text(800, 1100, 'R-B', fontweight='bold', fontsize=14, color='b', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(1000+ 200*h[:,800], range(len(h[:,800])),'r')
    plt.text(1000, 1100, 'H', fontweight='bold', fontsize=14, color='b', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(1200+ 200*s[:,800], range(len(h[:,800])),'r')
    plt.text(1200, 1100, 'S', fontweight='bold', fontsize=14, color='b', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(1400+ 200*v[:,800], range(len(h[:,800])),'r')
    plt.text(1400, 1100, 'V', fontweight='bold', fontsize=14, color='b', bbox=dict(facecolor='white', alpha=0.5))
    skimage.feature.canny(v[1000:2000,750:1000])
    plt.figure(figsize=(14,8))

    hgauss = scipy.ndimage.gaussian_filter1d(h[1200:1800,800], 7)
    sgauss = scipy.ndimage.gaussian_filter1d(s[1200:1800,800], 7)
    vgauss = scipy.ndimage.gaussian_filter1d(v[1200:1800,800], 7)
    hobs = np.abs(scipy.ndimage.sobel(hgauss))
    hobs = hobs * 1/hobs.max()
    sobs = np.abs(scipy.ndimage.sobel(sgauss))
    sobs = sobs * 1/sobs.max()
    vobs = np.abs(scipy.ndimage.sobel(vgauss))
    vobs = vobs * 1/vobs.max()
    rbgauss = scipy.ndimage.gaussian_filter1d(r[1200:1800,800]-b[1200:1800,800], 7)
    rbobs = np.abs(scipy.ndimage.sobel(rbgauss))
    rbobs = rbobs * 10/rbobs.max()
    plt.plot(mask)
    plt.subplot(2,2,1)
    plt.plot(h[1200:1800,800])
    plt.plot(hgauss)
    plt.plot(hobs)
    plt.title(np.argmax(hobs))
    mask = hgauss > skimage.filters.threshold_otsu(hgauss)
    # %timeit movavg(r[1200:1800,800]-b[1200:1800,800], 7)
    # %timeit scipy.ndimage.gaussian_filter1d(r[1200:1800,800]-b[1200:1800,800], 7)
    plt.plot(mask)
    plt.subplot(2,2,2)
    plt.plot(s[1200:1800,800])
    plt.plot(sgauss)
    plt.plot(sobs)
    plt.title(np.argmax(sobs))
    mask = sgauss > skimage.filters.threshold_otsu(sgauss)
    plt.plot(mask)
    plt.subplot(2,2,3)
    plt.plot(v[1200:1800,800])
    plt.plot(vgauss)
    plt.plot(vobs)
    plt.title(np.argmax(vobs))
    mask = vgauss > skimage.filters.threshold_otsu(vgauss)
    plt.plot(mask)
    plt.subplot(2,2,4)
    plt.plot(r[1200:1800,800]-b[1200:1800,800])
    plt.plot(rbgauss)
    plt.plot(rbobs)
    mask = rbgauss > skimage.filters.threshold_otsu(rbgauss)
    plt.plot(mask*10)
    plt.title(np.argmax(rbobs))
    plt.show()
# %%
""" PCA stuff"""
from sklearn.decomposition import PCA
t
for t in np.random.choice(ts, 1): # ['1535571001']
    t = str(t)
    ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
    img = np.rot90(imageio.imread(ifile))

    hsv = cv2.cvtColor(np.flip(img,2), cv2.COLOR_BGR2HSV).astype(float) / 255
    ims = hsv[1200:2000,300:1050,:]
    imsrgb = img[1200:2000,300:1050,:]
    ims.shape
    impca = np.hstack([np.reshape(ims[:,:,0], (ims.shape[0]*ims.shape[1], 1)),
                       np.reshape(ims[:,:,1], (ims.shape[0]*ims.shape[1], 1)),
                       np.reshape(ims[:,:,2], (ims.shape[0]*ims.shape[1], 1))])
    impca.shape
    pca = PCA(n_components=3)
    fitted = pca.fit_transform(impca)
    print(pca.explained_variance_ratio_)

    plt.figure(figsize=(8,10))
    plt.imshow(imsrgb)
    plt.contourf(np.reshape(fitted[:,0], (ims.shape[0], ims.shape[1])), [-50, 0, 50], alpha=0., hatches=['-', '/', '\\', '//'])
    plt.colorbar()
# %%
""" make a contour plot """


auxn = aux.sel(time=slice(pd.Timestamp('2018-08-26'), pd.Timestamp('2018-08-31')))
auxn = auxn.where(auxn.ssoblvs.std(dim='xlocs') < 30, drop=True)
idx = auxn.wl.argmax()
print(auxn.wl.max())
t = str(auxn.timestamp[idx].values.astype(int))# 3450
ifile = fildir + 'proc/rect/' + product + '/' + t + '.' + camera + '.' + product + '.rect.png'
img = np.rot90(imageio.imread(ifile))

gb = auxn.groupby_bins(auxn.wl, np.arange(.75, 2, .2))
xs = []
ys = []
zs = []

for a, b in gb:
    # print(a, len(b.time))
    if len(b.time) > 6:
        # goods = b.ssoblvs.std(dim='time').values/10 < 6
        xtmp = y[aux.xlocs.values]
        # xtmp[~goods] = np.nan
        # print(b.ssoblvs)
        if np.all(np.isnan(b.ssoblvs.mean(dim='time'))):
            continue
        ytmp = x[b.ssoblvs.mean(dim='time').values.astype(int)]
        # ytmp[~goods] = np.nan
        ztmp = b.wl.median(dim='time').values * np.ones_like(y[aux.xlocs.values])
        ztmp[~goods] = np.nan
        plt.figure()
        plt.plot(xtmp, ytmp)
        plt.plot(xtmp, x[b.vsoblvs.mean(dim='time').values.astype(int)])
        # plt.title(b.ssoblvs.std(dim='xlocs'))
        plt.show()
        xs.append(xtmp)
        ys.append(ytmp)
        zs.append(ztmp)
    # plt.plot(y[aux.xlocs.values], x[b.ssoblvs.mean(dim='time').values.astype(int)], '-*')
xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)
idxorder = np.argsort(np.nanmean(zs, axis=-1))
xs = xs[idxorder, :]
ys = ys[idxorder, :]
zs = zs[idxorder, :]
plt.figure(figsize=(10,8))
plt.imshow(img, extent=(y.max(), y.min(), x.min(), x.max()))
plt.title(auxn.wl[auxn.timestamp == int(t)].values)
plt.contourf(xs, ys, zs, alpha=0.5)
# plt.gca().invert_xaxis()
plt.xlabel('alongshore (y) [m]')
plt.ylabel('offshore (x) [m]')

plt.colorbar(label='Beach elevation [m NAVD88]')
plt.axis('equal')
plt.ylim(50,120)
plt.xlim(25,-125)
plt.show()
    # b.ssoblvs.mean(dim='time').plot()
# %%
t = str(1563143401)# 3450
ifile = fildir + 'proc/rect/' + product + '/' + t + '.' + camera + '.' + product + '.rect.png'
img = imageio.imread(ifile)
plt.figure(figsize=(14,8))
plt.subplot(1,4,1)
plt.imshow(img)
plt.xlim(500,1300)
plt.xticks([])
plt.yticks([])
plt.title('RGB')
hsv = cv2.cvtColor(np.flip(img,2), cv2.COLOR_BGR2HSV).astype(float) / 255
plt.subplot(1,4,2)
plt.imshow(hsv[:,:,0], cmap=plt.cm.gray)
plt.xlim(500,1300)
plt.xticks([])
plt.yticks([])
plt.title('H')
plt.subplot(1,4,3)
plt.imshow(hsv[:,:,1], cmap=plt.cm.gray)
plt.xlim(500,1300)
plt.xticks([])
plt.yticks([])
plt.title('S')
plt.subplot(1,4,4)
plt.imshow(hsv[:,:,2], cmap=plt.cm.gray)
plt.xlim(500,1300)
plt.xticks([])
plt.yticks([])
plt.title('V')
plt.show()
