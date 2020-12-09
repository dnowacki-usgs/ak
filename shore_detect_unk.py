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
n9468333 = xr.load_dataset('/Volumes/Backstaff/field/unk/n9468333.nc')
wavesunk = xr.load_dataset('/Volumes/Backstaff/field/unk/ak_4m_hs.nc')
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

tens = to_epoch(gb[10])
elevens = to_epoch(gb[11])
noons = to_epoch(gb[12])
thirteens = to_epoch(gb[13])
fourteens = to_epoch(gb[14])
fifteens = to_epoch(gb[15])
sixteens = to_epoch(gb[16])
# %%
t = str(sixteens[0])

ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
imgboth = imageio.imread(ifile)
# cv2lab_image = cv2.cvtColor(np.flip(imgboth,2), cv2.COLOR_BGR2Lab)
# scilab_image = rgb2lab(imgboth)
# need to do the flip to get from rgb to bgr
# cv2hsv_image = cv2.cvtColor(np.flip(imgboth,2), cv2.COLOR_BGR2HSV)
# scihsv_image = rgb2hsv(imgboth)

# need to do the flip to get from rgb to bgr
lab = cv2.cvtColor(np.flip(imgboth,2), cv2.COLOR_BGR2Lab)
# hsv = rgb2hsv(imgboth)
hsv = cv2.cvtColor(np.flip(imgboth,2), cv2.COLOR_BGR2HSV)
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
x = np.arange(0, 250, .1)
y = np.arange(-150,40,.1)
wls = []
ylocs = []
sylocs = []
# syotsulocs = []
hylocs = []
times = []
stds = []
ts = np.hstack([tens, elevens, noons, thirteens, fourteens, fifteens])
# ENSURE ONLY 2018 DATA
goods = np.array([pd.Timestamp(x, unit='s').isocalendar()[0] == 2018 for x in ts])# need the funny isocalendar for week of year computations later
# ENSURE ONLY 2019 DATA
goods = np.array([pd.Timestamp(x, unit='s').isocalendar()[0] == 2019 for x in ts])
ts = ts[goods]
goods = (np.array([pd.Timestamp(x, unit='s').weekofyear for x in ts]) <= 48) & (np.array([pd.Timestamp(x, unit='s').month for x in ts]) >= 5)
ts = ts[goods]
len(ts)
for t in np.random.choice(ts, 400):
    # if (pd.Timestamp(t, unit='s').month > 10) or (pd.Timestamp(t, unit='s').month < 5):
        # continue
    t = str(t)
    ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
    # imgboth = np.rot90(imageio.imread(ifile))

    hsv = cv2.cvtColor(np.flip(imgboth,2), cv2.COLOR_BGR2HSV).astype(float) / 255
    rboth = imgboth[:,:,0]
    gboth = imgboth[:,:,1]
    bboth = imgboth[:,:,2]

    plt.figure(figsize=(16,8))
    wl = n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))]
    hs = wavesunk['swh'][np.argmin(np.abs(pd.DatetimeIndex(wavesunk.valid_time.values) - pd.to_datetime(t, unit='s')))]
    plt.imshow(imgboth)
    firstguess = np.where(x == int(8*wl.values + 151))[0][0] # eyeballed slope from a first pass
    xstart = firstguess - 125
    xend = firstguess + 125
    print(xstart, xend)
    plt.axhline(xstart, c='grey')
    plt.axhline(xend, c='grey')
    for xloc in np.arange(200,1201,100):
        h = movavg(hsv[:,xloc, 0].copy())
        s = movavg(hsv[:,xloc, 1].copy())
        v = hsv[:,xloc, 2].copy()
        h[0:xstart] = np.nan
        h[xend:] = np.nan
        s[0:xstart] = np.nan
        s[xend:] = np.nan
        v[0:xstart] = np.nan
        v[xend:] = np.nan
        # l[0:xstart] = np.nan
        # l[xend:] = np.nan
        # a[0:xstart] = np.nan
        # a[xend:] = np.nan
        # b[0:xstart] = np.nan
        # b[xend:] = np.nan

        # kde = scipy.stats.gaussian_kde(rboth[:,xloc]-bboth[:,xloc])
        # pdf_locs = np.linspace(min(rboth[:,xloc]-bboth[:,xloc]),max(rboth[:,xloc]-bboth[:,xloc]),100) # These locations are equivalent to the locations given by Matlabs ksdensity function
        # pdf_values = kde(pdf_locs)
        # thresh_otsu = skimage.filters.threshold_otsu(rboth[:,xloc]-bboth[:,xloc])
        # plt.plot(pdf_locs, pdf_values)

        # print((np.nanmax(s)-np.nanmin(s))/np.nanmean(s))

        try:
            lm = np.argwhere(v < np.nanmean(v))[0][0]
            if np.nanmean(s[xstart:xstart+10]) < np.nanmean(s[xend-10:xend]):
                mins = np.argwhere(s > np.nanmean(s))[0][0]
            else:
                mins = np.argwhere(s < np.nanmean(s))[0][0]
            if np.nanmean(h[xstart:xstart+10]) < np.nanmean(h[xend-10:xend]):
                minh = np.argwhere(h > np.nanmean(h))[0][0]
            else:
                minh = np.argwhere(h < np.nanmean(h))[0][0]
        except IndexError:
            continue
        # plt.plot(lm,yloc,  'c*') # 8 is first value that isn't one of the info lines
        gv = np.gradient(v)
        gv[0:lm-40] = np.nan
        gv[lm+40:] = np.nan

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
        # otsuloc = np.nanargmin(np.abs(s - skimage.filters.threshold_otsu(s[xstart:xend])))

        if np.nanmean(h[xstart:xstart+10]) < np.nanmean(h[xend-10:xend]):
            ghloc = np.nanargmax(gh)
        else:
            ghloc = np.nanargmin(gh)
        try:
            gvloc = np.nanargmin(gv)
            # plt.plot(xloc+20, gvloc, 'rs')
            plt.plot(xloc, gsloc, 's', c='C0')
            plt.plot(xloc+200*s, np.arange(len(s)), 'r')
            # plt.plot(xloc-20, ghloc, 'bo')
            if xloc == 800:
                ylocs.append(gvloc)
                sylocs.append(gsloc)
                hylocs.append(ghloc)
                stds.append(np.nanstd(s))
                # syotsulocs.append(otsuloc)
                wls.append(wl.values)
                times.append(t)
        except ValueError:
            if xloc == 800:
                ylocs.append(np.nan)
                sylocs.append(np.nan)
                hylocs.append(np.nan)
                stds.append(np.nan)
                # syotsulocs.append(np.nan)
                wls.append(np.nan)
                tims.append(np.nan)
            continue

    plt.ylim(2000,1200)
    plt.title(f"{t} --- {pd.to_datetime(t, unit='s').tz_localize(pytz.utc).tz_convert(pytz.timezone('US/Alaska'))}, WL = {wl.values} m NAVD88, Hs = {hs.values:.2} m")
    print(t)
    # plt.savefig('unk' + t + '.png', bbox_inches='tight')
    plt.show()
    # skimage.filters.threshold_otsu(s[xstart:xend])
    # plt.plot(s)
    # plt.axhline(skimage.filters.threshold_otsu(s[xstart:xend]))
# %%
times = np.array(times)
ylocs = np.array(ylocs)
sylocs = np.array(sylocs)
# syotsulocs = np.array(syotsulocs)
hylocs = np.array(hylocs)
stds = np.array(stds)
wls = np.array(wls)

df = pd.DataFrame({'time': pd.DatetimeIndex([pd.Timestamp(int(x), unit='s') for x in times]),
                   'timestamp': times,
                   'sylocs': sylocs,
                   # 'syotsulocs': syotsulocs,
                   'stds': stds,
                   'wls': wls}).sort_values('time').set_index('time')

# QAQC the values very close to the edge of the allowable limit
df['min_ys'] = df['wls'] * 8 + 151 - 150*.1
# %%
goods = (np.abs(x[df['sylocs']] - df['min_ys']) > 4) & (stds >= 0.04)
# goods = np.abs(x[df['sylocs']] - df['min_ys']) > 4
# goods = stds >= 0.04
sum(goods)/len(wls)
plt.scatter(wls, x[sylocs], c=stds)
plt.plot(wls[goods], x[sylocs][goods], 'r.')
plt.fill_between(xs, ys-150*.1, ys+150*.1, color='lightgrey', zorder=0)
plt.colorbar()

pd.Timestamp('2018-12-31T21:50+00:00').isocalendar()
# %%
goods = (np.abs(x[df['sylocs']] - df['min_ys']) > 4) & (df['stds'] >= 0.04)

plt.figure(figsize=(8,6))
xs = np.arange(wls.min(), wls.max(), .1)
ys = 8*xs + 151
# plt.subplot(1,3,1)
# plt.scatter(wls[goods], x[hylocs[goods]], c=pd.DatetimeIndex([pd.Timestamp(x, unit='s') for x in times[goods].astype(int)]), label='Aug–Oct 2018', marker='s')
# cbar = plt.colorbar()
# cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%Y-%m-%d'))

# plt.plot(xs, ys, '--', lw=1, c='grey')

# plt.subplot(1,3,2)
plt.scatter(wls[goods], x[sylocs[goods]], c=pd.DatetimeIndex([pd.Timestamp(x, unit='s') for x in times[goods].astype(int)]), label='Aug–Oct 2018')
# plt.scatter(wls[goods], x[syotsulocs[goods]], c=pd.DatetimeIndex([pd.Timestamp(x, unit='s') for x in times[goods].astype(int)]), label='Aug–Oct 2018', marker='s')
cbar = plt.colorbar()
cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%Y-%m-%d'))
plt.plot(xs, ys, '--', lw=1, c='grey')
plt.xlabel('Unalakleet water level [m NAVD88]')
siegel = djn.siegel(wls[goods], x[sylocs[goods]])
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

df.index.min().isocalendar()
df.index.isocalendar().year
for month, gb in df.groupby(df.index.isocalendar().week):
    weeks.append(month)
    print(gb)
    goods = (np.abs(x[gb['sylocs']] - gb['min_ys']) > 4) & (gb['stds'] >= 0.04)
    plt.subplot(4,8,n)

    xs = np.arange(df['wls'].min(), df['wls'].max(), .1)
    ys = 8*xs + 151

    plt.scatter(x[gb['sylocs'][goods]], gb['wls'][goods], c=gb.index[goods], label='Aug–Oct 2018')

    if n == 28:
        plt.xlabel('Cross-shore position [m]')

    if sum(goods) > 1:
        theil = scipy.stats.theilslopes(gb['wls'][goods], x[gb['sylocs'][goods]])
        theils.append(theil)
        siegel = djn.siegel(x[gb['sylocs'][goods]], gb['wls'][goods] )
        slopes.append(siegel)
    else:
        theils.append((np.nan, np.nan, np.nan, np.nan))
        slopes.append((siegel, siegel))
    r2s.append(np.corrcoef(x[gb['sylocs'][goods]], gb['wls'][goods])[0,1]**2)
    plt.plot(ys, siegel[0]*ys + siegel[1])
    plt.title(f"{month}: {siegel[0]:.3f}")
    # plt.fill_between(xs, ys-150*.1, ys+150*.1, color='lightgrey', zorder=0)
    xlims = plt.xlim(140, 185)
    ylims = plt.ylim(0,5)
    if (n != 1) and (n != 9) and (n != 17) and (n != 25):
        plt.gca().set_yticklabels([])
    if n < 25:
        plt.gca().set_xticklabels([])
    if n == 9:
        plt.ylabel('Unalakleet water level [m NAVD88]')


    plt.text(xlims[0] + 0.1*np.diff(xlims), ylims[0] + 0.9*np.diff(ylims), f"N = {len(gb)}, r2 = {r2s[-1]:.2f}", va='top')
    n += 1
plt.suptitle(f"{df.index[0]} — {df.index[-1]}")
plt.savefig(f'week_of_year_{df.index[0].year}.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
slopes = np.array(slopes)
weeks = np.array(weeks)
r2s = np.array(r2s)
theils = np.array(theils)
#
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
goods = r2s > 0.
# plt.plot(weeks[goods], slopes[:,0][goods], '.')
# plt.plot(weeks[goods], theils[:,0][goods], '.')
plt.errorbar(weeks[goods], theils[goods,0], np.vstack([theils[goods,0]-theils[goods,2], theils[goods,3]-theils[goods,0]]))
plt.ylabel('Foreshore slope')
plt.xlabel('Week of year')
plt.ylim(ymin=0)
plt.grid()
plt.subplot(2,1,2)
175*theils[goods, 0]+ theils[goods,1]
elev = 175*theils[goods, 0]+ theils[goods,1]
elevlo = 175*theils[goods, 2]+ theils[goods,1]
elevhi = 175*theils[goods, 3]+ theils[goods,1]

plt.errorbar(weeks[goods], elev, np.vstack([elev-elevlo, elevhi-elev]))
plt.ylim(-20,20)
plt.grid()
# for w, s, r, ls, us in zip(weeks[goods], theils[:,0][goods], r2s[goods], theils[:,2][goods],theils[:,3][goods],):
    # plt.text(w, s, f"{s:.2f}pm{(s-ls):.2f}", ha='center')
# %%



# def imadjust(img):
#     lower = np.percentile(img, 1)
#     upper = np.percentile(img, 99)
#     out = (img - lower) * (255 / (upper - lower))
#     return np.clip(out, 0, 255, out) # in-place clipping
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

t = str(1569371401)
ifile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
img = np.rot90(imageio.imread(ifile))
hsv = cv2.cvtColor(np.flip(img,2), cv2.COLOR_BGR2HSV).astype(float) / 255
gray = cv2.cvtColor(np.flip(img,2), cv2.COLOR_BGR2GRAY)[1200:2000,400:1000]
plt.figure(figsize=(10,8))
plt.imshow(gray)
from skimage.filters import threshold_otsu
gray2 = imadjust(gray).astype(np.uint8)
plt.imshow(cv2.equalizeHist(gray))
plt.imshow(cv2.equalizeHist(gray2))

eq = cv2.equalizeHist(gray2)

thresh = threshold_otsu(eq)*.9
binary = gray2 > thresh
canny = skimage.feature.canny(binary, sigma=3)
plt.imshow(canny)
np.argwhere(canny)
r, c = np.where(canny)

loc = []
for n in range(canny.shape[1]):
    try:
        loc.append(np.where(canny[:,n])[0][0])
    except:
        loc.append(np.nan)
loc = np.array(loc)
# plt.imshow(skimage.morphology.thin(canny))
plt.imshow(gray, cmap=plt.cm.gray)
plt.plot(range(canny.shape[1]), loc, 'r')
# plt.imshow(gray)
# plt.colorbar()
# plt.imshow(imadjust(gray))
# plt.colorbar()
#
# # edges = skimage.feature.canny(hsv[:,:,1], sigma=3)
# # plt.imshow(edges)
# plt.show()
