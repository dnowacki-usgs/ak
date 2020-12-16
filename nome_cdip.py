import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from MesoPy import Meso
m = Meso(token='a5d4ab17621848e08d97208b499f649e')
fildir = '/Volumes/Backstaff/field/unk/'
# %%

url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/241p1/241p1_historic.nc'
ds = xr.load_dataset(url)
dsbit = xr.Dataset()
for k in ['waveHs', 'waveTp', 'waveTa', 'waveTp', 'waveTz']:
    dsbit[k] = ds[k]
dsbit
dsbit.to_netcdf('/Volumes/Backstaff/field/unk/cdip_241.nc')
# %%
ds = xr.load_dataset('/Volumes/Backstaff/field/unk/cdip_241.nc')
# %%
ds = ds.sel(waveTime=slice('2018-01-01', '2018-12-31'))
ax = plt.subplot(2,1,1)
(ds.waveHs/3.5).plot()
dsunk.swh.plot(x='valid_time')
plt.subplot(2,1,2, sharex=ax)
ds.waveTp.plot(c='C1')
# plt.xlim(pd.Timestamp('2018-07'), pd.Timestamp('2018-11'))
plt.show()
# %%
# https://nomads.ncep.noaa.gov
# https://polar.ncep.noaa.gov/waves/hindcasts/prod-multi_1.php
# ftp://polar.ncep.noaa.gov/pub/history/waves/multi_1/201809/gribs/
hs   = xr.concat([xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.hs.201808.grb2', engine='cfgrib'),
                  xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.hs.201809.grb2', engine='cfgrib'),
                  xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.hs.201810.grb2', engine='cfgrib'),
                  xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.hs.201811.grb2', engine='cfgrib')],
                  dim='step')
wind = xr.concat([xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.wind.201808.grb2', engine='cfgrib'),
                  xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.wind.201809.grb2', engine='cfgrib'),
                  xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.wind.201810.grb2', engine='cfgrib'),
                  xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.wind.201811.grb2', engine='cfgrib'),],
                  dim='step')
tp   = xr.concat([xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.tp.201808.grb2', engine='cfgrib'),
                  xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.tp.201809.grb2', engine='cfgrib'),
                  xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.tp.201810.grb2', engine='cfgrib'),
                  xr.open_dataset(fildir + 'ww3_multi_1/multi_1.ak_4m.tp.201811.grb2', engine='cfgrib'),],
                  dim='step')
ds = xr.merge([hs, wind, tp], combine_attrs='override')
# %%
def getlon(lon):
    return (lon + 180) % 360 + 180
idxlon = np.argmin(np.abs(getlon(-160.89)-ds.longitude.values))
idxlat = np.argmin(np.abs(63.85-ds.latitude.values))
ds.isel(longitude=idxlon, latitude=idxlat).to_netcdf('/Volumes/Backstaff/field/unk/ak_4m.nc')
# %%
dsunk = xr.load_dataset('/Volumes/Backstaff/field/unk/ak_4m.nc')
# %%
data = m.timeseries(stid='paun',
                    vars='pressure,wind_speed,wind_direction,wind_gust',
                    start='201808010000',
                    end='201912310000')
# %%
# data['STATION'][0]['OBSERVATIONS'].keys()
ds = xr.Dataset()
ds['time'] = pd.to_datetime(data['STATION'][0]['OBSERVATIONS']['date_time'])
ds['time'] = pd.DatetimeIndex(ds['time'].values, tz=None)
for k, suffix in zip(['pressure', 'wind_speed', 'wind_direction', 'wind_gust'],
                     ['_set_1d',  '_set_1',     '_set_1',         '_set_1']):
    ds[k] = xr.DataArray(
        np.array(data['STATION'][0]['OBSERVATIONS'][k + suffix]).astype(float),
        dims='time')
    ds[k].attrs['units'] = data['UNITS'][k]
for k in ['STATUS',
        'MNET_ID',
        'ELEVATION',
        'NAME',
        'STID',
        'ELEV_DEM',
        'LONGITUDE',
        'STATE',
        'LATITUDE',
        'TIMEZONE',
        'ID'
        ]:
    if k == 'NAME':
        ds.attrs['STATION_NAME'] = data['STATION'][0][k]
    else:
        ds.attrs[k] = data['STATION'][0][k]
ds.to_netcdf('/Volumes/Backstaff/field/unk/paun_timeseries.nc')
# %%
paun = xr.load_dataset('/Volumes/Backstaff/field/unk/paun_timeseries.nc')
# %%
era5 = xr.open_dataset(fildir + 'adaptor.mars.internal-1607638906.448238-23511-7-2fda0881-4b12-46ac-80cc-2b0f27e2ccca.grib', engine='cfgrib')
# %%
plt.figure(figsize=(10,8))

# ds['swh'] = ds.swh.assign_coords(longitude=(((ds.swh.longitude + 180) % 360) - 180))
# 63.85370440754068, -160.89278197708578
# %%
plt.figure(figsize=(10,8))

ws = xr.Dataset()

ws['time'] = dsunk.valid_time.values
len(ws['time'])
_, index = np.unique(ws['time'], return_index=True)
ws['time'] = ws['time'][index]
ws['ws'] = xr.DataArray(((dsunk.u[index].values**2 + dsunk.v[index].values**2)**0.5), dims='time')
paun.wind_speed.reindex_like(ws, method='nearest', tolerance='5min').plot()
ws.ws.plot( c='C1')
plt.xlim(pd.Timestamp('2018-08'), pd.Timestamp('2018-12'))

plt.figure()
plt.plot(paun.wind_speed.reindex_like(ws, method='nearest', tolerance='5min'), ws.ws, '.')
plt.plot([0, 12], [0, 12])

# %%
dsunk.swh.plot(x="valid_time")
plt.twinx()
((dsunk.u**2 + dsunk.v**2)**0.5).plot(x="valid_time", c='C1')
# dsunk.perpw.plot(x="valid_time", c='C1')
# %%
plt.figure(figsize=(10,8))
# ds.swh.isel(step=10).plot(vmin=0, vmax=.5)
dsunk.perpw.isel(step=10).plot(vmin=0, vmax=5)
# ds.u.isel(step=10).plot(vmin=0, vmax=2)
plt.plot(getlon(-160.89), 63.85, 'r*')
plt.xlim(197, 200)
plt.grid()
plt.ylim(62, 66)
# %%
# %%
era5.swh.isel(time=-1)
