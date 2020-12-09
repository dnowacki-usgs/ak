import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
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
# ftp://polar.ncep.noaa.gov/pub/history/waves/multi_1/201809/gribs/
ds = xr.concat([xr.open_dataset('/Users/dnowacki/Downloads/multi_1.ak_4m.hs.201808.grb2', engine='cfgrib'),
                xr.open_dataset('/Users/dnowacki/Downloads/multi_1.ak_4m.hs.201809.grb2', engine='cfgrib'),
                xr.open_dataset('/Users/dnowacki/Downloads/multi_1.ak_4m.hs.201810.grb2', engine='cfgrib'),
                xr.open_dataset('/Users/dnowacki/Downloads/multi_1.ak_4m.hs.201811.grb2', engine='cfgrib')],
                dim='step')
# %%
idxlon = np.argmin(np.abs(getlon(-160.89)-ds.longitude.values))
idxlat = np.argmin(np.abs(63.85-ds.latitude.values))
ds.isel(longitude=idxlon, latitude=idxlat).to_netcdf('/Volumes/Backstaff/field/unk/ak_4m_hs.nc')
# %%
dsunk = xr.load_dataset('/Volumes/Backstaff/field/unk/ak_4m_hs.nc')

# %%
plt.figure(figsize=(10,8))

# ds['swh'] = ds.swh.assign_coords(longitude=(((ds.swh.longitude + 180) % 360) - 180))
# 63.85370440754068, -160.89278197708578
def getlon(lon):
    return (lon + 180) % 360 + 180


import numpy as np
# %%
dsunk.swh.plot(x="valid_time")
# %%
plt.figure(figsize=(10,8))
ds.swh.isel(step=10).plot(vmin=0, vmax=.5)
plt.plot(getlon(-160.89), 63.85, 'r*')
plt.xlim(197, 200)
plt.grid()
plt.ylim(62, 66)
# %%
