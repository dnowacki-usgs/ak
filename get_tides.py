%load_ext autoreload
%autoreload 2
import sys
sys.path.append('/Users/dnowacki/Documents/python')
import noaa
import os
import xarray as xr
# %%

# %%
# https://tidesandcurrents.noaa.gov/waterlevels.html?id=9468333
# n9468333 = noaa.get_long_coops_data('9468333', start_date='20180801', end_date='20191201', datum='NAVD', product='water_level')
# n9468333.to_netcdf('/Volumes/Backstaff/field/unk/n9468333.nc')

n9468333 = xr.load_dataset('/Volumes/Backstaff/field/unk/n9468333.nc')
