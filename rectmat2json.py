import sys
sys.path.append('/Users/dnowacki/Documents/python')
import matlabtools
import json
# %%
fildir = '/Volumes/Backstaff/field/bti/from_srharrison/geom/'

# %%
cam = 'c2'
mat = matlabtools.loadmat(fildir + '1531020639.c2.bright.barter.meta.mat')

cam = 'c1'
mat = matlabtools.loadmat(fildir + '1531020672.c1.bright.barter.meta.mat')
# %%
with open(f'intrinsic_{cam}.json', 'w') as f:
    json.dump(lcps, f)

with open(f'extrinsic_{cam}.json', 'w') as f:
    json.dump(betas, f)
# %%
lcpkeys = ['NU', 'NV', "c0U", "c0V", "fx", "fy", "d1", "d2", "d3", "t1", "t2"]
lcps = {}
for k in lcpkeys:
    lcps[k] = mat['meta']['globals']['lcp'][k]

betakeys = ["x", "y", "z", "a", "t", "r"]
betas = {}
for k, v in zip(betakeys, mat['meta']['betas']):
    betas[k] = v

print(lcps)
print(betas)

# %%
cam = 'c1'
mat = matlabtools.loadmat('/Volumes/Backstaff/field/unk/unalakleet/1535238001.c1.timex.unalakleet.meta.mat')
cam = 'c2'
mat = matlabtools.loadmat('/Volumes/Backstaff/field/unk/unalakleet/1535238001.c2.timex.unalakleet.meta.mat')
# %%
with open(f'unk_intrinsic_{cam}.json', 'w') as f:
    json.dump(lcps, f)

with open(f'unk_extrinsic_{cam}.json', 'w') as f:
    json.dump(betas, f)
