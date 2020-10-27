%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import cv2
import skimage
import imutils
import matplotlib.pyplot as plt
params = {'mathtext.fontset': 'custom', 'mathtext.it': 'Helvetica:italic'}
plt.rcParams.update(params)
from skimage.metrics import structural_similarity
import numpy as np
import glob
import os
from datetime import datetime
import glob
# %%
def get_contours(thresh, min_area=-1):
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [x for x in contours if cv2.contourArea(x) > min_area]

def get_area(contours):
    return [cv2.contourArea(x) for x in contours]

def imshow(img, extent=(10,200,-200,25)):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=extent)
    # plt.xticks([])
    # plt.yticks([])

# fildir = '/Users/dnowacki/Projects/CoastCam/'
fildir = '/Volumes/Backstaff/field/bti/'
camera = 'c1'
fils = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'proc/rect/*' + camera + '*snap*png')][0:2]

xmin = 10
xmax = 200
ymin = -200
ymax = 5
dx = 0.1
dy = 0.1
x = np.arange(xmin, xmax+dx, dx)
y = np.arange(ymin, ymax+dy, dy)

xgoods = np.where(x < 150)[0]
ygoods = np.where(np.isfinite(y))[0]

nets = []
for n in range(len(fils) - 1):
    timestampA = fils[n]
    timestampB = fils[n+1]
    filA = fildir + 'proc/rect/' + timestampA + '.' + camera + '.snap.rect.png'
    filB = fildir + 'proc/rect/' + timestampB + '.' + camera + '.snap.rect.png'

    # load and crop
    imageA = cv2.imread(filA)[ygoods[:,None], xgoods[None,:], :]
    imageB = cv2.imread(filB)[ygoods[:,None], xgoods[None,:], :]

    # grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    grayA = imageA[:,:,2]
    grayB = imageB[:,:,2]

    BminusA = cv2.normalize(cv2.subtract(grayB, grayA), None, 255, 0, cv2.NORM_MINMAX)
    AminusB = cv2.normalize(cv2.subtract(grayA, grayB), None, 255, 0, cv2.NORM_MINMAX)

    # plt.figure(figsize=(16,8))
    # plt.subplot(1,2,1)
    # AminusB.max()
    # plt.imshow(BminusA, vmin=0, vmax=255, cmap=plt.cm.get_cmap('RdBu', 8) )
    # plt.colorbar(ticks=np.arange(0,257,32))
    # plt.title('B-A')
    # plt.subplot(1,2,2)
    # plt.imshow(AminusB, vmin=0, vmax=255, cmap=plt.cm.get_cmap('RdBu', 8) )
    # plt.title('A-B')
    # plt.colorbar(ticks=np.arange(0,257,32))
    #
    # bigDiff = BminusA.astype(int) - AminusB.astype(int)
    # normBigDiff = cv2.normalize(bigDiff.copy(), None, 255, 0, cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.normalize
    # plt.figure(figsize=(10,6))
    # plt.subplot(1,2,1)
    # plt.imshow(bigDiff, cmap=plt.cm.RdBu)
    # plt.title('Big difference')
    # plt.colorbar()
    #
    # plt.subplot(1,2,2)
    # plt.imshow(normBigDiff, cmap=plt.cm.RdBu)
    # plt.colorbar()
    # plt.title('normalized big difference')
    # plt.show()

    # _, bdgain = cv2.threshold(normBigDiff, 175, 255, 0)
    # _, bdloss = cv2.threshold(normBigDiff, 81, 255, 0)
    _, bdgain = cv2.threshold(BminusA, 80, 255, 0)
    _, bdloss = cv2.threshold(AminusB, 80, 255, 0)
    gaincontours = get_contours(bdgain)
    losscontours = get_contours(bdloss)
    gainsum = np.sum(get_area(gaincontours)) * .1 * .1
    losssum = np.sum(get_area(losscontours)) * .1 * .1
    nets.append(gainsum-losssum)
    print(f'{timestampA}, {gainsum-losssum:.1f}')

    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    imshow(cv2.drawContours(
            cv2.drawContours(imageA.copy(), losscontours, -1, (12,36,255), 2),
           gaincontours, -1, (36,255,12), 2),
           extent=(x[xgoods].min(), x[xgoods].max(), y[ygoods].min(), y[ygoods].max()))
    plt.title(datetime.utcfromtimestamp(int(timestampA)).isoformat())
    plt.text(20, -6, f'+ {gainsum:.1f} m$^2$', color='w')
    plt.subplot(1,2,2)
    imshow(cv2.drawContours(
            cv2.drawContours(imageB.copy(), losscontours, -1, (12,36,255), 2),
           gaincontours, -1, (36,255,12), 2),
           extent=(x[xgoods].min(), x[xgoods].max(), y[ygoods].min(), y[ygoods].max()))
    plt.title(datetime.utcfromtimestamp(int(timestampB)).isoformat())
    plt.text(20, -6, f'- {losssum:.1f} m$^2$\nnet $\Delta$: {gainsum-losssum:.1f} m$^2$', color='w')
    plt.plot([20, 45, 45, 20, 20], [-175, -175, -125, -125, -175])
    plt.text(25, -150, f'{50*25}\nm$^2$', color='w')
    # plt.subplot(1,3,3)
    # imshow(normBigDiff, extent=(x[xgoods].min(), x[xgoods].max(), y[ygoods].min(), y[ygoods].max()))
    # plt.title(datetime.utcfromtimestamp(int(timestampB)).isoformat())
    # plt.text(20, 5, f'{losssum:.1f} m$^2$ lost\n{gainsum-losssum:.1f} m$^2$ net change', color='w')
    # plt.plot([20, 45, 45, 20, 20], [-175, -175, -125, -125, -175])
    # plt.text(25, -150, f'{50*25}\nm$^2$', color='w')
    plt.savefig(fildir + 'proc/figs/' + timestampA + '.' + camera + '.snap.gainloss.png', bbox_inches='tight', dpi=300)
    plt.show()

nets = np.array(nets)
filarr = np.array(fils).astype(int)

# %%
plt.plot(filarr[0:-1], np.cumsum(nets))
# %%
plt.figure(figsize=(16,13))

plt.subplot(2,2,1)
# threshA[:300,:] = 0
cntnew = get_contours(threshA, min_area=25)
areasAall = np.sum(get_area(cntnew))
imshow(cv2.drawContours(imageA.copy(), cntnew, -1, (36,255,12), 2))
plt.text(.05, .9, f'{(areasAall):.0f}', transform=plt.gca().transAxes, fontsize=14, color='w', va='top')
plt.title(filA)

plt.subplot(2,2,2)
# threshB[:300,:] = 0
cntnew = get_contours(threshB.copy(), min_area=25)
areasBall = np.sum(get_area(cntnew))
print(np.sum(get_area(losscontours))/areasAall)
imshow(cv2.drawContours(imageB.copy(), cntnew, -1, (36,12,255), 2))
plt.text(.05, .9, f'{(areasBall):.0f}', transform=plt.gca().transAxes, fontsize=14, color='w', va='top')
plt.title(filB)

plt.subplot(2,2,3)
imshow(imgnewA)
plt.text(.05, .9, f'{(areasA):.0f}\nnet ice gained: {100*(areasA-areasB)/areasAall:.3f}%', transform=plt.gca().transAxes, fontsize=14, color='w', va='top')
plt.title('contours show ice gained')
plt.subplot(2,2,4)
imshow(imgnewB)
plt.text(.05, .9, f'{(areasB):.0f}', transform=plt.gca().transAxes, fontsize=14, color='w', va='top')
plt.title('contours show ice lost')
print(areasA - areasB)
plt.subplots_adjust(wspace=.01, hspace=.05)
# plt.colorbar()


# %%
