%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import cv2
import skimage
import imutils
import matplotlib.pyplot as plt
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

def imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=(10,200,-200,25))
    # plt.xticks([])
    # plt.yticks([])

# fildir = '/Users/dnowacki/Projects/CoastCam/'
fildir = '/Volumes/Backstaff/field/bti/'
camera = 'c1'
fils = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'proc/rect/*' + camera + '*snap*png')]

nets = []
for n in range(len(fils)):
    timestampA = fils[n]
    timestampB = fils[n+1]
    filA = fildir + 'proc/rect/' + timestampA + '.' + camera + '.snap.rect.png'
    filB = fildir + 'proc/rect/' + timestampB + '.' + camera + '.snap.rect.png'

    imageA = cv2.imread(filA)
    imageB = cv2.imread(filB)

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    diff = cv2.subtract(grayB, grayA)

    # plt.figure(figsize=(10,8))
    # plt.subplot(1,3,1)
    # imshow(cv2.absdiff(grayB, grayA))
    # plt.title('absdiff')
    # plt.subplot(1,3,2)
    # imshow(cv2.subtract(grayB, grayA))
    # plt.title('B-A')
    # plt.subplot(1,3,3)
    # imshow(cv2.subtract(grayA, grayB))
    # plt.title('A-B')

    BminusA = cv2.subtract(grayB, grayA)
    AminusB = cv2.subtract(grayA, grayB)

    bigDiff = BminusA.astype(int) - AminusB.astype(int)
    normBigDiff = cv2.normalize(bigDiff.copy(), None, 255, 0, cv2.NORM_MINMAX).astype(np.uint8)

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

    _, bdgain = cv2.threshold(normBigDiff, 175, 255, 0)
    _, bdloss = cv2.threshold(normBigDiff, 81, 255, 0)
    gaincontours = get_contours(bdgain)
    losscontours = get_contours(cv2.bitwise_not(bdloss))
    gainsum = np.sum(get_area(gaincontours)) * .1 * .1
    losssum = np.sum(get_area(losscontours)) * .1 * .1
    nets.append(gainsum-losssum)
    print(timestampA, gainsum-losssum)

    # plt.figure(figsize=(10,8))
    # plt.subplot(1,2,1)
    # imshow(cv2.drawContours(imageA.copy(), gaincontours, -1, (36,255,12), 2))
    # plt.title(datetime.utcfromtimestamp(int(timestampA)).isoformat())
    # plt.text(20, 10, f'{gainsum:.1f} m2 gained', color='w')
    # plt.subplot(1,2,2)
    # imshow(cv2.drawContours(imageB.copy(), losscontours, -1, (36,255,12), 2))
    # plt.title(datetime.utcfromtimestamp(int(timestampB)).isoformat())
    # plt.text(20, 10, f'{losssum:.1f} m2 lost\n{gainsum-losssum:.1f} m2 net change', color='w')
    # plt.plot([20, 45, 45, 20, 20], [-175, -175, -125, -125, -175])
    # plt.text(25, -150, f'{50*25}\nm2', color='w')
    # plt.savefig(fildir + 'proc/figs/' + timestampA + '.' + camera + '.gainloss.png', bbox_inches='tight', dpi=300)
    # plt.show()

nets = np.array(nets)
len(fils)
nets.shape
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
