# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 19:41:24 2017

@author: Charles
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from PIL import Image


img = cv2.imread('../input/train/0cdf5b5d0ce1_04.jpg')
assert img is not None, "Failed to read image"
plt.imshow(img)
img = cv2.resize(img, (1024, 1024))
plt.imshow(img)
             

mask = cv2.imread('../input/train_masks/0cdf5b5d0ce1_04_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
plt.imshow(mask)
mask = cv2.resize(mask, (1024, 1024))
plt.imshow(mask)
plt.imshow(mask)
mask = np.expand_dims(mask, axis=2)

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image



img = cv2.imread('../input/train/0cdf5b5d0ce1_04.jpg')

img = cv2.resize(img, (1024, 1024))
plt.imshow(img)

img = randomHueSaturationValue(img,
                               hue_shift_limit=(-50, 50),
                               sat_shift_limit=(-5, 5),
                               val_shift_limit=(-15, 15))

img2 = randomHueSaturationValue(img,
                               hue_shift_limit=(-50, 50),
                               sat_shift_limit=(-5, 5),
                               val_shift_limit=(-15, 15))

img3 = randomHueSaturationValue(img,
                               hue_shift_limit=(-50, 50),
                               sat_shift_limit=(-5, 5),
                               val_shift_limit=(-15, 15))
img4 = randomHueSaturationValue(img,
                               hue_shift_limit=(-50, 50),
                               sat_shift_limit=(-5, 5),
                               val_shift_limit=(-15, 15))
img5 = randomHueSaturationValue(img,
                               hue_shift_limit=(-50, 50),
                               sat_shift_limit=(-5, 5),
                               val_shift_limit=(-15, 15))
img6 = randomHueSaturationValue(img,
                               hue_shift_limit=(-50, 50),
                               sat_shift_limit=(-5, 5),
                               val_shift_limit=(-15, 15))

#plt.imshow(img3)

test =img3+img2+img+img4

#test = test/3
#test =cv2.cvtColor(test, cv2.COLOR_BayerRG2GRAY)

plt.imshow(test)







































