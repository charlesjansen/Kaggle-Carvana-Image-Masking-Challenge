import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import img_as_ubyte


import params

nameWeightAndOutput = "u_renorm_incep_eco_1024_alt_adam_img3sum1024_val20.0_bat2_retrained3"

input_width = params.input_width
input_height = params.input_height
#input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model

df_test = pd.read_csv('../input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

model.load_weights(filepath='weights/' + nameWeightAndOutput + '.hdf5')

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

"""
for _id in tqdm(ids_test.values):
    img = cv2.imread('../input/layer1Pred/1280/{}.png'.format(_id))
    img = cv2.resize(img, (128, 128))
    cv2.imwrite('../input/layer1Pred/1280to128/{}.png'.format(_id), img_as_ubyte(img))
"""

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

  
rles = []
predsListMask = []
print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
for start in tqdm(range(0, len(ids_test), batch_size)):
    x_batch = []
    end = min(start + batch_size, len(ids_test))
    ids_test_batch = ids_test[start:end]
    for id in ids_test_batch.values:
        img = cv2.imread('../input/test/{}.jpg'.format(id))
        img = cv2.resize(img, (input_width, input_height))
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
        img = img+img2+img3+img4
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    for pred in preds:
        prob = cv2.resize(pred, (orig_width, orig_height))
        mask = prob > threshold
        #predsListMask.append(mask)
        cv2.imwrite('../input/layer1Pred/1024NSAT/{}.png'.format(id), img_as_ubyte(pred))
        #cv2.imwrite('../input/layer1Prob/{}.png'.format(id), img_as_ubyte(prob))
        #cv2.imwrite('../input/pseudo_masks/{}_mask.png'.format(id), img_as_ubyte(mask))
        rle = run_length_encode(mask)
        rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/' + nameWeightAndOutput + '.csv', index=False)
#df.to_csv('submit/' + nameWeightAndOutput + '.csv.gz', index=False, compression='gzip')
