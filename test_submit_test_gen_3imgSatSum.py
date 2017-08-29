import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import params

nameWeight = "u_renorm_incep_eco_1024_alt_adam_img3sum1024_val20.0_bat2_retrained3"
nameOutput = nameWeight + "_testGen"

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

model.load_weights(filepath='weights/' + nameWeight + '.hdf5')

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


rles = []

#100064 ids_test = 32*53*59
test_splits = 106  # Split test set (number of splits must be multiple of 2-->non! voir au dessus) 16 for 512; 8 initialy
ids_test_splits = np.split(ids_test, indices_or_sections=test_splits)


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

split_count = 0
for ids_test_split in ids_test_splits:
    split_count += 1

    def test_generator():
        while True:
            for start in range(0, len(ids_test_split), batch_size):
                x_batch = []
                end = min(start + batch_size, len(ids_test_split))
                ids_test_split_batch = ids_test_split[start:end]
                for id in ids_test_split_batch.values:
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
                yield x_batch
                
    print("Predicting on {} samples (split {}/{})".format(len(ids_test_split), split_count, test_splits))
    print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
    
    preds = model.predict_generator(generator=test_generator(),
                                steps=np.ceil(float(len(ids_test_split)) / float(batch_size)))
    preds = np.squeeze(preds, axis=3)
    
    print("Generating masks...")
    for pred in tqdm(preds, miniters=100):
        prob = cv2.resize(pred, (orig_width, orig_height))
        mask = prob > threshold
        rle = run_length_encode(mask)
        rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/' + nameOutput + '.csv', index=False)
