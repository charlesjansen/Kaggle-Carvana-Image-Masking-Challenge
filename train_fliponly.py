import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

#import os
#os.listdir("F:\DS-main\Kaggle-main\Carvana Image Masking Challenge\input\pseudo_masks")

import params

input_width = params.input_width
input_height = params.input_height
#input_size = params.input_size
epochs = params.max_epochs
batch_size = params.batch_size
model = params.model
test_size =  params.test_size


model.load_weights(filepath='weights/unet_renorm_incep_eco_alternate_adam_RMS_1280_val20_b2.hdf5')


df_train = pd.read_csv('../input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=test_size, random_state=42)

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def train_generator():
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch.values:
                img = cv2.imread('../input/train/{}.jpg'.format(id))
                img = cv2.resize(img, (input_width, input_height))
                mask = cv2.imread('../input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_width, input_height))
                img, mask = randomHorizontalFlip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                img = cv2.imread('../input/train/{}.jpg'.format(id))
                img = cv2.resize(img, (input_width, input_height))
                mask = cv2.imread('../input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_width, input_height))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


callbacks = [EarlyStopping(monitor='val_dice_coeff',
                           patience=10,
                           verbose=1,
                           min_delta=1e-5,
                           mode='max'),
             ReduceLROnPlateau(monitor='val_dice_coeff',
                               factor=0.3,
                               patience=3,
                               verbose=1,
                               epsilon=1e-5,
                               mode='max'),
             ModelCheckpoint(monitor='val_dice_coeff',
                             filepath='weights/unet_renorm_incep_eco_alternate_adam'+str(input_width)+'_val20_b'+str(batch_size)+'.hdf5', 
                             save_best_only=True,
                             save_weights_only=True,
                             verbose = 1,
                             mode='max'),
             TensorBoard(log_dir='logs')]

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
