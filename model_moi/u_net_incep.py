from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.losses import binary_crossentropy
import keras.backend as K
from batch_renorm import BatchRenormalization



def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))

def conv2D_BatchRenorm_Relu(inputs, size):
    down = Conv2D(size, (3, 3), padding='same')(inputs)
    down = BatchRenormalization()(down)
    down = Activation('relu')(down)
    return down
	
def conv2D_BatchRenorm_Relu_twice(inputs, size):
    down = conv2D_BatchRenorm_Relu(inputs, size)
    down = conv2D_BatchRenorm_Relu(down, size)
    return down
	
def conv2D_BatchRenorm_Relu_three(inputs, size):
    down = conv2D_BatchRenorm_Relu(inputs, size)
    down = conv2D_BatchRenorm_Relu(down, size)
    down = conv2D_BatchRenorm_Relu(down, size)
    return down

def conv2D_Relu_BatchRenorm(inputs, size):
    down = Conv2D(size, (3, 3), padding='same')(inputs)
    down = Activation('relu')(down)
    down = BatchRenormalization()(down)
    return down

def conv2D_Relu_BatchRenorm_twice(inputs, size):
    down = conv2D_Relu_BatchRenorm(inputs, size)
    down = conv2D_Relu_BatchRenorm(down, size)
    return down
	
def conv2D_Relu_BatchRenorm_three(inputs, size):
    down = conv2D_Relu_BatchRenorm(inputs, size)
    down = conv2D_Relu_BatchRenorm(down, size)
    down = conv2D_Relu_BatchRenorm(down, size)
    return down

def conv2D_inception_BatchRenorm_Relu(inputs, size):
    tower_0 = Conv2D(size, (1, 1), padding='same')(inputs)
    tower_0 = BatchRenormalization()(tower_0)
    tower_0 = Activation('relu')(tower_0)
    
    tower_1 = Conv2D(size, (1, 1), padding='same')(inputs)
    tower_1 = BatchRenormalization()(tower_1)
    tower_1 = Activation('relu')(tower_1)
    tower_1 = Conv2D(size, (3, 3), padding='same')(tower_1)
    tower_1 = BatchRenormalization()(tower_1)
    tower_1 = Activation('relu')(tower_1)
    
    tower_2 = Conv2D(size, (1, 1), padding='same')(inputs)
    tower_2 = BatchRenormalization()(tower_2)
    tower_2 = Activation('relu')(tower_2)
    tower_2 = Conv2D(size, (5, 5), padding='same')(tower_2)
    tower_2 = BatchRenormalization()(tower_2)
    tower_2 = Activation('relu')(tower_2)
    
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    tower_3 = Conv2D(size, (1, 1), padding='same')(tower_3)
    tower_3 = BatchRenormalization()(tower_3)
    tower_3 = Activation('relu')(tower_3)
    
    output = concatenate([tower_0, tower_1, tower_2, tower_3], axis=3)
    return output

def conv2D_inception_Relu_BatchRenormUnited(inputs, size):
    tower_0 = Conv2D(size, (1, 1), padding='same')(inputs)
    tower_0 = Activation('relu')(tower_0)
    
    tower_1 = Conv2D(size, (1, 1), padding='same')(inputs)
    tower_1 = BatchRenormalization()(tower_1)
    tower_1 = Activation('relu')(tower_1)
    tower_1 = Conv2D(size, (3, 3), padding='same')(tower_1)
    tower_1 = Activation('relu')(tower_1)
    
    tower_2 = Conv2D(size, (1, 1), padding='same')(inputs)
    tower_2 = BatchRenormalization()(tower_2)
    tower_2 = Activation('relu')(tower_2)
    tower_2 = Conv2D(size, (5, 5), padding='same')(tower_2)
    tower_2 = Activation('relu')(tower_2)
    
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    tower_3 = Conv2D(size, (1, 1), padding='same')(tower_3)
    tower_3 = Activation('relu')(tower_3)
    
    merged = concatenate([tower_0, tower_1, tower_2, tower_3], axis=3)
    merged = BatchRenormalization()(merged)
    merged = Dropout(0.25)(merged)

    return merged


def conv2D_inception_eco_BatchRenorm_Relu(inputs, size):#using the same conv 1x1
    conv1x1 = Conv2D(size, (1, 1), padding='same')(inputs)
    conv1x1 = BatchRenormalization()(conv1x1)
    conv1x1 = Activation('relu')(conv1x1)
    
    tower_1 = Conv2D(size, (3, 3), padding='same')(conv1x1)
    tower_1 = BatchRenormalization()(tower_1)
    tower_1 = Activation('relu')(tower_1)
    
    tower_2 = Conv2D(size, (5, 5), padding='same')(conv1x1)
    tower_2 = BatchRenormalization()(tower_2)
    tower_2 = Activation('relu')(tower_2)
    
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    tower_3 = Conv2D(size, (1, 1), padding='same')(tower_3)
    tower_3 = BatchRenormalization()(tower_3)
    tower_3 = Activation('relu')(tower_3)
    
    output = concatenate([conv1x1, tower_1, tower_2, tower_3], axis=3)
    return output

def conv2D_inception_eco_Relu_BatchRenorm(inputs, size):#using the same conv 1x1
    conv1x1 = Conv2D(size, (1, 1), padding='same')(inputs)
    conv1x1 = Activation('relu')(conv1x1)
    conv1x1 = BatchRenormalization()(conv1x1)
    
    tower_1 = Conv2D(size, (3, 3), padding='same')(conv1x1)
    tower_1 = Activation('relu')(tower_1)
    tower_1 = BatchRenormalization()(tower_1)
    
    tower_2 = Conv2D(size, (5, 5), padding='same')(conv1x1)
    tower_2 = Activation('relu')(tower_2)
    tower_2 = BatchRenormalization()(tower_2)
    
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    tower_3 = Conv2D(size, (1, 1), padding='same')(tower_3)
    tower_3 = Activation('relu')(tower_3)
    tower_3 = BatchRenormalization()(tower_3)
    
    output = concatenate([conv1x1, tower_1, tower_2, tower_3], axis=3)
    return output

def get_wnet_renorm_incep_128(input_shape=(128, 128, 3),
                 num_classes=1):
    # 128			 
    inputs = Input(shape=input_shape)
    down1 = conv2D_BatchRenorm_Relu_twice(inputs, 64)
    # 64
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    down2 = conv2D_inception_BatchRenorm_Relu(down1_pool, 128)
    # 32
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    down3 = conv2D_inception_BatchRenorm_Relu(down2_pool, 256)
    # 16
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    down4 = conv2D_inception_BatchRenorm_Relu(down3_pool, 512)
    # 8	 # center
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    center = conv2D_inception_BatchRenorm_Relu(down4_pool, 1024)
    # 16
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = conv2D_inception_BatchRenorm_Relu(up4, 512)
    # 32
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = conv2D_inception_BatchRenorm_Relu(up3, 256)
    # 64
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = conv2D_inception_BatchRenorm_Relu(up2, 128)
    # 128
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = conv2D_inception_BatchRenorm_Relu(up1, 64)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    return model

	
def get_unet_renorm_128(input_shape=(128, 128, 3),
                 num_classes=1):
    # 128			 
    inputs = Input(shape=input_shape)
    down1 = conv2D_BatchRenorm_Relu_twice(inputs, 64)
    # 64
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    down2 = conv2D_BatchRenorm_Relu_twice(down1_pool, 128)
    # 32
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    down3 = conv2D_BatchRenorm_Relu_twice(down2_pool, 256)
    # 16
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    down4 = conv2D_BatchRenorm_Relu_twice(down3_pool, 512)
    # 8	 # center
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    center = conv2D_BatchRenorm_Relu_twice(down4_pool, 1024)
    # 16
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = conv2D_BatchRenorm_Relu_three(up4, 512)
    # 32
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = conv2D_BatchRenorm_Relu_three(up3, 256)
    # 64
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = conv2D_BatchRenorm_Relu_three(up2, 128)
    # 128
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = conv2D_BatchRenorm_Relu_three(up1, 64)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    return model




def get_unet_512(input_shape=(512, 512, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])

    return model


def get_unet_1024(input_shape=(1024, 1024, 3),
                  num_classes=1):
    inputs = Input(shape=input_shape)
    # 1024

    down0b = Conv2D(8, (3, 3), padding='same')(inputs)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b = Conv2D(8, (3, 3), padding='same')(down0b)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(down0b_pool)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    # 1024

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])

    return model

def get_unet_renorm_incep_1024(input_shape=(1024, 1024, 3),
                 num_classes=1):
    # 1024			 
    inputs = Input(shape=input_shape)
    down0b = conv2D_BatchRenorm_Relu_twice(inputs, 8)
    # 512
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    down0a = conv2D_inception_BatchRenorm_Relu(down0b_pool, 16)
    # 256
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    down0 = conv2D_inception_BatchRenorm_Relu(down0a_pool, 32)
    # 128
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    down1 = conv2D_inception_BatchRenorm_Relu(down0_pool, 64)
    # 64
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    down2 = conv2D_inception_BatchRenorm_Relu(down1_pool, 128)
    # 32
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    down3 = conv2D_inception_BatchRenorm_Relu(down2_pool, 256)
    # 16
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    down4 = conv2D_inception_BatchRenorm_Relu(down3_pool, 512)
    # 8	 # center
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    center = conv2D_inception_BatchRenorm_Relu(down4_pool, 1024)
    # 16
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = conv2D_inception_BatchRenorm_Relu(up4, 512)
    # 32
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = conv2D_inception_BatchRenorm_Relu(up3, 256)
    # 64
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = conv2D_inception_BatchRenorm_Relu(up2, 128)
    # 128
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = conv2D_inception_BatchRenorm_Relu(up1, 64)
    # 256
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = conv2D_inception_BatchRenorm_Relu(up0, 32)
    # 512
    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = conv2D_inception_BatchRenorm_Relu(up0a, 16)
    # 1024
    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = conv2D_inception_BatchRenorm_Relu(up0b, 8)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    return model

def get_unet_renorm_incep_eco_1024(input_shape=(1024, 1024, 3),
                 num_classes=1):
    # 1024			 
    inputs = Input(shape=input_shape)
    down0b = conv2D_BatchRenorm_Relu_twice(inputs, 8)
    # 512
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    down0a = conv2D_inception_eco_BatchRenorm_Relu(down0b_pool, 16)
    # 256
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    down0 = conv2D_inception_eco_BatchRenorm_Relu(down0a_pool, 32)
    # 128
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    down1 = conv2D_inception_eco_BatchRenorm_Relu(down0_pool, 64)
    # 64
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    down2 = conv2D_inception_eco_BatchRenorm_Relu(down1_pool, 128)
    # 32
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    down3 = conv2D_inception_eco_BatchRenorm_Relu(down2_pool, 256)
    # 16
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    down4 = conv2D_inception_eco_BatchRenorm_Relu(down3_pool, 512)
    # 8	 # center
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    center = conv2D_inception_eco_BatchRenorm_Relu(down4_pool, 1024)
    # 16
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = conv2D_inception_eco_BatchRenorm_Relu(up4, 512)
    # 32
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = conv2D_inception_eco_BatchRenorm_Relu(up3, 256)
    # 64
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = conv2D_inception_eco_BatchRenorm_Relu(up2, 128)
    # 128
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = conv2D_inception_eco_BatchRenorm_Relu(up1, 64)
    # 256
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = conv2D_inception_eco_BatchRenorm_Relu(up0, 32)
    # 512
    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = conv2D_inception_eco_BatchRenorm_Relu(up0a, 16)
    # 1024
    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = conv2D_inception_eco_BatchRenorm_Relu(up0b, 8)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=Adam(lr=0.0001), loss=bce_dice_loss, metrics=[dice_loss])    
    
    return model

def get_unet_renorm_incep_eco_1024_alternate_adam(input_shape=(1024, 1024, 3),#batch2
                 num_classes=1):
    # 1024			 
    inputs = Input(shape=input_shape)
    down0b = conv2D_Relu_BatchRenorm_twice(inputs, 8)
    # 512
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    down0a = conv2D_inception_eco_Relu_BatchRenorm(down0b_pool, 16)
    # 256
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    down0 = conv2D_Relu_BatchRenorm_twice(down0a_pool, 32)
    # 128
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    down1 = conv2D_inception_eco_Relu_BatchRenorm(down0_pool, 64)
    # 64
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    down2 = conv2D_Relu_BatchRenorm_twice(down1_pool, 128)
    # 32
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    down3 = conv2D_inception_eco_Relu_BatchRenorm(down2_pool, 256)
    # 16
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    down4 = conv2D_Relu_BatchRenorm_twice(down3_pool, 512)
    # 8	 # center
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    center = conv2D_inception_eco_Relu_BatchRenorm(down4_pool, 1024)
    # 16
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = conv2D_Relu_BatchRenorm_three(up4, 512)
    # 32
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = conv2D_inception_eco_Relu_BatchRenorm(up3, 256)
    # 64
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = conv2D_Relu_BatchRenorm_three(up2, 128)
    # 128
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = conv2D_inception_eco_Relu_BatchRenorm(up1, 64)
    # 256
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = conv2D_Relu_BatchRenorm_three(up0, 32)
    # 512
    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = conv2D_inception_eco_Relu_BatchRenorm(up0a, 16)
    # 1024
    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = conv2D_Relu_BatchRenorm_three(up0b, 8)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=Adam(lr=0.000034), loss=bce_dice_loss, metrics=[dice_loss])    
    
    return model

def get_unet_renorm_1024_alternateEnd_adam(input_shape=(1024, 1024, 3),
                 num_classes=1):
    # 1024			 
    inputs = Input(shape=input_shape)
    down0b = conv2D_Relu_BatchRenorm_twice(inputs, 8)
    # 512
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    down0a = conv2D_Relu_BatchRenorm_twice(down0b_pool, 16)
    # 256
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    down0 = conv2D_Relu_BatchRenorm_twice(down0a_pool, 32)
    # 128
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    down1 = conv2D_Relu_BatchRenorm_twice(down0_pool, 64)
    # 64
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    down2 = conv2D_Relu_BatchRenorm_twice(down1_pool, 128)
    # 32
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    down3 = conv2D_Relu_BatchRenorm_twice(down2_pool, 256)
    # 16
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    down4 = conv2D_Relu_BatchRenorm_twice(down3_pool, 512)
    # 8	 # center
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    center = conv2D_Relu_BatchRenorm_twice(down4_pool, 1024)
    # 16
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = conv2D_inception_eco_Relu_BatchRenorm(up4, 512)
    # 32
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = conv2D_inception_eco_Relu_BatchRenorm(up3, 256)
    # 64
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = conv2D_inception_eco_Relu_BatchRenorm(up2, 128)
    # 128
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = conv2D_inception_eco_Relu_BatchRenorm(up1, 64)
    # 256
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = conv2D_inception_eco_Relu_BatchRenorm(up0, 32)
    # 512
    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = conv2D_inception_eco_Relu_BatchRenorm(up0a, 16)
    # 1024
    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = conv2D_inception_eco_Relu_BatchRenorm(up0b, 8)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=Adam(lr=0.0001), loss=bce_dice_loss, metrics=[dice_loss])    
    
    return model
def get_unet_renormUnited_incep_layer1incep_1024(input_shape=(1024, 1024, 3),
                 num_classes=1):#too big
    # 1024			 
    inputs = Input(shape=input_shape)
    down0b = conv2D_inception_Relu_BatchRenormUnited(inputs, 8)
    # 512
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    down0a = conv2D_inception_Relu_BatchRenormUnited(down0b_pool, 16)
    # 256
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    down0 = conv2D_inception_Relu_BatchRenormUnited(down0a_pool, 32)
    # 128
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    down1 = conv2D_inception_Relu_BatchRenormUnited(down0_pool, 64)
    # 64
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    down2 = conv2D_inception_Relu_BatchRenormUnited(down1_pool, 128)
    # 32
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    down3 = conv2D_inception_Relu_BatchRenormUnited(down2_pool, 256)
    # 16
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    down4 = conv2D_inception_Relu_BatchRenormUnited(down3_pool, 512)
    # 8	 # center
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    center = conv2D_inception_Relu_BatchRenormUnited(down4_pool, 1024)
    # 16
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = conv2D_inception_Relu_BatchRenormUnited(up4, 512)
    # 32
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = conv2D_inception_Relu_BatchRenormUnited(up3, 256)
    # 64
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = conv2D_inception_Relu_BatchRenormUnited(up2, 128)
    # 128
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = conv2D_inception_Relu_BatchRenormUnited(up1, 64)
    # 256
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = conv2D_inception_Relu_BatchRenormUnited(up0, 32)
    # 512
    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = conv2D_inception_Relu_BatchRenormUnited(up0a, 16)
    # 1024
    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = conv2D_inception_Relu_BatchRenormUnited(up0b, 8)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    return model



def get_unet_renorm_1024(input_shape=(1024, 1024, 3),
                  num_classes=1):
    inputs = Input(shape=input_shape)
    # 1024

    down0b = Conv2D(8, (3, 3), padding='same')(inputs)
    down0b = BatchRenormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b = Conv2D(8, (3, 3), padding='same')(down0b)
    down0b = BatchRenormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(down0b_pool)
    down0a = BatchRenormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchRenormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchRenormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchRenormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchRenormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchRenormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchRenormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchRenormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchRenormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchRenormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchRenormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchRenormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchRenormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchRenormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchRenormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchRenormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchRenormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchRenormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchRenormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchRenormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchRenormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchRenormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchRenormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchRenormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchRenormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchRenormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchRenormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchRenormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchRenormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchRenormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchRenormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchRenormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchRenormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchRenormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchRenormalization()(up0b)
    up0b = Activation('relu')(up0b)
    # 1024

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer = SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])

    return model