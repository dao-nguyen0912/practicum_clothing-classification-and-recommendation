import shutil
import os
import re
import cv2
# will use them for creating custom directory iterator
import numpy as np
import pandas as pd
from six.moves import range
# regular expression for splitting by whitespace
splitter = re.compile("\s+")
base_path = 'DATA/DeepFashion_crop_cate1/'
import cv2
import numpy as np
from PIL import Image
import random
import PIL.Image
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Softmax, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD,RMSprop
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
import threading
from tensorflow.keras.utils import  Sequence
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

print('TB_inceptionV3_sgd')

train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()


# Fixed for Cats & Dogs color images
CHANNELS = 3

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 200
EARLY_STOP_PATIENCE = 20

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 64000//64
STEPS_PER_EPOCH_VALIDATION = 16000//64
# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 64
BATCH_SIZE_VALIDATION = 64

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=20,
                               factor=0.5,
                               verbose=1)
tensorboard = TensorBoard(log_dir='./logs')
cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = 'DAO/output/TB_inceptionV3_sgd.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')


#model.load_weights('/DATA/model_cate1.hdf5')             
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'DATA/DeepFashion_crop_cate1/img/train',
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary')
        
print(train_generator.class_indices)
validation_generator = test_datagen.flow_from_directory(
        'DATA/DeepFashion_crop_cate1/img/val',
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary')
        
from tensorflow.keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(input_shape = (150, 150, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')
for layer in base_model.layers:
    layer.trainable = False
    
# Flatten the output layer to 1 dimension
x = Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = Dropout(0.5)(x)

# Add a final sigmoid layer for classification
x = Dense(1, activation='sigmoid')(x)

model = Model(base_model.input, x)

#model.compile(optimizer = RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy',metrics = ['accuracy'])
model.compile(optimizer = SGD(learning_rate=0.0001), loss = 'binary_crossentropy',metrics = ['accuracy'])


fit_history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs = NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH_VALIDATION,
        callbacks=[lr_reducer,cb_checkpointer, cb_early_stopper,tensorboard]
    )

model.load_weights('DAO/output/TB_inceptionV3_sgd.hdf5')
model.save('DAO/output/TB_inceptionV3_sgd.h5')