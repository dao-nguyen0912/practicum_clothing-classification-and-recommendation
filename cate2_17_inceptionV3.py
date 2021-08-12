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
base_path = 'DATA/DeepFashion_crop_cate2/'
import cv2
import numpy as np
from PIL import Image
import random
import PIL.Image
import tensorflow
from tensorflow.keras.preprocessing import image
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
from tensorflow.keras.preprocessing import image,image_dataset_from_directory
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
x = Dense(17, activation='softmax')(x)

model = Model(base_model.input, x)

model.compile(optimizer = RMSprop(learning_rate=0.0001), loss = 'categorical_crossentropy',metrics = ['accuracy', 'top_k_categorical_accuracy'])


        
                            
train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()

classes =['Blazer','Blouse','Cardigan','Cutoffs','Hoodie','Jacket','Jeans','Joggers','Leggings','Shorts','Skirt','Sweater','Sweatpants','Sweatshorts','Tank','Tee','Top']

train_iterator = train_datagen.flow_from_directory("DATA/DeepFashion_crop_cate2/img/train", classes =classes,  target_size=(150, 150),class_mode='categorical',batch_size =64)
test_iterator = test_datagen.flow_from_directory("DATA/DeepFashion_crop_cate2/img/val", classes = classes, target_size=(150, 150),class_mode='categorical',batch_size =64)


lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)
tensorboard = TensorBoard(log_dir='./logs')
early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)
checkpoint = ModelCheckpoint('DAO/output/model_17cate2_inceptionV3.hdf5')


#model.load_weights('DAO/output/model_22cate2_crop_iterator.h5')
model.fit(train_iterator,
                          steps_per_epoch=26222//64,
                          epochs=200, validation_data=test_iterator,
                          validation_steps=6555//64,
                          verbose=1,
                          shuffle=True, 
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard]
                         )
model.load_weights('DAO/output/model_17cate2_inceptionV3.hdf5')
model.save('DAO/output/model_17cate2_inceptionV3.h5')