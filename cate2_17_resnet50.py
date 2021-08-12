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
from tensorflow.keras.optimizers import SGD
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


print('C2_17_resnet50')
        
model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
for layer in model_resnet.layers[:-12]:
    # 6 - 12 - 18 have been tried. 12 is the best.
    layer.trainable = False
x = model_resnet.output
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
y = Dense(17, activation='softmax', name='img')(x)



final_model = Model(inputs=model_resnet.input,
                    outputs=[y])

opt = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
import tensorflow_addons as tfa

final_model.compile(optimizer=opt,
                    loss='categorical_crossentropy',   
                    #loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[#tensorflow.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                                #tensorflow.keras.metrics.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy'),
                               'accuracy', 'top_k_categorical_accuracy'
                                #tensorflow.keras.metrics.Precision(name='precision', top_k=5),
                                #tensorflow.keras.metrics.Recall(name='recall', top_k=5)
                                #tfa.metrics.F1Score(num_classes=22)
                                ] # default: top-5
                            )

                             
train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()

classes =['Blazer','Blouse','Cardigan','Cutoffs','Hoodie','Jacket','Jeans','Joggers','Leggings','Shorts','Skirt','Sweater','Sweatpants','Sweatshorts','Tank','Tee','Top']

train_iterator = DirectoryIterator("DATA/DeepFashion_crop_cate2/img/train", classes =classes, image_data_generator=train_datagen,  target_size=(150, 150),class_mode='categorical',batch_size =64)
test_iterator = DirectoryIterator("DATA/DeepFashion_crop_cate2/img/val", classes = classes, image_data_generator=test_datagen, target_size=(150, 150),class_mode='categorical',batch_size =64)


lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)
tensorboard = TensorBoard(log_dir='./logs')
early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)
checkpoint = ModelCheckpoint('DAO/output/model_17cate2_resnet50.hdf5')
def custom_generator(iterator):
    while True:
        batch_x, batch_y = iterator.next()
        yield (batch_x, batch_y)

final_model.load_weights('DAO/output/model_17cate2_resnet50.hdf5')
final_model.fit(custom_generator(train_iterator),
                          steps_per_epoch=26222//64,
                          epochs=200, validation_data=custom_generator(test_iterator),
                          validation_steps=6555//64,
                          verbose=1,
                          shuffle=True, 
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard]
                         )
final_model.load_weights('DAO/output/model_17cate2_resnet50.hdf5')
final_model.save('DAO/output/model_17cate2_resnet50.h5')